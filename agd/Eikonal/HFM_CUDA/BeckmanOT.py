import os
import numpy as np
import time
import cupy as cp

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from ... import FiniteDifferences as fd

"""
This file provides a GPU solver for a very particular instance of the (W1, unbalanced) 
optimal transport problem : minimizing the quantity
int |σ| + (λ/2)|div(σ)+ν-μ|^2 
where μ and ν are (possibly vector valued) measures, and λ is a relaxation parameter.

More generally, this file is a basic GPU implementation of the Chambolle-Pock primal-dual
optimization algorithm, that can be modified to address other optimization problems.
"""

def solve_ot(λ,ξ,dx,relax_norm_constraint=0.01,
	τ_primal=None,ρ_overrelax=1.8, ϕ0=None, σ0=None,
	atol=0,rtol=1e-6,E_rtol=1e-3,maxiter=5000,
	shape_i = None,stop_period=10, verbosity=2):
	"""
	Numerically solves a relaxed Bellman formulation of optimal transport
	- λ (positive) : the relaxation parameter
	- ξ (array) : the difference of measures ν-μ
	- dx (array) : the grid scales (one per dimension) 
	"""

	# traits and sizes
	int_t = np.int32
	float_t = np.float32
	assert np.ndim(dx)==1 # One grid scale per dimension
	vdim = len(dx)
	ξ = cp.asarray(ξ,dtype=float_t)
	shape_s = ξ.shape[-vdim:] # shape for vector fields
	shape_v = tuple(s-1 for s in shape_s)
	multichannel_shape = ξ.shape[:-vdim]
	nchannels = np.prod(multichannel_shape,dtype=int)
	ξ = ξ.reshape((nchannels,*shape_s)) # All channels in first dimension

	gradnorm2 = 4.*np.sum(dx**-2) # Squared norm of the gradient operator
	if τ_primal is None: τ_primal = 5./np.sqrt(gradnorm2)
	assert τ_primal>0
	if shape_i is None: shape_i = {1:(64,), 2:(8,8), 3:(4,4,4)}[vdim]
	assert len(shape_i)==vdim
	size_i = np.prod(shape_i)

	# Format suitably for the cupy kernel
	ξ = cp.ascontiguousarray(fd.block_expand(cp.asarray(ξ,dtype=float_t),shape_i,
		constant_values=np.nan))
	shape_o = ξ.shape[1:1+vdim]
	size_o = np.prod(shape_o)

	if ϕ0 is None: ϕ = cp.zeros((nchannels,)+shape_o+shape_i,dtype=float_t) # primal point
	else: ϕ = cp.ascontiguousarray(fd.block_expand(cp.asarray(ϕ0,dtype=float_t).reshape((
		nchannels,)+shape_s),shape_i,shape_o))

	if σ0 is None:σ=cp.zeros((nchannels,vdim,*shape_o,*shape_i),dtype=float_t)#dual point
	else: σ = cp.ascontiguousarray(fd.block_expand(cp.asarray(σ0,dtype=float_t).reshape((
		nchannels,vdim)+shape_v),shape_i,shape_o))

	ϕ_ext = cp.zeros((nchannels,)+shape_o+shape_i,dtype=float_t) # extrapolated primal point
	primal_value = cp.zeros(shape_o,dtype=float_t) # primal objective value, by block
	dual_value = cp.zeros(shape_o,dtype=float_t) # dual objective value, by block
	stabilized = cp.zeros(shape_o,dtype=np.int8)

	# --------------- cuda header construction and compilation ----------------
	# Generate the load order for the boundary of shared data
	x_top_e = fd.block_neighbors(shape_i,True)
	x_bot_e = fd.block_neighbors(shape_i,False)

	# Generate the kernels
	traits = {
		'ndim_macro':vdim,
		'Int':int_t,
		'Scalar':float_t,
		'shape_i':shape_i,
		'shape_e':tuple(s+1 for s in shape_i),
		'size_bd_e':len(x_top_e),
		'x_top_e':x_top_e,
		'x_bot_e':1+x_bot_e,
		'nchannels':nchannels,
	}

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits,size_of_shape=True,log2_size=True)

	source += [
	'#include "Kernel_BeckmanOT.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") #,"-lineinfo") ,"--device-debug"

	source = "\n".join(source)
	module = cupy_module_helper.GetModule(
		"#define checkstop_macro false\n"+source,cuoptions)
	module_checkstop = cupy_module_helper.GetModule(
		"#define checkstop_macro true\n"+source,cuoptions)
	# -------------- cuda module generated (may be reused...) ----------------

	def setcst(*args): 
		cupy_module_helper.SetModuleConstant(module,*args)
		cupy_module_helper.SetModuleConstant(module_checkstop,*args)
	setcst('shape_tot_s',shape_s,int_t)
	setcst('shape_tot_v',shape_v,int_t)
	setcst('shape_o',shape_o,int_t)
	setcst('size_io',size_i*size_o,int_t)
	setcst('tau_primal',τ_primal,float_t)
	setcst('tau_dual',1./(gradnorm2*τ_primal),float_t)
	setcst('idx',1./dx,float_t)
	setcst('lambda',λ,float_t)
	setcst('ilambda',1./λ,float_t)
	setcst('irelax_norm_constraint',1./relax_norm_constraint,float_t)
	setcst('rho_overrelax',ρ_overrelax,float_t)
	setcst('atol',atol,float_t)
	setcst('rtol',rtol,float_t)


	primal_step = module.get_function("primal_step")
	dual_step = module.get_function("dual_step")
	primal_step_checkstop = module_checkstop.get_function("primal_step")
	dual_step_checkstop = module_checkstop.get_function("dual_step")
	primal_values,dual_values = [],[]

	# Main loop
	for arr in (ξ,ϕ,ϕ_ext,σ,primal_value,dual_value,stabilized): 
		assert arr.flags['C_CONTIGUOUS'] # Just to be sure (Common source of silent bugs.)
	top = time.time()
	for niter in range(maxiter):
		if niter%stop_period!=0:
			primal_step((size_o,),(size_i,),(ϕ,ϕ_ext,σ,ξ))
			dual_step((size_o,),(size_i,),(σ,ϕ_ext))
		else: 
			primal_step_checkstop((size_o,),(size_i,),(ϕ,ϕ_ext,σ,ξ,
				primal_value,dual_value,stabilized))
#			print(f"{niter=},primal={primal_value.sum()},dual={dual_value.sum()}")
			dual_step_checkstop((size_o,),(size_i,),(σ,ϕ_ext,
				primal_value,dual_value,stabilized))
#			print(f"primal={primal_value.sum()},dual={dual_value.sum()}")		
			e_primal =  float(primal_value.sum())
			e_dual   = -float(dual_value.sum())
			primal_values.append(e_primal)
			dual_values.append(e_dual)
			if E_rtol>0 and e_primal-e_dual<E_rtol*np.abs(e_primal): break
			if np.all(stabilized): break
			stabilized.fill(0)
#		if niter==10: print(f"First {niter} iterations took {time.time()-top} seconds")
	else: 
		if E_rtol>0 and verbosity>=1: 
			print("Exhausted iteration budget without satisfying convergence criterion") 
	if verbosity>=2:
		print(f"GPU primal-dual solver completed {niter+1} steps in {time.time()-top} seconds")

	ϕ = ϕ.reshape(multichannel_shape+ϕ.shape[1:])
	ϕ = fd.block_squeeze(ϕ,shape_s)
	σ = np.moveaxis(σ,0,1).reshape((vdim,)+multichannel_shape+σ.shape[2:])
	σ = fd.block_squeeze(σ,shape_v)

	return {'ϕ':ϕ,'σ':σ, #'ϕ_ext':fd.block_squeeze(ϕ_ext,shape_s),
	'stabilized':stabilized,'niter':niter+1,
	'stopping_criterion':'stabilized' if np.all(stabilized) else 'gap',
	'primal_values':np.array(primal_values),'dual_values':np.array(dual_values)}