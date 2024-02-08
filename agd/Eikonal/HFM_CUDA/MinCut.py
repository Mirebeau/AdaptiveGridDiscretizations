import os
import numpy as np
from ... import FiniteDifferences as fd
import time

try:
	import cupy as cp
	from . import cupy_module_helper
	from .cupy_module_helper import SetModuleConstant
	from . import Eigh
except ImportError:
	pass # Allow using the CPU implementation on a machine without cuda

"""
This file provides a GPU solver of the mincut optimization problem, which reads
minimize int_Omega F(grad ϕ) + g ϕ subject to -1 <= ϕ <= 1,
and arises in image segmentation methods.
This problem is dual to 
minimize int_Omega |div(eta) - g| subject to F^*(eta) <=1,
which is solved numerically. 
"""

def stag_grids(shape,corners,sparse=False,xp=np):
	"""
	Generates the grid, staggered grid, and boundary conditions, as suitable for the mincut
	problem, for a rectangular domain. Grids use 'ij' indexing.
	Input : 
	 - shape (tuple) : the domain dimensions (n1,...,nd)
	 - corners (array of shape (2,d)) : the extreme points of the domain
	 - sparse (bool) : wether to generate a dense or a sparse grid 
	Output : Xm,Xϕ,dx,weights (last element only if retweights=True)
	 - Xs : standard grid coordinates (for the scalar potential)
	 - Xv : staggered grid coordinates, at the center of each cell (for the gradient)
	 - dx : the discretization scale
	"""
	if isinstance(shape,int): shape = (shape,); corners = tuple((c,) for c in corners)
	float_t = np.float64 if xp is np else np.float32
	corners = xp.asarray(corners,dtype=float_t)
	assert corners.shape == (2,len(shape))
	bot,top = corners
	dx = (top-bot)/(xp.asarray(shape,dtype=float_t)-1)
	Xs = tuple(boti+dxi*xp.arange(si,dtype=float_t) for (boti,dxi,si) in zip(bot,dx,shape))
	Xv = tuple(dxi/2.+axi[:-1] for (dxi,axi) in zip(dx,Xs))

	def make_grid(X):
		if sparse: return tuple(axi.reshape((1,)*i+(-1,)+(1,)*(bc.ndim-i-1)) for i,axi in enumerate(X))
		else: return xp.asarray(xp.meshgrid(*X,indexing='ij'),dtype=float_t)

	return make_grid(Xs),make_grid(Xv),dx

def _preproc_metric(metric,use_numpy_eigh):
	from ... import Metrics
	if isinstance(metric,Metrics.Isotropic): return metric.cost[None]
	elif isinstance(metric,Metrics.Diagonal): return metric.costs
	elif metric.model_HFM().startswith("AsymIso"): return [metric.a[None],metric.w]

	elif isinstance(metric,Metrics.Rander): 
		return _preproc_metric(Metrics.Riemann(metric.m),use_numpy_eigh),metric.w
	elif metric.model_HFM().startswith("AsymRander"): #Not imported by default in Metrics
		assert cp.allclose(metric.v,0)
		return _preproc_metric(Metrics.AsymQuad(metric.m,metric.u),use_numpy_eigh),metric.w

	def eigh(m):
		λ,v = Eigh.eigh(m,quaternion=True,flatsym=True,use_numpy=use_numpy_eigh)
		return [np.moveaxis(λ,-1,0),np.moveaxis(v,-1,0)]

	m = np.concatenate([np.moveaxis(metric.m[i,:(i+1)],0,-1)
		for i in range(metric.vdim)],axis=-1)
	eigm = eigh(m)
	if isinstance(metric,Metrics.Riemann): return eigm
	assert isinstance(metric,Metrics.AsymQuad)
	k=0 # compute m + w w^T
	for i in range(metric.vdim): 
		for j in range(i+1): 
			m[...,k] += metric.w[i]*metric.w[j]
			k+=1
	return eigm+eigh(m)+[metric.w]
	

def mincut(g,metric,dx=None,
	grad="gradb",τ_primal=0.2,ρ_overrelax=1.8,
	maxiter=5000,E_rtol=1e-2, #,rtol=1e-6,
	shape_i=None,use_numpy_eigh=False,verbosity=2):
	"""
	Numerically solves the mincut problem.
	- g (array) : the ground cost functions
	- metric : the geometric metric 
	- grad (optional, string): gradient discretization. Possible values : 
	   - 'gradb' -> upwind
	   - 'gradc' -> centered (accurate but unstable)
	   - 'grad2' -> use both upwind and downwind
	- τ_primal : time step for the primal proximal operator
	"""

	# traits and sizes
	int_t = np.int32
	float_t = np.float32
	shape_s = g.shape # shape for vector fields
	shape_v = tuple(s-1 for s in shape_s)
	vdim = len(shape_s) # Number of space dimensions
	qdim = {1:np.nan, 2:2, 3:4}[vdim] # Data size for a rotation matrix (compact format)
	assert τ_primal>0

	# Prepare the metric
	w_randers = None
	if not isinstance(metric,cp.ndarray):
		assert metric.shape==() or metric.shape==shape_v
		from ... import AutomaticDifferentiation as ad
		metric = ad.cupy_generic.cupy_set(metric,dtype32=True,iterables=type(metric)) 
		if dx is not None:
			dx = cp.asarray(dx,dtype=float_t)
			if   np.ndim(dx)==0: metric = metric.with_speed(dx) 
			elif np.ndim(dx)==1: metric = metric.with_speeds(dx)
		metric = _preproc_metric(metric,use_numpy_eigh)
		if isinstance(metric,tuple): metric,w_randers = metric
		if isinstance(metric,list):  metric = np.concatenate(metric,axis=0)
	
	def asfield(a): return np.moveaxis(np.broadcast_to(a,shape_v+a.shape),-1,0)
	if np.ndim(metric)==1:metric = asfield(metric)
	if np.ndim(w_randers)==1:w_randers = asfield(w_randers)

	assert metric.shape[1:]==shape_v # Metric is provided at cell centers
	geomsize = len(metric)
	metric_type = {1:'iso', (1+vdim):'iso_asym', (vdim+qdim):'riemann', 
	(vdim+qdim+vdim+qdim+vdim):'riemann_asym'}[geomsize]

	if shape_i is None: shape_i = {1:(64,), 2:(8,8), 3:(4,4,4)}[vdim]
	assert len(shape_i)==vdim
	size_i = np.prod(shape_i)

	if vdim==1: grad='gradc' # Only one meaningful discretization in dimension 1
	else: assert grad in ('gradb','gradc','grad2')
	if grad=='grad2':
		graddim = 2*vdim
		gradnorm2 = 2*vdim
		if w_randers is not None: w_randers = np.concatenate([w_randers,w_randers],axis=0)
	else:
		graddim = vdim
		gradnorm2 = 4*vdim # Squared norm of gradient operator

	# Format suitably for the cupy kernel
	g = cp.ascontiguousarray(fd.block_expand(cp.asarray(g,dtype=float_t),
		shape_i,constant_values=np.nan))
	shape_o = g.shape[:vdim]
	metric = cp.ascontiguousarray(fd.block_expand(cp.asarray(metric,dtype=float_t)
		,shape_i,shape_o,constant_values=np.nan))

	# Create the optimization variables
	ϕ = cp.zeros(shape_o+shape_i,dtype=float_t) # primal point
	ϕ_ext = cp.zeros(shape_o+shape_i,dtype=float_t) # extrapolated primal point
	η = cp.zeros((graddim,*shape_o,*shape_i),dtype=float_t) # dual point
	primal_value = cp.zeros(shape_o,dtype=float_t) # primal objective value, by block
	dual_value = cp.zeros(shape_o,dtype=float_t) # dual objective value, by block

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
		'newton_maxiter':7,
		'metric_type_macro':{'iso':1,'iso_asym':2,'riemann':3,'riemann_asym':4}[metric_type],
		'size_bd_e':len(x_top_e),
		'x_top_e':x_top_e,
		'x_bot_e':1+x_bot_e,
		'grad_macro':{'gradb':1,'gradc':2,'grad2':3}[grad],
		'preproc_randers_macro':w_randers is not None,
	}

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits,size_of_shape=True,log2_size=True)

	source += [
	'#include "Kernel_MinCut.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source = "\n".join(source)
#	print(source)
	module = cupy_module_helper.GetModule(source,cuoptions)
	# ------------- cuda module generated (may be reused...) ----------------

	def setcst(*args): cupy_module_helper.SetModuleConstant(module,*args)
	setcst('shape_tot_s',shape_s,int_t)
	setcst('shape_tot_v',shape_v,int_t)
	setcst('shape_o',shape_o,int_t)
	setcst('size_io',g.size,int_t)
	setcst('tau_primal',τ_primal,float_t)
	setcst('tau_dual',1./(gradnorm2*τ_primal),float_t)
	setcst('rho_overrelax',ρ_overrelax,float_t)
	
	primal_step = module.get_function("primal_step")
	dual_step = module.get_function("dual_step")
	size_o = np.prod(shape_o)
	primal_values,dual_values = [],[]

	if w_randers is not None:
		w_randers = cp.ascontiguousarray(
		fd.block_expand(w_randers,shape_i,shape_o,constant_values=np.nan),dtype=float_t)
		assert w_randers.shape == η.shape
		dummy = cp.array([0.],dtype=float_t)
#		print(f"g_gpu={fd.block_squeeze(g,shape_s)[-5:,-5:]}")
#		g.fill(0)
		setcst('preproc_randers',1,int_t)
		primal_step((size_o,),(size_i,),(g,dummy,w_randers,dummy,dummy,dummy))
		setcst('preproc_randers',0,int_t)
#		print(f"g_gpu={fd.block_squeeze(g,shape_s)[-5:,-5:]}")
#		print(np.any(np.isnan(g)))

	# Main loop
	top = time.time()
	for niter in range(maxiter):
#		setcst('niter',niter,int_t)
#		print(f"{niter=}")
#		print(f"{ϕ=}")
#		print(f"η*dx={η.get()*dx}")
		primal_step((size_o,),(size_i,),(ϕ,ϕ_ext,η,g,primal_value,dual_value))
#		print(f"{ϕ=}")
#		print(primal_value)
		dual_step((size_o,),(size_i,),(η,ϕ_ext,metric,primal_value))
#		print(primal_value)

#		print(f"(After dual step η={η.get()}")
#		print(f"(After dual step η*dx={η.get()*dx}")

		# Check objectives and termination

		e_primal =  float(primal_value.sum())
		e_dual   = -float(dual_value.sum())
		primal_values.append(e_primal)
		dual_values.append(e_dual)
		if E_rtol>0 and e_primal-e_dual<E_rtol*np.abs(e_primal): break
	else: 
		if E_rtol>0 and verbosity>=1: 
			print("Exhausted iteration budget without satisfying convergence criterion") 
	if verbosity>=2:
		print(f"GPU primal-dual solver completed {niter+1} steps in {time.time()-top} seconds")

	# Reshape, rescale the results
	ϕ = fd.block_squeeze(ϕ,shape_s)
	η = fd.block_squeeze(η,shape_v)
	if grad=='grad2': η=np.moveaxis(η.reshape((2,vdim)+shape_v),0,1)
	if dx is not None: 
		if np.ndim(dx)==0: η*=dx
		else: 
			for ηi,dxi in zip(η,dx): ηi*=dxi

	return {'ϕ':ϕ,'η':η,'niter':niter+1,
	'primal_values':np.array(primal_values),'dual_values':np.array(dual_values)}