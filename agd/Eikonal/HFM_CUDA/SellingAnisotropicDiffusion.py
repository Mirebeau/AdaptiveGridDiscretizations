# Copyright 2022 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
import numbers

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from ... import AutomaticDifferentiation as ad

def anisotropic_diffusion_scheme(D,dx=1.,ced=None,periodic=False,block_size=256,traits=None):
	"""
	First stage of solving D_t u = div( D grad u ), with Neumann b.c.
	Input : 
	- D : positive definite diffusion tensors. Format : (n1,...,nd,d*(d+1)/2)
	- ced (optional) : a dictionary with keys
	  	- alpha, gamma (real) : weickerts coherence enhancing diffusion parameters
	  	- cond_max,cond_amplification_threshold (real, optional) : additional parameters
	  	- retD (bool, optional) : wether to return the diffusion tensors 

	  	If ced is present, then D is regarded as a structure tensor, and modified 
		according to Weickert's coherence enhancing diffusion principle
	- dx (optional): grid scale
	
	Output : 
	- dt_max : the largest stable time step for the scheme
	- scheme_data : data to be used in anisotropic_diffusion_steps 
	- D (optional) : the diffusion tensors, returned if ced['retD']=true 
	"""
	if D.shape[0]==D.shape[1]==D.ndim-2: # Also allow common format (d,d,n1,...,nd)  
		from ... import Metrics
		D = np.moveaxis(Metrics.misc.flatten_symmetric_matrix(D),0,-1)
	D = cp.ascontiguousarray(D)
	shape = D.shape[:-1]
	ndim = len(shape)
	size = np.prod(shape)
	symdim = (ndim*(ndim+1))//2
	decompdim = symdim # changes in dimension 4
	assert ndim in (1,2,3) and D.shape[-1]==symdim
	int_t = np.int32
	float_t = np.float32
	assert D.dtype==float_t

	traits_default = {
		'ndim_macro':ndim,
		'Int':int_t,
		'Scalar':float_t,
		'prev_coef':1, # Set prev_coef = 0 for the pure linear operator
		'fourth_order_macro':False,
	}
	if traits is not None: traits_default.update(traits)
	traits = traits_default

	if isinstance(periodic,numbers.Number): periodic = (periodic,)*ndim
	if any(periodic): 
		traits['periodic_macro'] = 1
		traits['periodic_axes'] = periodic
	if ced is not None: 
		assert ndim>1
		ced = ced.copy()
		traits['ced_macro']=1
		traits['retD_macro'] = ced.pop('retD',False)
	retD = (np.empty_like(D),) if traits.get('retD_macro',False) else tuple()

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits,size_of_shape=True)

	source += [
	'#include "Kernel_SellingAnisotropicDiffusion.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)
	SetModuleConstant(module,'shape_tot',shape,int_t)
	SetModuleConstant(module,'size_tot',size,int_t)
	if isinstance(dx,numbers.Number): dx = (dx,)*ndim
	SetModuleConstant(module,'dx',dx,float_t)
	if ced is not None:
		SetModuleConstant(module,"ced_alpha",ced.pop("α"),float_t)
		SetModuleConstant(module,"ced_gamma",ced.pop("γ"),float_t)
		SetModuleConstant(module,"ced_cond_max",ced.pop("cond_max",100),float_t)
		SetModuleConstant(module,"ced_cond_amplification_threshold",
			ced.pop("cond_amplification_threshold",2),float_t)
		if ced: print(f"Warning : unused ced keys {list(ced.keys())}")

	cupy_kernel = module.get_function("anisotropic_diffusion_scheme")
	wdiag  = np.full_like(D,0.,shape=shape)
	wneigh = np.empty_like(D,shape=shape+(decompdim,));
	nneigh = 4 if traits['fourth_order_macro'] else 2
	ineigh = np.full_like(D,-1,shape=shape+(decompdim,nneigh),dtype=int_t);

	grid_size = int(np.ceil(size/block_size))
	cupy_kernel((grid_size,),(block_size,),(D,wdiag,wneigh,ineigh)+retD)

	dt_max = 2./np.max(wdiag)
	return (dt_max,(wdiag,wneigh,ineigh,module))+retD

def anisotropic_diffusion_steps(u,dt,ndt,scheme_data,
	block_size=1024,overwrite_u=False,out=None):
	"""
	Solving D_t u = div( D grad u ), with Neumann b.c.
	Input : 
	- u : initial data
	- dt : time step
	- ndt : number of time steps
	- scheme_data : output of anisotropic_diffusion_scheme 
	"""
	if not overwrite_u: u=u.copy()
	u = cp.ascontiguousarray(u)
	if out is None: out = np.empty_like(u)
	wdiag,wneigh,ineigh,module = scheme_data
	float_t = wneigh.dtype 
	assert u.dtype==float_t
	assert u.shape==scheme_data[0].shape
	assert out.shape==u.shape and out.dtype==u.dtype 
	assert u.flags['C_CONTIGUOUS'] and out.flags['C_CONTIGUOUS']
	SetModuleConstant(module,'dt',dt,float_t)
	cupy_kernel = module.get_function("anisotropic_diffusion_step")
	uold,unew = u,out
	grid_size = int(np.ceil(u.size/block_size))
	for i in range(ndt):
		unew[:]=0.
		cupy_kernel((grid_size,),(block_size,),(uold,unew,wdiag,wneigh,ineigh))
		uold,unew = unew,uold
	if out is not uold: out[:]=uold # unew and uold were just swapped
	return out
