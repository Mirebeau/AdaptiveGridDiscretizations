# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
import numbers

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from ... import AutomaticDifferentiation as ad

def dtype_sup(dtype):
	dtype=np.dtype(dtype)
	if dtype.kind in ('i','u'): return np.iinfo(dtype).max
	elif dtype.kind=='f': return dtype(np.inf)
	else: raise ValueError("Unsupported dtype")
def dtype_inf(dtype):
	dtype=np.dtype(dtype)
	if dtype.kind in ('i','u'): return np.iinfo(dtype).min
	elif dtype.kind=='f': return dtype(-np.inf)
	else: raise ValueError("Unsupported dtype")

def distance_kernel(radius,ndim,dtype=np.float32,ord=2,mult=1):
	rg = range(-radius,radius+1)
	axes = (rg,)*ndim
	X = np.meshgrid(*axes)
	dist = mult*ad.Optimization.norm(X,axis=0,ord=ord)
	if np.dtype(dtype).kind in ('i','u'): dist = np.round(dist)
	return dist.astype(dtype)

def inf_convolution(arr,kernel,out=None,niter=1,periodic=False,
	upper_saturation=None, lower_saturation=None, mix_is_min=True,
	overwrite=False,block_size=1024):
	"""
	Perform an inf convolution of an input with a given kernel, on the GPU.
	- arr : the input array
	- kernel : the convolution kernel. A centered kernel will be used.
	- niter (optional) : number of iterations of the convolution.
	- periodic (optional, bool or tuple of bool): axes using periodic boundary conditions.
	- mix_is_min : if false, use sup_convolution instead
	"""
	if niter>=2 and not overwrite: arr=arr.copy()
	arr = cp.ascontiguousarray(arr)

	conv_t = arr.dtype.type
	int_t = np.int32

	traits = {
		'T':conv_t,
		'shape_c':kernel.shape,
		'mix_is_min_macro':int(mix_is_min),
		'ndim':arr.ndim,
		}

	if upper_saturation is None: upper_saturation = dtype_sup(conv_t)
	else: traits['upper_saturation_macro']=1
	traits['T_Sup']=(upper_saturation,conv_t)
	if lower_saturation is None: lower_saturation = dtype_inf(conv_t)
	else: traits['lower_saturation_macro']=1
	traits['T_Inf']=(lower_saturation,conv_t)

	if isinstance(periodic,numbers.Number): periodic = (periodic,)*arr.ndim
	if any(periodic): 
		traits['periodic_macro'] = 1
		traits['periodic_axes'] = periodic

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits,size_of_shape=True)

	source += [
	'#include "Kernel_InfConvolution.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source="\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)
	SetModuleConstant(module,'kernel_c',kernel,conv_t)
	SetModuleConstant(module,'shape_tot',arr.shape,int_t)
	SetModuleConstant(module,'size_tot',arr.size,int_t)

	cupy_kernel = module.get_function("InfConvolution")

	if out is None: out = np.empty_like(arr)
	else: assert out.dtype==arr.dtype and out.size==arr.size and out.flags['C_CONTIGUOUS']
	grid_size = int(np.ceil(arr.size/block_size))

	for i in range(niter):
		cupy_kernel((grid_size,),(block_size,),(arr,out))
		arr,out = out,arr

	return arr
		





