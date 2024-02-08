# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
from ...Eikonal.HFM_CUDA import cupy_module_helper as cmh


#Compile times for the cupy kernel can be a bit long, presumably due to the merge sort.
#Fortunately, this happens only once.

def simplify_ad(x,atol,rtol,blockSize=256):
	"""Calls the GPU implementation of the simplify_ad method"""
	
	# Get the data
	coef,index = map(cp.ascontiguousarray,(x.coef,x.index))
	size_ad = x.size_ad
	if size_ad==0: return
	bound_ad = int(2**np.ceil(np.log2(size_ad)))

	# Set the traits
	int_t = np.int32
	size_t = int_t
	index_t = index.dtype.type
	scalar_t = coef.dtype.type
	tol_macro = atol is not None
	traits = {
		'Int':int_t,
		'IndexT':index_t,
		'SizeT':size_t,
		'Scalar':scalar_t,
		'bound_ad':bound_ad,
		'tol_macro':tol_macro,
	}

	# Setup the cupy kernel
	source = cmh.traits_header(traits) #integral_max=True # needed for fixed length sort
	cuda_rpaths = "cuda","../../Eikonal/HFM_CUDA/cuda"
	cuda_paths = [os.path.join(os.path.dirname(os.path.realpath(__file__)),rpath) for rpath in cuda_rpaths]
	date_modified = max(cmh.getmtime_max(path) for path in cuda_paths)

	source += ['#include "simplify_ad.h"',
	f"// Date cuda code last date_modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_paths[0]}", f"-I {cuda_paths[1]}") 

	source="\n".join(source)
	module = cmh.GetModule(source,cuoptions)
	cmh.SetModuleConstant(module,'size_ad',x.size_ad,int_t)
	cmh.SetModuleConstant(module,'size_tot',x.size,size_t)
	if tol_macro: 
		cmh.SetModuleConstant(module,'atol',atol,scalar_t)
		cmh.SetModuleConstant(module,'rtol',rtol,scalar_t)
	cupy_kernel = module.get_function("simplify_ad")

	# Call the kernel
	gridSize = int(np.ceil(x.size/blockSize))
	new_size_ad = cp.zeros(x.shape,dtype=np.int32)
#	print("i,c",index,coef)
	cupy_kernel((gridSize,),(blockSize,),(index,coef,new_size_ad))
	new_size_ad = np.max(new_size_ad)
	
	x.coef  = coef[...,:new_size_ad]
	x.index = index[...,:new_size_ad]

