# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os

from . import cupy_module_helper
from ...Metrics.misc import flatten_symmetric_matrix,expand_symmetric_matrix
from ... import LinearParallel as lp

def Reconstruct(coefs,offsets):
     return lp.mult(coefs,lp.outer_self(offsets)).sum(2)

def VoronoiDecomposition(m,offset_t=np.int32,
	flattened=False,blockDim=None,traits=None,steps="Both",retry64_tol=2e-5):
	"""
	Returns the Voronoi decomposition of a family of quadratic forms. 
	- m : the (symmetric) matrices of the quadratic forms to decompose.
	- offset_t : the type of offsets to be returned. 
	- flattened : wether the input matrices are flattened
	- retry64_tol (optional) : retries decomposition using 64bit floats if this error 
	 is exceeded relative to matrix trace. (Set retry64_tol = 0 to use double precision.)
	"""

	# Prepare the inputs and outputs
	if flattened: m_exp = None
	else: m_exp = m; m = flatten_symmetric_matrix(m)
	symdim = len(m)
	ndim = int(np.sqrt(2*symdim))
	assert symdim==ndim*(ndim+1)/2
	shape = m.shape[1:]
	size = m.size/symdim

	if not (2<=ndim and ndim<=6): 
		raise ValueError(f"Voronoi decomposition not implemented in dimension {ndim}")
	decompdim = [0,1,3,6,12,15,21][ndim]

	float_t = np.float32 if retry64_tol else np.float64
	tol = {np.float32:1e-5, np.float64:2e-14}[float_t]
	int_t = np.int32
	m = cp.ascontiguousarray(m,dtype=float_t)
	weights = cp.empty((decompdim,*shape),dtype=float_t)
	offsets = cp.empty((ndim,decompdim,*shape),dtype=offset_t)

	weights,offsets=map(cp.ascontiguousarray,(weights,offsets))

	# Setup the GPU kernel
	if traits is None: traits = {}
	traits.update({
		'ndim_macro':ndim,
		'OffsetT':offset_t,
		'Scalar':float_t,
		'Int':np.int32,
		'SIMPLEX_TOL_macro':tol,
		})

	source = cupy_module_helper.traits_header(traits)
	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)

	source += [
	'#include "VoronoiDecomposition.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source="\n".join(source)
	
	module = cupy_module_helper.GetModule(source,cuoptions)
	cupy_module_helper.SetModuleConstant(module,'size_tot',size,int_t)

	if blockDim is None: blockDim = [128,128,128,128,128,32,32][ndim]
	gridDim = int(np.ceil(size/blockDim))

	def retry64():
		if retry64_tol==0 or ndim!=6: return
		nonlocal m_exp
		if m_exp is None: m_exp = expand_symmetric_matrix(m)
		mrec = Reconstruct(weights,offsets)
		error = np.sum(np.abs(m_exp-mrec),axis=(0,1))
		bad = np.logical_not(error < (retry64_tol * lp.trace(m_exp)))
		if np.any(bad):
#				print(f"nrecomputed {np.sum(bad)}")
			out64 = VoronoiDecomposition(m[:,bad],offset_t=offset_t,
				flattened=True,traits=traits,retry64_tol=0.,steps=steps)
			for a,b in zip(out,out64): a[...,bad]=b

	if steps=="Both":
		cupy_kernel = module.get_function("VoronoiDecomposition")
		cupy_kernel((gridDim,),(blockDim,),(m,weights,offsets))
		out = weights,offsets
		retry64()
		return out

	# Two steps approach. First minimize
	a = cp.empty((ndim,ndim,*shape),dtype=float_t)
	vertex = cp.empty(shape,dtype=int_t)
	objective = cp.empty(shape,dtype=float_t)
	a,vertex,objective = map(cp.ascontiguousarray,(a,vertex,objective))

	cupy_kernel = module.get_function("VoronoiMinimization")
	cupy_kernel((gridDim,),(blockDim,),(m,a,vertex,objective))

	if steps=="Single": 
		out = a,vertex,objective
		retry64()
		return out

	cupy_kernel = module.get_function("VoronoiKKT")
	cupy_kernel((gridDim,),(blockDim,),(m,a,vertex,objective,weights,offsets))
	if shape==(): vertex,objective = (e.reshape(()) for e in (vertex,objective))

	out = a,vertex,objective,weights,offsets
	retry64()
	return out

