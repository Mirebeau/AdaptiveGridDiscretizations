# Copyright 2022 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import cupy as cp

from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant

def eigh(m,quaternion=False,flatsym=None,use_numpy=False):
	"""
	Computes the eigenvalues of the symmetric matrix m, of dimension d<=3.
	The cuda routines are likely more accurate, but they are often slow and memory
	intensive for large arrays of small matrices.
	Input : 
	 - m : array of shape (n1,...,nk,d,d) symmetric w.r.t the last two entries, or of 
	  shape (n1,...,nk, d(d+1)/2) if flatsym==True
	 - flatsym (bool, default: autodetect): format for the symmetric matrices, see above.
	 - quaternion (bool): format specifier for the eigenvectors, see below
	 - use_numpy : use the numpy.linalg.eigh routine (calls cuda)
	Output :
	 - λ : array of shape (n1,...,nk,d). The eigenvalues of m, sorted increasingly.
	 - v : the eigenvectors of m, in the following format
	   quaternion == false : shape (n1,...,nk,d,d)
	   - true : export the eigenvectors compactly using the quaternion format
	    (d=3), or omitting the second eigenvector (d=2).
	"""
	float_t = np.float32
	int_t = np.int32
	block_size = 1024

	if flatsym is None: flatsym = not(m.ndim>=2 and m.shape[-1]==m.shape[-2]<=3)
	shape = m.shape[:(-1 if flatsym else -2)]
	size = np.prod(shape,dtype=int)
	d = int(np.floor(np.sqrt(2*m.shape[-1]))) if flatsym else m.shape[-1]

	assert m.shape == ( (*shape,(d*(d+1))//2) if flatsym else (*shape,d,d) )
	assert isinstance(quaternion,bool) or quaternion is None
	assert 1<=d<=3
	vshape = (*shape,{1:1,2:2,3:4}[d]) if quaternion else (*shape,d,d) 

	if d==1: # Trivial 1D case
		v=np.broadcast_to(cp.ones(1,dtype=float_t),vshape)
		λ = m.reshape((*shape,d)).astype(float_t)
		return λ,v
	if use_numpy:
		from ... import Sphere
		assert not flatsym
		if quaternion is None: return np.linalg.eigvalsh(m)
		λ,v = np.linalg.eigh(m)
		if not quaternion: return λ,v 
		v[np.linalg.det(v)<0,:,-1] *= -1 # Turn v into a direct orthogonal basis
		v = np.moveaxis(v,(-2,-1),(0,1))
		v = Sphere.sphere1_from_rotation2(v) if d==2 else Sphere.sphere3_from_rotation3(v)
		v = np.moveaxis(v,0,-1)
		return λ,v
	if not flatsym: 
		m = cp.stack([m[...,i,j] for i in range(d) for j in range(i+1)], axis=-1)
	m = cp.ascontiguousarray(m,dtype=float_t)

	traits = {
	'Scalar':float_t,
	'Int':int_t,
	'quaternion_macro':bool(quaternion),
	'ndim_macro':d,
	}

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += [
	'#include "Kernel_Eigh.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 

	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)
	def setcst(*args): SetModuleConstant(module,*args)
	setcst('size_tot',size,int_t)

	λ = cp.empty((*shape,d),dtype=float_t)
	grid_size = int(np.ceil(size/block_size))
	if quaternion is None: # Special value to get the eigenvalues alone
		f = module.get_function('kernel_eigvalsh')
		f((grid_size,),(block_size,),(m,λ))
		return λ
	
	v = cp.empty(vshape,dtype=float_t)
	f = module.get_function('kernel_eigh')
	f((grid_size,),(block_size,),(m,λ,v))
	return λ,v

def eigvalsh(m,flatsym=False):
	"""
	Computes the eigenvalues of the symmetric matrix m, of dimension d<=3.
	The cuda routines are likely more accurate, but often slow and memory intensive.
	Input : 
	 - m : array of shape (n1,...,nk,d,d) symmetric w.r.t the last two entries, or of 
	  shape (n1,...,nk, d(d+1)/2) if flatsym==True
	 - flatsym (bool): format for the symmetric matrices, see above

	Output :
	 - λ : array of shape (n1,...,nk,d). The eigenvalues of m, sorted increasingly.
	"""
	return eigh(m,flatsym=flatsym,quaternion=None)