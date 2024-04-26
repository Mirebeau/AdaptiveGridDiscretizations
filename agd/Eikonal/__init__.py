# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The Eikonal package embeds CPU and GPU numerical solvers of (generalized) eikonal 
equations. These are variants of the fast marching and fast sweeping method, based on 
suitable discretizations of the PDE, and written and C++.

Please see the illustrative notebooks for detailed usage instructions and examples:
https://github.com/Mirebeau/AdaptiveGridDiscretizations

Main object : 
- dictIn : a dictionary-like structure, used to gathers the arguments of the 
eikonal solver, and eventually call it.
"""

import numpy as np
import importlib
import functools

from .LibraryCall import GetBinaryDir
from .run_detail import Cache
from .DictIn import dictIn,dictOut,CenteredLinspace


def _VoronoiDecomposition_noAD(arr,mode=None,steps='Both',*args,**kwargs):
	"""
	Calls the FileVDQ library to decompose the provided quadratic form(s),
	as based on Voronoi's first reduction of quadratic forms.
	- mode : 'cpu' or 'gpu' or 'gpu_transfer'. 
	 Defaults to VoronoiDecomposition.default_mode if specified, or to gpu/cpu adequately.
	- args,kwargs : passed to gpu decomposition method
	"""
	from ..AutomaticDifferentiation.cupy_generic import cupy_set,cupy_get,from_cupy
	if mode is None: mode = VoronoiDecomposition.default_mode
	if mode is None: mode = 'gpu' if from_cupy(arr) else 'cpu'
	if mode in ('gpu','gpu_transfer'):
		from .HFM_CUDA.VoronoiDecomposition import VoronoiDecomposition as VD
		if mode=='gpu_transfer': arr = cupy_set(arr)
		out = VD(arr,*args,steps=steps,**kwargs)
		if mode=='gpu_transfer': out = cupy_get(out,iterables=(tuple,))
		return out
	elif mode=='cpu':
		from ..Metrics import misc
		from . import FileIO
		bin_dir = GetBinaryDir("FileVDQ",None)
		dim = arr.shape[0]; shape = arr.shape[2:]
		if dim==1: # Trivial case
			from .. import Selling
			return Selling.Decomposition(arr)
		arr = arr.reshape( (dim,dim,np.prod(shape,dtype=int)) )
		arr = np.moveaxis(misc.flatten_symmetric_matrix(arr),0,-1)
		vdqIn ={'tensors':arr,'steps':steps}
		if 'smooth' in kwargs: vdqIn['smooth']=kwargs['smooth']
		vdqOut = FileIO.WriteCallRead(vdqIn, "FileVDQ", bin_dir)
		weights = np.moveaxis(vdqOut['weights'],-1,0)
		offsets = np.moveaxis(vdqOut['offsets'],(-1,-2),(0,1)).astype(int)
		weights,offsets = (e.reshape(e.shape[:depth]+shape) 
			for (e,depth) in zip((weights,offsets),(1,2)))
		if steps=='Both': return weights,offsets
		objective = vdqOut['objective'].reshape(shape)
		vertex = vdqOut['vertex'].reshape(shape).astype(int)
		chg = np.moveaxis(vdqOut['chg'],(-1,-2),(0,1)) 
		chg=chg.reshape((dim,dim)+shape)
		return chg,vertex,objective,weights,offsets
	else: raise ValueError(f"VoronoiDecomposition unsupported mode {mode}")

def VoronoiDecomposition(arr,*args,**kwargs):
	"""
	Voronoi decomposition of arr, an array of dxd symmetric positive definite matrices, 
	with shape (d,d,n1,...nk), and possibly with AD.
	args,kwargs : see _VoronoiDecomposition_noAD 
	"""
	from .. import AutomaticDifferentiation as ad
	from ..Metrics.misc import flatten_symmetric_matrix as fltsym
	from .. import LinearParallel as lp
	res = _VoronoiDecomposition_noAD(ad.remove_ad(arr),*args,**kwargs)
	if not ad.is_ad(arr): return res
	if len(arr)==4: raise ValueError("AD unsupported in dimension 4, sorry")
	λ,e = res[-2],res[-1]
	D_flat = arr if kwargs.get('flattened',False) else fltsym(arr)
	eet_flat = fltsym(lp.outer_self(res[-1]))
	λ_ad = lp.solve_AV(eet_flat.astype(D_flat.dtype),D_flat) # e is integer valued
	return *res[:-2],λ_ad,e

VoronoiDecomposition.default_mode = None

