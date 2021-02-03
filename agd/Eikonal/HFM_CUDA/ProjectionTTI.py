# Copyright 2021 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module attempts to find the parameters of a TTI norm, given a hooke tensor.
This boils down to minimizing a polynomial over a sphere. We follow a naive approach, where
Newton methods are started from a large number of points in the sphere, and the best result 
is selected. In principle however, SDP relaxation techniques are applicable.
"""


import os
import numpy as np
import cupy as cp
from . import cupy_module_helper
from ...Metrics import misc

def ProjectionTTI(hooke,n_newton=10,samples=None):
	"""
	Attempts to produce a TTI norm matching the given Hooke tensors.
	- n_newton = number of Newton steps in the optimization.
	- samples = seed points in unit sphere (or number of them)
	"""

	# Check and prepare the Hooke tensors
	hdim = len(hooke)
	assert hooke.shape[1]==hdim 
	assert hdim in (3,6)
	vdim = {3:2,6:3}[hdim]
	shape = hooke.shape[2:]
	n_hooke = np.prod(shape,dtype=int)

	float_t = np.float32
	int_t = np.int32
	hooke = misc.flatten_symmetric_matrix(hooke.reshape((hdim,hdim,-1)))
	hooke = cp.asarray(np.moveaxis(hooke,0,-1).astype(float_t))

	# Prepare the samples
	xdim = {2:1,3:3}[vdim] # Dimension of ball
	if samples is None: samples = {1:10,3:400}[xdim]
	if np.ndim(samples)==0:
		# Sample the unit sphere
		nX = {1:samples,3:int(np.round((2*samples)**(1/3)))}[xdim]
		aX,dx = np.linspace(-1,1,nX,retstep=True,endpoint=False)
		samples = np.array(np.meshgrid(*(dx/2+aX,)*xdim,indexing='ij'))
		# TODO : Optionally add some noise to the samples ?
		inside = (samples**2).sum(axis=0)<=1
		samples = samples[:,inside]
	assert len(samples)==xdim
	samples = samples.reshape(xdim,-1).astype(float_t).T
	n_samples = len(samples)
	n_samples_bound = int(2**np.ceil(np.log2(n_samples)))

	# Setup the cuda kernel
	traits = {'Scalar':float_t, 'ndim':vdim, 'n_samples_bound':n_samples_bound}

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += ['#include "ProjectionTTI.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 
	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)

	def SetCst(name,var,dtype):cupy_module_helper.SetModuleConstant(module,name,var,dtype)
	SetCst('n_newton',n_newton,int_t)
	SetCst('n_hooke',n_hooke,int_t)
	SetCst('n_samples',n_samples,int_t)
	SetCst('x_in',np.pad(samples,[(0,n_samples_bound-n_samples),(0,0)]),float_t)

	ProjTTI = module.get_function('ProjectionTTI')

	# Prepare arguments
	score = cp.full((n_hooke,),np.nan,dtype=float_t)
	x_out = cp.full((n_hooke,xdim),np.nan,dtype=float_t)

	hooke,score,x_out = [cp.ascontiguousarray(e) for e in (hooke,score,x_out)]
	size_i = 64
	size_o = int(np.ceil(n_hooke/size_i))
	ProjTTI( (size_o,),(size_i,), (hooke,score,x_out))

	# TODO : Convert to TTI norm

	return score,x_out.T



