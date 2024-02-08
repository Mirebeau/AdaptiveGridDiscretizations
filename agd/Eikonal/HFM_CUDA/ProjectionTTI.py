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
from ...Metrics import misc
from ... import Sphere
from ... import LinearParallel
from ...Metrics import Seismic 
from ... import AutomaticDifferentiation as ad

def rotation_xy(α,β):
	"""
	A composition of two rotations, along the x and y axes, with the given angles.
	"""
	cα,sα,cβ,sβ,z = np.cos(α),np.sin(α),np.cos(β),np.sin(β),np.zeros_like(α)
	return ad.array([
		[cβ,z,sβ],
		[sα*sβ,cα,-cβ*sα],
		[-cα*sβ,sα,cα*cβ]])

def rotation_TTI(x,vdim=None):
	"""Construction of the rotation associated to the parameter x returned by ProjectionTTI"""
	xdim = 1 if vdim==2 else len(x)
	if xdim==1: r = LinearParallel.rotation(x)
	elif xdim==2: r = rotation_xy(*x)
	elif xdim==3: r = Sphere.rotation3_from_ball3(x)[0]
	return np.moveaxis(r,0,1) # Reverse the rotation, to match the notebook convention


def ProjectionTTI(hooke,ret_rot=True,n_newton=10,samples=None,quaternion=False):
	"""
	Attempts to produce a TTI norm matching the given Hooke tensors.
	Inputs : 
	- n_newton = number of Newton steps in the optimization.
	- samples = seed points in unit sphere (or number of them)
	
	Output : 
	Returns the score (squared projection error), an HexagonalMaterial, and 
	if retx==False : a rotation matrix
	if retx==True  : the parameters of the optimal rotation
	"""
	import cupy as cp
	from . import cupy_module_helper
	# Check and prepare the Hooke tensors
	hdim = len(hooke)
	assert hooke.shape[1]==hdim 
	assert hdim in (3,6)
	vdim = {3:2,6:3}[hdim]
	shape = hooke.shape[2:]
	n_hooke = np.prod(shape,dtype=int)

	float_t = np.float32
	int_t = np.int32
#	if retx==False: hooke_in = hooke
	hooke = misc.flatten_symmetric_matrix(hooke.reshape((hdim,hdim,-1)))
	hooke = cp.asarray(np.moveaxis(hooke,0,-1).astype(float_t))

	# Prepare the samples, used to initialize the Newton method, which finds the frame
	xdim = 1 if vdim==2 else (3 if quaternion else 2)
	if samples is None: samples = {1:40,2:400,3:400}[xdim] # default number of samples
	if np.ndim(samples)==0:
		if xdim==1: # Sample the rotation angle
			nX = samples
			samples = np.linspace(0,np.pi/2,nX,endpoint=False)[np.newaxis] 
		elif xdim==2: # Sample two rotation angles, along the x and y axes 
			nX = int(np.round(np.sqrt(samples))) # recall : vti is z-invariant
			aX = np.linspace(-np.pi/2,np.pi/2,nX,endpoint=False)
			samples = np.array(np.meshgrid(aX,aX,indexing='ij')).reshape(2,-1)
		elif xdim==3: # Sample a point in the three dimensional sphere
			nX = int(np.round((2*samples)**(1/3))) # Turned quaternion, then rotation
			aX,dx = np.linspace(-1,1,nX,retstep=True,endpoint=False)
			samples = np.array(np.meshgrid(*(dx/2+aX,)*xdim,indexing='ij'))
			inside = (samples**2).sum(axis=0)<=1
			samples = samples[:,inside]
	assert len(samples)==xdim
	samples = samples.reshape(xdim,-1).astype(float_t).T
	n_samples = len(samples)
	samples = cp.asarray(samples,dtype=float_t)

	# Setup the cuda kernel
	traits = {'Scalar':float_t, 'xdim_macro':xdim } 

	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += ['#include "Kernel_ProjectionTTI.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 
	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)

	def SetCst(name,var,dtype):cupy_module_helper.SetModuleConstant(module,name,var,dtype)
	SetCst('n_newton',n_newton,int_t)
	SetCst('n_hooke',n_hooke,int_t)
	SetCst('n_samples',n_samples,int_t)

	ProjTTI = module.get_function('ProjectionTTI')

	# Prepare arguments
	score = cp.full((n_hooke,),np.nan,dtype=float_t)
	x_out = cp.full((n_hooke,xdim),np.nan,dtype=float_t)
	hexa = cp.full((n_hooke,5),np.nan,dtype=float_t)

	hooke,score,x_out,samples,hexa = [cp.ascontiguousarray(e) 
		for e in (hooke,score,x_out,samples,hexa)]
	size_i = 64
	size_o = int(np.ceil(n_hooke/size_i))
	ProjTTI( (size_o,),(size_i,), (hooke,score,x_out,samples,hexa))
	del hooke

	# Post processing
	score.reshape(shape)
	x_out = x_out.T.reshape((xdim,*shape))
	hexa = hexa.T.reshape((5,*shape))
	if xdim==1: x_out = x_out[0]; score*=2
	return score,hexa,(rotation_TTI(x_out,vdim) if ret_rot else x_out)



