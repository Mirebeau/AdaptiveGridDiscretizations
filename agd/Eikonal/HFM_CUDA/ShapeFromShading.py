# Copyright 2021 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module solves the shape from shading PDE, with a non-vertical light source. 
The PDE reads 
(alpha u_x + beta u_y + gamma) / sqrt(1+u_x^2+u_y^2) = rhs
where alpha, beta, gamma are parameters related to the sun direction.
"""

import os
import numpy as np
import cupy as cp
from . import cupy_module_helper
from ... import FiniteDifferences as fd

def EvalScheme(cp,u,params,uc=None,mask=None):
	"""
	Evaluates the (piecewise) quadratic equation defining the numerical scheme.
	Inputs :
	 - uc : plays the role of λ
	 - params : alpha,beta,gamma,h (grid scale)
	"""
	alpha,beta,gamma,h = params
	if uc is None: uc=u
	
	sa,alpha=int(np.sign(alpha)),np.abs(alpha)
	sb,beta =int(np.sign(beta)), np.abs(beta)

	wx = np.roll(u,-sa,axis=0)
	wy = np.roll(u,-sb, axis=1)
	vx = np.minimum(wx,np.roll(u,sa,axis=0))
	vy = np.minimum(wy,np.roll(u,sb,axis=1))

	residue = (cp*np.sqrt(1+(np.maximum(0,uc-vx)**2+np.maximum(0,uc-vy)**2)/h**2) 
		+ alpha*(uc-wx)/h+beta*(uc-wy)/h-gamma)
	return residue if mask is None else np.where(mask,residue,0.)


def Solve(rhs,mask,u0,params,niter=300,traits=None):
	"""
	Iterative solver for the shape from shading equation.
	"""
	float_t = np.float32
	int_t = np.int32
	boolatom_t = np.uint8
	assert len(params)==4

	# Reshape data
	rhs = cp.asarray(rhs,dtype=float_t)
	mask = cp.asarray(mask,dtype=boolatom_t)
	u0 = cp.asarray(u0,dtype=float_t)
	shape = rhs.shape
	assert mask.shape==shape and u0.shape==shape

	traits_default = {'side_i':8,'niter_i':8}
	if traits is None: traits = traits_default
	else: traits = {**traits_default,**traits}

	shape_i = (traits['side_i'],)*2
	rhs,mask,u0 = [fd.block_expand(e,shape_i,constant_values=v) 
		for e,v in [(rhs,np.nan),(mask,False),(u0,0.)] ]
	
	# Find active blocks
	shape_o = rhs.shape[:2]
	update_o = np.flatnonzero(np.any(mask,axis=(-2,-1))).astype(int_t)

	# Setup the cuda module
	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += ['#include "Kernel_ShapeFromShading.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 
	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)

	def SetCst(name,var,dtype):cupy_module_helper.SetModuleConstant(module,name,var,dtype)
	SetCst('params',params,float_t)
	SetCst('shape_o',shape_o,int_t)
	SetCst('shape_tot', np.array(shape_o)*np.array(shape_i), int_t)

	sfs = module.get_function('JacobiUpdate')

	# Call the kernel
	rhs,mask,u,update_o = [cp.ascontiguousarray(e) for e in (rhs,mask,u0,update_o)]
	for i in range(niter):
		sfs((update_o.size,),shape_i,(u,rhs,mask,update_o))

	return fd.block_squeeze(u,shape)

	