# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
from collections import OrderedDict
import copy

from . import kernel_traits
from . import _solvers
from . import cupy_module_helper
from .graph_reverse import graph_reverse
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad

# This file implements some member functions of the Interface class of HFM_CUDA

def values_expand(self):
	"""
	Returns the solution to the eikonal equation, expanded in the kernel adapted block format.
	"""
	if self._values_expand is None:
		eikonal = self.kernel_data['eikonal']
		values = eikonal.args['values']
		if eikonal.policy.multiprecision:
			valuesq = eikonal.args['valuesq']
			if eikonal.policy.values_float64:
				float64_t = np.dtype('float64').type
				self._values_expand = (values.astype(float64_t) 
					+ float64_t(self.multip_step) * valuesq)
			else:
				self._values_expand = (values+valuesq.astype(self.float_t)*self.multip_step)
		else:
			self._values_expand = values
	return self._values_expand

def values(self):
	"""
	Returns the solution to the eikonal equation.
	"""
	if self._values is None:
		self._values = fd.block_squeeze(self.values_expand,self.shape)
	return self._values

def PostProcess(self):
	if self.verbosity>=1: print("Post-Processing")
	eikonal = self.kernel_data['eikonal']

	# values are now extracted only if needed
	self._values_expand=None
	self._values=None

	if not self.flow_needed: return

	# Compute the geodesic flow, if needed, and related quantities
	shape_oi = self.shape_o+self.shape_i
	nact = self.nscheme['nact']
	ndim = self.ndim

	flow = self.kernel_data['flow']
	flow.policy.atol = np.inf # Not an iterative solver, solution is already computed
	flow.policy.rtol = 0.
	if flow.traits.get('flow_weights_macro',False):
		flow.args['flow_weights']   = cp.empty((nact,)+shape_oi,dtype=self.float_t)
	if flow.traits.get('flow_weightsum_macro',False):
		flow.args['flow_weightsum'] = cp.empty(shape_oi,dtype=self.float_t)
	if flow.traits.get('flow_weightpos_macro',False):
		flow.args['flow_weightpos'] = cp.empty(shape_oi,dtype=self.int_t) 
	if flow.traits.get('flow_state_macro',False):
		flow.args['flow_state'] = cp.empty((self.nstates,)+shape_oi,dtype=self.float_t) 
	if flow.traits.get('flow_offsets_macro',False):
		flow.args['flow_offsets']   = cp.empty((ndim,nact,)+shape_oi,dtype=self.offset_t)
	if flow.traits.get('flow_indices_macro',False):
		flow.args['flow_indices']   = cp.empty((nact,)+shape_oi,dtype=self.int_t)
	if flow.traits.get('flow_vector_macro',False):
		flow.args['flow_vector']    = cp.empty((ndim,)+shape_oi,dtype=self.float_t)

	self.Solve('flow')
	flow_normalization_needed = (self.forwardAD or self.reverseAD) and self.drift_model

	if flow_normalization_needed or self.exportGeodesicFlow:
		flow_vector = fd.block_squeeze(flow.args['flow_vector'],self.shape)
		if self.hasTips and np.may_share_memory(flow_vector,flow.args['flow_vector']): 
			flow_vector = flow_vector.copy()
		if self.drift_model:
			self.flow_normalization = np.where(self.seedTags,1.,self.metric.norm(-flow_vector))
			flow_vector/=self.flow_normalization # Flow on adimensionized grid
		if self.exportGeodesicFlow:
			flow_vector *= -self.h_broadcasted  
			self.hfmOut['flow'] = flow_vector
	if self.exportActiveNeighs:
		self.hfmOut['activeNeighs']=fd.block_squeeze(flow.args['flow_weightpos'],self.shape)
	if self.exportFlowState:
		self.hfmOut['flow_state']=fd.block_squeeze(flow.args['flow_state'],self.shape)
	
def SolveLinear(self,rhs,diag,indices,weights,chg,kernelName):
	"""
	A linear solver for the systems arising in automatic differentiation of the HFM.
	"""

	data = self.kernel_data[kernelName]
	eikonal = self.kernel_data['eikonal']

	# Set the linear solver traits
	data.traits = self.GetValue(kernelName+'_traits',default=dict())
	data.traits.update({key:value for key,value in eikonal.traits.items() 
		if key in ('shape_i','niter_i','periodic','pruning_macro','minchg_freeze_macro')})
	data.traits.update({'ndim':self.ndim,'nrhs':len(rhs),'nindex':len(indices)})
	data.source = cupy_module_helper.traits_header(data.traits,join=True,
		size_of_shape=True,log2_size=True)+"\n"
	data.source += '#include "Kernel_LinearUpdate.h"\n'+self.cuda_date_modified

#	print(data.traits,data.source)
	data.module = cupy_module_helper.GetModule(data.source, self.cuoptions)
	data.policy = copy.copy(eikonal.policy)
	if data.policy.solver=='fast_iterative_method': 
		data.policy.solver='adaptive_gauss_siedel_iteration'

	# Setup the kernel
	def SetCst(*args): cupy_module_helper.SetModuleConstant(data.module,*args)
	SetCst('shape_o', self.shape_o,       self.int_t)
	SetCst('size_o',  self.size_o,        self.int_t)
	SetCst('size_tot',np.prod(self.shape_o)*np.prod(self.shape_i),self.int_t)

	float_res = np.finfo(self.float_t).resolution
	if not hasattr(self,'linear_rtol'):
		self.linear_rtol = self.GetValue('linear_rtol',default=float_res*5,
			help="Relative convergence tolerance for the linear systems")
	if not hasattr(self,'linear_atol'):
		self.linear_atol = self.GetValue('linear_atol',default=None,
			help="Absolute convergence tolerance for the linear systems")
		if self.linear_atol is None: 
			self.linear_atol=self.linear_rtol*np.mean(np.abs(rhs[np.isfinite(rhs)]))
		self.hfmOut['keys']['default']['linear_atol']=self.linear_atol

	SetCst('rtol',self.linear_rtol,self.float_t)
	SetCst('atol',self.linear_atol,self.float_t)

	# We use a dummy initialization, to infinity, to track non-visited values
	sol = cp.full(rhs.shape,np.inf,dtype=self.float_t) 
	# Trigger is from the seeds (forward), or farthest points (reverse), excluding walls
	domain = np.isfinite(self.kernel_data['eikonal'].args['values'])
	data.trigger = np.logical_and(np.all(weights==0.,axis=0),domain)

	# Call the kernel
	data.args = OrderedDict({
		'sol':sol,'rhs':rhs,'diag':diag,'indices':indices,'weights':weights})
	if data.policy.bound_active_blocks: data.args['chg']=chg

	self.Solve(kernelName)
	return sol


def SolveAD(self):
	"""
	Forward and reverse differentiation of the HFM.
	"""
	if not (self.forwardAD or self.reverseAD): return
	if self.hasChart: 
		raise ValueError("Sorry, forward and reversed AD not yet supported with charts")
	eikonal = self.kernel_data['eikonal']
	flow = self.kernel_data['flow']
	traits = eikonal.traits
	if traits.get('order2_macro') or traits.get('factor_macro'):
		self.Warn("Eikonal AD ignores order2 and source factorization")

	if eikonal.policy.bound_active_blocks:
		dist = eikonal.args['values']
		if self.multiprecision: 
			dist += self.float_t(self.multip_step) * eikonal.args['valuesq']
	else: dist=0.

	diag = flow.args['flow_weightsum'].copy() # diagonal preconditioner
	self.boundary = diag==0. #seeds, or walls, or out of domain
	diag[self.boundary]=1.
	
	indices = flow.args['flow_indices'] 
	weights = flow.args['flow_weights']
	
	if self.forwardAD:
		grad = self.rhs.gradient()
		rhs = np.where(self.seedTags, grad, grad*self.rhs.value)

		if self.drift_model: rhs*=self.flow_normalization
		rhs = cp.ascontiguousarray(fd.block_expand(rhs,self.shape_i,
			mode='constant',constant_values=np.nan))
		valueVariation = self.SolveLinear(rhs,diag,indices,weights,dist,'forwardAD')
		coef = np.moveaxis(fd.block_squeeze(valueVariation,self.shape),0,-1)
		val = self.values
		self._values = ad.Dense.denseAD(val,cp.asarray(coef,val.dtype))


	if self.reverseAD:
		# Get the rhs
		rhs = self.GetValue('sensitivity',help='Reverse automatic differentiation')
		if rhs.shape[:-1]!=self.shape: 
			raise ValueError(f"Reverse AD rhs shape {rhs.shape} does not start with {self.shape}")
		rhs = cp.ascontiguousarray(fd.block_expand(np.moveaxis(rhs,-1,0),self.shape_i,
			mode='constant',constant_values=np.nan))

		# Get the matrix structure
		invalid_index = np.iinfo(self.int_t).max
		indices[weights==0]=invalid_index
		indicesT,weightsT = graph_reverse(indices,weights,invalid_index=invalid_index)
		# By default, weightsT[indicesT==invalid_index]=0

		allSensitivity = self.SolveLinear(rhs,diag,indicesT,weightsT,-dist,'reverseAD')
		allSensitivity = np.moveaxis(fd.block_squeeze(allSensitivity,self.shape),0,-1)
		pos = tuple(self.seedIndices.T)
		seedSensitivity = allSensitivity[pos]
		allSensitivity[pos]=0
		if self.drift_model:
			allSensitivity/=np.expand_dims(self.flow_normalization,axis=-1)
		self.hfmOut['costSensitivity'] = allSensitivity 

		val,(row,col) = self.seedValues_rev.triplets()
		seedSensitivity = ad.misc.spapply((val,(col,row)),seedSensitivity)
		self.hfmOut['seedValueSensitivity'] = seedSensitivity



"""
# Failed attempt using a generic sparse linear solver. (Fails to converge or too slow.)
def SolveAD(self)
	if self.forward_ad or self.reverse_ad:
		spmod=self.xp.cupyx.scipy.sparse
		xp=self.xp
		diag = self.flow['flow_weightsum'].copy() # diagonal preconditioner
		self.boundary = diag==0. #seeds, or walls, or out of domain
		diag[self.boundary]=1.
		coef = xp.concatenate((xp.expand_dims(diag,axis=0),
				-self.flow['flow_weights']),axis=0)
		diag_precond=True
		if diag_precond: coef/=diag
		size_tot = np.prod(self.shape) # Not same as solver size_tot
		rg = xp.arange(size_tot).reshape((1,)+self.shape)
		row = self.xp.broadcast_to(rg,coef.shape)
		col = xp.concatenate((rg,self.flow['flow_indices']),axis=0)

		self.triplets = (npl.flat(coef),(npl.flat(row),npl.flat(col))) 
		self.spmat = spmod.coo_matrix(self.triplets)

	if self.forward_ad:
		if self.costVariation is None:
			self.costVariation = self.xp.zeros(self.shape+self.seedValues.size_ad,
				dtype=self.float_t)
		rhs=self.costVariation 
		if ad.is_ad(self.seedValues):
			rhs[tuple(self.seedIndices.T)] = self.seedValues.coef
#			rhs/=xp.expand_dims(diag,axis=-1)
		rhs=rhs.reshape(size_tot,-1)

		# Solve the linear system
		csrmat = self.spmat.tocsr()
		# In contrast with scipy, lsqr must do one solve per entry. 
		# Note : lsqr also assumes rhs contiguity
		self.forward_solutions = [ 
			spmod.linalg.lsqr(csrmat,self.xp.ascontiguousarray(r)) for r in rhs.T] 
		self.hfmOut['valueVariation'] = self.xp.stack(
			[s[0].reshape(self.shape) for s in self.forward_solutions],axis=-1) 

	if self.reverse_ad:
		rhs = self.GetValue('sensitivity',help='Reverse automatic differentiation')
		hfmOut['valueSensitivity'] = spmod.linalg.lsqr(self.spmat.T.tocsr(),rhs)
"""