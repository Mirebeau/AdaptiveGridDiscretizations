# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import itertools
import math
import numpy as np
import copy
from . import Dense
from . import Sparse
from . import ad_generic
from .ad_generic import remove_ad

def norm(arr,ord=2,axis=None,keepdims=False,averaged=False):
	"""
	Returns L^p norm of array, seen as a vector, w.r.t. weights.
	Defined as : (sum_i x[i]^p)^(1/p)

	Remark : not a matrix operator norm
	
	Inputs:
	 - ord : exponent p
	 - axis : int or None, axis along which to compute the norm. 
	 - keepdims : wether to keep singleton dimensions.
	 - averaged : wether to introduce a normalization factor, so that norm(ones(...))=1

	Compatible with automatic differentiation.
	"""
	arr = ad_generic.array(arr)
	if ord==np.inf or ord%2!=0:
		try: arr = np.abs(arr)
		except TypeError: arr = np.vectorize(np.abs)(arr)
	if ord==np.inf: return np.max(arr,axis=axis,keepdims=keepdims)
	sum_pow = np.sum(arr**ord,axis=axis,keepdims=keepdims)
	
	if averaged:
		size = arr.size if axis is None else arr.shape[axis]
		sum_pow/=size

	return sum_pow**(1./ord)

def norm_infinity(arr,*args,**kwargs):
	"""
	L-Infinity norm (largest absolute value)
	"""
	return norm(arr,np.inf,*args,**kwargs)

def norm_average(arr,*args,**kwargs):
	"""
	Averaged L1 norm (sum of absolute values divided by array size)
	"""
	return norm(arr,1,*args,**kwargs,averaged=True)

class stop_default:
	"""
	Default stopping criterion for the newton method.
	Parameters : 
	* residue_tol : target tolerance on the residue infinity norm
	* niter_max : max iterations before aborting
	* raise_on_abort : wether to raise an exception if aborting
	* niter_print : generator for which iterations to print the state
	"""	
	def __init__(
		self,residue_tol=1e-8,niter_max=50,raise_on_abort=True,
		niter_print="Default",
		verbosity=3
		):
		self.residue_tol	=residue_tol
		self.niter_max 		=niter_max
		self.raise_on_abort	=raise_on_abort

		if niter_print=="Default":
			niter_print = itertools.chain(range(1,6),range(6,16,2),itertools.count(16,4))
		self.niter_print_iter =iter(copy.deepcopy(niter_print))
		self.niter_print_next = next(self.niter_print_iter) # Next iteration to print
		self.niter_print_last = None # Last iteration printed

		self.residue_norms = []
		self.verbosity=verbosity

	def abort(self):
		if self.raise_on_abort:
			raise ValueError("Newton solver did not reach convergence")
		return True


	def __call__(self,residue,niter):
		residue_norm = norm_infinity(remove_ad(residue))
		self.residue_norms.append(residue_norm)

		def print_state():
			if niter!=self.niter_print_last:
				self.niter_print_last = niter
				if self.verbosity>=3: print("Iteration:",niter," Residue norm:",residue_norm)


		if niter>=self.niter_print_next:
			print_state()
			self.niter_print_next = next(self.niter_print_iter)
		
		
		if residue_norm<self.residue_tol:
			print_state()
			if self.verbosity>=2: print("Target residue reached. Terminating.")
			return True

		if np.isnan(residue_norm):
			print_state()
			if self.verbosity>=1: print("Residue has NaNs. Aborting.")
			return self.abort()
		
		if niter>=self.niter_max:
			print_state()
			if self.verbosity>=1: print("Max iterations exceeded. Aborting.")
			return self.abort()

		return False		

class damping_default:
	def __init__(self,criterion=None,refine_factor=2.,step_min=2.e-3,raise_on_abort=False):
		self.criterion=criterion
		self.refine_factor=refine_factor
		self.step_min=step_min
		self.raise_on_abort=raise_on_abort
		self.steps = []

	def __call__(self,x,direction,*params):
		step = 1.
		while self.criterion(x+step*direction,*params):
			step/=2.
			if step<self.step_min:
				print("Minimal damping undershot. Aborting.")
				if self.raise_on_abort:
					raise ValueError
				break	
		self.steps.append(step)
		return step


def newton_root(func,x0,params=tuple(),stop="Default",
	relax=None,damping=None,ad="Sparse",solver=None,in_place=False):
	"""
	Newton's method, for finding a root of a given function.
	func : function to be solved
	x0 : initial guess for the root
	fprime : method for computin
	stop : stopping criterion
	relax : added to the jacobian before inversion
	damping : criterion for step reduction
	ad : is either 
	   - keyword "Sparse" for using Sparse AD (Default)
	   - keyword "Dense" for using Dense AD
	   - a shape_bound given as a tuple, for Dense AD with few independent variables
	"""

	if stop == "Default": 	stop = stop_default()
	if ad == "Dense":		ad = tuple()

	# Create a variable featuring AD information
	def ad_var(x0):
		if not in_place:
			x0=np.copy(x0)

		if ad == "Sparse":			
			return Sparse.identity(constant=x0)
		elif isinstance(ad,tuple):	
			return Dense.identity(constant=x0,shape_bound=ad)						
		assert False

	# Find Newton's descent direction
	def descent_direction(residue):
		if relax is not None:
#			residue=residue+relax
			residue+=relax

		if solver is not None:
			return solver(residue)
		elif ad == "Sparse":
			return residue.solve()
		elif isinstance(ad,tuple): # Dense ad
			return residue.solve(shape_bound=ad)

	x=ad_var(x0)
	for niter in itertools.count():
		residue = func(x,*params)
		if stop(residue,niter):
			return remove_ad(x)

		direction = descent_direction(residue)

		step = 1. if damping is None else damping(remove_ad(x),direction,*params)
		x += step*direction


#def newton_min():
#	pass