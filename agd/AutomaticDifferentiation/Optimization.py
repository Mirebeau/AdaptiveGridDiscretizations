# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import itertools
import math
import numpy as np
import copy
from . import Dense
from . import Sparse
from . import ad_generic
from .ad_generic import remove_ad,is_ad

# Newton minimize
from . import Dense2
from . import Sparse2

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
	Default stopping criterion for the Newton method, which uses the method __call__.
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
		if self.raise_on_abort: raise ValueError("Newton solver did not reach convergence")
		return True


	def __call__(self,residue,niter):
		"""
		Decides wether to stop the Newton solver, and which information to print. 
		Input : 
		 - residue : current error residual 
		 - niter : current iteration number 
		Output : 
		 - a boolean : wether to stop the Newton solver at the current iteration
		"""
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

	def __call__(self,x,direction,*cargs):
		step = 1.
		while self.criterion(x+step*direction,*cargs):
			step/=2.
			if step<self.step_min:
				print("Minimal damping undershot. Aborting.")
				if self.raise_on_abort:
					raise ValueError
				break	
		self.steps.append(step)
		return step


def newton_root(f,x0,fargs=tuple(),stop="Default",
	relax=None,damping=None,ad="Sparse",solver=None,in_place=False):
	"""
	Newton's method, for finding a root of a given function.
	f : function to be solved
	x0 : initial guess for the root
	stop : stopping criterion, presented as either
	   - keyword "Default" for using the stop_default class
	   - a dict for using the stop_class with specified initialization arguments
	   - a callable,  
	relax : added to the jacobian before inversion
	damping : criterion for step reduction
	ad : is either 
	   - keyword "Sparse" for using Sparse AD (Default)
	   - keyword "Dense" for using Dense AD
	   - a shape_bound given as a tuple, for Dense AD with few independent variables
	"""

	if stop == "Default": 	stop = stop_default()
	elif isinstance(stop,dict): stop = stop_default(**stop)
	if ad == "Dense":		ad = tuple() # Tuple represents shape_bound

	# Create a variable featuring AD information
	def ad_var(x0):
		if not in_place: x0=np.copy(x0)

		if ad == "Sparse":		   return Sparse.identity(constant=x0)
		elif isinstance(ad,tuple): return Dense.identity( constant=x0,shape_bound=ad)						
		assert False

	# Find Newton's descent direction
	def descent_direction(residue):
		if relax is not None: residue+=relax # Relax is usually a multiple of identity

		if solver is not None:     return solver(residue)
		elif ad == "Sparse":       return residue.solve() # Sparse ad
		elif isinstance(ad,tuple): return residue.solve(shape_bound=ad) # Dense ad

	x=ad_var(x0)
	for niter in itertools.count():
		residue = f(x,*fargs)
		if stop(residue,niter): return remove_ad(x)

		direction = descent_direction(residue)

		step = 1. if damping is None else damping(remove_ad(x),direction,*fargs)
		x += step*direction


def newton_minimize(f,x0,fargs=tuple(),
	f_value_gradient_direction="Sparse2",
	step_min=0.01,maxiter=50,verbosity=1,
	δ_atol=1e-9,δ_rtol=1e-9,δ_ntol=3,δ_nneg=3):
	"""
	Inputs
	- f : function to evaluate
	- x0 : initial guess
	- fargs (optional) : additional arguments for f
	- f_value_gradient_direction (optional) : method for computing the value, gradient, 
	   and descent direction of f, which can be either of 
	   - "Sparse2" or "Dense2", automatic differentiation is applied to f
	   - a callable, with the same arguments as f, returning a Sparse2 or Dense2 result
	   - a callable, with the same arguments as f, returning a tuple (value,gradient,direction)
	- step_min (optional) : minimum admissible step for the damped newton method
	- maxiter (optional) : max number of iterations
	- verbosity (optional) : amount of information displayed
	- δ_atol,δ_rtol,δ_ntol,δ_nneg (optional) : parameters of the stopping criterion
	"""
	if verbosity<=0: kiter_print=iter([-1,-1])
	elif verbosity==1: kiter_print = itertools.chain(range(1,6),range(6,16,2),itertools.count(16,4))
	elif verbosity>=2: kiter_print = itertools.count(1)
	kiter_print_next=next(kiter_print)

	step_min = 2.**np.ceil(np.log2(step_min))
	def tol(v): return δ_atol+np.abs(v)*δ_rtol

	x = x0.copy() # Updated variable throughout the iterations
	f_vgd = f_value_gradient_direction
	if isinstance(f_vgd,str):
		if   f_vgd=="Sparse2": x_ad = Sparse2.identity(x.shape)
		elif f_vgd=="Dense2":  x_ad = Dense2.identity(x.shape)
		else: raise ValueError(f"Unrecognized value {f_vgd} of f_value_gradient_direction")
		def f_vgd(x,*fargs): # Overrides previous definition
			x_ad.value=x
			return f(x_ad,*fargs)

	y = f_vgd(x,*fargs)
	if isinstance(y,tuple): 
		assert len(y)==3 # Directly get value,gradient,direction
		vgd = lambda x : f_vgd(x,*fargs)
	else:
		def split(y): 
			if not is_ad(y): # Allow returning np.nan of np.inf in case of error
				assert not np.isfinite(y)
				return y,None,None
			return y.value,y.to_first().to_dense().coef,y.solve_stationnary()
		y = split(y)
		def vgd(x): return split(f_vgd(x,*fargs))
	if verbosity>=0: print(f"Initialization, objective {y[0]}")

	δ_ktol = 0 # Number of consecutive improvements below tolerance 
	δ_kneg = 0 # Number of consecutive deteriorations of the solution
	exit=False

	s = 1. # Newton step size, can be damped, always a power of two
	for i in range(1,maxiter):
		v,g,d = y 
#		print(x,d,"x,d,iter=",i)
		# Try with current step
		x += s*d
		y = vgd(x)
		δ = v - y[0]

		# Adjust damping for this step, if needed
		δtol = tol(v)
		if not δ > -δtol and s>step_min: 
			while s>step_min:
				s/=2.
				x-=s*d
				δ = v - f(x,*fargs)
				if δ > -δtol: break
			else:
				if verbosity>=-1: print(f"Warning minimal step size {s} deteriorates objective")
			y = vgd(x)

		# Adjust damping for next step
		gd = np.dot(g,d) # Expected to be non-negative
		δ0 = gd*(s-s**2/2.) # Expected improvement if the function was quadratic
		if δ>=δ0/2 and s<1: s*=2. # Looks quadratic, increase step
		if δ<=δ0/4 and s>step_min: s/=2. # Looks non-linear, decrease step

		# Stopping criteria
		v=y[0]
		δtol = tol(v)
		if not np.isfinite(v): 
			if verbosity>=-2: print("Infinite objective value, aborting.")
			exit=True

		if np.abs(δ)<δtol: 
			δ_ktol+=1
			if δ_ktol==δ_ntol:
				if verbosity>=0: print("Convergence criterion satisfied, terminating.")
				exit=True
		else: δ_ktol=0

		if δ<=-δtol: 
			δ_kneg+=1
			if δ_kneg==δ_nneg:
				if verbosity>=-2: print("Stalling, aborting")
				exit=True
		else: δ_kneg=0

		# Display
		if i==kiter_print_next or exit:
			kiter_print_next = next(kiter_print)
			if verbosity>=0: print(f"Iteration {i}, Newton step {s}, objective {v}.")

		if exit: break

	return x
