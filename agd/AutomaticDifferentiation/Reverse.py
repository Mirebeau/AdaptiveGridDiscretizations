# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import copy
from . import functional
from . import misc
from . import Sparse

class reverseAD(object):
	"""
	A class for reverse first order automatic differentiation.

	Fields : 
	- input_iterables : tuple, subset of {tuple,list,dict,set}.
		Which input structures should be explored when looking for AD information
	- output_iterables : tuple subset of (tuple,list,dict).
		Which output structures should be explored looking for AD information
	"""

	def __init__(self,operator_data=None,input_iterables=None,output_iterables=None):
		self.operator_data=operator_data
		self.deepcopy_states = False

		self.input_iterables  = (tuple,) if input_iterables is None else input_iterables
		self.output_iterables = (tuple,) if output_iterables is None else output_iterables
		assert hasattr(self.input_iterables,'__iter__') and hasattr(self.output_iterables,'__iter__')

		self._size_ad = 0
		self._size_rev = 0
		self._states = []
		self._shapes_ad = tuple()

	@property
	def size_ad(self): return self._size_ad
	@property
	def size_rev(self): return self._size_rev

	# Variable creation
	def register(self,a):
		return misc.register(self.identity,a,self.input_iterables)

	def identity(self,*args,**kwargs):
		"""Creates and register a new AD variable"""
		result = Sparse.identity(*args,**kwargs,shift=self.size_ad)
		self._shapes_ad += (functional.pair(self.size_ad,result.shape),)
		self._size_ad += result.size
		return result

	def _identity_rev(self,*args,**kwargs):
		"""Creates and register an AD variable with negative indices, 
		used as placeholders in reverse AD"""
		result = Sparse.identity(*args,**kwargs,shift=self.size_rev)
		self._size_rev += result.size
		result.index = -result.index-1
		return result

	def _index_rev(self,index):
		"""Turns the negative placeholder indices into positive ones, 
		for sparse matrix creation."""
		index=index.copy()
		pos = index<0
		index[pos] = -index[pos]-1+self.size_ad
		return index

	# Applying a function
	def apply(self,func,*args,**kwargs):
		"""
		Applies a function on the given args, saving adequate data
		for reverse AD.
		"""
		if self.operator_data == "PassThrough": return func(*args,**kwargs)
		_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse.spAD,self.input_iterables)
		if len(corresp)==0: return func(*args,**kwargs)
		_output = func(*_args,**_kwargs)
		output,shapes = misc._apply_output_helper(self,_output,self.output_iterables)
		self._states.append((shapes,func,
			copy.deepcopy(args) if self.deepcopy_states else args,
			copy.deepcopy(kwargs) if self.deepcopy_states else kwargs))
		return output

	def apply_linear_mapping(self,matrix,rhs,niter=1):
		return self.apply(linear_mapping_with_adjoint(matrix,niter=niter),rhs)
	def apply_linear_inverse(self,solver,matrix,rhs,niter=1):
		return self.apply(linear_inverse_with_adjoint(solver,matrix,niter=niter),rhs)
	def simplify(self,rhs):
		return self.apply(identity_with_adjoint,rhs)

	def iterate(self,func,var,*args,**kwargs):
		"""
		Input: function, variable to be updated, niter, nrec, optional args
		Iterates a function, saving adequate data for reverse AD. 
		If nrec>0, a recursive strategy is used to limit the amount of data saved.
		"""
		niter = kwargs.pop('niter')
		nrec = 0 if niter<=1 else kwargs.pop('nrec',0)
		assert nrec>=0
		if nrec==0:
			for i in range(niter):
				var = self.apply(func,
					var if self.deepcopy_states else copy.deepcopy(var),
					*args,**kwargs)
			return var
		else:
			assert False #TODO. See ODE.RecurseRewind for the strategy.
		"""
			def recursive_iterate():
				other = reverseAD()
				return other.iterate(func,
			niter_top = int(np.ceil(niter**(1./(1+nrec))))
			for rec_iter in (niter//niter_top,)*niter_top + (niter%niter_top,)
				
				var = self.apply(recursive_iterate,var,*args,**kwargs,niter=rec_iter,nrec=nrec-1)

		for 
		"""


	# Adjoint evaluation pass

	def to_inputshapes(self,a):
		return tuple(misc._to_shapes(a,shape,self.input_iterables) for shape in self._shapes_ad)

	def gradient(self,a):
		"""Computes the gradient of the scalar spAD variable a"""
		assert(isinstance(a,Sparse.spAD) and a.shape==tuple())
		coef = Sparse.spAD(a.value,a.coef,self._index_rev(a.index)).to_dense().coef
		size_total = self.size_ad+self.size_rev
		if coef.size<size_total:  coef = misc._pad_last(coef,size_total)
		for outputshapes,func,args,kwargs in reversed(self._states):
			co_output_value = misc._to_shapes(coef[self.size_ad:],outputshapes,self.output_iterables)
			_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse.spAD,self.input_iterables)
			co_arg_request = [a for _,a in corresp]
			co_args = func(*_args,**_kwargs,co_output=functional.pair(co_output_value,co_arg_request))
			for a_sparse,a_value2 in corresp:
				found=False
				for a_value,a_adjoint in co_args:
					if a_value is a_value2:
						val,(row,col) = a_sparse.triplets()
						coef_contrib = misc.spapply(
							(val,(self._index_rev(col),row)),
							misc.as_flat(a_adjoint))
						# Possible improvement : shift by np.min(self._index_rev(col)) to avoid adding zeros
						coef[:coef_contrib.shape[0]] += coef_contrib
						found=True
						break
				if not found:
					raise ValueError(f"ReverseAD error : sensitivity not provided for input value {id(a_sparse)} equal to {a_sparse}")
		return coef[:self.size_ad]

	def output(self,a):
		"""Computes the gradient of the output a, times the co_state, for an operator_like reverseAD"""
		assert not(self.operator_data is None)
		if self.operator_data == "PassThrough":
			return a
		inputs,(co_output_value,_) = self.operator_data
		grad = self.gradient(misc.sumprod(a,co_output_value,self.output_iterables))
		grad = self.to_inputshapes(grad)
		co_arg=[]
		def f(input):
			nonlocal co_arg
			input,to_ad = misc.ready_ad(input)
			if to_ad:
				co_arg.append( (input,grad[len(co_arg)]) )
		misc.map_iterables(f,inputs,self.input_iterables)
		return co_arg
#		return [(x,y) for (x,y) in zip(inputs,self.to_inputshapes(grad))]



# End of class reverseAD

def empty(inputs=None,**kwargs):
	rev = reverseAD(**kwargs)
	return rev if inputs is None else (rev,rev.register(inputs))

# Elementary operators with adjoints

def operator_like(inputs=None,co_output=None,**kwargs):
	"""
	Operator_like reverseAD (or reverseAD2 depending on co_output): 
	- has a fixed co_output
	"""
	mode = misc.reverse_mode(co_output)
	if mode == "Forward": 
		return reverseAD(operator_data="PassThrough",**kwargs),inputs
	elif mode == "Reverse":
		rev = reverseAD(operator_data=(inputs,co_output),**kwargs)
		return rev,rev.register(inputs)
	elif mode == "Reverse2": 
		from . import Reverse2
		return Reverse2.operator_like(inputs,co_output,**kwargs)

def linear_inverse_with_adjoint(solver,matrix,niter=1):
	from . import apply_linear_inverse
	def operator(x):	return apply_linear_inverse(solver,matrix,  x,niter=niter)
	def adjoint(x): 	return apply_linear_inverse(solver,matrix.T,x,niter=niter)
	def method(u,co_output=None):
		mode = misc.reverse_mode(co_output)
		if   mode == "Forward":	return operator(u)
		elif mode == "Reverse": c,_ 		= co_output; return [(u,adjoint(c))]
		elif mode == "Reverse2":(c1,c2),_ 	= co_output; return [(u,adjoint(c1),adjoint(c2))]
	return method

def linear_mapping_with_adjoint(matrix,niter=1):
	from . import apply_linear_mapping
	def operator(x):	return apply_linear_mapping(matrix,  x,niter=niter)
	def adjoint(x): 	return apply_linear_mapping(matrix.T,x,niter=niter)
	def method(u,co_output=None):
		mode = misc.reverse_mode(co_output)
		if   mode == "Forward":	return operator(u)
		elif mode == "Reverse": c,_ 		= co_output; return [(u,adjoint(c))]
		elif mode == "Reverse2":(c1,c2),_ 	= co_output; return [(u,adjoint(c1),adjoint(c2))]
	return method

def identity_with_adjoint(u,co_output=None):
		mode = misc.reverse_mode(co_output)
		if mode == "Forward":	return u
		elif mode == "Reverse": c,_ 		= co_output; return [(u,c)]
		elif mode == "Reverse2":(c1,c2),_ 	= co_output; return [(u,c1,c2)]