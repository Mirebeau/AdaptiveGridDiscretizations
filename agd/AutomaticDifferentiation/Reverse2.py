# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import copy
from . import functional
from . import misc
from . import Dense
from . import Sparse
from . import Reverse
from . import Sparse2
from .cupy_generic import isndarray

class reverseAD2(object):
	"""
	A class for reverse second order automatic differentiation
	"""

	def __init__(self,operator_data=None,input_iterables=None,output_iterables=None):
		self.operator_data=operator_data
		self.deepcopy_states = False

		self.input_iterables  = (tuple,) if input_iterables  is None else input_iterables
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
		"""Creates and registers a new AD variable"""
		assert (self.operator_data is None) or kwargs.pop("operator_initialization",False)
		result = Sparse2.identity(*args,**kwargs,shift=self.size_ad)
		self._shapes_ad += (functional.pair(self.size_ad,result.shape),)
		self._size_ad += result.size
		return result

	def _identity_rev(self,*args,**kwargs):
		"""Creates and register an AD variable with negative indices, 
		used as placeholders in reverse AD"""
		result = Sparse2.identity(*args,**kwargs,shift=self.size_rev)
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

	def apply(self,func,*args,**kwargs):
		"""
		Applies a function on the given args, saving adequate data
		for reverse AD.
		"""
		if self.operator_data == "PassThrough": return func(*args,**kwargs)
		_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse2.spAD2,self.input_iterables)
		if len(corresp)==0: return f(args,kwargs)
		_output = func(*_args,**_kwargs)
		output,shapes = misc._apply_output_helper(self,_output,self.output_iterables)
		self._states.append((shapes,func,
			copy.deepcopy(args) if self.deepcopy_states else args,
			copy.deepcopy(kwargs) if self.deepcopy_states else kwargs))
		return output

	def apply_linear_mapping(self,matrix,rhs,niter=1):
		return self.apply(Reverse.linear_mapping_with_adjoint(matrix,niter=niter),rhs)
	def apply_linear_inverse(self,matrix,solver,rhs,niter=1):
		return self.apply(Reverse.linear_inverse_with_adjoint(matrix,solver,niter=niter),rhs)
	def simplify(self,rhs):
		return self.apply(Reverse.identity_with_adjoint,rhs)

	# Adjoint evaluation pass
	def gradient(self,a):
		"""Computes the gradient of the scalar spAD2 variable a"""
		assert(isinstance(a,Sparse2.spAD2) and a.shape==tuple())
		coef = Sparse.spAD(a.value,a.coef1,self._index_rev(a.index)).to_dense().coef
		for outputshapes,func,args,kwargs in reversed(self._states):
			co_output_value = misc._to_shapes(coef[self.size_ad:],outputshapes,self.output_iterables)
			_args,_kwargs,corresp = misc._apply_input_helper(args,kwargs,Sparse2.spAD2,self.input_iterables)
			co_arg_request = [a for _,a in corresp]
			co_args = func(*_args,**_kwargs,co_output=functional.pair(co_output_value,co_arg_request))
			for a_sparse,a_value2 in corresp:
				found = False
				for a_value,a_adjoint in co_args:
					if a_value is a_value2:
						val,(row,col) = a_sparse.to_first().triplets()
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

	def _hessian_forward_input_helper(self,args,kwargs,dir):
		"""Replaces Sparse AD information with dense one, based on dir_hessian."""
		from . import is_ad
		corresp = []
		def _make_arg(a):
			nonlocal dir,corresp
			if is_ad(a):
				assert isinstance(a,Sparse2.spAD2)
				a1=Sparse.spAD(a.value,a.coef1,self._index_rev(a.index))
				coef = misc.spapply(a1.triplets(),dir[:a1.bound_ad()])
				a_value = Dense.denseAD(a1.value, coef.reshape(a.shape+(1,)))
				corresp.append((a,a_value))
				return a_value
			else:
				return a
		def make_arg(a):
			return misc.map_iterables(_make_arg,a,self.input_iterables)
		_args = tuple(make_arg(a) for a in args)
		_kwargs = {key:make_arg(val) for key,val in kwargs.items()}
		return _args,_kwargs,corresp

	def _hessian_forward_make_dir(self,values,shapes,dir):
		def f(val,s):
			nonlocal self,dir
			if s is not None:
				start,shape = s
				assert isinstance(val,Dense.denseAD) and val.size_ad==1
				assert val.shape==shape
				sstart = self.size_ad+start
				dir[sstart:(sstart+val.size)] = val.coef.flatten()
		misc.map_iterables2(f,values,shapes,self.output_iterables)

	def hessian(self,a):
		"""Returns the hessian operator associated with the scalar spAD2 variable a"""
		assert(isinstance(a,Sparse2.spAD2) and a.shape==tuple())
		def hess_operator(dir_hessian,coef2_init=None,with_grad=False):
			nonlocal self,a
			# Forward pass : propagate the hessian direction
			size_total = self.size_ad+self.size_rev
			dir_hessian_forwarded = np.zeros(size_total)
			dir_hessian_forwarded[:self.size_ad] = dir_hessian
			denseArgs = []
			for outputshapes,func,args,kwargs in self._states:
				# Produce denseAD arguments containing the hessian direction
				_args,_kwargs,corresp = self._hessian_forward_input_helper(args,kwargs,dir_hessian_forwarded)
				denseArgs.append((_args,_kwargs,corresp))
				# Evaluate the function 
				output = func(*_args,**_kwargs)
				# Collect the forwarded hessian direction
				self._hessian_forward_make_dir(output,outputshapes,dir_hessian_forwarded)

			# Reverse pass : evaluate the hessian operator
			# TODO : avoid the recomputation of the gradient
			coef1 = Sparse.spAD(a.value,a.coef1,self._index_rev(a.index)).to_dense().coef
			coef2 = misc.spapply((a.coef2,(self._index_rev(a.index_row),self._index_rev(a.index_col))),dir_hessian_forwarded, crop_rhs=True)
			if coef1.size<size_total:  coef1 = misc._pad_last(coef1,size_total)
			if coef2.size<size_total:  coef2 = misc._pad_last(coef2,size_total)
			if not(coef2_init is None): coef2 += misc._pad_last(coef2_init,size_total)
			for (outputshapes,func,_,_),(_args,_kwargs,corresp) in zip(reversed(self._states),reversed(denseArgs)):
				co_output_value1 = misc._to_shapes(coef1[self.size_ad:],outputshapes,self.output_iterables)
				co_output_value2 = misc._to_shapes(coef2[self.size_ad:],outputshapes,self.output_iterables)
				co_arg_request = [a for _,a in corresp]
				co_args = func(*_args,**_kwargs,co_output=functional.pair(functional.pair(co_output_value1,co_output_value2),co_arg_request))
				for a_value,a_adjoint1,a_adjoint2 in co_args:
					for a_sparse,a_value2 in corresp:
						if a_value is a_value2:
							# Linear contribution to the gradient
							val,(row,col) = a_sparse.to_first().triplets()
							triplets = (val,(self._index_rev(col),row))
							coef1_contrib = misc.spapply(triplets,misc.as_flat(a_adjoint1))
							coef1[:coef1_contrib.shape[0]] += coef1_contrib

							# Linear contribution to the hessian
							linear_contrib = misc.spapply(triplets,misc.as_flat(a_adjoint2))
							coef2[:linear_contrib.shape[0]] += linear_contrib

							# Quadratic contribution to the hessian
							obj = (a_adjoint1*a_sparse).sum()
							quadratic_contrib = misc.spapply((obj.coef2,(self._index_rev(obj.index_row),self._index_rev(obj.index_col))), 
								dir_hessian_forwarded, crop_rhs=True)
							coef2[:quadratic_contrib.shape[0]] += quadratic_contrib

							break
			return (coef1[:self.size_ad],coef2[:self.size_ad]) if with_grad else coef2[:self.size_ad]
		return hess_operator


	def to_inputshapes(self,a):
		return tuple(misc._to_shapes(a,shape,self.input_iterables) for shape in self._shapes_ad)

	def output(self,a):
		assert not(self.operator_data is None)
		if self.operator_data == "PassThrough":
			return a
		inputs,((co_output_value1,co_output_value2),_),dir_hessian = self.operator_data
		_a = misc.sumprod(a,co_output_value1,self.output_iterables)
		_a2 = misc.sumprod(a,co_output_value2,self.output_iterables,to_first=True)
		coef2_init = Sparse.spAD(_a2.value,_a2.coef,self._index_rev(_a2.index)).to_dense().coef

		hess = self.hessian(_a)
		coef1,coef2 = hess(dir_hessian,coef2_init=coef2_init,with_grad=True)

		coef1 = self.to_inputshapes(coef1)
		coef2 = self.to_inputshapes(coef2)
		co_arg = []
		def f(input):
			nonlocal co_arg
			if isndarray(input):
				assert isinstance(input,Dense.denseAD) and input.size_ad==1
				l = len(co_arg)
				co_arg.append( (input,coef1[l],coef2[l]) )
		misc.map_iterables(f,inputs,self.input_iterables)
		return co_arg

# End of class reverseAD2

def empty(inputs=None,**kwargs):
	rev = reverseAD2(**kwargs)
	return rev if inputs is None else (rev,rev.register(inputs))

def operator_like(inputs=None,co_output=None,**kwargs):
	"""
	Operator_like reverseAD2 (or Reverse depending on reverse mode): 
	- should not register new inputs (conflicts with the way dir_hessian is provided)
	- fixed co_output 
	- gets dir_hessian from inputs
	"""
	mode = misc.reverse_mode(co_output)
	if mode == "Forward":
		return reverseAD2(operator_data="PassThrough",**kwargs),inputs
	elif mode == "Reverse":
		from . import Reverse
		return Reverse.operator_like(inputs,co_output,**kwargs)
	elif mode=="Reverse2":
		dir_hessian = tuple()
		def reg_coef(a):
			nonlocal dir_hessian
			if isndarray(a):
				assert isinstance(a,Dense.denseAD) and a.size_ad==1
				dir_hessian+=(a.coef.flatten(),)
		input_iterables = kwargs.get('input_iterables',(tuple,))
		misc.map_iterables(reg_coef,inputs,input_iterables)
		dir_hessian = np.concatenate(dir_hessian)
		rev = reverseAD2(operator_data=(inputs,co_output,dir_hessian),**kwargs)
		def reg_value(a):
			nonlocal rev
			if isinstance(a,Dense.denseAD):
				return rev.identity(constant=a.value,operator_initialization=True)
			else: return a
		return rev,misc.map_iterables(reg_value,inputs,rev.input_iterables)