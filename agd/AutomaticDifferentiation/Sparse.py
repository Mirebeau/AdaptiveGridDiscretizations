# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import functools
from . import functional
from . import Base
from . import cupy_support as cps
from . import ad_generic
from . import cupy_generic
from . import misc
from . import Dense

_add_dim = misc._add_dim; _pad_last = misc._pad_last; _concatenate=misc._concatenate;

class spAD(Base.baseAD):
	"""
	A class for sparse forward automatic differentiation
	"""

	def __init__(self,value,coef=None,index=None,broadcast_ad=False):
		if self.is_ad(value): # Case where one should just reproduce value
			assert coef is None and index is None
			self.value,self.coef,self.index = value.value,value.coef,value.index
			return
		if ad_generic.is_ad(value):
			raise Base.ADCastError(f"Attempting to cast {type(value)} to incompatible type spAD")
			
		assert ((coef is None) and (index is None)) or (coef.shape==index.shape)

		self.value = ad_generic.asarray(value)
		shape =self.shape
		shape2=shape+(0,)
		self.coef  = (np.zeros_like(value,shape=shape2) if coef is None 
			else misc._test_or_broadcast_ad(coef,shape,broadcast_ad) ) 
		self.index = (np.zeros_like(value,shape=shape2,
			dtype=cupy_generic.samesize_int_t(value.dtype))  
			if index is None else misc._test_or_broadcast_ad(index,shape,broadcast_ad) )

	@classmethod
	def order(cls): return 1
	def copy(self,order='C'):
		return self.new(self.value.copy(order=order),
			self.coef.copy(order=order),self.index.copy(order=order))
	def as_tuple(self): return self.value,self.coef,self.index

	def __copy__(self): return self.copy(order='K')
	def __deepcopy__(self,*args): 
		return self.new(self.value.__deepcopy__(*args),
			self.coef.__deepcopy__(*args),self.index.__deepcopy__(*args))

	# Representation 
	def __iter__(self):
		for value,coef,index in zip(self.value,self.coef,self.index):
			yield self.new(value,coef,index)

	def __str__(self):
		return "spAD"+str((self.value,self.coef,self.index))
	def __repr__(self):
		return "spAD"+repr((self.value,self.coef,self.index))	

	# Operators
	def as_func(self,h=None):
		"""Replaces the symbolic perturbation with h, if specified."""
		if h is None: 
			lin = self.tangent_operator()
			return lambda h : (lin*h).reshape(self.shape) + misc.add_ndim(self.value,h.ndim-1)
		value,coef = (misc.add_ndim(e,h.ndim-1) for e in (self.value,self.coef))
		return value+(coef*h[self.index]).sum(axis=self.ndim)

	def __add__(self,other):
		if self.is_ad(other):
			value = self.value+other.value
			return self.new(value, _concatenate(self.coef,other.coef,value.shape), 
				_concatenate(self.index,other.index,value.shape))
		else:
			return self.new(self.value+other, self.coef, self.index, broadcast_ad=True)

	def __sub__(self,other):
		if self.is_ad(other):
			value = self.value-other.value
			return self.new(self.value-other.value, _concatenate(self.coef,-other.coef,value.shape), _concatenate(self.index,other.index,value.shape))
		else:
			return self.new(self.value-other, self.coef, self.index, broadcast_ad=True)

	def __mul__(self,other):
		if self.is_ad(other):
			value = self.value*other.value
			coef1,coef2 = _add_dim(other.value)*self.coef,_add_dim(self.value)*other.coef
			index1,index2 = np.broadcast_to(self.index,coef1.shape),np.broadcast_to(other.index,coef2.shape)
			return self.new(value,_concatenate(coef1,coef2),_concatenate(index1,index2))
		elif self.isndarray(other):
			value = self.value*other
			coef = _add_dim(other)*self.coef
			index = np.broadcast_to(self.index,coef.shape)
			return self.new(value,coef,index)
		else:
			return self.new(self.value*other,other*self.coef,self.index)

	def __truediv__(self,other):
		if self.is_ad(other):
			return self.new(self.value/other.value,
				_concatenate(self.coef*_add_dim(1/other.value),other.coef*_add_dim(-self.value/other.value**2)),
				_concatenate(self.index,other.index))
		elif self.isndarray(other):
			return self.new(self.value/other,self.coef*_add_dim(1./other),self.index)
		else:
			return self.new(self.value/other,self.coef/other,self.index)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other): 	
		value = other/self.value
		coef = self.coef*_add_dim(-other/self.value**2)
		index = np.broadcast_to(self.index,coef.shape)
		return self.new(value,coef,index)

	def __neg__(self):		return self.new(-self.value,-self.coef,self.index)

	# Math functions
	def _math_helper(self,deriv):
		a,b=deriv
		return self.new(a,_add_dim(b)*self.coef,self.index)

	@classmethod
	def compose(cls,a,t):
		assert isinstance(a,Dense.denseAD) and all(cls.is_ad(b) for b in t)
		lens = tuple(len(b) for b in t)
		assert a.size_ad == sum(lens)
		t = tuple(np.moveaxis(b,0,-1) for b in t)
		a_coefs = np.split(a.coef,np.cumsum(lens[:-1]),axis=-1)
		def FlattenLast2(arr): return arr.reshape(arr.shape[:-2]+(np.prod(arr.shape[-2:],dtype=int),))
		coef = tuple(_add_dim(c)*b.coef for c,b in zip(a_coefs,t) )
		coef = np.concatenate( tuple(FlattenLast2(c) for c in coef), axis=-1)
		index = np.broadcast_to(np.concatenate( tuple(FlattenLast2(b.index) 
			for b in t), axis=-1),coef.shape)
		return cls.new(a.value,coef,index)

	#Indexing
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def __getitem__(self,key):
		ekey = misc.key_expand(key)
		return self.new(self.value[key], self.coef[ekey], self.index[ekey])

	def __setitem__(self,key,other):
		ekey = misc.key_expand(key)
		if self.is_ad(other):
			self.value[key] = other.value
			pad_size = max(self.coef.shape[-1],other.coef.shape[-1])
			if pad_size>self.coef.shape[-1]:
				self.coef = _pad_last(self.coef,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef[ekey] = _pad_last(other.coef,pad_size)
			self.index[ekey] = _pad_last(other.index,pad_size)
		else:
			self.value[key] = other
			self.coef[ekey] = 0.

	def reshape(self,shape,order='C'):
		value = self.value.reshape(shape,order=order)
		shape2 = value.shape+(self.size_ad,)
		return self.new(value,self.coef.reshape(shape2,order=order),
			self.index.reshape(shape2,order=order))

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return self.new(np.broadcast_to(self.value,shape), 
			np.broadcast_to(self.coef,shape2), np.broadcast_to(self.index,shape2))

	def pad(self,pad_width,*args,constant_values=0,**kwargs):
		return self.new(
			np.pad(self.value,pad_width,*args,constant_values=constant_values,**kwargs),
			np.pad(self.coef, pad_width+((0,0),),*args,constant_values=0,**kwargs),
			np.pad(self.index,pad_width+((0,0),),*args,constant_values=0,**kwargs))
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return self.new(self.value.transpose(axes),
			self.coef.transpose(axes2),self.index.transpose(axes2))

	def allclose(self,other,*args,**kwargs):
		raise ValueError("Unsupported, sorry, please try allclose(a.to_dense(),b.to_dense())")
		 
	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		value = self.value.sum(axis,**kwargs)
		shape = value.shape +(self.size_ad * self.shape[axis],)
		coef = np.moveaxis(self.coef, axis,-1).reshape(shape)
		index = np.moveaxis(self.index, axis,-1).reshape(shape)
		out = self.new(value,coef,index)
		return out

	# Conversion
	def bound_ad(self):
		return 1+int(cps.max(self.index,initial=-1))

	def to_dense(self,dense_size_ad=None):
		if self.ndim!=1: 
			return self.reshape(-1).to_dense(dense_size_ad=dense_size_ad).reshape(self.shape)
		if dense_size_ad is None: dense_size_ad = self.bound_ad()
		index = self.index+dense_size_ad*np.arange(self.size)[:,None]
		coef = np.bincount(index.reshape(-1),self.coef.reshape(-1),dense_size_ad*self.size)
		return Dense.denseAD(self.value,coef.reshape(self.size,dense_size_ad))

	#Linear algebra
	def triplets(self,tol=0):
		"""
		Returns the triplets defining the sparse linear operator. 
		- tol : remove coefficients whose magnitude is below this threshold
		"""
		xp = cupy_generic.get_array_module(self.value)
		coef = self.coef.reshape(-1)
		row = np.broadcast_to(_add_dim(xp.arange(self.size).reshape(self.shape)), self.index.shape).reshape(-1)
		column = self.index.reshape(-1)

		pos = np.logical_or(np.abs(coef)>tol,np.isnan(coef))
		return (coef[pos],(row[pos],column[pos]))

	def tangent_operator(self,bound_ad=None,**kwargs):
		"""
		The tangent linear operator as a sparse matrix.
		- **kwargs : passed to triplets
		"""
		if bound_ad is None: bound_ad = self.bound_ad()
		return misc.tocsr(self.triplets(**kwargs),shape=(self.size,bound_ad))
		
	def adjoint_operator(self,bound_ad=None,**kwargs):
		"""
		The adjoint of the tangent linear operator as a sparse matrix.
		- **kwargs : passed to triplets
		"""
		if bound_ad is None: bound_ad = self.bound_ad()
		coef,(row,column) = self.triplets(**kwargs)
		return misc.tocsr((coef,(column,row)),shape=(bound_ad,self.size))

	def solve(self,raw=False):
		"""
		Assume that the spAD instance represents the variable y = x + A*delta,
		where delta is a symbolic perturbation. 
		Solves the system x + A*delta = 0, assuming compatible shapes.
		"""
		mat = self.triplets()
		rhs = -self.value.flatten()
		return (mat,rhs) if raw else misc.spsolve(mat,rhs).reshape(self.shape)

	def is_elliptic(self,tol=None,identity_var=None):
		"""
		Tests wether the variable encodes a (linear) degenerate elliptic scheme.
		Output :
		- sum of the coefficients at each position (must be non-negative for 
		degenerate ellipticity, positive for strict ellipticity)
		- maximum of off-diagonal coefficients at each position (must be non-positive)
		Output (if tol is specified) : 
		- min_sum >=-tol and max_off <= tol
		Side effect warning : AD simplification, which is also possibly costly
		"""
		self.simplify_ad()
		min_sum = self.coef.sum(axis=-1)

		rg = (np.arange(self.size).reshape(self.shape+(1,))
			if identity_var is None else identity_var.index)
		coef = self.coef.copy()
		coef[self.index==rg] = -np.inf
		coef[coef==0.] = -np.inf
		max_off = coef.max(axis=-1)

		if tol is None: return min_sum,max_off
		return min_sum.min()>=-tol and max_off.max()<=tol

	@classmethod
	def concatenate(cls,elems,axis=0):
		axis1 = axis if axis>=0 else axis-1
		elems2 = tuple(cls(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		return cls( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(_pad_last(e.coef,size_ad)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.index,size_ad) for e in elems2),axis=axis1))

	# Memory optimization
	def simplify_ad(self,atol=None,rtol=None):
		"""
		Compresses the AD information by merging suitable coefficients, and optionally 
		removing negligible ones.
		- atol : absolute tolerance to discard a coefficient. (True -> sensible default.)
		- rtol : relative tolerance to discard a coefficient (compared to largest in row)
		Operates in place, but also returns itself.
		"""
		# TODO : investigate possible optimizations using np.bincount or scipy.ndimage.sum 
		if atol is True: atol = np.finfo(self.value.dtype).resolution
		if rtol is True: rtol = np.finfo(self.value.dtype).resolution
		if atol is None and rtol is not None: atol=0.
		if rtol is None and atol is not None: rtol=0.

		if self.size_ad==0: # Nothing to simplify
			return self
		if len(self.shape)==0: # Add dimension to scalar-like arrays
			other = self.reshape((1,))
			other.simplify_ad(atol=atol,rtol=rtol)
			other = other.reshape(tuple())
			self.coef,self.index = other.coef,other.index
			return self

		if self.cupy_based(): 
			from .AD_CUDA.simplify_ad import simplify_ad
			return simplify_ad(self,atol=atol,rtol=rtol)

		if not self.index.flags['WRITEABLE']: 
			self.index = self.index.copy()

		# Sort indices by increasing values
		bad_index = np.iinfo(self.index.dtype).max
		bad_pos = self.coef==0
		self.index[bad_pos] = bad_index
		ordering = self.index.argsort(axis=-1)
		self.coef  = np.take_along_axis(self.coef, ordering,axis=-1)
		self.index = np.take_along_axis(self.index,ordering,axis=-1)

		# Accumulate coefficients associated with identical indices
		cum_coef = np.zeros_like(self.value)
		indices  = np.zeros_like(self.index,shape=self.shape)
		size_ad = self.size_ad
		self.coef  = np.moveaxis(self.coef, -1,0)
		self.index = np.moveaxis(self.index,-1,0)
		prev_index = np.copy(self.index[0])

		for i in range(size_ad):
			 # Note : self.index, self.coef change during iterations
			ind,co = self.index[i],self.coef[i]
			pos_new_index = np.logical_and(prev_index!=ind, ind!=bad_index)
			pos_old_index = np.logical_not(pos_new_index)
			prev_index[pos_new_index] = ind[pos_new_index]
			cum_coef[pos_new_index]  = co[pos_new_index]
			cum_coef[pos_old_index] += co[pos_old_index]
			indices[pos_new_index] += 1
			indices_exp = np.expand_dims(indices,axis=0)
			cps.put_along_axis(self.index,indices_exp,prev_index,axis=0)
			cps.put_along_axis(self.coef, indices_exp,cum_coef,axis=0)

		# Eliminate meaningless coefficients, after largest of indices
		indices[self.index[0]==bad_index]=-1
		indices_max = int(np.max(indices,axis=None))
		size_ad_new = indices_max+1
		self.coef  = self.coef[:size_ad_new]
		self.index = self.index[:size_ad_new]
		if size_ad_new==0:
			self.coef  = np.moveaxis(self.coef,0,-1)
			self.index = np.moveaxis(self.index,0,-1)
			return self

		coef_end  = self.coef[ np.maximum(indices_max,0)]
		index_end = self.index[np.maximum(indices_max,0)]
		coef_end[ indices<indices_max] = 0.
		index_end[indices<indices_max] = -1
		while np.min(indices,axis=None)<indices_max:
			indices=np.minimum(indices_max,1+indices)
			indices_exp = np.expand_dims(indices,axis=0)
			cps.put_along_axis(self.coef, indices_exp,coef_end,axis=0)
			cps.put_along_axis(self.index,indices_exp,index_end,axis=0)

		self.coef  = np.moveaxis(self.coef,0,-1)
		self.index = np.moveaxis(self.index,0,-1)
		self.coef  = self.coef.reshape( self.shape+(size_ad_new,))
		self.index = self.index.reshape(self.shape+(size_ad_new,))

		self.index[self.index==-1]=0 # Corresponding coefficient is zero anyway.

		# Optionally remove coefficients below tolerance threshold
		if atol is not None:
			tol = atol
			if rtol!=0:
				max_coef = np.max(np.abs(self.coef),axis=-1)
				tol = np.expand_dims( tol + max_coef*rtol, axis=-1)

			bad_pos = np.abs(self.coef) <= tol
			self.index[bad_pos] = bad_index
			self.coef[ bad_pos] = 0.
			ordering = self.index.argsort(axis=-1)
			self.coef  = np.take_along_axis(self.coef, ordering,axis=-1)
			self.index = np.take_along_axis(self.index,ordering,axis=-1)

			new_size_ad = self.size_ad - np.min(np.sum(bad_pos,axis=-1))
			self.coef = self.coef[...,:new_size_ad]
			self.index = self.index[...,:new_size_ad]
			self.index[self.index==bad_index]=0 

		return self

# -------- End of class spAD -------

# -------- Factory methods -----

def identity(shape=None,constant=None,shift=0):
	shape,constant = misc._set_shape_constant(shape,constant)
	shape2 = shape+(1,)
	xp = cupy_generic.get_array_module(constant)
	int_t=cupy_generic.samesize_int_t(constant.dtype)
	return spAD(constant,np.ones_like(constant,shape=shape2),
		xp.arange(shift,shift+np.prod(shape,dtype=int),dtype=int_t).reshape(shape2))

def register(inputs,iterables=None,shift=0,ident=identity):
	if iterables is None:
		iterables = (tuple,)
	def reg(a):
		nonlocal shift
		a,to_ad = misc.ready_ad(a)
		if to_ad:
			result = ident(constant=a,shift=shift)
			shift += result.size
			return result
		else:
			return a
	return functional.map_iterables(reg,inputs,iterables)

