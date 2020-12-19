# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import functools
from . import ad_generic
from . import cupy_support as cps
from . import misc
from . import Dense
from . import Base

_add_dim = misc._add_dim; _add_dim2 = misc._add_dim2; _add_coef=misc._add_coef;

class denseAD2(Base.baseAD):
	"""
	A class for dense forward second order automatic differentiation
	"""

	def __init__(self,value,coef1=None,coef2=None,broadcast_ad=False):
		if self.is_ad(value): # Case where one should just reproduce value
			assert coef1 is None and coef2 is None
			self.value,self.coef1,self.coef2=value.value,value.coef1,value.coef2
			return
		if ad_generic.is_ad(value):
			raise ValueError("Attempting to cast between different AD types")
		self.value = ad_generic.asarray(value)
		self.coef1 = (cps.zeros_like(value,shape=self.shape+(0,)) if coef1 is None 
			else misc._test_or_broadcast_ad(coef1,self.shape,broadcast_ad) )
		self.coef2 = (cps.zeros_like(value,shape=self.shape+(0,0)) if coef2 is None 
			else misc._test_or_broadcast_ad(coef2,self.shape,broadcast_ad,2) )

	@classmethod
	def order(cls): return 2
	def copy(self,order='C'):
		return self.new(self.value.copy(order=order),
			self.coef1.copy(order=order),self.coef2.copy(order=order))
	def as_tuple(self): return self.value,self.coef1,self.coef2

	# Representation 
	def __iter__(self):
		for value,coef1,coef2 in zip(self.value,self.coef1,self.coef2):
			yield self.new(value,coef1,coef2)

	def __str__(self):
		return "denseAD2("+str(self.value)+","+misc._prep_nl(str(self.coef1))+","+misc._prep_nl(str(self.coef2)) +")"
	def __repr__(self):
		return "denseAD2("+repr(self.value)+","+misc._prep_nl(repr(self.coef1))+","+misc._prep_nl(repr(self.coef2)) +")"

	# Operators
	def as_func(self,h):
		"""Replaces the symbolic perturbation with h"""
		value,coef1,coef2 = (misc.add_ndim(e,h.ndim-1) for e in (self.value,self.coef1,self.coef2))
		hh = cps.expand_dims(h,axis=0) * cps.expand_dims(h,axis=1)
		return value+(coef1*h).sum(axis=self.ndim) + 0.5*(hh*coef2).sum(axis=(self.ndim,self.ndim+1))

	def __add__(self,other):
		if self.is_ad(other):
			return self.new(self.value+other.value,_add_coef(self.coef1,other.coef1),_add_coef(self.coef2,other.coef2))
		else:
			return self.new(self.value+other, self.coef1, self.coef2, broadcast_ad=True)
	def __sub__(self,other):
		if self.is_ad(other):
			return self.new(self.value-other.value,_add_coef(self.coef1,-other.coef1),_add_coef(self.coef2,-other.coef2))
		else:
			return self.new(self.value-other, self.coef1, self.coef2, broadcast_ad=True)

	def __mul__(self,other):
		if self.is_ad(other):
			mixed = cps.expand_dims(self.coef1,axis=-1)*cps.expand_dims(other.coef1,axis=-2)
			return self.new(self.value*other.value, _add_coef(_add_dim(self.value)*other.coef1,_add_dim(other.value)*self.coef1),
				_add_coef(_add_coef(_add_dim2(self.value)*other.coef2,_add_dim2(other.value)*self.coef2),_add_coef(mixed,np.moveaxis(mixed,-2,-1))))
		elif self.isndarray(other):
			return self.new(self.value*other,_add_dim(other)*self.coef1,_add_dim2(other)*self.coef2)
		else:
			return self.new(self.value*other,other*self.coef1,other*self.coef2)

	def __truediv__(self,other):
		if self.is_ad(other):
			return self.__mul__(other.__pow__(-1))
		elif self.isndarray(other):
			inv = 1./other
			return self.new(self.value*inv,_add_dim(inv)*self.coef1,_add_dim2(inv)*self.coef2)
		else:
			inv = 1./other
			return self.new(self.value*inv,self.coef1*inv,self.coef2*inv)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other):	return self.__pow__(-1).__mul__(other)


	def __neg__(self):		return self.new(-self.value,-self.coef1,-self.coef2)

	# Math functions
	def _math_helper(self,deriv): # Inputs : a=f(x), b=f'(x), c=f''(x), where x=self.value
		a,b,c=deriv
		mixed = cps.expand_dims(self.coef1,axis=-1)*cps.expand_dims(self.coef1,axis=-2)
		return self.new(a,_add_dim(b)*self.coef1,_add_dim2(b)*self.coef2+_add_dim2(c)*mixed)

	@classmethod
	def compose(cls,a,t):
		assert cls.is_ad(a) and all(cls.is_ad(b) for b in t)
		b = np.moveaxis(cls.concatenate(t,axis=0),0,-1)
		coef1 = (_add_dim(a.coef1)*b.coef1).sum(axis=-2)
		coef2_pure = (_add_dim2(a.coef1)*b.coef2).sum(axis=-3)
		shape_factor = b.shape[:-1]
		mixed = ( b.coef1.reshape(shape_factor+(a.size_ad,1,b.size_ad,1))
			     *b.coef1.reshape(shape_factor+(1,a.size_ad,1,b.size_ad)) )
		coef2_mixed = (_add_dim2(a.coef2)*mixed).sum(axis=-3).sum(axis=-3)
		return cls.new(a.value,coef1,coef2_pure+coef2_mixed)

	#Indexing
	@property
	def size_ad(self):  return self.coef1.shape[-1]

	def to_first(self): return Dense.denseAD(self.value,self.coef1)
	def gradient(self,i=None): 
		"""Returns the gradient, or the i-th component of the gradient if specified."""
		grad = np.moveaxis(self.coef1,-1,0)
		return grad if i is None else grad[i]
	def hessian(self,i=None,j=None): 
		"""Returns the hessian, or component (i,j) of the hessian if specified."""
		assert (i is None) == (j is None)
		hess = np.moveaxis(self.coef2,(-2,-1),(0,1))
		return hess if i is None else hess[i,j]

	def __getitem__(self,key):
		ekey1,ekey2 = misc.key_expand(key,1),misc.key_expand(key,2)
		return self.new(self.value[key], self.coef1[ekey1], self.coef2[ekey2])

	def __setitem__(self,key,other):
		ekey1,ekey2 = misc.key_expand(key,1),misc.key_expand(key,2)
		if self.is_ad(other):
			osad = other.size_ad
			if osad==0: return self.__setitem__(key,other.value)
			elif self.size_ad==0: 
				self.coef1=cps.zeros_like(self.value,shape=self.coef1.shape[:-1]+(osad,))
				self.coef2=cps.zeros_like(self.value,shape=self.coef2.shape[:-2]+(osad,osad))
			self.value[key] = other.value
			self.coef1[ekey1] = other.coef1
			self.coef2[ekey2] = other.coef2
		else:
			self.value[key] = other
			self.coef1[ekey1] = 0.
			self.coef2[ekey2] = 0.


	def reshape(self,shape,order='C'):
		shape = misc._to_tuple(shape)
		shape1 = shape+(self.size_ad,)
		shape2 = shape+(self.size_ad,self.size_ad)
		return self.new(self.value.reshape(shape,order=order),
			self.coef1.reshape(shape1,order=order), self.coef2.reshape(shape2,order=order))

	def broadcast_to(self,shape):
		shape1 = shape+(self.size_ad,)
		shape2 = shape+(self.size_ad,self.size_ad)
		return self.new(np.broadcast_to(self.value,shape), 
			np.broadcast_to(self.coef1,shape1), np.broadcast_to(self.coef2,shape2))

	def pad(self,pad_width,*args,constant_values=0,**kwargs):
		return self.new(
			np.pad(self.value,pad_width,*args,constant_values=constant_values,**kwargs),
			np.pad(self.coef1,pad_width+((0,0),),*args,constant_values=0,**kwargs),
			np.pad(self.coef2,pad_width+((0,0),(0,0),),*args,constant_values=0,**kwargs),)
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes1 = tuple(axes) +(self.ndim,)
		axes2 = tuple(axes) +(self.ndim,self.ndim+1)
		return self.new(self.value.transpose(axes),
			self.coef1.transpose(axes1),self.coef2.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		axis=misc.normalize_axis(axis,self.ndim)
		out = self.new(self.value.sum(axis,**kwargs), 
			self.coef1.sum(axis,**kwargs), self.coef2.sum(axis,**kwargs))
		return out

	@classmethod
	def concatenate(cls,elems,axis=0):
		axis1,axis2 = (axis,axis) if axis>=0 else (axis-1,axis-2)
		elems2 = tuple(cls(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return cls( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(e.coef1 if e.size_ad==size_ad else 
			cps.zeros_like(e.coef1,shape=e.shape+(size_ad,)) for e in elems2),axis=axis1),
		np.concatenate(tuple(e.coef2 if e.size_ad==size_ad else 
			cps.zeros_like(e.coef2,shape=e.shape+(size_ad,size_ad)) for e in elems2),axis=axis2))

	def apply_linear_operator(self,op):
		return self.new(op(self.value),
			misc.apply_linear_operator(op,self.coef1,flatten_ndim=1),
			misc.apply_linear_operator(op,self.coef2,flatten_ndim=2))
# -------- End of class denseAD2 -------

# -------- Factory method -----

denseAD2_cupy,new = Base.cupy_variant(denseAD2)

def identity(*args,**kwargs):
	arr = Dense.identity(*args,**kwargs)
	return new(arr.value,arr.coef,
		cps.zeros_like(arr.value,shape=arr.shape+(arr.size_ad,arr.size_ad)))

def register(*args,**kwargs):
	return Dense.register(*args,**kwargs,ident=identity)
