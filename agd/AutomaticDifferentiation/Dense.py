# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import functools
import numpy as np
from . import functional
from . import Base
from . import cupy_support as cps
from . import ad_generic
from . import misc


_add_dim = misc._add_dim; _add_coef=misc._add_coef

class denseAD(Base.baseAD):
	"""
	A class for dense forward automatic differentiation
	"""

	def __init__(self,value,coef=None,broadcast_ad=False):
		if self.is_ad(value): # Case where one should just reproduce value
			assert coef is None
			self.value,self.coef=value.value,value.coef
			return
		if ad_generic.is_ad(value):
			raise ValueError("Attempting to cast between different AD types")
		self.value = ad_generic.asarray(value)
		self.coef = (cps.zeros_like(value,shape=self.shape+(0,)) if coef is None 
			else misc._test_or_broadcast_ad(coef,self.shape,broadcast_ad) )
		self._init_cupy()

	@classmethod
	def order(cls): return 1
	def copy(self,order='C'):
		return self.new(self.value.copy(order=order),self.coef.copy(order=order))
	def as_tuple(self): return self.value,self.coef
	
	# Representation 
	def __iter__(self):
		for value,coef in zip(self.value,self.coef):
			yield self.new(value,coef)
	def __str__(self):
		return "denseAD("+str(self.value)+","+misc._prep_nl(str(self.coef))+")"
	def __repr__(self):
		return "denseAD("+repr(self.value)+","+misc._prep_nl(repr(self.coef))+")"

	# Operators
	def as_func(self,h):
		"""Replaces the symbolic perturbation with h"""
		value,coef = (misc.add_ndim(e,h.ndim-1) for e in (self.value,self.coef))
		return value+(coef*h).sum(axis=self.ndim)

	def __add__(self,other):
		if self.is_ad(other):
			return self.new(self.value+other.value, _add_coef(self.coef,other.coef))
		else:
			return self.new(self.value+other, self.coef, broadcast_ad=True)

	def __sub__(self,other):
		if self.is_ad(other):
			return self.new(self.value-other.value, _add_coef(self.coef,-other.coef))
		else:
			return self.new(self.value-other, self.coef, broadcast_ad=True)

	def __mul__(self,other):
		if self.is_ad(other):
			return self.new(self.value*other.value,_add_coef(_add_dim(other.value)*self.coef,
				_add_dim(self.value)*other.coef))
		elif self.isndarray(other):
			return self.new(self.value*other,_add_dim(other)*self.coef)
		else:
			return self.new(self.value*other,other*self.coef)

	def __truediv__(self,other):
		if self.is_ad(other):
			return self.new(self.value/other.value,
				_add_coef(_add_dim(1/other.value)*self.coef,
					_add_dim(-self.value/other.value**2)*other.coef))
		elif self.isndarray(other):
			return self.new(self.value/other,_add_dim(1./other)*self.coef)
		else:
			return self.new(self.value/other,(1./other)*self.coef) 

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other):     return -(self-other)
	def __rtruediv__(self,other): return self.new(other/self.value,
		_add_dim(-other/self.value**2)*self.coef)

	def __neg__(self): return self.new(-self.value,-self.coef)

	# Math functions
	def _math_helper(self,deriv):
		a,b=deriv
		return self.new(a,_add_dim(b)*self.coef)

	@classmethod
	def compose(cls,a,t):
		assert cls.is_ad(a) and all(cls.is_ad(b) for b in t)
		b = np.moveaxis(cls.concatenate(t,axis=0),0,-1)
		coef = (_add_dim(a.coef)*b.coef).sum(axis=-2)
		return cls(a.value,coef)

	#Indexing
	@property
	def size_ad(self):  return self.coef.shape[-1]

	def gradient(self,i=None): 
		"""Returns the gradient, or the i-th component of the gradient if specified."""
		grad = np.moveaxis(self.coef,-1,0)
		return grad if i is None else grad[i]

	def __getitem__(self,key):
		ekey = misc.key_expand(key)
		return self.new(self.value[key], self.coef[ekey])

	def __setitem__(self,key,other):
		ekey = misc.key_expand(key)
		if self.is_ad(other):
			if other.size_ad==0: return self.__setitem__(key,other.value)
			elif self.size_ad==0: 
				self.coef=cps.zeros_like(self.value,shape=self.shape+(other.size_ad,))
			self.value[key] = other.value
			self.coef[ekey] =  other.coef
		else:
			self.value[key] = other
			self.coef[ekey] = 0.

	def reshape(self,shape,order='C'):
		shape = misc._to_tuple(shape)
		shape2 = shape+(self.size_ad,)
		return self.new(self.value.reshape(shape,order=order),self.coef.reshape(shape2,order=order))

	def broadcast_to(self,shape):
		shape2 = shape+(self.size_ad,)
		return self.new(np.broadcast_to(self.value,shape), np.broadcast_to(self.coef,shape2) )

	def pad(self,pad_width,*args,constant_values=0,**kwargs):
		return self.new(
			np.pad(self.value,pad_width,*args,constant_values=constant_values,**kwargs),
			np.pad(self.coef,pad_width+((0,0),),*args,constant_values=0,**kwargs))
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes)+(self.ndim,)
		return self.new(self.value.transpose(axes),self.coef.transpose(axes2))

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		axis=misc.normalize_axis(axis,self.ndim)
		out = self.new(self.value.sum(axis,**kwargs), self.coef.sum(axis,**kwargs))
		return out

	# Numerical 
	def solve(self,shape_free=None,shape_bound=None):
		shape_free,shape_bound = ad_generic._set_shape_free_bound(self.shape,shape_free,shape_bound)
		assert np.prod(shape_free)==self.size_ad
		v = np.moveaxis(np.reshape(self.value,(self.size_ad,)+shape_bound),0,-1)
		a = np.moveaxis(np.reshape(self.coef,(self.size_ad,)+shape_bound+(self.size_ad,)),0,-2)
		return -np.reshape(np.moveaxis(np.linalg.solve(a,v),-1,0),self.shape)

	@classmethod
	def concatenate(cls,elems,axis=0):
		axis1 = axis if axis>=0 else axis-1
		elems2 = tuple(cls(e) for e in elems)
		size_ad = max(e.size_ad for e in elems2)
		assert all((e.size_ad==size_ad or e.size_ad==0) for e in elems2)
		return cls( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(e.coef if e.size_ad==size_ad else 
			cps.zeros_like(e.value,shape=e.shape+(size_ad,)) for e in elems2),axis=axis1))

	def associate(self,squeeze_free_dims=-1,squeeze_bound_dims=-1):
		from . import associate
		sq_free = squeeze_free_dims
		sq_free1= sq_free if sq_free>=0 else (sq_free-1)
		value = associate(self.value,sq_free, squeeze_bound_dims)
		coef  = associate(self.coef, sq_free1,squeeze_bound_dims)
		coef = np.moveaxis(coef,self.ndim if sq_free is None else (self.ndim-1),-1)
		return self.new(value,coef)

	def apply_linear_operator(self,op):
		return self.new(op(self.value),misc.apply_linear_operator(op,self.coef,flatten_ndim=1))

# -------- End of class denseAD -------
# ------ Cupy support ------

denseAD_cupy,new = Base.cupy_variant(denseAD)

# -------- Factory methods -----


def identity(shape=None,shape_free=None,shape_bound=None,constant=None,shift=(0,0)):
	"""
	Creates a dense AD variable with independent symbolic perturbations for each coordinate
	(unless some are tied together as specified by shape_free and shape_bound)
	"""
	shape,constant = misc._set_shape_constant(shape,constant)
	shape_free,shape_bound = ad_generic._set_shape_free_bound(shape,shape_free,shape_bound)

	ndim_elem = len(shape)-len(shape_bound)
	shape_elem = shape[:ndim_elem]
	size_elem = int(np.prod(shape_elem))
	size_ad = shift[0]+size_elem+shift[1]
	coef1 = cps.zeros_like(constant,shape=(size_elem,size_ad))
	for i in range(size_elem):
		coef1[i,shift[0]+i]=1.
	coef1 = coef1.reshape(shape_elem+(1,)*len(shape_bound)+(size_ad,))
	if coef1.shape[:-1]!=constant.shape: 
		coef1 = np.broadcast_to(coef1,shape+(size_ad,))
	return new(constant,coef1)

def register(inputs,iterables=None,shape_bound=None,shift=(0,0),ident=identity,considered=None):
	"""
	Creates a series of dense AD variables with independent symbolic perturbations for each coordinate,
	and adequate intermediate shifts.
	"""
	if iterables is None:
		iterables = (tuple,)
	boundsize = 1 if shape_bound is None else np.prod(shape_bound,dtype=int)
	def is_considered(a):
		return considered is None or a in considered

	start=shift[0]
	starts = []
	def setstart(a):
		nonlocal start,starts
		if considered is None or any(a is val for val in considered):
			a,to_ad = misc.ready_ad(a)
			if to_ad: 
				starts.append(start)
				start += a.size//boundsize
				return a
		starts.append(None)
		return a
	inputs = functional.map_iterables(setstart,inputs,iterables)

	end = start+shift[1]

	starts_it = iter(starts)
	def setad(a):
		start = next(starts_it)
		if start is None: return a
		else: return ident(constant=a,shift=(start,end-start-a.size//boundsize),
			shape_bound=shape_bound)
	return functional.map_iterables(setad,inputs,iterables)

