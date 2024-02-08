# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import functools
from . import cupy_generic
from . import ad_generic
from . import cupy_support as cps
from . import misc
from . import Sparse
from . import Dense
from . import Dense2
from . import Base

_add_dim = misc._add_dim; _pad_last = misc._pad_last; _concatenate=misc._concatenate;


class spAD2(Base.baseAD):
	"""
	A class for sparse forward second order automatic differentiation
	"""

	def __init__(self,value,coef1=None,index=None,coef2=None,index_row=None,index_col=None,broadcast_ad=False):
		if self.is_ad(value):
			assert (coef1 is None and index is None 
				and coef2 is None and index_row is None and index_col is None)
			self.value,self.coef1,self.index,self.coef2,self.index_row,self.index_col \
				= value.value,value.coef1,value.index,value.coef2,value.index_row,value.index_col
			return
		if ad_generic.is_ad(value):
			raise Base.ADCastError(f"Attempting to cast {type(value)} to incompatible type spAD2")

		# Create instance 
		self.value = ad_generic.asarray(value)
		shape = self.shape
		shape2 = shape+(0,)
		int_t = cupy_generic.samesize_int_t(value.dtype)
		assert ((coef1 is None) and (index is None)) or (coef1.shape==index.shape)
		self.coef1 = (np.zeros_like(value,shape=shape2) if coef1  is None 
			else misc._test_or_broadcast_ad(coef1,shape,broadcast_ad) )
		self.index = (np.zeros_like(value,shape=shape2,dtype=int_t)  if index is None  
			else misc._test_or_broadcast_ad(index,shape,broadcast_ad) )
		
		assert (((coef2 is None) and (index_row is None) and (index_col is None)) 
			or ((coef2.shape==index_row.shape) and (coef2.shape==index_col.shape)))
		self.coef2 = (np.zeros_like(value,shape=shape2) if coef2  is None 
			else misc._test_or_broadcast_ad(coef2,shape,broadcast_ad) )
		self.index_row = (np.zeros_like(value,shape=shape2,dtype=int_t)  if index_row is None 
			else misc._test_or_broadcast_ad(index_row,shape,broadcast_ad) )
		self.index_col = (np.zeros_like(value,shape=shape2,dtype=int_t)  if index_col is None 
			else misc._test_or_broadcast_ad(index_col,shape,broadcast_ad) )

	@classmethod
	def order(cls): return 2
	def copy(self,order='C'):
		return self.new(self.value.copy(order=order),
			self.coef1.copy(order=order),self.index.copy(order=order),
			self.coef2.copy(order=order),self.index_row.copy(order=order),self.index_col.copy(order=order))
	def as_tuple(self): return self.value,self.coef1,self.index,self.coef2,self.index_row,self.index_col

	# Representation 
	def __iter__(self):
		for value,coef1,index,coef2,index_row,index_col in zip(self.value,self.coef1,self.index,self.coef2,self.index_row,self.index_col):
			yield self.new(value,coef1,index,coef2,index_row,index_col)

	def __str__(self):
		return "spAD2"+str((self.value,self.coef1,self.index,self.coef2,self.index_row,self.index_col))
	def __repr__(self):
		return "spAD2"+repr((self.value,self.coef1,self.index,self.coef2,self.index_row,self.index_col))	

	# Operators
	def as_func(self,h):
		"""Replaces the symbolic perturbation with h"""
		value,coef1,coef2 = (misc.add_ndim(e,h.ndim-1) for e in (self.value,self.coef1,self.coef2))
		return (value+(coef1*h[self.index]).sum(axis=self.ndim)
			+0.5*(coef2*h[self.index_row]*h[self.index_col]).sum(axis=self.ndim))

	def __add__(self,other):
		if self.is_ad(other):
			value = self.value+other.value; shape = value.shape
			return self.new(value, 
				_concatenate(self.coef1,other.coef1,shape), _concatenate(self.index,other.index,shape),
				_concatenate(self.coef2,other.coef2,shape), _concatenate(self.index_row,other.index_row,shape), _concatenate(self.index_col,other.index_col,shape))
		else:
			return self.new(self.value+other, self.coef1, self.index, 
				self.coef2, self.index_row, self.index_col, broadcast_ad=True)

	def __sub__(self,other):
		if self.is_ad(other):
			value = self.value-other.value; shape = value.shape
			return self.new(value, 
				_concatenate(self.coef1,-other.coef1,shape), _concatenate(self.index,other.index,shape),
				_concatenate(self.coef2,-other.coef2,shape), _concatenate(self.index_row,other.index_row,shape), _concatenate(self.index_col,other.index_col,shape))
		else:
			return self.new(self.value-other, self.coef1, self.index, 
				self.coef2, self.index_row, self.index_col, broadcast_ad=True)

	def __mul__(self,other):
		if self.is_ad(other):
			value = self.value*other.value
			coef1_a,coef1_b = _add_dim(other.value)*self.coef1,_add_dim(self.value)*other.coef1
			index_a,index_b = np.broadcast_to(self.index,coef1_a.shape),np.broadcast_to(other.index,coef1_b.shape)
			coef2_a,coef2_b = _add_dim(other.value)*self.coef2,_add_dim(self.value)*other.coef2
			index_row_a,index_row_b = np.broadcast_to(self.index_row,coef2_a.shape),np.broadcast_to(other.index_row,coef2_b.shape)
			index_col_a,index_col_b = np.broadcast_to(self.index_col,coef2_a.shape),np.broadcast_to(other.index_col,coef2_b.shape)

			len_a,len_b = self.coef1.shape[-1],other.coef1.shape[-1]
			coef2_ab = np.repeat(self.coef1,len_b,axis=-1) * np.tile(other.coef1,len_a) 
			index2_a = np.broadcast_to(np.repeat(self.index,len_b,axis=-1),coef2_ab.shape)
			index2_b = np.broadcast_to(np.tile(other.index,len_a),coef2_ab.shape)

			return self.new(value,_concatenate(coef1_a,coef1_b),_concatenate(index_a,index_b),
				np.concatenate((coef2_a,coef2_b,coef2_ab,coef2_ab),axis=-1),
				np.concatenate((index_row_a,index_row_b,index2_a,index2_b),axis=-1),
				np.concatenate((index_col_a,index_col_b,index2_b,index2_a),axis=-1))
		elif self.isndarray(other):
			value = self.value*other
			coef1 = _add_dim(other)*self.coef1
			index = np.broadcast_to(self.index,coef1.shape)
			coef2 = _add_dim(other)*self.coef2
			index_row = np.broadcast_to(self.index_row,coef2.shape)
			index_col = np.broadcast_to(self.index_col,coef2.shape)
			return self.new(value,coef1,index,coef2,index_row,index_col)
		else:
			return self.new(self.value*other,other*self.coef1,self.index,
				other*self.coef2,self.index_row,self.index_col)

	def __truediv__(self,other):
		if self.is_ad(other):
			return self.__mul__(other.__pow__(-1))
		elif self.isndarray(other):
			inv = 1./other
			return self.new(self.value*inv,self.coef1*_add_dim(inv),self.index,
				self.coef2*_add_dim(inv),self.index_row,self.index_col)
		else:
			inv = 1./other
			return self.new(self.value*inv,self.coef1*inv,self.index,
				self.coef2*inv,self.index_row,self.index_col)

	__rmul__ = __mul__
	__radd__ = __add__
	def __rsub__(self,other): 		return -(self-other)
	def __rtruediv__(self,other):
		return self.__pow__(-1).__mul__(other)

	def __neg__(self):		return self.new(-self.value,-self.coef1,self.index,
		-self.coef2,self.index_row,self.index_col)

	# Math functions
	def _math_helper(self,deriv): # Inputs : a=f(x), b=f'(x), c=f''(x), where x=self.value
		a,b,c=deriv
		len_1 = self.coef1.shape[-1]
		coef1_r,index_r = np.repeat(self.coef1,len_1,axis=-1),np.repeat(self.index,len_1,axis=-1)
		coef1_t,index_t = np.tile(self.coef1,len_1),np.tile(self.index,len_1) 
		return self.new(a,_add_dim(b)*self.coef1,self.index,
			_concatenate(_add_dim(b)*self.coef2,_add_dim(c)*(coef1_r*coef1_t)),
			_concatenate(self.index_row, index_r),_concatenate(self.index_col, index_t))
	
	@classmethod
	def compose(cls,a,t):
		assert isinstance(a,Dense2.denseAD2) and all(cls.is_ad(b) for b in t)
		b = np.moveaxis(cls.concatenate(t,axis=0),0,-1) # Possible performance hit if ad sizes are inhomogeneous
		coef1 = _add_dim(a.coef1)*b.coef1
		index1 = np.broadcast_to(b.index,coef1.shape)
		coef2_pure = _add_dim(a.coef1)*b.coef2
		index_row_pure = np.broadcast_to(b.index_row,coef2_pure.shape)
		index_col_pure = np.broadcast_to(b.index_col,coef2_pure.shape)

		s = b.shape[:-1]; na = a.size_ad; nb = b.size_ad1;
		coef2_mixed = misc._add_dim2(a.coef2)*np.reshape(b.coef1,s+(na,1,nb,1))*np.reshape(b.coef1,s+(1,na,1,nb))
		s2 = coef2_mixed.shape
		index_row_mixed = np.broadcast_to(b.index.reshape(s+(na,1,nb,1)),s2)
		index_col_mixed = np.broadcast_to(b.index.reshape(s+(1,na,1,nb)),s2)
		#s3 = s2[:-4]+(na*na*nb*nb,) a.reshape(s3)

		coef1,index1,coef2_pure,index_row_pure,index_col_pure = (
			_flatten_nlast(a,2) for a in (coef1,index1,coef2_pure,index_row_pure,index_col_pure))
		coef2_mixed,index_row_mixed,index_col_mixed = (
			_flatten_nlast(a,4) for a in (coef2_mixed,index_row_mixed,index_col_mixed))
		
		return cls.new(a.value,coef1,index1,
			_concatenate(coef2_pure,coef2_mixed),_concatenate(index_row_pure,index_row_mixed),
			_concatenate(index_col_pure,index_col_mixed))

	#Indexing
	@property
	def size_ad1(self):  return self.coef1.shape[-1]
	@property
	def size_ad2(self):  return self.coef2.shape[-1]

	def __getitem__(self,key):
		ekey = misc.key_expand(key)
		try:
			return self.new(self.value[key], 
				self.coef1[ekey], self.index[ekey], 
				self.coef2[ekey], self.index_row[ekey], self.index_col[ekey])
		except ZeroDivisionError: # Some cupy versions fail indexing correctly if size==0
			value = self.value[ekey]
			shape = value.shape
			def take(arr): 
				size_ad = arr.shape[-1]
				return arr[ekey] if size_ad>0 else np.zeros_like(arr,shape=(*shape,size_ad))
			return self.new(self.value[key], 
				take(self.coef1), take(self.index), 
				take(self.coef2), take(self.index_row), take(self.index_col))

	def __setitem__(self,key,other):
		ekey = misc.key_expand(key)
		if self.is_ad(other):
			self.value[key] = other.value

			pad_size = max(self.coef1.shape[-1],other.coef1.shape[-1])
			if pad_size>self.coef1.shape[-1]:
				self.coef1 = _pad_last(self.coef1,pad_size)
				self.index = _pad_last(self.index,pad_size)
			self.coef1[ekey] = _pad_last(other.coef1,pad_size)
			self.index[ekey] = _pad_last(other.index,pad_size)

			pad_size = max(self.coef2.shape[-1],other.coef2.shape[-1])
			if pad_size>self.coef2.shape[-1]:
				self.coef2 = _pad_last(self.coef2,pad_size)
				self.index_row = _pad_last(self.index_row,pad_size)
				self.index_col = _pad_last(self.index_col,pad_size)
			self.coef2[ekey] = _pad_last(other.coef2,pad_size)
			self.index_row[ekey] = _pad_last(other.index_row,pad_size)
			self.index_col[ekey] = _pad_last(other.index_col,pad_size)
		else:
			self.value[key] = other
			self.coef1[ekey] = 0.
			self.coef2[ekey] = 0.

	def reshape(self,shape,order='C'):
		value = self.value.reshape(shape,order=order)
		shape1 = value.shape+(self.size_ad1,)
		shape2 = value.shape+(self.size_ad2,)
		return self.new(value,
			self.coef1.reshape(shape1,order=order), self.index.reshape(shape1,order=order),
			self.coef2.reshape(shape2,order=order),self.index_row.reshape(shape2,order=order),
			self.index_col.reshape(shape2,order=order))

	def broadcast_to(self,shape):
		shape1 = shape+(self.size_ad1,)
		shape2 = shape+(self.size_ad2,)
		return self.new(np.broadcast_to(self.value,shape), 
			np.broadcast_to(self.coef1,shape1), np.broadcast_to(self.index,shape1),
			np.broadcast_to(self.coef2,shape2), np.broadcast_to(self.index_row,shape2), 
			np.broadcast_to(self.index_col,shape2))

	def pad(self,pad_width,*args,constant_values=0,**kwargs):
		def _pad(arr):return np.pad(arr,pad_width+((0,0),),*args,constant_values=0,**kwargs)
		return self.new(
			np.pad(self.value,pad_width,*args,constant_values=constant_values,**kwargs),
			_pad(self.coef1),_pad(self.index),
			_pad(self.coef2),_pad(self.index_row),_pad(self.index_col))
	
	def transpose(self,axes=None):
		if axes is None: axes = tuple(reversed(range(self.ndim)))
		axes2 = tuple(axes) +(self.ndim,)
		return self.new(self.value.transpose(axes),
			self.coef1.transpose(axes2),self.index.transpose(axes2),
			self.coef2.transpose(axes2),self.index_row.transpose(axes2),
			self.index_col.transpose(axes2))

	def allclose(self,other,*args,**kwargs):
		raise ValueError("Unsupported, sorry, please try allclose(a.to_dense(),b.to_dense())")

	# Reductions
	def sum(self,axis=None,out=None,**kwargs):
		if axis is None: return self.flatten().sum(axis=0,out=out,**kwargs)
		value = self.value.sum(axis,**kwargs)

		shape1 = value.shape + (self.size_ad1 * self.shape[axis],)
		coef1 = np.moveaxis(self.coef1, axis,-1).reshape(shape1)
		index = np.moveaxis(self.index, axis,-1).reshape(shape1)

		shape2 = value.shape + (self.size_ad2 * self.shape[axis],)
		coef2 = np.moveaxis(self.coef2, axis,-1).reshape(shape2)
		index_row = np.moveaxis(self.index_row, axis,-1).reshape(shape2)
		index_col = np.moveaxis(self.index_col, axis,-1).reshape(shape2)

		out = self.new(value,coef1,index,coef2,index_row,index_col)
		return out

	# Conversion
	def bound_ad(self):
		def maxi(a): return int(cps.max(a,initial=-1))
		return 1+np.max((maxi(self.index),maxi(self.index_row),maxi(self.index_col)))
	def to_dense(self,dense_size_ad=None):
		# Can be accelerated using np.bincount, similarly to spAD.to_dense
		def mvax(arr): return np.moveaxis(arr,-1,0)
		dsad = self.bound_ad() if dense_size_ad is None else dense_size_ad
		coef1 = np.zeros_like(self.value,shape=self.shape+(dsad,))
		for c,i in zip(mvax(self.coef1),mvax(self.index)):
			np.put_along_axis(coef1,_add_dim(i),np.take_along_axis(coef1,_add_dim(i),axis=-1)+_add_dim(c),axis=-1)
		coef2 = np.zeros_like(self.value,shape=self.shape+(dsad*dsad,))
		for c,i in zip(mvax(self.coef2),mvax(self.index_row*dsad+self.index_col)):
			np.put_along_axis(coef2,_add_dim(i),np.take_along_axis(coef2,_add_dim(i),axis=-1)+_add_dim(c),axis=-1)
		return Dense2.denseAD2(self.value,coef1,np.reshape(coef2,self.shape+(dsad,dsad)))

	def to_first(self):
		return Sparse.spAD(self.value,self.coef1,self.index)

	#Linear algebra
	def triplets(self):
		"""The hessian operator, presented as triplets"""
		return (self.coef2,(self.index_row,self.index_col))

	def hessian_operator(self,shape=None):
		"""
		The hessian operator, presented as an opaque matrix class, supporting mul.
		Implicitly sums over all axes. Recommendation : apply simplify_ad before call.
		"""
		return misc.tocsr(self.sum().triplets(),shape=shape)

	def tangent_operator(self): return self.to_first().tangent_operator()
	def adjoint_operator(self): return self.to_first().adjoint_operator()

	def solve_stationnary(self,raw=False):
		"""
		Finds a stationnary point to a quadratic function, provided as a spAD2 array scalar. 
		Use "raw = True" to obtain the raw linear system and use your own solver.
		"""
		mat = self.triplets()
		rhs = - self.to_first().to_dense(self.bound_ad()).coef
		return (mat,rhs) if raw else misc.spsolve(mat,rhs)

	def solve_weakform(self,raw=False):
		"""
		Assume that a spAD2 array scalar represents the quadratic function
		Q(u,v) = a0 + a1.(u,v) + (u,v).a2.(u,v) of the variable (u,v).
		Finds u such that Q(u,v) is independent of v.
		Use "raw = True" to obtain the raw linear system and use your own solver.
		"""
		(coef,(row,col)),rhs = self.solve_stationnary(raw=True)
		n = rhs.size//2
		rhs = rhs[n:]
		pos = np.logical_and(row>=n,col<n)
		mat = (coef[pos],(row[pos]-n,col[pos]))
		return (mat,rhs) if raw else misc.spsolve(mat,rhs)
	
	# Static methods
	@classmethod
	def concatenate(cls,elems,axis=0):
		axis1 = axis if axis>=0 else axis-1
		elems2 = tuple(cls(e) for e in elems)
		size_ad1 = max(e.size_ad1 for e in elems2)
		size_ad2 = max(e.size_ad2 for e in elems2)
		return cls( 
		np.concatenate(tuple(e.value for e in elems2), axis=axis), 
		np.concatenate(tuple(_pad_last(e.coef1,size_ad1)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.index,size_ad1)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.coef2,size_ad2)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.index_row,size_ad2)  for e in elems2),axis=axis1),
		np.concatenate(tuple(_pad_last(e.index_col,size_ad2)  for e in elems2),axis=axis1))

	def simplify_ad(self,*args,**kwargs):
		spHelper1 = Sparse.spAD(self.value,self.coef1,self.index)
		spHelper1.simplify_ad(*args,**kwargs)
		self.coef1,self.index = spHelper1.coef,spHelper1.index

		if self.size_ad2>0: # Otherwise empty max
			n_col = 1+np.max(self.index_col)
			index = self.index_row.astype(np.int64)*n_col + self.index_col.astype(np.int64)
			spHelper2 = Sparse.spAD(self.value,self.coef2,index)
			spHelper2.simplify_ad(*args,**kwargs)
			self.coef2 = spHelper2.coef
			int_t = self.index_row.dtype.type
			self.index_row,self.index_col = spHelper2.index//n_col, spHelper2.index%n_col
			if int_t!=np.int64: self.index_row,self.index_col = (
				e.astype(int_t) for e in (self.index_row,self.index_col))
			
		return self

# -------- End of class spAD2 -------

# -------- Utilities ------

def _flatten_nlast(a,n):
	assert n>0
	s=a.shape
	return a.reshape(s[:-n]+(np.prod(s[-n:]),))

# -------- Factory method -----

def identity(*args,**kwargs):
	arr = Sparse.identity(*args,**kwargs)
	shape2 = arr.shape+(0,)
	return spAD2(arr.value,arr.coef,arr.index,
		np.zeros_like(arr.coef,shape=shape2),
		np.zeros_like(arr.index,shape=shape2),
		np.zeros_like(arr.index,shape=shape2))

def register(*args,**kwargs):
	return Sparse.register(*args,**kwargs,ident=identity)

# ---------- Sparse hessian extraction ---------

def _hessian_operator_noAD(f,x,simplify_ad=None,fargs=tuple()):
	"""Same as Hessian operator, but does not support f with AD values"""
	x_ad = identity(constant=x)
	try: f_ad = f(x_ad,*fargs)
	except Base.ADCastError: return hessian_operator_denseAD(f,x,simplify_ad=simplify_ad,fargs=fargs)
	if simplify_ad or simplify_ad is None and f_ad.ndim > 0: f_ad.simplify_ad(atol=True,rtol=True)
	return f_ad.hessian_operator( shape=(x_ad.size,x_ad.size) )

def hessian_operator(f,x,simplify_ad=None,fargs=tuple(),rev_ad=False):
	"""
	Returns the sparse matrix associated to the hessian of f at x,
	generated using automatic differentiation.
	Typically used to obtain the sparse matrix of a quadratic form.
	Output of f is summed, if non-scalar.
	- simplify_ad (optional): wether to simplify the ad information 
	   before generating the sparse matrix


	*Autodiff support* 
	Consider the functional D * u**2, written with the following convention
	>>> def f(u,D,ad_channel=lambda x:x): return ad_channel(D) * u**2
	See Eikonal.cuda.AnisotropicWave, classes AcousticHamiltonian_Sparse and 
	ElasticHamiltonian_Sparse for non-trivial examples.

	- Foward autodiff. Returns an denseAD_Lin class (operator plus perturbation), 
	if f(x) is a first order dense forward AD variable, with the following convention/example :   
	>>> ad.Sparse2.hessian_operator(f,np.zeros(10),fargs=(ad.Dense.identity(constant=2),))

	- Reverse autodiff support (rev_ad=True). The components of f(x,ad_channel=ones_like)
	are regarded as independent contributions w.r.t which to compute the sensitivity of the result.
	"""
	f_dense = f(x,*fargs)
	if simplify_ad is None: simplify_ad = f_dense.ndim>0
	if isinstance(f_dense,Dense.denseAD): # ---- Forward autodiff support ----
		size_ad = f_dense.size_ad
		x_ad = identity(constant=x)
		shape = x_ad.size,x_ad.size
		ops = []
		for i in range(-1,size_ad):
			f_sparse = f(x_ad,*fargs,ad_channel=lambda x:x.value if i==-1 else x.coef[...,i])
			if simplify_ad: f_sparse.simplify_ad(atol=True,rtol=True)
			ops.append(f_sparse.hessian_operator(shape=shape))
		return Dense.denseAD_Lin(ops[0],ops[1:])

	op = _hessian_operator_noAD(f,x,simplify_ad,fargs)
	if not rev_ad : return op #  ---- No autodiff ----
	
	# ---- Reverse autodiff support ----
	x_ad = identity(constant=x)
	f_sparse = f(x_ad,*fargs,ad_channel=lambda x:np.ones_like(x))
	if simplify_ad: f_sparse.simplify_ad(atol=True,rtol=True)
	coef2,row,col = f_sparse.coef2,f_sparse.index_row,f_sparse.index_col
	f_sparse = None # deallocate first and zero-th order coefficients
	
	def sensitivity(x):
		x=x.reshape((x_ad.size,*x.shape[x_ad.ndim:]))
		return np.sum(coef2.reshape(coef2.shape+(1,)*x.ndim)
			*x.value[row][...,None]*x.coef[col],axis=coef2.ndim-1)

	return op, sensitivity



