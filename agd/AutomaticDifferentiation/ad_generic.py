# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import itertools
import numpy as np
import functools

from . import functional
from .Base import is_ad,isndarray,array,asarray,_cp_ndarray

"""
This file implements functions which apply indifferently to several AD types.
"""

def adtype(data,iterables=tuple()):
	"""
	Returns None if no ad variable found, or the adtype if one is found.
	Also checks consistency of the ad types.
	"""
	result = None
	for value in rec_iter(data,iterables):
		t=type(x)
		if is_ad(t): 
			if result is None: result=t
			else: assert result==t
	return result

def precision(x):
	"""
	Precision of the floating point type of x.
	"""
	if not isinstance(x,type): x = array(x).dtype.type
	return np.finfo(x).precision	

def remove_ad(data,iterables=tuple()):
	def f(a): return a.value if is_ad(a) else a
	return functional.map_iterables(f,data,iterables)

def as_writable(a):
	"""
	Returns a writable array containing the same elements as a.
	If the array a, or a field of a for an AD type, is flagged as 
	non-writable, then it is copied.
	"""
	if isinstance(a,(np.ndarray,_cp_ndarray)):
		return a if a.flags['WRITEABLE'] else a.copy()
	return a.new(*tuple(as_writable(e) for e in a.as_tuple() ))

def common_cast(*args):
	"""
	If any of the arguments is an AD type, casts all other arguments to that type.
	Casts to ndarray if no argument is an AD type. 
	Usage : if a and b may or may not b AD arrays, 
	a,b = common_cast(a,b); a[0]=b[0]
	"""
	args = tuple(array(x) for x in args)
	common_type = None
	for x in args: 
		if is_ad(x):
			if common_type is None:
				common_type = type(x)
			if not isinstance(x,common_type):
				raise ValueError("Error : several distinct AD types")
	return args if common_type is None else tuple(common_type(x) for x in args)


def min_argmin(array,axis=None):
	if axis is None: return min_argmin(array.reshape(-1),axis=0)
	ai = np.argmin(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

def max_argmax(array,axis=None):
	if axis is None: return max_argmax(array.reshape(-1),axis=0)
	ai = np.argmax(array,axis=axis)
	return np.squeeze(np.take_along_axis(array,np.expand_dims(ai,
		axis=axis),axis=axis),axis=axis),ai

# ------- Linear operators ------


def apply_linear_mapping(matrix,rhs,niter=1):
	"""
	Applies the provided linear operator, to a dense AD variable of first or second order.
	"""
	def step(x): return np.dot(matrix,x) if isinstance(matrix,np.ndarray) else (matrix*x)
	operator = functional.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

def apply_linear_inverse(solver,matrix,rhs,niter=1):
	"""
	Applies the provided linear inverse to a dense AD variable of first or second order.
	"""
	def step(x): return solver(matrix,x)
	operator = functional.recurse(step,niter)
	return rhs.apply_linear_operator(operator) if is_ad(rhs) else operator(rhs)

# ------- Shape manipulation -------

def squeeze_shape(shape,axis):
	if axis is None:
		return shape
	assert shape[axis]==1
	if axis==-1:
		return shape[:-1]
	else:
		return shape[:axis]+shape[(axis+1):]

def expand_shape(shape,axis):
	if axis is None:
		return shape
	if axis==-1:
		return shape+(1,)
	if axis<0:
		axis+=1
	return shape[:axis]+(1,)+shape[axis:]

def _set_shape_free_bound(shape,shape_free,shape_bound):
	if shape_free is not None:
		assert shape_free==shape[0:len(shape_free)]
		if shape_bound is None: 
			shape_bound=shape[len(shape_free):]
		else: 
			assert shape_bound==shape[len(shape_free):]
	if shape_bound is None: 
		shape_bound = tuple()
	assert len(shape_bound)==0 or shape_bound==shape[-len(shape_bound):]
	if shape_free is None:
		if len(shape_bound)==0:
			shape_free = shape
		else:
			shape_free = shape[:len(shape)-len(shape_bound)]
	return shape_free,shape_bound

def disassociate(array,shape_free=None,shape_bound=None,
	expand_free_dims=-1,expand_bound_dims=-1):
	"""
	Turns an array of shape shape_free + shape_bound 
	into an array of shape shape_free whose elements 
	are arrays of shape shape_bound.
	Typical usage : recursive automatic differentiation.
	Caveat : by defaut, singleton dimensions are introduced 
	to avoid numpy's "clever" treatment of scalar arrays.

	Arguments: 
	- array : reshaped array
	- (optional) shape_free, shape_bound : outer and inner array shapes. One is deduced from the other.
	- (optional) expand_free_dims, expand_bound_dims. 
	"""
	shape_free,shape_bound = _set_shape_free_bound(array.shape,shape_free,shape_bound)
	shape_free  = expand_shape(shape_free, expand_free_dims)
	shape_bound = expand_shape(shape_bound,expand_bound_dims)
	
	size_free = np.prod(shape_free)
	array = array.reshape((size_free,)+shape_bound)
	result = np.zeros(size_free,object)
	for i in range(size_free): result[i] = array[i]
	return result.reshape(shape_free)

def associate(array,squeeze_free_dims=-1,squeeze_bound_dims=-1):
	"""
	Turns an array of shape shape_free, whose elements 
	are arrays of shape shape_bound, into an array 
	of shape shape_free+shape_bound.
	Inverse opeation to disassociate.
	"""
	if is_ad(array): 
		return array.associate(squeeze_free_dims,squeeze_bound_dims)
	result = np.stack(array.reshape(-1),axis=0)
	shape_free  = squeeze_shape(array.shape,squeeze_free_dims)
	shape_bound = squeeze_shape(result.shape[1:],squeeze_bound_dims) 
	return result.reshape(shape_free+shape_bound)
