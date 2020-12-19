# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import numbers
from .functional import map_iterables,map_iterables2,pair
from .cupy_generic import isndarray,from_cupy,cp
from .ad_generic import is_ad,remove_ad
from . import ad_generic
from . import cupy_support as cps

# ------- Ugly utilities -------
def normalize_axis(axis,ndim,allow_tuple=True):
	if allow_tuple and isinstance(axis,tuple): 
		return tuple(normalize_axis(ax,ndim,False) for ax in axis)
	if axis<0: return axis+ndim
	return axis

def add_ndim(arr,n): return np.reshape(arr,arr.shape+(1,)*n)
def _add_dim(a):     return cps.expand_dims(a,axis=-1)	
def _add_dim2(a):    return _add_dim(_add_dim(a))

def _to_tuple(a): return tuple(a) if hasattr(a,"__iter__") else (a,)

def key_expand(key,depth=1): 
	"""Modifies a key to access an array with more dimensions. Needed if ellipsis is used."""
	if isinstance(key,tuple):
		if any(a is ... for a in key):
			return key + (slice(None),)*depth
	return key

def _pad_last(a,pad_total): # Always makes a deep copy
		return np.pad(a, pad_width=((0,0),)*(a.ndim-1)+((0,pad_total-a.shape[-1]),), mode='constant', constant_values=0)
def _add_coef(a,b):
	if a.shape[-1]==0: return b
	elif b.shape[-1]==0: return a
	else: return a+b
def _prep_nl(s): return "\n"+s if "\n" in s else s

def _concatenate(a,b,shape=None):
	if shape is not None:
		if a.shape[:-1]!=shape: a = np.broadcast_to(a,shape+a.shape[-1:])
		if b.shape[:-1]!=shape: b = np.broadcast_to(b,shape+b.shape[-1:])
	return np.concatenate((a,b),axis=-1)

def _set_shape_constant(shape=None,constant=None):
	if isndarray(shape): shape=tuple(shape)
	if constant is None:
		if shape is None:
			raise ValueError("Error : unspecified shape or constant")
		constant = np.full(shape,0.)
	else:
		if not isndarray(constant):
			constant = ad_generic.asarray(constant)
		if shape is not None and shape!=constant.shape: 
			raise ValueError("Error : incompatible shape and constant")
		else:
			shape=constant.shape
	return shape,constant

def _test_or_broadcast_ad(array,shape,broadcast,ad_depth=1):
	if broadcast:
		if array.shape[:-ad_depth]==shape:
			return array
		else:
			return np.broadcast_to(array,shape+array.shape[-ad_depth:])
	else:
		assert array.shape[:-ad_depth]==shape
		return array



# -------- For Dense and Dense2 -----

def apply_linear_operator(op,rhs,flatten_ndim=0):
	"""
	Applies a linear operator to an array with more than two dimensions,
	by flattening the last dimensions
	"""
	assert (rhs.ndim-flatten_ndim) in [1,2]
	shape_tail = rhs.shape[1:]
	op_input = rhs.reshape((rhs.shape[0],np.prod(shape_tail,dtype=int)))
	op_output = op(op_input)
	return op_output.reshape((op_output.shape[0],)+shape_tail)


# -------- Functional iteration, mainly for Reverse and Reverse2 -------

def ready_ad(a):
	"""
	Readies a variable for adding ad information, if possible.
	Returns : readied variable, boolean (wether AD extension is possible)
	"""
	if is_ad(a):
		raise ValueError("Variable a already contains AD information")
	elif isinstance(a,numbers.Real) and not isinstance(a,numbers.Integral):
		return np.array(a),True
	elif isndarray(a) and not issubclass(a.dtype.type,numbers.Integral):
		return a,True
	else:
		return a,False

# Applying a function
def _apply_output_helper(rev,val,iterables):
	"""
	Adds 'virtual' AD information to an output (with negative indices), 
	in selected places.
	"""
	def f(a):
		a,to_ad = ready_ad(a)
		if to_ad:
			shape = pair(rev.size_rev,a.shape)
			return rev._identity_rev(constant=a),shape		
		else:
			return a,None
	return map_iterables(f,val,iterables,split=True)


def register(identity,data,iterables):
	def reg(a):
		a,to_ad = ready_ad(a)
		if to_ad: return identity(constant=a)
		else: return a 
	return map_iterables(reg,data,iterables)


def _to_shapes(coef,shapes,iterables):
	"""
	Reshapes a one dimensional array into the given shapes, 
	given as a tuple of pair(start,shape) 
	"""
	def f(s):
		if s is None:
			return None
		else:
			start,shape = s
			return coef[start : start+np.prod(shape,dtype=int)].reshape(shape)
	return map_iterables(f,shapes,iterables)

def _apply_input_helper(args,kwargs,cls,iterables):
	"""
	Removes the AD information from some function input, and provides the correspondance.
	"""
	corresp = []
	def _make_arg(a):
		nonlocal corresp
		if is_ad(a):
			assert isinstance(a,cls)
			a_value = remove_ad(a)
			corresp.append((a,a_value))
			return a_value
		else:
			return a
	_args = tuple(map_iterables(_make_arg,val,iterables) for val in args)
	_kwargs = {key:map_iterables(_make_arg,val,iterables) for key,val in kwargs.items()}
	return _args,_kwargs,corresp


def sumprod(u,v,iterables,to_first=False):
	acc=0.
	def f(u,v):
		nonlocal acc
		if u is not None: 
			U = u.to_first() if to_first else u
			acc=acc+(U*v).sum()
	map_iterables2(f,u,v,iterables)
	return acc

def reverse_mode(co_output):
	if co_output is None: 
		return "Forward"
	else:
		assert isinstance(co_output,pair)
		c,_ = co_output
		if isinstance(c,pair):
			return "Reverse2"
		else: 
			return "Reverse"

# ----- Functionnal -----

def recurse(step,niter=1):
	def operator(rhs):
		nonlocal step,niter
		for i in range(niter):
			rhs=step(rhs)
		return rhs
	return operator

# ------- Common functions -------

def as_flat(a):
	return a.reshape(-1) if isndarray(a) else ad_generic.array([a])

def tocsr(triplets,shape=None):
	"""Turns sparse matrix given as triplets into a csr (compressed sparse row) matrix"""
	if from_cupy(triplets[0]): import cupyx; spmod = cupyx.scipy.sparse
	else: import scipy.sparse as spmod
	return spmod.coo_matrix(triplets,shape=shape).tocsr()	

def spsolve(triplets,rhs):
	"""
	Solves a sparse linear system where the matrix is given as triplets.
	"""
	if from_cupy(triplets[0]): 
		import cupyx; 
		solver = cupyx.scipy.sparse.linalg.lsqr # Only available solver
	else:
		import scipy.sparse.linalg
		solver = scipy.sparse.linalg.spsolve
	return solver(tocsr(triplets),rhs)		

def spapply(triplets,rhs,crop_rhs=False):
	"""
	Applies a sparse matrix, given as triplets, to an rhs.
	"""
	if crop_rhs: 
		cols = triplets[1][1]
		if len(cols)==0: 
			return cps.zeros_like(rhs,shape=(0,))
		size = 1+np.max(cols)
		if rhs.shape[0]>size:
			rhs = rhs[:size]
	return tocsr(triplets)*rhs