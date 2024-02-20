# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements functionalities needed to make the agd library generic to cupy/numpy usage.
It does not import cupy, unless absolutely required.
"""

import itertools
import numpy as np
import sys
import functools
import types
from copy import copy

from . import functional
from .Base import cp,isndarray,from_cupy,is_ad,array,cupy_alt_overloads

def get_array_module(data,iterables=tuple()):
	"""Returns the cupy module or the numpy module, depending on data"""
	if cp is None: return np
	return cp if any(from_cupy(x) for x in functional.rec_iter(data,iterables)) else np

def samesize_int_t(float_t):
	"""
	Returns an integer type of the same size (32 or 64 bits) as a given float type
	"""
	float_t = np.dtype(float_t).type
	float_name = str(float_t)
	if   float_t==np.float32: return np.int32
	elif float_t==np.float64: return np.int64
	else: raise ValueError(
		f"Type {float_t} is not a float type, or has no default matching int type")

# ----------- Retrieving data from a cupy array ------------

dtype32to64 = {np.float32:np.float64,np.int32:np.int64,np.uint32:np.uint64,}
dtype64to32 = {np.float64:np.float32,np.int64:np.int32,np.uint64:np.uint32,}

def cupy_get(x,dtype64=False,iterables=tuple()):
	"""
	If argument is a cupy ndarray, returns output of 'get' member function, 
	which is a numpy ndarray. Likewise for AD types. Returns unchanged argument otherwise.
	- dtype64 : convert 32 bit floats and ints to their 64 bit counterparts
	"""
	def caster(x):
		if from_cupy(x):
			if is_ad(x): return type(x)(*(caster(z) for z in x.as_tuple()))
			x = x.get()
			if x.dtype.type in dtype32to64: x=x.astype(dtype32to64[x.dtype.type])
		return x
	return functional.map_iterables(caster,x,iterables)

def cupy_set(x,dtype32=True,iterables=tuple()):
	"""
	If argument is a numpy ndarray, converts it to a cupy ndarray. Applies to AD Types.
	- dtype32 : convert 64 bit floats and ints to their 32 bit counterparts
	"""
	def caster(x):
		if isndarray(x) and not from_cupy(x):
			if is_ad(x): return type(x)(*(caster(z) for z in x.as_tuple()))
			dtype = dtype64to32.get(x.dtype.type,x.dtype.type)
			return cp.asarray(x,dtype=dtype)
		return x
	return functional.map_iterables(caster,x,iterables)

@functional.decorator_with_arguments
def cupy_get_args(f,*args,**kwargs):
	"""
	Decorator applying cupy_get to all arguments of the given function.
	 - *args, **kwargs : passed to cupy_get
	"""
	@functools.wraps(f)
	def wrapper(*fargs,**fkwargs):
		fargs = tuple(cupy_get(arg,*args,**kwargs) for arg in fargs)
		fkwargs = {key:cupy_get(value,*args,**kwargs) for key,value in fkwargs.items()}
		return f(*fargs,**fkwargs)
	return wrapper

# ----- Casting data to appropriate floating point and integer types ------

def has_dtype(arg,dtype="dtype",iterables=(tuple)):
	"""
	Wether one member of args is an ndarray with the provided dtype.
	"""
	dtype = np.dtype(dtype)
	has_dtype_ = False
	def find_dtype(x):
		nonlocal has_dtype_
		has_dtype_ = has_dtype_ or (isndarray(x) and x.dtype==dtype)
	for x in functional.rec_iter(arg,iterables=iterables): find_dtype(x)
	return has_dtype_
			
def get_float_t(arg,**kwargs):
	"""
	Returns float32 if found in any argument, else float64.
	- kwargs : passed to has_dtype
	"""
	return np.float32 if has_dtype(arg,dtype=np.float32,**kwargs) else np.float64

def array_float_caster(arg,**kwargs):
	"""
	returns lambda arr : xp.asarray(arr,dtype=float_t) 
	where xp and float_t are in consistency with the arguments.
	"""
	xp = get_array_module(arg,**kwargs)
	float_t = get_float_t(arg,**kwargs)
	return lambda arr:xp.asarray(arr,dtype=float_t)

@functional.decorator_with_arguments
def set_output_dtype32(f,silent=True,iterables=(tuple,)):
	"""
	If the output of the given funtion contains ndarrays with 64bit dtype,
	int or float, they are converted to 32 bit dtype.
	"""
	def caster(a):
		if isndarray(a) and a.dtype in (np.float64,np.int64):
			xp = get_array_module(a)
			dtype = np.float32 if a.dtype==np.float64 else np.int32
			if not silent: print(
				f"Casting output of function {f.__name__} " 
				f"from {a.dtype} to {np.dtype(dtype)}")
			return xp.asarray(a,dtype=dtype)
		return a

	@functools.wraps(f)
	def wrapper(*args,**kwargs):
		output = f(*args,**kwargs)
		return functional.map_iterables(caster,output,iterables=iterables)

	return wrapper

# ------------ A helper function for cupy/numpy notebooks -------------

def cupy_friendly(arg):
	"""
	Returns a "cupy-friendly" copy of the input module, function, or object,
	following arbitrary and ad-hoc rules.
	"""

	if isinstance(arg,types.ModuleType):
		# Special cases
		if arg is np:
			print("Replacing numpy with cupy, set to output 32bit ints and floats by default.")
			if cp is None: 
				raise ValueError("cupy module not found.\n"
					"If your are using Google Colab, go to modify->notebook parameters and activate GPU acceleration.")
			cp32 = functional.decorate_module_functions(cp,set_output_dtype32)
			print("Using cp.asarray(*,dtype=np.float32) as the default caster in ad.array.")
			array.caster = lambda x: cp.asarray(x,dtype=np.float32)
			return cp32
		if arg.__name__ == 'scipy.ndimage':
			print("Replacing module scipy.ndimage with cupyx.scipy.ndimage .")
			from cupyx.scipy import ndimage
			return ndimage
		if arg.__name__ == 'agd.Eikonal':
			print("Setting dictIn.default_mode = 'gpu' in module agd.Eikonal .")
			arg.dictIn.default_mode = 'gpu'
			arg.VoronoiDecomposition.default_mode = 'gpu'
			return arg

		# Default behavior
		print(f"Returning a copy of module {arg.__name__} whose functions accept cupy arrays as input.")
		return functional.decorate_module_functions(arg,cupy_get_args)


	if arg is np.allclose:
		print("Setting float32 compatible default values atol=rtol=1e-5 in np.allclose")
		def allclose(*args,**kwargs):
			kwargs.setdefault('atol',1e-5)
			kwargs.setdefault('rtol',1e-5)
			return np.allclose(*args,**kwargs)
		return allclose

	if isinstance(arg,types.FunctionType):
		if arg in cupy_alt_overloads: 
			alt,exception = cupy_alt_overloads[arg] 
			print("Adding (partial) support for (old versions of) cupy"
			f" versions to function {arg.__name__}")
			return alt

		# Default behavior
		print(f"Returning a copy of function {arg.__name__} which accepts cupy arrays as input.")
		return cupy_get_args(arg)

	if isndarray(arg):
		print(f"Replacing ndarray object with cupy variant, for object of type {type(arg)}")
		return cupy_set(arg)
	else:
		print("Replacing ndarray members with their cupy variants, "
			f"for object of type {type(arg)}")
		return cupy_set(arg,iterables=(type(arg),))





