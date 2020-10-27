# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module implements some finite differences operations, as well as related array 
reshaping and broadcasting operations. The main functions are the following, see the 
the detailed help below.

Elementary finite differences:
- DiffUpwind
- DiffCentered
- Diff2

Array broadcasting:
- as_field
- common_field

Block based array reshaping:
- block_expand
- block_squeeze

"""

import numpy as np
import itertools
from . import AutomaticDifferentiation as ad
import functools
import operator

# --- Domain shape related methods --------

def as_field(u,shape,conditional=True,depth=None):
	"""
	Checks if the last dimensions of u match the given shape. 
	If not, u is extended with these additional dimensions.
	conditional : if False, reshaping is always done
	depth (optional) : the depth of the geometrical tensor field (1: vectors, 2: matrices)
	"""
	u=ad.asarray(u)
	ndim = len(shape)
	def as_is():
		if not conditional: return False
		elif depth is None: return u.ndim>=ndim and u.shape[-ndim:]==shape
		else: assert u.shape[depth:] in (tuple(),shape); return u.shape[depth:]==shape
	if as_is(): return u
	else: return np.broadcast_to(u.reshape(u.shape+(1,)*ndim), u.shape+shape)

def common_field(arrays,depths,shape=None):
	"""
	Adds trailing dimensions, and broadcasts the given arrays, for suitable interoperation.
	
	Inputs: 
	- arrays : a list [a_1,...,a_n], or iterable, of numeric arrays such that
	 a_i.shape = shape_i + shape, or a_i.shape = shape_i, for each 1<=i<=n.
	- depths : defined as [len(shape_i) for 1<=i<=n]
	- shape (optional) : the trailing shape.
	
	Output:
	- the arrays, with added trailing and dimensions broadcasting so that
	 a_i.shape = shape_i + shape for each 1<=i<=n.

	"""
	if shape is None:
		assert len(arrays)==len(depths)
		for arr,depth in zip(arrays,depths):
			if arr is None: continue
			arr=ad.asarray(arr)
			shape = arr.shape[depth:]
			if shape!=tuple(): break
	return tuple(None if arr is None else as_field(arr,shape,depth=depth) 
		for (arr,depth) in zip(arrays,depths))

def round_up_ratio(num,den):
	"""
	Returns the least multiple of den after num.
	num and den must be integers, with den>0. 
	"""
	num,den = np.asarray(num),np.asarray(den)
	return (num+den-1)//den

def block_expand(arr,shape_i,renumber_ad=False,**kwargs):
	"""
	Reshape an array so as to factor shape_i (the inner shape),
	and move these axes last.
	Inputs : 
	 - **kwargs : passed to np.pad
	Output : 
	 - padded and reshaped array
	"""
	ndim = len(shape_i)
	if ndim==0: return arr # Empty block
	shape_pre = arr.shape[:-ndim]
	ndim_pre = len(shape_pre)
	shape_tot=np.array(arr.shape[-ndim:])
	shape_i = np.array(shape_i)

	# Extend data
	shape_o = round_up_ratio(shape_tot,shape_i)
	shape_pad = (0,)*ndim_pre + tuple(shape_o*shape_i - shape_tot)
	arr = np.pad(arr, tuple( (0,s) for s in shape_pad), **kwargs) 

	# Reshape
	shape_interleaved = tuple(np.stack( (shape_o,shape_i), axis=1).flatten())
	arr = arr.reshape(shape_pre + shape_interleaved)

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = ndim_pre + 1+2*rg
	axes_split = ndim_pre + ndim+rg
	arr = np.moveaxis(arr,axes_interleaved,axes_split)

	return arr

def block_squeeze(arr,shape,renumber_ad=False):
	"""
	Inverse operation to block_expand.
	"""
	ndim = len(shape)
	if ndim==0: return arr # Empty block
	shape_pre = arr.shape[:-2*ndim]
	ndim_pre = len(shape_pre)
	shape_o = arr.shape[(-2*ndim):-ndim]
	shape_i = arr.shape[-ndim:]

	# Move axes
	rg = np.arange(ndim)
	axes_interleaved = ndim_pre + 1+2*rg
	axes_split = ndim_pre + ndim+rg
	arr = np.moveaxis(arr,axes_split,axes_interleaved)

	# Reshape
	arr = arr.reshape(shape_pre
		+tuple(s_o*s_i for (s_o,s_i) in zip(shape_o,shape_i)) )

	# Extract subdomain
	region = tuple(slice(0,s) for s in (shape_pre+shape))
	arr = arr.__getitem__(region)

	return arr

# ----- Utilities for finite differences ------

def BoundedSlices(slices,shape):
	"""
	Returns the input slices with None replaced with the upper bound
	from the given shape
	"""
	if slices[-1]==Ellipsis:
		slices=slices[:-1]+(slice(None,None,None),)*(len(shape)-len(slices)+1)
	def BoundedSlice(s,n):
		if not isinstance(s,slice):
			return slice(s,s+1)
		else:
			return slice(s.start, n if s.stop is None else s.stop, s.step)
	return tuple(BoundedSlice(s,n) for s,n in zip(slices,shape))

def OffsetToIndex(shape,offset, mode='clip', uniform=None, where=(Ellipsis,)):
	"""
	Returns the index corresponding to position + offset, 
	and a boolean for wether it falls inside the domain.
	Set padding=None for periodic boundary conditions
	"""
	ndim = len(shape) # Domain dimension
	assert(offset.shape[0]==ndim)
	if uniform is None: # Uniform = True iff offsets are independent of the position in the domain
		uniform = not ((offset.ndim > ndim) and (offset.shape[-ndim:]==shape))

	odim = (offset.ndim-1) if uniform else (offset.ndim - ndim-1) # Dimensions for distinct offsets 
	everywhere = where==(Ellipsis,)
	xp=ad.cupy_generic.get_array_module(offset)

	grid = (xp.mgrid[tuple(slice(n) for n in shape)]
		if everywhere else
		np.squeeze(xp.mgrid[BoundedSlices(where,shape)],
		tuple(1+i for i,s in enumerate(where) if not isinstance(s,slice)) ) )
	grid = grid.reshape( (ndim,) + (1,)*odim+grid.shape[1:])

	if not everywhere and not uniform:
		offset = offset[(slice(None),)*(1+odim)+where]

	neigh = grid + (offset.reshape(offset.shape + (1,)*ndim) if uniform else offset)
	bound = xp.array(shape,dtype=neigh.dtype).reshape((ndim,)+(1,)*(neigh.ndim-1))

	if mode=='wrap': neigh = np.mod(neigh,bound); inside=True
	else: inside = np.logical_and(np.all(neigh>=0,axis=0),np.all(neigh<bound,axis=0))

	neighIndex = ad.cupy_support.ravel_multi_index(neigh, shape, mode=mode)
	return neighIndex, inside

def TakeAtOffset(u,offset, padding=np.nan, **kwargs):
	mode = 'wrap' if padding is None else 'clip'
	neighIndex, inside = OffsetToIndex(u.shape,offset,mode=mode, **kwargs)

	values = u.reshape(-1)[neighIndex]
	if padding is not None:
		if ad.isndarray(values):
			values[np.logical_not(inside)] = padding
		elif not inside:
			values = padding
	return values

def AlignedSum(u,offset,multiples,weights,**kwargs):
	"""Returns sum along the direction offset, with specified multiples and weights"""
	return sum(TakeAtOffset(u,mult*ad.asarray(offset),**kwargs)*weight 
		for mult,weight in zip(multiples,weights))


# --------- Degenerate elliptic finite differences -------

def Diff2(u,offset,gridScale=1.,order=2,**kwargs):
	"""
	Approximates <offset, (d^2 u) offset> with second order accuracy.
	Second order finite difference in the specidied direction.
	"""
	if   order<=2: multiples,weights = (1,0,-1),(1.,-2.,1.)
	elif order<=4: multiples,weights = (2,1,0,-1,-2),(-1/12.,4/3.,-15/6.,4/3.,-1/12.)
	else: raise ValueError("Unsupported order") 
	return AlignedSum(u,offset,multiples,np.array(weights)/gridScale**2,**kwargs)


def DiffUpwind(u,offset,gridScale=1.,order=1,**kwargs):
	"""
	Approximates <grad u, offset> with specified accuracy order.
	Upwind first order finite difference in the specified direction.
	Note: only order=1 yields degenerate elliptic schemes.
	"""
	if   order==1: multiples,weights = (1,0),    ( 1.,-1.)
	elif order==2: multiples,weights = (2,1,0),  (-0.5,2.,-1.5)
	elif order==3: multiples,weights = (3,2,1,0),(1./3.,-1.5,3.,-11./6.)
	else: raise ValueError("Unsupported order")
	return AlignedSum(u,offset,multiples,np.array(weights)/gridScale,**kwargs)

# --------- Non-Degenerate elliptic finite differences ---------

def DiffCentered(u,offset,gridScale=1.,order=2,**kwargs):
	"""
	Approximates <d u, offset> with second order accuracy.
	Centered first order finite difference in the specified direction.
	"""
	if   order<=2: multiples,weights = ( 1,-1),( 1.,-1.)
	elif order<=4: multiples,weights = ( 2, 1,-1,-2),(-1/6., 4/3.,-4/3., 1/6.)
	else: raise ValueError("Unsupported order")
	return AlignedSum(u,offset,multiples,np.array(weights)/(2*gridScale),**kwargs)

def DiffCross(u,offset0,offset1,gridScale=1.,order=2,**kwargs):
	"""
	Approximates <offsets0, (d^2 u) offset1> with second order accuracy.
	Centered finite differences scheme, but lacking the degenerate ellipticity property.
	"""
	if   order<=2: multiples,weights = ( 1,-1),(1.,1.)
	elif order<=4: multiples,weights = ( 2, 1,-1,-2),(-1/12.,4/3.,4/3.,-1/12.)
	else: raise ValueError("Unsupported order")
	weights = np.array(weights)/(4*gridScale**2)
	return (AlignedSum(u,offset0+offset1,multiples,weights,**kwargs) 
		- AlignedSum(u,offset0-offset1,multiples,weights,**kwargs) )

# ------------ Composite finite differences ----------

def AxesOffsets(u=None,offsets=None,dimension=None):
	"""
	Returns the offsets corresponding to the axes.
		Inputs : 
	 - offsets (optional). Defaults to np.eye(dimension)
	 - dimension (optional). Defaults to u.ndim
	"""
	if offsets is None:
		if dimension is None:
			dimension = u.ndim
		offsets = np.eye(dimension).astype(int)
	return offsets



def DiffHessian(u,offsets=None,dimension=None,**kwargs):
	"""
	Approximates the matrix (<offsets[i], (d^2 u) offsets[j]> )_{ij}, using AxesOffsets as offsets.
	Centered and cross finite differences are used, thus lacking the degenerate ellipticity property.
	"""
	from . import Metrics
	offsets=AxesOffsets(u,offsets,dimension)
	return Metrics.misc.expand_symmetric_matrix([
		Diff2(u,offsets[i],**kwargs) if i==j else DiffCross(u,offsets[i],offsets[j],**kwargs) 
		for i in range(len(offsets)) for j in range(i+1)])

def DiffGradient(u,offsets=None,dimension=None,**kwargs):
	"""
	Approximates the vector (<d u, offsets[i]>)_i, using AxesOffsets as offsets
	Centered finite differences are used, thus lacking the degerate ellipticity property.
	"""
	return DiffCentered(u,AxesOffsets(u,offsets,dimension),**kwargs)

# ----------- Interpolation ---------

def UniformGridInterpolator1D(bounds,values,mode='clip',axis=-1):
	"""
	Interpolation on a uniform grid. mode is in ('clip','wrap', ('fill',fill_value) )
	"""
	val = values.swapaxes(axis,0)
	fill_value = None
	if isinstance(mode,tuple):
		mode,fill_value = mode		
	def interp(position):
		endpoint=not (mode=='wrap')
		size = val.size
		index_continuous = (size-int(endpoint))*(position-bounds[0])/(bounds[-1]-bounds[0])
		index0 = np.floor(index_continuous).astype(int)
		index1 = np.ceil(index_continuous).astype(int)
		index_rem = index_continuous-index0
		
		fill_indices=False
		if mode=='wrap':
			index0=index0%size
			index1=index1%size
		else: 
			if mode=='fill':
				 fill_indices = np.logical_or(index0<0, index1>=size)
			index0 = np.clip(index0,0,size-1) 
			index1 = np.clip(index1,0,size-1)
		
		index_rem = index_rem.reshape(index_rem.shape+(1,)*(val.ndim-1))
		result = val[index0]*(1.-index_rem) + val[index1]*index_rem
		if mode=='fill': result[fill_indices] = fill_value
		result = np.moveaxis(result,range(position.ndim),range(-position.ndim,0))
		return result
	return interp

# def AxesOrderingBounds(grid):
# 	dim = len(grid)
# 	lbounds = grid.__getitem__((slice(None),)+(0,)*dim)
# 	ubounds = grid.__getitem__((slice(None),)+(-1,)*dim)

# 	def active(i):
# 		di = grid.__getitem__((slice(None),)+(0,)*i+(1,)+(0,)*(dim-1-i))
# 		return np.argmax(np.abs(di-lbounds))
# 	axes = tuple(active(i) for i in range(dim))

# 	return axes,lbounds,ubounds 