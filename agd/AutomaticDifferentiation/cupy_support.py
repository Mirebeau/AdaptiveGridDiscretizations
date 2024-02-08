import numpy as np
from .Base import implements_cupy_alt,cp,is_ad

"""
This file implements a few numpy functions that not well supported by the
cupy version (6.0, thus outdated) that is available on windows by conda at the 
time of writing.
"""

# -- NOT -- fixed in cupy 8.2
@implements_cupy_alt(np.max,TypeError)
def max(a,*args,**kwargs):
	initial=kwargs.pop('initial') # cupy (old version ?) does not accept initial argument
	return np.maximum(initial,np.max(a,*args,**kwargs))

def _along_axis(arr,indices,axis):
	axis%=arr.ndim
	def indices_(ax):
		if ax==axis: return indices
		sax = arr.shape[ax]
		ind = np.arange(sax).reshape((1,)*ax + (sax,)+(1,)*(arr.ndim-ax-1))
		return np.broadcast_to(ind,indices.shape)
	return tuple(indices_(ax) for ax in range(arr.ndim))

# --- NOT --- fixed in cupy 8.2
@implements_cupy_alt(np.put_along_axis,TypeError)
def put_along_axis(arr,indices,values,axis):
	arr[_along_axis(arr,indices,axis)]=values

# -- NOT -- fixed in cupy 8.2
@implements_cupy_alt(np.packbits,TypeError)
def packbits(arr,bitorder='big'):
	"""Implements bitorder option in cupy""" 
	if bitorder=='little':
		shape = arr.shape
		arr = arr.reshape(-1,8)
		arr = arr[:,::-1]
		arr = arr.reshape(shape)
	return cp.packbits(arr)

