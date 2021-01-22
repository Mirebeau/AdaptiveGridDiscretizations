# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module implements some basic linear algebra routines, with the following characteristics.
- The geometry comes first, a.k.a vector has shape (vdim, n1,...,nk) where vdim is the 
 ambient vector space dimension, and n1,...,nk are arbitrary. 
 Note that *numpy uses the opposite convention*, putting vdim in last position.
- The routines are compatible with forward automatic differentiation, see module
 AutomaticDifferentiation.
"""

import numpy as np
from . import AutomaticDifferentiation as ad
from . import FiniteDifferences as fd
from .AutomaticDifferentiation import cupy_support as cps

def identity(shape):
	dim = len(shape)
	a = np.full((dim,dim)+shape,0.)
	for i in range(dim):
		a[i,i]=1.
	return a

def rotation(theta,axis=None):
	"""
	Dimension 2 : by a given angle.
	Dimension 3 : by a given angle, along a given axis.
	Three dimensional rotation matrix, with given axis and angle.
	Adapted from https://stackoverflow.com/a/6802723
	"""
	if axis is None:
		c,s=np.cos(theta),np.sin(theta)
		return ad.asarray([[c,-s],[s,c]])
	else:
		theta,axis = (ad.asarray(e) for e in (theta,axis))
		axis = axis / np.linalg.norm(axis,axis=0)
		theta,axis=fd.common_field((theta,axis),(0,1))
		a = np.cos(theta / 2.0)
		b, c, d = -axis * np.sin(theta / 2.0)
		aa, bb, cc, dd = a * a, b * b, c * c, d * d
		bc, ad_, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
		return ad.asarray([
			[aa + bb - cc - dd, 2 * (bc + ad_), 2 * (bd - ac)],
			[2 * (bc - ad_), aa + cc - bb - dd, 2 * (cd + ab)],
			[2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
#	return scipy.linalg.expm(np.cross(np.eye(3), axis/scipy.linalg.norm(axis)*theta)) # Alternative

# Dot product (vector-vector, matrix-vector and matrix-matrix) in parallel
def dot_VV(v,w):
	"""
	Dot product <v,w> of two vectors.
	Inputs : 
	- v,w : arrays of shape (vdim, n1,...,nk), 
	 where vdim is the ambient vector space dimension
	"""
	v=ad.asarray(v); w=ad.asarray(w)
	if v.shape[0]!=w.shape[0]: raise ValueError('dot_VV : Incompatible shapes')
	return (v*w).sum(0)

def dot_AV(a,v):
	"""
	Dot product a.v of a matrix and vector.
	Inputs : 
	- a : array of shape (wdim,vdim, n1,...,nk)
	- v : array of shape (vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	if a.shape[1]!=v.shape[0]: raise ValueError("dot_AV : Incompatible shapes")
	return (a*cps.expand_dims(v,axis=0)).sum(1)

def dot_VA(v,a):
	"""
	Dot product v^T.a of a vector and matrix.
	Inputs : 
	- v : array of shape (vdim, n1,...,nk)
	- a : array of shape (vdim,wdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	m,n = a.shape[:2]
	bounds = a.shape[2:]
	if v.shape != (m,)+bounds:
		raise ValueError("dot_VA : Incompatible shapes")

	return (v.reshape((m,1)+bounds)*a).sum(0)


def dot_AA(a,b):
	"""
	Dot product a.b of two matrices.
	Inputs : 
	- a: array of shape (vdim,wdim, n1,...,nk),
	- a: array of shape (wdim,xdim, n1,...,nk),	
	"""
	m,n=a.shape[:2]
	bounds = a.shape[2:]
	k = b.shape[1]
	if b.shape!=(n,k,)+bounds:
		raise ValueError("dot_AA error : Incompatible shapes")
	return (a.reshape((m,n,1)+bounds)*b.reshape((1,n,k)+bounds)).sum(1)

def dot_VAV(v,a,w):
	"""
	Dot product <v,a.w> of two vectors and a matrix (usually symmetric).
	Inputs (typical): 
	- v: array of shape (vdim, n1,...,nk),	
	- a: array of shape (vdim,vdim, n1,...,nk),
	- w: array of shape (vdim, n1,...,nk),	
	"""
	return dot_VV(v,dot_AV(a,w))
	
def mult(k,x):
	"""Multiplication by scalar, of a vector or matrix"""
	# Quite useless, as it is directly handled by broadcasting
	bounds = k.shape
	dim = x.ndim-k.ndim
	if x.shape[dim:]!=bounds:
		raise ValueError("mult error : incompatible shapes")
	return k.reshape((1,)*dim+bounds)*x
	

def perp(v):
	"""
	Rotates a vector by pi/2, producing [v[1],v[0]]
	Inputs: 
	- v: array of shape (2, n1,...,nk)
	"""
	if v.shape[0]!=2:
		raise ValueError("perp error : Incompatible dimension")		
	return ad.asarray( (-v[1],v[0]) )
	
def cross(v,w):
	"""
	Cross product v x w of two vectors.
	Inputs: 
	- v,w: arrays of shape (3, n1,...,nk)
	"""
	if v.shape[0]!=3 or v.shape!=w.shape:
		raise ValueError("cross error : Incompatible dimensions")
	return ad.asarray( (v[1]*w[2]-v[2]*w[1], \
	v[2]*w[0]-v[0]*w[2], v[0]*w[1]-v[1]*w[0]) )
	
def outer(v,w):
	"""
	Outer product v w^T of two vectors.
	Inputs : 
	- v,w: arrays of shape (vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	if v.shape[1:] != w.shape[1:]:
		raise ValueError("outer error : Incompatible dimensions")
	m,n=v.shape[0],w.shape[0]
	bounds = v.shape[1:]
	return v.reshape((m,1)+bounds)*w.reshape((1,n)+bounds)

def outer_self(v):
	"""
	Outer product v v^T of a vector with itself.
	Inputs : 
	- v: array of shape (vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	v=ad.asarray(v)
	return outer(v,v)

def transpose(a):
	"""
	Transpose a^T of a matrix.
	Input : 
	- a: array of shape (vdim,wdim, n1,...,nk),
	"""
	return a.transpose( (1,0,)+tuple(range(2,a.ndim)) )
	
def trace(a):
	"""
	Trace tr(a) of a square matrix, a.k.a sum of the diagonal elements.
	Input : 
	- a: array of shape (vdim,vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	vdim = a.shape[0]
	if a.shape[1]!=vdim:
		raise ValueError("trace error : incompatible dimensions")
	return sum(a[i,i] for i in range(vdim))

# Low dimensional special cases

def det(a):
	"""
	Determinant of a square matrix.
	Input : 
	- a: array of shape (vdim,vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	a=ad.asarray(a)

	dim = a.shape[0]
	if a.shape[1]!=dim:
		raise ValueError("inverse error : incompatible dimensions")
	if dim==1:
		return a[0,0]
	elif dim==2:
		return a[0,0]*a[1,1]-a[0,1]*a[1,0]
	elif dim==3:
		return a[0,0]*a[1,1]*a[2,2]+a[0,1]*a[1,2]*a[2,0]+a[0,2]*a[1,0]*a[2,1] \
		- a[0,2]*a[1,1]*a[2,0] - a[1,2]*a[2,1]*a[0,0]- a[2,2]*a[0,1]*a[1,0]
	else:
		raise ValueError("det error : unsupported dimension") 

# Suppressed due to extreme slowness, at least in cupy 6
#	if not (ad.is_ad(a) or a.dtype==np.dtype('object')): 
#		return np.linalg.det(np.moveaxis(a,(0,1),(-2,-1)))


def inverse(a):
	"""
	Inverse of a square matrix.
	Input : 
	- a: array of shape (vdim,vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	a=ad.asarray(a)
	if not (ad.is_ad(a) or a.dtype==np.dtype('object')):
		try: return np.moveaxis(np.linalg.inv(np.moveaxis(a,(0,1),(-2,-1))),(-2,-1),(0,1))
		except np.linalg.LinAlgError: pass # Old cupy versions do not support arrays of dimension>2

	if isinstance(a,ad.Dense.denseAD):
		b = inverse(a.value)
		b_ = fd.as_field(b,(a.size_ad,),conditional=False) 
		h = a.coef
		return ad.Dense.denseAD(b,-dot_AA(b_,dot_AA(h,b_)))
	elif isinstance(a,ad.Dense2.denseAD2):
		b = inverse(a.value)
		b1 = fd.as_field(b,(a.size_ad,),conditional=False)
		h = a.coef1
		h2 = a.coef2

		bh = dot_AA(b1,h)
		bhb = dot_AA(bh,b1)
		bhbhb = dot_AA(np.broadcast_to(cps.expand_dims(bh,-1),h2.shape),
			np.broadcast_to(cps.expand_dims(bhb,-2),h2.shape))

		b2 = fd.as_field(b,(a.size_ad,a.size_ad),conditional=False)
		bh2b = dot_AA(b2,dot_AA(h2,b2))
		return ad.Dense2.denseAD2(b,-bhb,bhbhb+np.swapaxes(bhbhb,-1,-2)-bh2b)
	elif ad.is_ad(a):
		d=len(a)
		return ad.apply(inverse,a,shape_bound=a.shape[2:])
	elif a.shape[:2] == (2,2):
		return ad.asarray([[a[1,1],-a[0,1]],[-a[1,0],a[0,0]]])/det(a)
	elif a.shape[:2] == (3,3):
		return ad.asarray([[
			a[(i+1)%3,(j+1)%3]*a[(i+2)%3,(j+2)%3]-a[(i+1)%3,(j+2)%3]*a[(i+2)%3,(j+1)%3]
			for i in range(3)] for j in range(3)])/det(a)
	raise ValueError(f"Unsupported inverse for {type(a)} with dtype {a.dtype} and dimensions {a.shape}")

def solve_AV(a,v):
	"""
	Solution to a linear system (preferably low dimensional).
	Input : 
	- a: array of shape (vdim,vdim, n1,...,nk),
	- v: array of shape (vdim,vdim, n1,...,nk),
	 where vdim is the ambient vector space dimension
	"""
	a=ad.asarray(a)
	if ad.is_ad(v) or a.dtype==np.dtype('object') or (len(a)<=3 and ad.cupy_generic.from_cupy(a)): 
		# Inefficient, but compatible with ndarray subclasses
		# Also cupy.linalg.solve as a performance issue (cupy version 7.8) 
		return dot_AV(inverse(a),v) 
	return np.moveaxis(np.linalg.solve(np.moveaxis(a,(0,1),(-2,-1)),np.moveaxis(v,0,-1)),-1,0)			


