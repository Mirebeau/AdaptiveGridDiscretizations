# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements some spline interpolation methods, on uniform grids,
in a manner compatible with automatic differentiation.

If none of the involved arrays use automatic differentiation, and if the options are 
compatible, then a bypass through ndimage may be used.
"""


import numpy as np
import itertools
import scipy.linalg

from . import AutomaticDifferentiation as ad
from .AutomaticDifferentiation import cupy_support as cps


#TODO : use a smaller stencil (3 pts instead of 4) for 2nd degree interpolation 
# when far from the boundary, in non-periodic mode. (Introduce interior nodes.)

def origin_scale_shape(grid):
	"""
	This function is indended for extracting the origin, scale, and shape,
	of a uniform coordinate system provided as a meshgrid.
	"""
	def _orig_step_len(a,axis):
		a = ad.asarray(a)
		assert a.ndim==1 or a.ndim>=axis
		if a.ndim>1:a=a.__getitem__((0,)*axis+(slice(None,),)+(0,)*(a.ndim-axis-1))
		return a[0],a[1]-a[0],len(a)
	origin_scale_shape = [_orig_step_len(a,axis) for axis,a in enumerate(grid)]
	origin,scale,shape = [list(l) for l in zip(*origin_scale_shape)]
	caster = ad.cupy_generic.array_float_caster(grid,iterables=(list,tuple))
	return caster(origin),caster(scale),tuple(shape)


def _append_dims(x,ndim):
	"""Appends specified number of singleton dimensions"""
	return np.reshape(x,x.shape+(1,)*ndim)

def map_coordinates(input,coordinates,*args,
	grid=None,origin=None,scale=None,depth=0,order=1,**kwargs):
	"""
	Thin wrapper over the ndimage.map_coordinates function, which adds the possibility of 
	rescaling the coordinates using a reference grid, and interpolating tensors.
	Will dispatch to cupyx.scipy.ndimage.map_coordinates if needed.

	Additional inputs : 
	- grid (optional) : reference coordinate system, which must be uniform
	- origin,scale (optional) : obtained from origin_scale_shape(grid)
	- depth : depth of interpolated objects 0->scalar, 1->vector, 2->matrix
	- order (optional) : set default 1 for better cupy/numpy reproducibility
	"""
	kwargs['order']=order
	if ad.cupy_generic.from_cupy(input):
		from cupyx.scipy.ndimage.interpolation import map_coordinates as mc
		def map_coordinates(arr,x,*args,**kwargs):
			# Cupy (version 7.8) requires the coordinates array to be flattened
			shape = x.shape[1:]
			return mc(arr,x.reshape((len(x),-1)),*args,**kwargs).reshape(shape)
	else: from scipy.ndimage.interpolation import map_coordinates

	if grid is not None: origin,scale,_ = origin_scale_shape(grid)
	assert (origin is None) == (scale is None)
	if origin is None: return map_coordinates(input,coordinates,*args,**kwargs)
	origin,scale = (_append_dims(e,coordinates.ndim-1) for e in (origin,scale))
	y = (coordinates-origin)/scale

	if depth==0: return map_coordinates(input,y,*args,**kwargs)
	oshape = input.shape[:depth]
	input = input.reshape((-1,)+input.shape[depth:])
	out = ad.array([map_coordinates(input_,y,*args,**kwargs) for input_ in input])
	out.reshape(oshape+y.shape[1:])
	return out

class _spline_univariate:
	"""
	A univariate spline of a given order, with not-a-knot boundary conditions.
	"""
	def __init__(self,order,shape,periodic):
		assert isinstance(order,int)
		self.order=order
		if not (order>=1 and order<=3):
			raise ValueError("Unsupported spline order")

		assert isinstance(shape,int) and shape>=0
		self.shape = shape

		assert isinstance(periodic,bool)
		self.periodic=periodic


	def __call__(self,xa,xs):
		"""
		Weight at absolute position xa, of of spline centered at xs.
		"""
		xa=ad.asarray(xa)
		xs=ad.asarray(xs)
		if   self.order==1: return self._call1(xa,xs)
		elif self.order==2: return self._call2(xa,xs)
		elif self.order==3: return self._call3(xa,xs)
		else: assert False

	def nodes(self,interior):
		"""
		Returns range(a,b) where [x+a,x+b] contains the support
		 of the spline centered at some point x (interior or boundary) 
		"""
		if   self.order==1: return range(-1,1)
		elif self.order==2: #return range(-1,2)
			return range(-1,2) if interior else range(-2,2)
		elif self.order==3: 
			return range(-2,2) if interior else range(-3,4)
		assert False

	def interior(self,x,tol=None):
		"""
		Wether the interior nodes can be used, or one should fall back to boundary nodes.
		"""
		if tol is None: tol = ad.precision(x) * np.max(self.shape)
		if np.any(x<-tol) or np.any(x>self.shape-1+tol):
			raise ValueError("Interpolating data outside domain")
		if self.order==1 or self.periodic:
			return np.full_like(x,True,dtype='bool')
		elif self.order==2:
			return x>=1
		elif self.order==3:
			return np.logical_and(x>=1,x<=self.shape-2)

	def _call1(self,xa,xs):
		"""
		A piecewise linear spline, defined over [-1,1].
		"""
		x=xa-xs.astype(xa.dtype) # Avoid float32 + int32 -> float64 cast on GPU
		result = np.zeros_like(x)
		seg=ad.asarray(np.floor(x+1))
		
		# relative interval [-1,0[
		pos = seg==0
		result[pos] = 1+x[pos]

		# relative interval [0,1[
		pos = seg==1
		result[pos] = 1-x[pos]

		return result

		# Rejected because derivative at node points is inconsistent accross nodes
		# return np.maximum(0.,1.-np.abs(x))

	def _call2(self,xa,xs):
		"""
		A piecewise quadratic spline function, defined over [-1,2]
		"""
		x=ad.asarray(xa-xs)
		result = np.zeros_like(x)


		# Which spline segment to use
		seg = ad.asarray(np.floor(x+1))

		if not self.periodic:
			# Implements the not-a-knot boundary condition
			pos = np.logical_and(xa<=1,xs<=3)
			seg[pos] = 2-xs[pos]

		# relative interval [-1,0[
		pos = seg==0
		x_ = x[pos]+1
		result[pos] = x_**2

		# relative interval [0,1[
		pos = seg==1
		x_ = x[pos]-0.5
		result[pos] = 1.5 - 2.*x_**2

		# relative interval [1,2[
		pos = seg==2
		x_ = 2.-x[pos]
		result[pos] = x_**2 

		return result

	def _call3(self,xa,xs):
		"""
		A piecewise cubic spline function, defined over [-2,2]
		"""
		x=ad.asarray(xa-xs)
		result = np.zeros_like(x)
		s=self.shape-1

		# Which spline segment to use
		seg = ad.asarray(np.floor(x+2))
		if not self.periodic:
			# Implements the not-a-knot boundary condition
			pos = np.logical_and(xa<=1,xs<=3)
			seg[pos] = 3-xs[pos]
			pos = np.logical_and(xa>=s-1,xs>=s-3)
			seg[pos] = self.shape-1-xs[pos]

		def f0(y): return y**3
		def f1(y): return 1+3*y+3*y**2-3.*y**3

		# relative interval [-2,-1[
		pos = seg==0
		result[pos] = f0(x[pos]+2.)

		# relative interval [-1,0[
		pos = seg==1
		result[pos] = f1(x[pos]+1.)

		# relative interval [0,1[
		pos = seg==2
		result[pos] = f1(1.-x[pos])

		# relative interval [1,2[
		pos = seg==3
		result[pos] = f0(2.-x[pos])
		
		"""
		if not self.periodic:
			# End of not-a-knot boundary condition
			def g(y): return 4.-6.*y**2+(50./27.)*y**3
			xa=ad.broadcast_to(xa,x.shape)
			pos = np.logical_and(xa<=3,xs==3)
			result[pos] = g(3.-xa[pos])
			pos = np.logical_and(xa>s-3,xs==s-3)
			result[pos] = g(xa[pos]-(s-3))
		"""

		return result

	def _band(self):
		rg = np.arange(self.shape+0.)
		if self.order==2:
			band_tr = np.stack((self(rg,rg-1),self(rg,rg),self(rg,rg+1),self(rg,rg+2)),axis=0)
			return _banded_transpose((2,1),band_tr)
		elif self.order==3:
			band_tr = np.stack((self(rg,rg-3),self(rg,rg-2),self(rg,rg-1),
				self(rg,rg),self(rg,rg+1),self(rg,rg+2),self(rg,rg+3)),axis=0)
			return _banded_transpose((3,3),band_tr)
		else: assert False


	def make_coefs(self,values,overwrite_values=False):
		"""
		Produces the node coefficients corresponding to given values.
		!! Call convention : interpolation is along the first axis. !!
		"""
		if self.order==1: 
			return values

		if self.periodic: 
			raise ValueError("Periodic interpolation is not supported for degree > 1")
		if ad.is_ad(values):
			raise ValueError("AD interpolation is not supported for degree > 1")
		assert len(values)==self.shape

		return scipy.linalg.solve_banded(*self._band(),values,
				overwrite_ab=True,overwrite_b=overwrite_values) 

def _banded_transpose(lu,t):
	l,u=lu
	return (u,l),np.stack(tuple(np.roll(t[i],i-u) for i in reversed(range(l+u+1))))
#	return np.stack((np.roll(t[2],1),t[1],np.roll(t[0],-1)),axis=0)

def _banded_densify(lu,t):
	"""Turn banded matrix in to dense matrix (inefficient, for testing)"""
	l,u=lu
	n = t.shape[1]
	mat = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			k=u+i-j
			if 0<=k and k<len(t):
				mat[i,j]=t[k,j]
	return mat

class _spline_tensor:
	"""
	A tensor product of univariate splines.
	"""
	def __init__(self,orders,shape,periodic):
		assert all(isinstance(t,tuple) for t in (orders,shape,periodic))
		self.splines = tuple(_spline_univariate(order,s,per) 
			for (order,s,per) in zip(orders,shape,periodic))
		assert all(len(t)==self.vdim for t in (orders,shape,periodic))

	@property
	def order(self):
		return tuple(spline.order for spline in self.splines)
	@property
	def shape(self):
		return tuple(spline.shape for spline in self.splines)
	@property
	def periodic(self):
		return tuple(spline.periodic for spline in self.splines)
	
	@property
	def vdim(self):
		return len(self.splines)

	def __call__(self,xa,xs):
		"""
		Weight at absolute position xa, of of spline centered at xs.
		"""
		return np.prod( ad.asarray(tuple(spline(xai,xsi) 
			for (xai,xsi,spline) in zip(xa,xs,self.splines))), axis=0)

	def nodes(self,interior):
		"""
		The spline at x (interior or boundary) is supported 
		on the union of x+node+[0,1]**vdim
		"""
		_nodes = tuple(spline.nodes(interior) for spline in self.splines)
		return np.asarray(tuple(itertools.product(*_nodes))).T

	def interior(self,x):
		"""
		Wether the interior nodes can be used.
		"""
		return ad.asarray(tuple(spline.interior(xi) 
			for (spline,xi) in zip(self.splines,x))).all(axis=0)
#		return np.logical_and.reduce(tuple(spline.interior(xi) # cupy has no reduce
#			for (spline,xi) in zip(self.splines,x)))

	def make_coefs(self,values,overwrite_values=False):
		"""
		Produces the node coefficients corresponding to given values.
		!! Call convention : interpolation is along the first axes. 
		"""
		for i,spline in enumerate(self.splines): #reversed(list(enumerate(self.splines))):
			values = np.moveaxis(spline.make_coefs(
				np.moveaxis(values,i,0),overwrite_values=overwrite_values),0,i)
		return values

class UniformGridInterpolation:
	"""
	Interpolates values on a uniform grid, in arbitrary dimension, using splines of 
	a given order.
	"""

	def __init__(self,grid,values=None,order=1,periodic=False,check_grid=True):
		"""
		- grid (ndarray) : must be a uniform grid. E.g. np.meshgrid(aX,aY,indexing='ij')
		 where aX,aY have uniform spacing. Alternatively, provide only the axes.
		- values (ndarray) : interpolated values.
		- order (int, tuple of ints) : spline interpolation order (<=3), along each axis.
		- periodic (bool, tuple of bool) : wether periodic interpolation, along each axis.
		"""
		if isinstance(grid,dict):
			self.shape  = grid['shape']
			self.origin = ad.asarray(grid['origin'])
			self.scale  = ad.asarray(grid['scale'])
			if grid.get('cell_centered',False): 
				self.origin += self.scale/2 # Convert to node_centered origin
		else:
			self.origin,self.scale,self.shape = origin_scale_shape(grid)
			if check_grid and grid[0].ndim>1:
				assert np.allclose(grid,self._grid(),atol=1e-5) #Atol allows float32 types

		if order is None: order = 1
		if isinstance(order,int): order = (order,)*self.vdim

		if periodic is None: periodic=False
		if isinstance(periodic,bool): periodic = (periodic,)*self.vdim
		self.periodic=periodic

		self.spline = _spline_tensor(order,self.shape,periodic)
		assert self.spline.vdim == self.vdim
		self.interior_nodes = self.spline.nodes(interior=True)
		self.boundary_nodes = self.spline.nodes(interior=False)

		self.coef = None if values is None else self.make_coefs(values)

	@property
	def vdim(self):
		"""
		Vector dimension of the interpolation points.
		"""
		return len(self.shape)
	@property
	def oshape(self):
		"""
		Number of dimension of the interpolated objects.
		"""
		return self.coef.shape[self.vdim:]

	def __call__(self,x,interior=None):
		"""
		Interpolates the data at the position x.
		"""
		x=ad.asarray(x)
		assert len(x)==self.vdim
		
		pdim = x.ndim-1 # Number of dimensions of position
		origin,scale = (_append_dims(e,pdim) for e in (self.origin,self.scale))

		# Separate treatment of interior and boundary points
		if interior is None:
			y = (ad.remove_ad(x) - origin)/scale
			interior_x = self.spline.interior(y)
			boundary_x = np.logical_not(interior_x)
			interior_result = self(x[:,interior_x],True)
			boundary_result = self(x[:,boundary_x],False)

			result_shape = self.oshape+x.shape[1:]
			if result_shape==tuple(): #numpy zeros_like has a bug for empty shapes
				some_result = interior_result if interior_result.size>0 else boundary_result
				result = cps.zeros_like(some_result.reshape(-1)[0])
			else: result = cps.zeros_like(interior_result,shape=result_shape)

			try:
				result[...,interior_x] = interior_result
				result[...,boundary_x] = boundary_result
			except (ValueError,IndexError):
				# Some cupy versions do not handle Ellipsis correctly
				ellipsis = (slice(None),)*len(self.oshape)
				result.__setitem__((*ellipsis,interior_x),interior_result)
				result.__setitem__((*ellipsis,boundary_x),boundary_result)

			return result

		# Rescale the coordinates in reference rectangle
		y = np.expand_dims((x - origin)/scale,axis=1)
		# Bottom left pixel
		yf = np.floor(y).astype(int)
		# All pixels supporting an active spline
		nodes = self.interior_nodes if interior else self.boundary_nodes
		ys = yf - _append_dims(nodes,pdim)
		
		# Spline coefficients, taking care of out of domain 
		ys_ = ys.copy()
		out = np.full_like(ad.remove_ad(x[0]),False,dtype=bool)
		for i,(d,per) in enumerate(zip(self.shape,self.periodic)):
			if per: 
				ys_[i] = ys_[i]%d
			else: 
				bad = np.logical_or(ys_[i]<0,ys_[i]>=d)
				out = np.logical_or(out,bad)
				try: 
					ys_[i,bad] = 0 
				except ValueError: # Old cupy versions do not support such slices
					ys_i = ys_[i]
					ys_i[bad] = 0
					ys_[i] = ys_i

		coef = self.coef[tuple(ys_)]
		coef[out]=0.
		ondim = len(self.oshape)
		coef = np.moveaxis(coef,range(-ondim,0),range(ondim))
		
		# Spline weights
		weight = self.spline(y,ys)
		return (coef*weight).sum(axis=ondim)

	def set_values(self,values):
		self.coef = self.make_coefs(values)

	def make_coefs(self,values,overwrite_values=False):
		values = ad.asarray(values)
		xp = ad.cupy_generic.get_array_module(values)
		self.interior_nodes = xp.array(self.interior_nodes)
		self.boundary_nodes = xp.array(self.boundary_nodes)

		ondim = values.ndim - self.vdim
		# Internally, interpolation is along the first axes.
		# (Contrary to external interface)
		val = np.moveaxis(values,range(ondim),range(-ondim,0))

		return self.spline.make_coefs(val,overwrite_values=overwrite_values)

	def _grid(self):
		xp = ad.cupy_generic.get_array_module(self.origin)
		dtype = self.origin.dtype
		return ad.asarray(np.meshgrid(*(o+h*xp.arange(s+0.,dtype=dtype) 
			for (o,h,s) in zip(self.origin,self.scale,self.shape)), indexing='ij'))



