# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements some spline interpolation methods, on uniform grids,
in a manner compatible with automatic differentiation.

If you do not need to differentiate the interpolated value w.r.t. the position,
 then using ndimage_map_coordinates is in likely to be more efficient numerically (but uses only
 float32 accuracy).
"""


import numpy as np
import itertools
import scipy.linalg

from . import AutomaticDifferentiation as ad


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


def coordinates_on_grid(coordinates,grid=None,origin=None,scale=None):
	"""Rescale coordinates relatively to a uniform grid : (coordinates-origin)/scale"""
	coordinates = ad.asarray(coordinates) # Ensure coordinates are presented as array not tuple
	if grid is not None: origin,scale,_ = origin_scale_shape(grid)
	if origin is None and scale is None: return coordinates
	origin,scale = (_append_dims(e,coordinates.ndim-1) for e in (origin,scale))
	return (coordinates-origin)/scale


def _append_dims(x,ndim):
	"""Appends specified number of singleton dimensions"""
	return np.reshape(x,x.shape+(1,)*ndim)


def ndimage_map_coordinates(input,coordinates,*args,grid=None,**kwargs):
	"""
	Wrapper over the ndimage.map_coordinates function, with the following helper tools : 
	- rescale the coordinates using a reference grid.
	- dispatch to cupyx.scipy.ndimage.map_coordinates if needed.
	- interpolate tensor data (geometry first)
	- AD type supported on input
	- AD type supported on coordinates (dispatch to reimplementation below)

	Additional input : 
	- grid (optional) : reference coordinate system, which must be uniform
	- *args,**kwargs : passed to scipy.ndimage.map_coordinates (or cupyx equivalent)
	"""
	if ad.is_ad(coordinates): return map_coordinates(input,coordinates,*args,grid=grid,**kwargs)
	coordinates = coordinates_on_grid(coordinates,grid)

	if ad.is_ad(input):
		# Basic support for dense AD types
		value = ndimage_map_coordinates(input.value,coordinates,*args,**kwargs)
		coef = np.moveaxis(ndimage_map_coordinates(np.moveaxis(input.coef,-1,0),coordinates,
			*args,**kwargs),0,-1)
		if isinstance(input, ad.Dense.denseAD): return ad.Dense.denseAD(value,coef)
		elif isinstance(input, ad.Dense.denseAD2):
			coef2 = np.moveaxis(ndimage_map_coordinates(np.moveaxis(input.coef,(-2,-1),(0,1)),
				coordinates,*args,**kwargs),(0,1),(-2,-1))
			return ad.Dense2.denseAD2(value,coef,coef2)
		else: raise ValueError(f"Unsupported ad type {type(input)} in ndimage_map_coordinates")

	if ad.cupy_generic.from_cupy(input): 
		from cupyx.scipy.ndimage import map_coordinates as _map_coordinates
	else: from scipy.ndimage import map_coordinates as _map_coordinates

	depth = input.ndim-len(coordinates)
	if depth==0: return _map_coordinates(input,coordinates,*args,**kwargs)
	# In the vector case, we interpolate each component independently
	oshape = input.shape[:depth]
	input = input.reshape((-1,)+input.shape[depth:])
	out = ad.array([_map_coordinates(input_,coordinates,*args,**kwargs) for input_ in input])
	return out.reshape(oshape+coordinates.shape[1:])

#-------------------------------------------------------------------------
# Partial reimplementation, with support for ADtype in coordinates

def spline_base(x,order=3):
	"""Basic spline functions, evaluated at x,x+1,..., over their support, where 0<=x<=1""" 
	if order==0: return (np.ones_like(x),)
	if order==1: return (x,1-x)
	x2 = x*x  
	if order==2: return (x2,1+2*x-2*x2,1-2*x+x2)
	x3 = x*x2 
	if order==3: return (x3,1+3*x+3*x2-3*x3,4-6*x2+3*x3,1-3*x+3*x2-x3)
	x4 = x*x3 
	if order==4: return (x4,1+4*x+6*x2+4*x3-4*x4,11+12*x-6*x2-12*x3+6*x4,11-12*x-6*x2+12*x3-4*x4,1-4*x+6*x2-4*x3+x4)
	x5 = x*x4
	if order==5: return (x5,1+5*x+10*x2+10*x3+5*x4-5*x5,26+50*x+20*x2-20*x3-20*x4+10*x5,66-60*x2+30*x4-10*x5,26-50*x+20*x2+20*x3-20*x4+5*x5,1-5*x+10*x2-10*x3+5*x4-x5)
	# The last two are just for fun
	x6 = x*x5
	if order==6: return (x6,1+6*x+15*x2+20*x3+15*x4+6*x5-6*x6,57+150*x+135*x2+20*x3-45*x4-30*x5+15*x6,302+240*x-150*x2-160*x3+30*x4+60*x5-20*x6,302-240*x-150*x2+160*x3+30*x4-60*x5+15*x6,57-150*x+135*x2-20*x3-45*x4+30*x5-6*x6,1-6*x+15*x2-20*x3+15*x4-6*x5+x6)
	x7 = x*x6
	if order==7: return (x7,1+7*x+21*x2+35*x3+35*x4+21*x5+7*x6-7*x7,120+392*x+504*x2+280*x3-84*x5-42*x6+21*x7,1191+1715*x+315*x2-665*x3-315*x4+105*x5+105*x6-35*x7,2416-1680*x2+560*x4-140*x6+35*x7,1191-1715*x+315*x2+665*x3-315*x4-105*x5+105*x6-21*x7,120-392*x+504*x2-280*x3+84*x5-42*x6+7*x7,1-7*x+21*x2-35*x3+35*x4-21*x5+7*x6-x7)
	raise ValueError(f"Unsupported spline degree : {order=}")

def spline_weighted(c,x,order=3,overwrite_x=False,periodic=False):
	"""
	Evaluate splines at position x, weighted by coefficients coef, with reflected boundary conditions
	- c : spline coefficients, array of shape (m1,...,mk,n1,...,nd)
	- x : spline evaluation position, array of shape (d,p1,...,pl)

	Returns : 
	- weighted spline, array of shape (m1,...,mk,p1,...,pl)
	"""
	if not overwrite_x : x = x.copy()
	x -= (order-1)/2 # Account for shift in spline support # order//2
	int_t = np.int32 # Integer type. Enough for applications, and GPU friendly
	x_dim = len(x)
	x_shape = x[0].shape
	x = x.reshape((x_dim,-1))
	c_shape = c.shape[:-x_dim] # depth = len(c_shape) (0 for scalar, 1 for matrix, ...)
	dom_shape = c.shape[-x_dim:]
	c = c.reshape((-1,*dom_shape))
	x_i = np.floor(x).astype(int_t)
	x -= x_i # 0<=x<=1, so that we can evaluate the spline 
	xp = ad.cupy_generic.get_array_module(x)
	x_i = x_i[:,None] + xp.arange(1+order,dtype=int_t)[:,None] # shape = (d,1+order,p)
	dom_shape_arr = xp.array(dom_shape,dtype=int_t)[:,None,None]
	periodic = np.expand_dims(xp.array(periodic),axis=(-2,-1))
	x_i = x_i % np.where(periodic,dom_shape_arr,2*dom_shape_arr)
	refl = x_i>=dom_shape_arr
	x_i[refl] = (2*dom_shape_arr-1-x_i)[refl]
	x_i = tuple(z.reshape( (1,)*k + (order+1,) + (1,)*(x_dim-k-1)+(-1,)) for k,z in enumerate(x_i))
	c = c[:,*x_i] # Get the interpolation coefficients
	for k in reversed(range(x_dim)):
		spline_val = ad.array(spline_base(x[k],order)[::-1])
		c = np.sum(c*spline_val,axis=-2)
	return c.reshape(c_shape+x_shape)

def spline_coefs(c,order=3,depth=0,periodic=False,solver=None):
	"""
	Produces coefficients such that spline weighted better approximates the values of c. For odd 
	order, the values at the grid nodes (integer coordinates) are exactly reproduced.

	Input : 
	- c : values of the function to be interpolated
	- order (int, default=3) : spline interpolation order.
	- periodic (bool, or tuple of bools) : choose between periodic and reflected boundary conditions

	Output : 
	- spline weights, assuming reflect boundary conditions, for use in spline_weighted
	"""
	if order<=1: return c # Bypass for (zero-th and) first order splines (hat function)
	xp = ad.cupy_generic.get_array_module(c)
	if solver is None: solver = scipy.linalg.solve_circulant
	dom_shape = c.shape[depth:]
	ord2 = order//2 # Different treatment of even and odd orders in this line and the next one
	spline_vals = xp.asarray(spline_base(((1+order)%2)/2,order))
	if not isinstance(periodic,tuple): periodic=(periodic,)*len(dom_shape)
	for i,(s,per) in enumerate(zip(dom_shape,periodic)):
		c = np.moveaxis(c,depth+i,0)
		if per: circ = np.zeros_like(ad.remove_ad(c),shape=(s,))
		else: circ = np.zeros_like(ad.remove_ad(c),shape=(2*s,)); c = np.concatenate((c,np.flip(c,axis=0)))
		circ[0] = spline_vals[-1]; circ[-order:] = spline_vals[:-1]
		if isinstance(c,ad.Dense.denseAD):c=ad.Dense.denseAD(solver(circ,c.value), solver(circ,c.coef))
		elif isinstance(c,ad.Dense2.denseAD2):c=ad.Dense2.denseAD2(solver(circ,c.value), solver(circ,c.coef), solver(circ,c.coef2))
		else: c = solver(circ,c)
		c = np.roll(c,-ord2,axis=0) if per else c[ord2:ord2+s]
		c = np.moveaxis(c,0,depth+i)
	return c

def map_coordinates(input,coordinates,output=None,order=3,mode='constant',cval=0.0,prefilter=True,
	grid=None,periodic=False):
	"""
	Partial reimplementation of map_coordinates, which allows (input and) coordinates to be AD arrays.
	Appears to be much more accurate that ndimage.map_coordinates, which likely uses float32 internally

	- grid : rescale the coordinates according to this grid
	"""
	assert output is None
	assert mode in ('reflect','grid-mirror','grid-wrap')
	if mode=='grid-wrap': periodic=True
	coordinates = coordinates_on_grid(coordinates,grid)
	if prefilter: coefs = spline_coefs(input,order,depth=input.ndim-len(coordinates),periodic=periodic)
	else: coefs = input/math.factorial(order)
	return spline_weighted(coefs,coordinates,order,periodic=periodic,overwrite_x=grid is not None)

class UniformGridInterpolation:
	"""
	Interpolates values on a uniform grid, in arbitrary dimension, using splines of 
	a given order. Uses the reimplementation of map_coordinates, which allows evaluating at 
	AD types for position.
	"""

	def __init__(self,grid,values,order=1,mode='reflect',check_grid=False,periodic=False):
		"""
		- grid (ndarray) : must be a uniform grid. E.g. np.meshgrid(aX,aY,indexing='ij')
		 where aX,aY have uniform spacing. Alternatively, provide only the axes.
		- values (ndarray) : interpolated values.
		- order (int, tuple of ints) : spline interpolation order, along each axis.
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

		assert mode=='reflect'
		self.order = order
		self.periodic = periodic
		self.coefs = spline_coefs(values,order,depth=values.ndim-len(grid),periodic=periodic)

	def __call__(self,x):
		x = coordinates_on_grid(x,origin=self.origin,scale=self.scale)
		return spline_weighted(self.coefs,x,self.order,overwrite_x=True,periodic=self.periodic)
