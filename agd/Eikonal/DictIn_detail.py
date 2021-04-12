# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements additional functionality of the Eikonal.dictIn class, related to 
source factorization, and to solving PDEs on spheres.
"""

import numpy as np
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd
from .. import LinearParallel as lp
from .. import Metrics

def factoringPointChoice(self):
	if 'factoringPointChoice' in self: # 'Key' and 'Seed' are synonyms here
		assert self['factoringPointChoice'] in ('Both','Key','Seed') 
		return self['factoringPointChoice']
	elif 'factoringRadius' not in self: return None
	elif self.get('order',1)==3: return 'Both'
	else: return 'Seed'


def SetFactor(self,radius=None,value=None,gradient=None):
	"""
	This function setups additive factorization around the seeds.
	Inputs (optional): 
	- radius.
		Positive number -> approximate radius, in pixels, of source factorization.
		-1 -> Factorization over all the domain. 
		None -> source factorization over all the domain
	- value (optional).
	  callable, array -> approximate values of the solution. 
	  None -> reconstructed from the metric.
	- gradient (optional) 
	  callable, array -> approximate gradient of the solution.
	  Obtained from the values by automatic differentiation if unspecified.

	Outputs : the subgrid used for factorization
	Side effect : sets 'factoringValues', 'factoringGradients', 
	   and in the case of a subgrid 'factoringIndexShift'
	"""
	# Set the factoring grid points
	if radius is None:
		radius = self.get('factoringRadius',10)

	if radius<0 or radius>=np.max(self.shape):
		factGrid = self.Grid()
		self.pop('factoringIndexShift',None)
	else:
		seed_ci =  self.PointFromIndex(self['seed'],to=True)
		bottom = [max(0,int(np.floor(ci)-radius)) for ci in seed_ci]
		top = [min(si,int(np.ceil(ci)+radius)+1) for ci,si in zip(seed_ci,self.shape)]
		self['factoringIndexShift'] = bottom
		aX = [self.xp.arange(bi,ti,dtype=self.float_t) for bi,ti in zip(bottom,top)]
		factGrid = ad.array(np.meshgrid(*aX,indexing='ij'))
		factGrid = np.moveaxis(self.PointFromIndex(np.moveaxis(factGrid,0,-1)),-1,0)
#			raise ValueError("dictIn.SetFactor : unsupported radius type")

	# Set the values
	if value is None:
		value = self.get('factoringPointChoice','Seed')

	if isinstance(value,str):
		seed = self['seed']
		if 'metric' in self: metric = self['metric']
		elif self['model'].startswith('Isotropic'): metric = Metrics.Isotropic(1)
		else: raise ValueError("dictIn.SetFactor error : no metric found")
		if 'cost' in self: metric = metric.with_cost(self['cost'])
		fullGrid = factGrid if factGrid.shape[1:]==self.shape else self.Grid()

		diff = lambda x : x-fd.as_field(seed,x.shape[1:],depth=1)
		if value in ('Key','Seed'):
			metric.set_interpolation(fullGrid) # Default order 1 interpolation suffices
			value = metric.at(seed)
		elif value=='Current': # Not really working ?
			# Strictly speaking, we are not interpolating the metric, 
			# since the point x at which it is evaluated lies on the grid
			metric.set_interpolation(fullGrid) 
			value = lambda x : metric.at(x).norm(diff(x))
			#Cheating : not differentiating w.r.t position, but should be enough here
			gradient = lambda x: metric.at(x).gradient(diff(x)) 
		elif value=='Both':
			# Pray that AD goes well ...
			metric.set_interpolation(fullGrid,order=3) # Need to differentiate w.r.t x
			value = lambda x : 0.5*(metric.at(seed).norm(diff(x)) + 
				metric.at(x).norm(diff(x)))
		else:
			raise ValueError(f"dictIn.SetFactor error : unsupported "
				"factoringPointChoice : {value} .")

	if callable(value):
		if gradient is None:
			factGrid_ad = ad.Dense.identity(constant=factGrid, shape_free=(self.vdim,))
			value = value(factGrid_ad)
		else:
			value = value(factGrid)

	if isinstance(value,Metrics.Base):
		factDiff = factGrid - fd.as_field(self['seed'],factGrid.shape[1:],depth=1)
		if gradient is None: 
			gradient = value.gradient(factDiff)
			# Avoids recomputation, but generates NaNs at the seed points.
			value = lp.dot_VV(gradient,factDiff)
			value[np.isnan(value)]=0 
		else:
			value = value.norm(factDiff)

	if ad.is_ad(value):
		gradient = value.gradient()
		value = value.value

	if not ad.isndarray(value): 
		raise ValueError(f"dictIn.SetFactor : unsupported value type {type(value)}")

	#Set the gradient
	if callable(gradient):
		gradient = gradient(factGrid)

	if not ad.isndarray(gradient): 
		raise ValueError(f"dictIn.SetFactor : unsupported gradient type {type(gradient)}")

	self["factoringValues"] = value
	self["factoringGradients"] = np.moveaxis(gradient,0,-1) # Geometry last in c++ code...

	for key in ('factoringRadius', 'factoringPointChoice'): self.pop(key,None)

	return factGrid

#def proj_tocost(x): return 2./(1.+lp.dot_VV(x,x))
def proj_tosphere(x):
	"""
	Maps a point of the equatorial plane to the sphere, by projection : 
	x -> (2 x,1-|x∏^2)/(1+|x|^2)
	""" 
	sq = lp.dot_VV(x,x)
	return np.concatenate((x,np.expand_dims((1.-sq)/2.,axis=0)),axis=0) * (2./(1.+sq))
def proj_fromsphere(q): 
	"""
	Maps a point of the sphere to the equatorial plane, by projection.
	(q,qz) -> q/(1+qz)
	"""
	return q[:-1]/(1.+q[-1])

#def sphere_tocost(x,chart): return proj_tocost(x)
def sphere_tosphere(x,chart): 
	"""
	See proj_tosphere. Last component is reversed according to chart.
	"""
	y = proj_tosphere(x)
	y[-1] *= chart
	return y
def sphere_fromsphere(q,chart):
	"""
	See proj_fromsphere. Last component of q is reversed according to chart.
	"""
	q=q.copy()
	q[-1] *= chart
	return proj_fromsphere(q)

def SetSphere(self,dimsp,separation=5,radius=1.1):
	"""
	Setups the manifold R^(d-k) x S^k, using the equatorial projection for the sphere S^k.
	Only compatible with the GPU accelerated eikonal solver. 
	Inputs : 
		dimsp (int): the discretization of a half sphere involves dimsp^k pixels.
		radius (optional, float > 1): each local chart has base domain [-radius,radius]^k.
		  (This is NOT the radius of the sphere, which is 1, but of the parametrization.)
		separation (optional, int) : number of pixels separating the two local charts.
			Set to false to define a projective space using a single chart. 
	Side effects : 
		Sets chart_mapping,chart_jump, dimensions, origin, gridScales.
	Output : 
		- Conversion utilities, between the equatorial plane and the sphere 
			(and grid in the non-projective case)
	"""
	if 'chart_mapping' in self: # Some space R^(d-k) x S^k is already set
		vdim_rect = self.vdim - self['chart_mapping'].ndim+1
		dims_rect = self['dims'][:vdim_rect]
		origin_rect = self['origin'][:vdim_rect]
		gridScales_rect = self['gridScales'][:vdim_rect]
	elif 'dims' in self: # Some rectangle R^(d-k) is already set
		dims_rect = self['dims']
		origin_rect = self.get('origin',np.zeros_like(dims_rect))
		if 'gridScales' in self: gridScales_rect = self['gridScales']
		else: gridScales_rect = self['gridScale']*np.ones_like(dims_rect) 
	else: # Nothing set. All coordinates are sphere like, d=k
		dims_rect = self.xp.array(tuple(),dtype=self.float_t)
		origin_rect = dims_rect.copy()
		gridScales_rect = dims_rect.copy()

	vdim_rect = len(dims_rect)
	vdim_sphere = self.vdim-vdim_rect # Dimension of the sphere
	gridScale_sphere = 2*radius/dimsp

	if separation: # Sphere manifold, described using two charts
		separation_radius = separation*gridScale_sphere /2
		center_radius = radius + separation_radius

		dims_sphere = (2*dimsp + separation,) + (dimsp,)*(vdim_sphere-1)
		origin_sphere = (-2*radius-separation_radius,) + (-radius,)*(vdim_sphere-1)

		def sphere_fromgrid(x):
			"""
			Produces a point of the equatorial plane, and a boolean indicating which projection to 
			use (False:south pole, True:north pole), from a grid point.
			"""
			assert len(x)==vdim_sphere
			x = x.copy()
			chart = np.where(np.abs(x[0])>=separation_radius,np.sign(x[0]),np.nan)
			x[0] -= center_radius*chart
			return x,chart

		def sphere_togrid(x,chart):
			"""
			Produces a point of the original grid, from a point of the equatorial plane 
			and a boolean indicating the active chart.
			"""
			assert len(x)==vdim_sphere
			x=x.copy()
			mixed_chart = x[0]*chart < -radius # The two coordinate charts would get mixed
			x[:,mixed_chart] = np.nan
			x[0] += chart*center_radius
			return x
	else: # Projective space, described using a single chart
		dims_sphere = (dimsp,)*vdim_sphere
		origin_sphere = (-radius,)*vdim_sphere

	dims_sphere,origin_sphere,gridScales_sphere = [self.array_float_caster(e) 
		for e in (dims_sphere,origin_sphere, (gridScale_sphere,)*vdim_sphere)]

	self['dims'] = np.concatenate((dims_rect,dims_sphere),axis=0)
	self['gridScales'] = np.concatenate(
		(gridScales_rect,gridScales_sphere),axis=0)
	self.pop('gridScale',None)
	self['origin'] = np.concatenate((origin_rect,origin_sphere),axis=0)

	# Produce a coordinate system, and set the jumps, etc
	aX = self.Axes()[vdim_rect:]
	X = self.array_float_caster(np.meshgrid(*aX,indexing='ij'))
	if separation: X,chart = sphere_fromgrid(X)
	X2 = lp.dot_VV(X,X)

	# Geodesics jump when the are sufficiently far away from the fundamental domain.
	radius_jump = (1+radius)/2
	self['chart_jump'] = X2 > radius_jump**2
	if separation:
		self['chart_mapping'] = sphere_togrid(
			sphere_fromsphere(sphere_tosphere(X,chart),-chart),-chart)
		return {'from_grid':sphere_fromgrid,'to_grid':sphere_togrid,
			'from_sphere':sphere_fromsphere,'to_sphere':sphere_tosphere} 
	else:
		self['chart_mapping'] = proj_fromsphere(-proj_tosphere(X))
		return {'from_sphere':proj_fromsphere,'to_sphere':proj_tosphere}
