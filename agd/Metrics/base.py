# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .. import AutomaticDifferentiation as ad
from ..AutomaticDifferentiation import cupy_support as cps
from .. import LinearParallel as lp
from .. import FiniteDifferences as fd
from .. import Interpolation

class Base:
	"""
	Base class for a metric
	"""

	def __repr__(self):
		return type(self).__name__ + repr(tuple(self))

	def __str__(self):
		return type(self).__name__ + str(tuple(self))

	def norm(self,v):
		"""
		Norm defiend by the metric. 
		Expected to be 1-homogeneous w.r.t. v
		"""
		raise NotImplementedError("""Error : norm must be specialized in subclass""")

	def gradient(self,v):
		"""
		Gradient of the norm defined by the metric.
		"""
		if ad.is_ad(v) or ad.is_ad(self,iterables=(type(self),)):
			v_dis = ad.disassociate(v,shape_bound=v.shape[1:])
			grad_dis = self.disassociate().gradient(v_dis)
			return ad.associate(grad_dis)

		v_ad = ad.Dense.identity(constant=v,shape_free=(len(v),))
		return np.moveaxis(self.norm(v_ad).coef,-1,0)
	
	def dual(self):
		r"""
		Dual norm, mathematically defined by 
		$N^*(x) = max\\{ < x, y> ; N(y)\leq 1 \\}$
		"""
		raise NotImplementedError("dual is not implemented for this norm")

	@property
	def vdim(self):
		"""
		Ambient vector space dimension
		"""
		raise NotImplementedError("vdim is not implemented for this norm")

	@property
	def shape(self):
		"""
		Dimensions of the underlying domain.
		Expected to be the empty tuple, or a tuple of length vdim.
		"""
		raise NotImplementedError("Shape not implemented for this norm")
	
	def disassociate(self):
		"""
		Hide the automatic differentiation (AD) information of the member fields.
		See AutomaticDifferentiation.disassociate
		"""
		def dis(x):
			if ad.isndarray(x) and x.shape[-self.vdim:]==self.shape:
				return ad.disassociate(x,shape_bound=self.shape)
			return x
		return self.from_generator(dis(x) for x in self)
# ---- Well posedness related methods -----
	def is_definite(self):
		"""
		Wether norm(u)=0 implies u=0. 
		"""
		raise NotImplementedError("is_definite is not implemented for this norm")

	def anisotropy(self):
		"""
		Sharp upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		raise NotImplementedError("anisotropy is not implemented for this norm")

	def anisotropy_bound(self):
		"""
		Upper bound on norm(u)/norm(v), 
		for any unit vectors u and v.
		"""
		return self.anisotropy()
	def cost_bound(self):
		"""
		Upper bound on norm(u), for any unit vector u.
		"""
		raise NotImplementedError("cost_bound is not implemented for this norm")

# ---- Causality and acuteness related methods ----

	def cos_asym(self,u,v):
		"""
		Generalized cosine defined by the metric, defined as 
		< grad F(u), v> / F(v)
		"""
		u,v=(ad.asarray(e) for e in (u,v))
		return lp.dot_VV(self.gradient(u),v)/self.norm(v)

	def cos(self,u,v):
		"""
		Generalized cosine defined by the metric, defined as 
		min(cos_asym(u,v),cos_asym(v,u)).
		"""
		u,v=(ad.asarray(e) for e in (u,v))
		gu,gv=self.gradient(u),self.gradient(v)
		guu,guv = lp.dot_VV(gu,u),lp.dot_VV(gu,v)
		gvu,gvv = lp.dot_VV(gv,u),lp.dot_VV(gv,v)
		return np.minimum(guv/gvv,gvu/guu)

	def angle(self,u,v):
		"""
		Generalized unoriented angle defined by the metric,
		see the cos and cos_asym member functions.
		"""
		c = ad.asarray(self.cos(u,v))
		mask=c < -1.
		c[mask]=0.
		result = ad.asarray(np.arccos(c))
		result[mask]=np.inf
		return result

# ---- Geometric transformations ----

	def inv_transform(self,a):
		"""
		Affine transformation of the norm. 
		The new unit ball is the inverse image of the previous one.
		"""
		raise NotImplementedError("Affine transformation not implemented for this norm")

	def transform(self,a):
		"""
		Affine transformation of the norm.
		The new unit ball is the direct image of the previous one.
		"""
		return self.inv_transform(lp.inverse(a))

	def rotate(self,r):
		"""
		Rotation of the norm, by a given rotation matrix.
		The new unit ball is the direct image of the previous one.
		"""
		return self.transform(r)

	def rotate_by(self,*args,**kwargs):
		"""
		Rotation of the norm, based on rotation parameters : angle (and axis in 3D).
		"""
		return self.rotate(lp.rotation(*args,**kwargs))

	def with_costs(self,costs):
		"""
		Produces a norm N' obeying N'(x) = N(costs * x)
		where the multiplication is elementwise.
		"""
		a = cps.zeros_like(costs,shape=(len(costs),)+costs.shape)
		for i,cost in enumerate(costs): a[i,i] = cost
		return self.inv_transform(a)

	def with_speeds(self,speeds): 
		"""
		Produces a norm N' obeying N'(x) = N(x/speeds) 
		where the division is elementwise.
		"""
		return self.with_costs(1./speeds)
	
	def with_cost(self,cost): 
		"""
		Produces a norm N' obeying N'(x) = N(cost*x)
		"""
		cost = ad.asarray(cost)
		costs = np.broadcast_to(cost,(self.vdim,)+cost.shape)
		return self.with_costs(costs)


	def with_speed(self,speed): 
		"""
		Produces a norm N' obeying N'(x) = N(x/speed)
		"""
		return self.with_cost(1/speed)

# ---- Import and export ----

	def flatten(self):
		"""
		Flattens and concatenate the member fields into a single array.
		"""
		raise NotImplementedError("Flattening not implemented for this norm")

	@classmethod
	def expand(cls,arr):
		"""
		Inverse of the flatten member function. 
		Turns a suitable array into a metric.
		"""
		raise NotImplementedError("Expansion not implemented for this norm")

	def to_HFM(self):
		"""
		Formats a metric for the HFM library. 
		This may include flattening some symmetric matrices, 
		concatenating with vector fields, and moving the first axis last.
		"""
		return np.moveaxis(self.flatten(),0,-1)

	def model_HFM(self):
		"""
		The name of the 'model' for parameter, as input to the HFM library.
		"""
		raise NotImplementedError("HFM name is not specified for this norm")

	@classmethod
	def from_HFM(cls,arr):
		"""
		Inverse of the to_HFM member function.
		Turns a suitable array into a metric.
		"""
		return cls.expand(np.moveaxis(arr,-1,0))

	def __iter__(self):
		"""
		Iterate over the member fields.
		"""
		raise NotImplementedError("__iter__ not implemented for this norm")
		
	@classmethod
	def from_generator(cls,gen):
		"""
		Produce a metric from a suitable generator expression.
		"""
		return cls(*gen)

	@classmethod
	def from_cast(cls,metric):
		"""
		Produces a metric by casting another metric of a compatible type.
		"""
		raise NotImplementedError("from_cast not implemented for this norm")

	@property
	def array_float_caster(self):
		"""
		Returns a caster function, which can be used to turn lists, etc, into
		arrays with the suitable floating point type, from the suitable library 
		(numpy or cupy), depending on the member fields.
		"""
		return ad.cupy_generic.array_float_caster(self,iterables=type(self))

# ---- Related with Lagrandian and Hamiltonian interpretation ----

	def norm2(self,v):
		"""
		Half squared norm.
		"""
		n = self.norm(v)
		return 0.5*n**2

	def gradient2(self,v):
		"""
		Gradient of the half squared norm.
		"""
		g = self.gradient(v)
		return lp.dot_VV(g,v)*g

	def set_interpolation(self,grid,**kwargs):
		"""
		Sets interpolation_data, required to specialize the norm 
		at a given position.
		Inputs:
			- grid (optional). Coordinate system (required on first call). 
			- kwargs. Passed to UniformGridInterpolation (includes order)
		"""
		vdim = len(grid)
		try: assert self.vdim == vdim
		except ValueError: pass # Constant isotropic metrics have no dimension

		def make_interp(value):
			if hasattr(value,'shape') and value.shape[-vdim:]==grid.shape[1:]:
				return Interpolation.UniformGridInterpolation(grid,value,**kwargs)
			return value

		self.interpolation_data = tuple(make_interp(value) for value in self)

	def at(self,x):
		"""
		Interpolates the metric to a given position, on a grid given beforehand.
		Inputs : 
			- x. Place where interpolation is needed.
		"""
		return self.from_generator(
			field(x) if callable(field) else field
			for field in self.interpolation_data)
		# isinstance(field,Interpolation.UniformGridInterpolation)
	

#	def is_ad(self):
#		return ad.is_ad(self,iterables=(Base,))

#	def remove_ad(self):
#		return self.from_generator(ad.remove_ad(x) for x in self)
