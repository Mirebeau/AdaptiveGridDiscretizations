# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .base import Base
from .isotropic import Isotropic
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

class Diagonal(Base):
	r"""
	A Diagonal norm takes the form 
	$$
	F(x) = \sqrt{\sum_{0\leq i < d} c_i x_i^2 },
	$$
	where $(c_i)_{0\leq i < d}$, are given positive scalars

	Member fields and __init__ arguments : 
	- costs : the array of $(c_i)_{0\leq i < d}$ positive numbers. Required shape : $(d,n_1,..,n_k)$ where $d=$`vdim` is the ambient space dimension.
	"""

	def __init__(self,costs):
		self.costs = ad.asarray(costs)

	@classmethod
	def from_speed(cls,speeds): 
		"""
		Produces a metric whose costs equal 1/speeds
		"""
		return cls(1./speeds)

	def dual(self): return self.from_speed(self.costs)
	def with_costs(self,costs): 
		self_costs,costs = fd.common_field((self.costs,costs),(1,1))
		return Diagonal(self_costs*costs)

	def norm(self,v):
		costs,v = fd.common_field((self.costs,v),depths=(1,1))
		return ad.Optimization.norm(costs*v,ord=2,axis=0)

	def is_definite(self): return np.all(self.costs>0.,axis=0)
	def anisotropy(self): return np.max(self.costs,axis=0)/np.min(self.costs,axis=0)
	def cost_bound(self): return np.max(self.costs,axis=0)

	@property
	def vdim(self): return len(self.costs)
	@property
	def shape(self): return self.costs.shape[1:]

	def flatten(self):      return self.costs
	@classmethod
	def expand(cls,arr):    return cls(arr)

	def model_HFM(self):
		return "Diagonal"+str(self.vdim)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		iso = Isotropic.from_cast(metric)
		shape = (iso.vdim,) + iso.shape
		return cls(np.broadcast_to(iso.cost,shape))

	def __iter__(self):
		yield self.costs
