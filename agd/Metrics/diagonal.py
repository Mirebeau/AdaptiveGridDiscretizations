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

	def make_proj_dual(self,newton_maxiter=7):
		"""
		Returns the orthogonal projection onto the unit ball.
		- newton_maxiter : number of iterations for the inner subproblem
		"""
		# Implementation based on Benamou, J.-D., Carlier, G. & Hatchi, R. A numerical solution to 
		# Monge’s problem with a Finsler distance as cost. ESAIM: M2AN 52, 2133–2148 (2018).
		d_ = self.costs
		def proj(x):
			d,x = fd.common_field((d_,x),depths=(1,1))
			x2=x**2
			norm2 = np.sum(x2/d,axis=0)
			# Solve, via a Newton method, the equation f(β)=0 where
			# f(β) = -1 + sum_i xi^2/(1+di*β)^2 .
			# By convexity of f, no damping or fancy stopping criterion is needed
			β = np.min(d,axis=0)*(np.sqrt(norm2)-1) # Initial guess, exact in isotropic case.
			orig = β<=0.; β[orig]=0. # Those are projected onto the origin
			lc2 = np.where(orig[None],1.,d*x2)
			for i in range(newton_maxiter):
				ilb = 1./(d+β)
				a = lc2*ilb**2
				val = np.sum(a,axis=0)-1.
				val[orig]=0
				der = -2*np.sum(a*ilb,axis=0)
				β -= val/der
			return x*d/(d+β)
		return proj

