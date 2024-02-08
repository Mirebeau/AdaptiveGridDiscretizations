# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .base import Base
from .riemann import Riemann
from . import misc
from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

class Rander(Base):
	r"""
	A Rander norm takes the form 
	$$
	F(x) = \sqrt{< x,m.x>} + < w,x>,
	$$
	where m is a given symmetric positive definite tensor, 
	and w is a given vector subject to the consition $< w,m^{-1} w> < 1$.

	Member fields and __init__ arguments : 
	- m : an array of shape (vdim,vdim,n1,..,nk) where vdim is the ambient space dimension.
	The array must be symmetric, a.k.a m[i,j] = m[j,i] for all $0\leq i< j< vdim$.
	- w : an array of shape (vdim,n1,...,nk)
	"""
	def __init__(self,m,w):
		m,w = (ad.asarray(e) for e in (m,w))
		self.m,self.w = fd.common_field((m,w),(2,1))

	def norm(self,v):
		v,m,w = fd.common_field((ad.asarray(v),self.m,self.w),(1,2,1))
		return np.sqrt(lp.dot_VAV(v,m,v))+lp.dot_VV(w,v)


	def dual(self):
		# The dual of a Randers norm is also a Randers norm
		s = lp.inverse(self.m-lp.outer_self(self.w))
		omega = lp.dot_AV(s,self.w)
		return Rander((1+lp.dot_VV(self.w,omega))*s, -omega)

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.m.shape[2:]	
	
	def is_definite(self):
		return np.logical_and(Riemann(self.m).is_definite(),
			lp.dot_VV(lp.solve_AV(self.m,self.w),self.w) <1.)

	def anisotropy_bound(self):
		r = np.sqrt(lp.dot_VV(lp.solve_AV(self.m,self.w),self.w))
		return ((1.+r)/(1.-r))*Riemann(self.m).anisotropy_bound()
	def cost_bound(self):
		return Riemann(self.m).cost_bound() + ad.Optimization.norm(self.w,ord=2,axis=0)

	def inv_transform(self,a):
		w,a = fd.common_field((self.w,a),depths=(1,2))
		return Rander(Riemann(self.m).inv_transform(a).m,lp.dot_VA(w,a))

	def with_costs(self,costs):
		costs,m,w = fd.common_field((costs,self.m,self.w),depths=(1,2,1))
		return Rander(m*lp.outer_self(costs),w*costs)

	def flatten(self,inverse_m=False):
		m,w = self.m,self.w
		if inverse_m: m = lp.inverse(m)
		return np.concatenate((misc.flatten_symmetric_matrix(m),w),axis=0)
	
	@classmethod
	def expand(cls,arr):
		m = misc.expand_symmetric_matrix(arr,extra_length=True)
		d=len(m)
		d_sym = (d*(d+1))//2
		assert(len(arr)==d_sym+d)
		w=arr[d_sym:]
		return cls(m,w)

	def model_HFM(self):
		return "Rander"+str(self.vdim)

	@classmethod
	def from_cast(cls,metric): 
		if isinstance(metric,cls):	return metric
		riemann = Riemann.from_cast(metric)
		return cls(riemann.m,(0.,)*riemann.vdim)

	def __iter__(self):
		yield self.m
		yield self.w

	@classmethod
	def from_Zermelo(cls,metric,drift):
		r"""
		Zermelo's navigation problem consists in computing a minimal path for 
		whose velocity is unit w.r.t. a Riemannian metric, and which is subject 
		to a drift. The accessible velocities take the form 
			$x+drift$ where $< x,m.x> \leq 1$
		This function reformulates it as a shortest path problem 
		in a Rander manifold.

		Inputs : 
		- metric : Symmetric positive definite matrix (Riemannian metric)
		- drift : Vector field, obeying $< drift,metric.drift> < 1$ (Drift)

		Outputs : 
		- the Rander metric.
		"""
		return cls(lp.inverse(metric),drift).dual()

	def to_Zermelo(self):
		"""
		Input : Parameters of a Rander metric.

		Output : Parameters of the corresponding Zermelo problem, of motion on a 
		Riemannian manifold with a drift.
		"""
		self_dual = self.dual()
		return lp.inverse(self_dual.m), self_dual.w

	def to_Varadhan(eps=1):
		r"""
		The Rander eikonal equation can be reformulated in an (approximate)
		linear form, using a logarithmic transformation
			$u + 2 eps < omega,grad u> - eps**2 Tr(D hess u)$.
		Then -eps log(u) solves the Rander eikonal equation, 
		up to a small additional diffusive term.

		Inputs : 
		- m and w, parameters of the Rander metric
		- eps (optionnal), relaxation parameter

		Outputs : 
		- $D$ and $2 * omega$, parameters of the linear PDE. 
		 ($D * eps^2$ and $2 * omega * eps$ if eps is specified)
		"""
		return self.Varadhan_from_Zermelo(self.to_Zermelo(),eps)

	@staticmethod
	def Varadhan_from_Zermelo(metric,drift,eps=1):
		"""
		Zermelo's navigation problem can be turned into a Rander shortest path problem,
		which itself can be (approximately) expressed in linear form using the logarithmic
		transformation. This function composes the above two steps.
		"""
		return eps**2*(lp.inverse(metric)-lp.outer_self(drift)), 2.*eps*drift

	def make_proj_dual(self,**kwargs):
		"""kwargs : passed to Riemann.make_proj_dual"""
		# Based on the Zermelo transformation, by which the Randers unit ball can be expressed 
		# as a shifted ellipse
		proj_riemann = Riemann(self.m).make_proj_dual(**kwargs)
		def proj(x):
			x,w = fd.common_field((x,self.w),depths=(1,1))
			return proj_riemann(x-w)+w
		return proj
