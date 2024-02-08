# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .base import Base
from .riemann import Riemann
from .rander import Rander
from . import misc
from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from ..FiniteDifferences import common_field

class AsymQuad(Base):
	r"""
	An asymmetric quadratic norm takes the form 
	$
	F(x) = \sqrt{< x, m x > + max(0,< w, x >)^2},
	$
	where $m$ is a given symmetric positive definite tensor, 
	and w is a given vector.

	Member fields and __init__ arguments : 
	- m : an array of shape (vdim,vdim,n1,..,nk) where vdim is the ambient space dimension.
	The array must be symmetric, a.k.a m[i,j] = m[j,i] for all 0 <= i < j < vdim.
	- w : an array of shape (vdim,n1,...,nk)
	"""

	def __init__(self,m,w):
		m,w = (ad.asarray(e) for e in (m,w))
		self.m,self.w =common_field((m,w),(2,1))

	def norm(self,v):
		v,m,w = common_field((ad.asarray(v),self.m,self.w),(1,2,1))
		return np.sqrt(lp.dot_VAV(v,m,v) + np.maximum(lp.dot_VV(w,v),0.)**2)

	def gradient(self,v):
		v,m,w = common_field((ad.asarray(v),self.m,self.w),(1,2,1))
		a = lp.dot_AV(m,v) + w*np.maximum(0.,lp.dot_VV(w,v))
		return a/np.sqrt(lp.dot_VV(v,a))


	def dual(self):
		M = lp.inverse(self.m+lp.outer_self(self.w))
		wInv = lp.solve_AV(self.m,self.w)
		W = -wInv/np.sqrt(1.+lp.dot_VV(self.w,wInv))
		return AsymQuad(M,W)

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.m.shape[2:]	

	def is_definite(self):
		return Riemann(self.m).is_definite()

	def anisotropy(self):
		eMax = Riemann(self.m+lp.outer_self(self.w)).eigvals().max(axis=0)
		eMin = Riemann(self.m).eigvals().min(axis=0)
		return np.sqrt(eMax/eMin)

	def cost_bound(self):
		return Riemann(self.m + lp.outer_self(self.w)).cost_bound()

	def inv_transform(self,a):
		rander = Rander(self.m,self.w).inv_transform(a)
		return AsymQuad(rander.m,rander.w)
	def with_costs(self,costs):
		rander = Rander(self.m,self.w).with_costs(costs)
		return AsymQuad(rander.m,rander.w)

	def flatten(self,solve_w=False):
		m,w = self.m,self.w
		if solve_w: w = lp.solve_AV(m,w)
		return Rander(m,w).flatten()

	@classmethod
	def expand(cls,arr):
		rd = Rander.expand(arr)
		return cls(rd.m,rd.w)

	def model_HFM(self):
		return "AsymmetricQuadratic"+str(self.vdim)


	@classmethod
	def needle(cls,u,cost_forward,cost_orthogonal,cost_reverse=None):
		"""
		Defines a needle like metric
		- u : reference direction. Denote U = u/|u|, and V the orthogonal unit vector.
		- cost_forward =  norm(U)
		- cost_orthogonal = norm(V) = norm(-V)
		- cost_reverse = norm(-U). (Defaults to cost_orthogonal)
		"""
		if cost_reverse is None: cost_reverse = cost_orthogonal
		cost_parallel = np.minimum(cost_forward,cost_reverse)
		riem,_u = Riemann.needle(u,cost_parallel,cost_orthogonal,ret_u=True)
		cost_diff = cost_forward**2-cost_reverse**2
		return cls(riem.m,np.sign(cost_diff)*np.sqrt(np.abs(cost_diff))*_u)

	@classmethod
	def from_cast(cls,metric): 
		if isinstance(metric,cls):	return metric
		if metric.model_HFM().startswith('AsymIso'):
			vdim,a,s,w = metric.vdim,metric.a,np.sign(metric.a),metric.w
			eye = np.eye(vdim,like=a).reshape( (vdim,vdim)+(1,)*a.ndim )
			return cls(a**2*eye-(s<0)*lp.outer_self(w),s*w)
		riemann = Riemann.from_cast(metric)
		return cls(riemann.m,(0.,)*riemann.vdim)

	def __iter__(self):
		yield self.m
		yield self.w

	def make_proj_dual(self,**kwargs):
		"""kwargs : passed to Riemann.make_proj_dual"""
		proj_m = Riemann(self.m).make_proj_dual(**kwargs)
		proj_w = Riemann(self.m + lp.outer_self(self.w)).make_proj_dual(**kwargs)
		def proj(x):
			x,w = common_field((x,self.w),depths=(1,1))
			x_m = proj_m(x)
			x_w = proj_w(x)
			s = lp.dot_VV(w,x_m) > 0
			return np.where(s[None],x_w,x_m)
		return proj
