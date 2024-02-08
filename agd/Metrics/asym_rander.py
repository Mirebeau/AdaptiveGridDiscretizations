# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from . import misc
from .base import Base
from .riemann import Riemann
from .rander import Rander
from .asym_quad import AsymQuad

from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

class AsymRander(Base):
	r"""
	Asymmetric Rander norms take the form
	$$
	F(x) = \sqrt{<x,m x> + max(0,<u,x>)^2 + max(0,<v,x>)^2} + <w,x>
	$$
	where m is a given symmetric positive definite tensor, and 
	$u,v,w$ are given vectors. The vector $w$ must be small enough, so that $F(x)>0$ 
	for all $x\neq 0$.

	Asymmetric Rander norms generalize both Rander norms and Asymmetric quadratic norms.
	They were proposed by Da Chen in the context of image processing applications.

	Member fields and __init__ arguments : 
	- m : an array of shape (vdim,vdim,n1,..,nk) where vdim is the ambient space dimension.
	The array must be symmetric, a.k.a m[i,j] = m[j,i] for all 0<=i<j<vdim.
	- u,v,w : an array of shape (vdim,n1,...,nk)
	"""

	def __init__(self,m,u,v,w):
		m = ad.asarray(m)
		if u is None: u=np.zeros_like(m[0])
		if v is None: v=np.zeros_like(m[0])
		if w is None: w=np.zeros_like(m[0])
		m,u,v,w = (ad.asarray(e) for e in (m,u,v,w))
		self.m,self.u,self.v,self.w = fd.common_field((m,u,v,w),(2,1,1,1))

	def norm(self,x):
		x,m,u,v,w = fd.common_field((ad.asarray(x),self.m,self.u,self.v,self.w),(1,2,1,1,1))
		return np.sqrt(lp.dot_VAV(x,m,x) + np.maximum(lp.dot_VV(x,u),0.)**2
			+ np.maximum(lp.dot_VV(x,v),0.)**2 ) + lp.dot_VV(x,w)

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.m.shape[2:]	

	def flatten(self):
		m,u,v,w = self.m,self.u,self.v,self.w
		return np.concatenate((misc.flatten_symmetric_matrix(m),u,v,w),axis=0)

	def model_HFM(self):
		return "AsymRander"+str(self.vdim)

	def __iter__(self):
		yield self.m
		yield self.u
		yield self.v
		yield self.w

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		elif isinstance(metric,Rander): 
			z = np.zeros_like(metric.w)
			return cls(metric.m,z,z,metric.w)
		else:
			asym = AsymQuad.from_cast(metric)
			z = np.zeros_like(metric.w)
			return cls(metric.m,metric.w,z,z)

	def make_proj_dual(self,**kwargs):
		"""
		Orthogonal projection onto the dual unit ball.
		- **kwargs : passed to AsymQuad.make_proj_dual
		"""
		if not np.allclose(v,0.): raise ValueError("Sorry, prox compuation requires v=0")
		proj_mu = AsymQuad(self.m,self.u).make_proj_dual(**kwargs)
		return lambda x : proj_mu(x-self.w)+self.w

