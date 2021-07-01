# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

from . import misc
from .base import Base
from .diagonal import Diagonal
from .. import LinearParallel as lp
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd

class Riemann(Base):
	r"""
	A Riemann norm takes the form 
	$$
	F(x) = \sqrt{< x,m x>}, 
	$$
	where m is a given symmetric positive definite tensor.

	Member fields and __init__ arguments : 
	- m : an array of shape $(d,d,n_1,..,n_k)$ where d=`vdim` is the ambient space dimension.
	The array must be symmetric, a.k.a m[i,j] = m[j,i] for all $0\leq i < j < d$. 
	"""

	def __init__(self,m):
		self.m=ad.asarray(m)

	def norm(self,v):
		v,m = fd.common_field((v,self.m),(1,2))
		return np.sqrt(lp.dot_VAV(v,m,v))

	def dual(self):
		return Riemann(lp.inverse(self.m))

	@property
	def vdim(self): return len(self.m)

	@property
	def shape(self): return self.m.shape[2:]	

	def eigvals(self):
		"""
		Eigenvalues of self.m
		"""
		try: return np.moveaxis(np.linalg.eigvalsh(np.moveaxis(self.m,(0,1),(-2,-1))),-1,0)
		except ValueError: 
			assert ad.cupy_generic.from_cupy(self.m)
			import cupy
			return cupy.asarray(Riemann(self.m.get()).eigvals())
	def is_definite(self):
		return self.eigvals().min(axis=0)>0
	def anisotropy(self):
		ev = self.eigvals()
		return np.sqrt(ev.max(axis=0)/ev.min(axis=0))
	def cost_bound(self):
		return np.sqrt(lp.trace(self.m))

	def inv_transform(self,a):
		m,a = fd.common_field((self.m,a),depths=(2,2))
		return Riemann(lp.dot_AA(lp.transpose(a),lp.dot_AA(m,a)))
	def with_costs(self,costs):
		costs,m = fd.common_field((costs,self.m),depths=(1,2))
		return Riemann(m*lp.outer_self(costs))

	def flatten(self):
		return misc.flatten_symmetric_matrix(self.m)

	@classmethod
	def expand(cls,arr):
		return cls(misc.expand_symmetric_matrix(arr))

	def model_HFM(self):
		return "Riemann"+str(self.vdim)

	@classmethod
	def needle(cls,u,cost_parallel,cost_orthogonal,ret_u=False):
		"""
		Defines a Riemannian metric, with 
		- eigenvector u
		- eigenvalue cost_parallel**2 in the eigenspace spanned by u
		- eigenvalue cost_orthogonal**2 in the eigenspace orthogonal with u

		The metric is 
		- needle-like if cost_parallel < cost_orthogonal
		- plate-like otherwise

		Optional argument:
		- ret_u : wether to return the (normalized) vector u
		"""
		u,cost_parallel,cost_orthogonal = (ad.asarray(e) for e in (u,cost_parallel,cost_orthogonal))
		u,cost_parallel,cost_orthogonal = fd.common_field((u.copy(),cost_parallel,cost_orthogonal),(1,0,0))
		
		# Eigenvector normalization
		nu = ad.Optimization.norm(u,ord=2,axis=0)
		mask = nu>0
		u[:,mask] /= nu[mask]

		xp = ad.cupy_generic.get_array_module(u)
		ident = fd.as_field(xp.eye(len(u),dtype=u.dtype),cost_parallel.shape,conditional=False)

		m = (cost_parallel**2-cost_orthogonal**2) * lp.outer_self(u) + cost_orthogonal**2 * ident
		return (cls(m),u) if ret_u else cls(m)

	@classmethod
	def from_diagonal(cls,diag):
		"""
		Produces a Riemann norm whose tensors have the given diagonal.
		"""
		diag = ad.asarray(diag)
		z = np.zeros_like(diag[0])
		vdim = len(diag)
		arr = ad.asarray([[z if i!=j else diag[i] for i in range(vdim)] for j in range(vdim)])
		return cls(arr)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		diag = Diagonal.from_cast(metric)
		return Riemann.from_diagonal(diag.costs**2)

	@classmethod
	def from_mapped_eigenvalues(cls,matrix,mapping):
		"""
		Defines a Riemannian metric which has the same eigenvectors as the provided 
		matrix, but (possibly) distinct eigenvalues obtained by the provided mapping.

		Inputs : 
		- matrix: a symmetric matrix, with shape (dim,dim,...)
		- mapping: a function, taking as input an array of shape (dim,...),
			and returning a similarly shaped array. 
			Called with the eigenvalues of matrix, sorted from smallest to largest.
		"""

		# Get eigenvalues and eigenvectors, caring that numpy puts physical axes last
		m_ = np.moveaxis(matrix,(0,1),(-2,-1)) 
		eVal_,eVec_ = np.linalg.eigh(m_) # Not compatible with AD.
		eVal,eVec = np.moveaxis(eVal_,-1,0),np.moveaxis(eVec_,(-2,-1),(0,1))

		# Apply provided mapping and construct new matrix
		mVal = ad.asarray(mapping(eVal))
		m = lp.outer(eVec,mVal*eVec).sum(axis=2)
		return cls(m)

	def __iter__(self):
		yield self.m
