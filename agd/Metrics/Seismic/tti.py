# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import copy

from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from .. import misc
from ..riemann import Riemann
from .implicit_base import ImplicitBase
from .thomsen_data import HexagonalFromTEM


class TTI(ImplicitBase):
	"""
	A family of reduced models, known as Tilted Transversally Anisotropic,
	and arising in seismic tomography.

	The *dual* unit ball is defined by an equation of the form
	$$
	l(X^2+Y^2,Z^2) + q(X^2+Y^2,Z^2) = 1,
	$$
	where $l$ is linear and $q$ is quadratic, where $X,Y,Z$ are the coefficients 
	of the input vector, usually altered by a linear transformation.
	In two dimensions, ignore the $Y^2$ term.

	The primal norm is obtained implicitly, by solving an optimization problem.

	Member fields and __init__ arguments : 
	- linear : an array of shape (2,n1,...,nk) encoding the linear part l
	- quadratic : an array of shape (2,2,n1,...,nk) encoding the quadratic part q
	- vdim (optional) : the ambient space dimension
	- *args,**kwargs (optional) : passed to implicit_base
	"""

	def __init__(self,linear,quadratic,vdim=None,*args,**kwargs):
		super(TTI,self).__init__(*args,**kwargs)
		self.linear = ad.asarray(linear)
		self.quadratic = ad.asarray(quadratic)
		assert len(self.linear) == 2
		assert self.quadratic.shape[:2] == (2,2)
		self._to_common_field()
		
		if vdim is None:
			if self.inverse_transformation is not None: vdim=len(self.inverse_transformation)
			elif self.linear.ndim>1: vdim = self.linear.ndim-1
			else: raise ValueError("Unspecified dimension")
		self._vdim=vdim

	@property
	def vdim(self): return self._vdim
	
	@property
	def shape(self): return self.linear.shape[1:]

	def _dual_level(self,v,params=None,relax=0.):
		l,q = self._dual_params(v.shape[1:]) if params is None else params
		v2 = v**2
		if self.vdim==3: v2 = ad.array([v2[:2].sum(axis=0),v2[2]])
		return lp.dot_VV(l,v2) + np.exp(-relax)*lp.dot_VAV(v2,q,v2) - 1.
	
	def cost_bound(self):
		# Ignoring the quadratic term for now.
		return self.Riemann_approx().cost_bound()
	def Riemann_approx(self):
		diag = 1/self.linear
		if self.vdim==3: diag = diag[0],diag[0],diag[1] 
		return Riemann.from_diagonal(diag).inv_transform(self.inverse_transformation)

	def _dual_params(self,shape=None):
		return fd.common_field((self.linear,self.quadratic),depths=(1,2),shape=shape)

	def __iter__(self):
		yield self.linear
		yield self.quadratic
		yield self._vdim
		for x in super(TTI,self).__iter__(): yield x

	def _to_common_field(self,shape=None):
		self.linear,self.quadratic,self.inverse_transformation = fd.common_field(
			(self.linear,self.quadratic,self.inverse_transformation),
			depths=(1,2,2),shape=shape)

	@classmethod
	def from_cast(cls,metric):
		if isinstance(metric,cls): return metric
		else: raise ValueError("No casting operations supported towards the TTI model")
		# Even cast from Riemann is non-canonical

	def model_HFM(self):
		return f"TTI{self.vdim}"

	def extract_xz(self):
		"""
		Extract a two dimensional Hooke tensor from a three dimensional one, 
		corresponding to a slice through the X and Z axes.
		Axes transformation information (rotation) is discarded.
 		"""
		if len(self.shape)==3: raise ValueError("Three dimensional field")
		if self.inverse_transformation is not None:
			raise ValueError("Cannot extract XZ slice from tilted norms")
		other = copy.copy(self)
		other._vdim = 2
		return other

	def flatten(self,transposed_transformation=False):
		linear = self.linear
		quad = 2.*self.quadratic # Note the factor 2, used in HFM

		if self.inverse_transformation is None: 
			xp = ad.cupy_generic.get_array_module(linear)
			trans = fd.as_field(xp.eye(self.vdim,dtype=linear.dtype),self.shape,depth=2) 
		else: trans = self.inverse_transformation
		if transposed_transformation: trans = lp.transpose(lp.inverse(trans))

		return np.concatenate(
			(self.linear,misc.flatten_symmetric_matrix(quad),
				trans.reshape((self.vdim**2,)+self.shape)),
			axis=0)

	@classmethod
	def expand(cls,arr):
		vdim = np.sqrt(len(arr)-(2+3))
		assert(vdim==int(vdim))
		vdim = int(vdim)
		shape = arr.shape[1:]

		linear = arr[0:2]
		quadratic = 0.5*misc.expand_symmetric_matrix(arr[2:5])
		inv_trans = arr[5:].reshape((vdim,vdim)+shape)
		return cls(linear,quadratic,vdim=vdim,inverse_transformation=inv_trans)

	@classmethod
	def from_hexagonal(cls,c11,_,c13,c33,c44,vdim=3):
		linear = [c11+c44,c33+c44]
		mixed = 0.5*(c13**2-c11*c33)+c13*c44
		quadratic = [[-c11*c44,mixed],[mixed,-c33*c44]]
		return cls(linear,quadratic,vdim=vdim)

	@classmethod
	def from_Thomsen(cls,tem,vdim=3):
		"""
		Produces a norm from the given Thomsem elasticity parameters.
		"""
		hex,ρ = HexagonalFromTEM(tem)
		return cls.from_hexagonal(*hex,vdim),ρ

# See Hooke.py file for reference
TTI.mica = TTI.from_hexagonal(178.,42.4,14.5,54.9,12.2), 2.79
# Stishovite is tetragonal, but the P-Wave velocity in the XZ plane 
# is equivalent to an hexagonal model.
TTI.stishovite2 = TTI.from_hexagonal(453,np.nan,203,776,252).extract_xz(), 4.29 

