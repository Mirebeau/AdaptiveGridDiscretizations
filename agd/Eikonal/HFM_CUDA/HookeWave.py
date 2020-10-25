# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements the linear elastic wave equation, with a generic Hooke tensor.
Reading 
Dt q = Dp H - r q
Dt p = Dq H - r p 
where r is a damping factor, H is a separable quadratic Hamiltonian 
H(q,p) = A(grad q) + B(p)
where A is the elastic potential energy, defined by the Hooke tensor, 
and B the kinetic energy, defined by the metric tensor.

The operator DpH is "differentiated then discretized". Usually, 
"discretize then differentiate" methods are preferred, but here they are much more memory
intensive (at least in three dimensions with the fourth order scheme).
"""

import numpy as np
import cupy as cp

from .VoronoiDecomposition import VoronoiDecomposition
from ... import FiniteDifferences as fd
from ... import Metrics
from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant



class HookeWave:

	def __init__(self,shape,shape_i=None,shape_j=None,periodic=None,order=2):
			#hooke,metric,damping,q,p):
		"""
		Decompose the hooke tensor using Voronoi's reduction. 
		Reshape the results, and q0,p0, for GPU efficient matrix vector products.
		"""

		self._float_t = np.float32
	#	self._int_t = np.int32
		self._offsetpack_t = np.int32

		self._shape = shape
		if self.vdim not in (2,3):
			raise ValueError("Only dimensions 2 and 3 are supported")

		if shape_i is None:
			shape_i = (8,8) if self.vdim==2 else (4,4,4)
		assert len(shape_i) == self.vdim
		self._shape_i = shape_i
		self._size_i = np.prod(shape_i)

		assert shape_j is None # TODO. Another layer. Presumably (2,)*self.vdim
		self._shape_j = shape_j

		shape_o = fd.round_up_ratio(shape,shape_i)
		self._shape_o = shape_o
		self._size_o = np.prod(shape_o)

		assert periodic is None or len(periodic)==self.vdim
		self._periodic = periodic

		if self.order not in (2,4):
			raise ValueError("Only second and fourth order scheme supported")
		self._order = order

		# Voronoi decomposition of Hooke tensor
		self._weights = None
		self._offsets = None

		self._metric = None
		self._dtQ = None
		self._dtP = None
		self._gridScales = None
		self._q = None
		self._p = None
		self._tmp = self.expand(cp.zeros(self.shape,dtype=self.float_t),padding=np.nan)
		self._nocheck = False

		# Generate the cuda module
		traits = {
			'shape_i':self.shape_i,
			'Scalar':self.float_t,
			'ndim_macro':self.vdim,
			'fourth_order_macro':self.order==4,
		}
		if periodic is not None: traits['periodic'] = periodic

		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = cupy_module_helper.getmtime_max(cuda_path)
		source = cupy_module_helper.traits_header(traits,size_of_shape=True)

		source += [
		'#include "HookeWave.h"',
		f"// Date cuda code last modified : {date_modified}"]
		cuoptions = ("-default-device", f"-I {cuda_path}") 

		source="\n".join(source)
		module = cupy_module_helper.GetModule(source,cuoptions)
		self._module = module
		self._UpdateQ = module.get_function("UpdateQ")
		self._UpdateP = module.get_function("UpdateP")

	# Traits
	@getter
	def float_t(self):  return self._float_t
#	@getter
#	def int_t(self):    return self._int_t
	@getter
	def offsetpack_t(self): return self._offsetpack_t
	@getter
	def order(self):    return self._order
	@getter
	def periodic(self): return self._periodic
	@getter
	def shape(self):    return self._shape
	@getter
	def shape_o(self):  return self._shape_o
	@getter
	def shape_i(self):  return self._shape_i
	@getter
	def shape_j(self):  return self._shape_j
	@getter
	def vdim(self):     return len(self.shape_i)
	@getter
	def size_o(self):   return self._size_o
	@getter
	def size_i(self):   return self._size_i


	def block_expand(self,value,**kwargs):
		"""
		Reshapes the array so as to factor shape_i. Also moves the geometry axis before
		shape_i, following the convention of HookeWave.h
		"""
		value = fd.block_expand(value,self.shape_i,**kwargs)
		if value.ndim == 2*self.vdim: return value
		elif value.ndim == 2*self.vdim+1: return np.moveaxis(value,0,self.vdim)
		else: raise ValueError("Unsupported geometry depth")
	def block_squeeze(self,value):
		"""Inverse operation to block_expand"""
		if value.ndim==2*self.vdim: pass
		elif value.ndim==2*self.vdim+1: value = np.moveaxis(value,self.vdim,0)
		else: raise ValueError("Unsupported geometry depth")
		return fd.block_squeeze(value,self.shape)

	# PDE parameters
	@setter
	def hooke(self,value): self.set_hooke(value)
		
	def set_hooke(self,hooke,div_hooke=None):
		hooke = fd.as_field(hooke,self.shape,depth=2)
		if div_hooke is None: 
			# Compute the divergence of the Hooke tensor, using finite differences
			if self.vdim==2: # Reshape as 2x2x3
				hk = ad.asarray([
					[hooke[0],hooke[2]],
					[hooke[2],hooke[1]],
					])
			elif self.vdim==3: # Reshape as 3x3x6
				hk = ad.asarray([
					[hooke[0],hooke[5],hooke[4]],
					[hooke[0],hooke[5],hooke[4]],
					[hooke[0],hooke[5],hooke[4]],
					])
			else: raise ValueError("Unsupported dimension")
			offsets = cp.eye(vdim,dtype=self.float_t)
			# TODO : use one-sided finite differences instead on boundary
			div_hooke = sum([fd.DiffCentered(hk[i],offsets[i],gridScale=self.gridScales[i],
				padding = None if self.periodic[i] else np.nan) for i in range(self.vdim)])
			hk = None
			div_hooke[np.isnan(div_hooke)]=0 # Should not be too bad thanks to damping

		# Solve for the first order term, and save it. (self.vdim,self.symdim)+self.shape
		firstorder = np.moveaxis(0,1,lp.solve_AV(hooke,np.moveaxis(div_hooke,1,0)))
		div_hooke = None
		firstorder = np.reshape(firstorder,(-1,)+self.shape)
		self._firstorder = fd.block_expand(firstorder,padding=np.nan)
		firstorder = None

		# Decompose the Hooke tensors using Voronoi's first reduction.
		weights,offsets = VoronoiDecomposition(hooke,offset_t=np.int8)
		self._weights = fd.block_expand(weights,padding=np.nan)
		weights = None
		
		# Reorder the offset components, compress as integer
		offsets2 = cp.zeros(weights.shape,dtype=self.offsetpack_t)
		if self.vdim==2:   nbits=10; order = (0,2,1) # Voigt to lower triangular ordering
		elif self.vdim==3: nbits=5;  order = (0,5,1,4,3,2)
		else: raise ValueError("Unsupported dimension")

		for i,o in enumerate(order):
			offsets2 += (offsets[o]+2**(nbits-1)) * 2**(nbits*i)
		self._offsets = offsets2


	@setter
	def metric(self,value):
		value = fd.as_field(value,self.shape,depth=2)
		value = Metrics.misc.flatten_symmetric_matrix(value)
		self._metric = self.block_expand(value,padding=np.nan)
	@setter
	def damping(self,value):
		value = fd.as_field(value,self.shape,depth=0)
		self._damping = self.block_expand(value,padding=np.nan)

	@setter
	def dtQ(self,value):
		assert np.ndim(value)==0
		self.SetCst('dtQ',value,self.float_t)
		self._dtQ = value

	@setter
	def dtP(self,value):
		assert np.ndim(value)==0
		self.SetCst('dtP',value,self.float_t)
		self._dtP = value

	@setter
	def dt(self,value): 
		self.dtQ = value
		self.dtP = value

	@setter
	def gridScales(self,value):
		assert np.ndim(value)==1 and len(value)==self.vdim
		self.SetCst('idx',1./cp.asarray(value),float_t)
		self._gridScales = gridScales

	@setter
	def gridScale(self,value): self.gridScales = (value,)*self.vdim
	
	def SetCst(self,name,value,dtype):
		SetModuleConstant(self._module,name,value,dtype)

	# Unknowns
	@setter
	def q(self,value): 
		value = fd.as_field(value,self.shape,depth=1)
		self._q = self.block_expand(value,padding=np.nan)
	@setter
	def p(self,value): 
		value = fd.as_field(value,self.shape,depth=1)
		self._p = self.block_expand(value,padding=np.nan)

	@getter
	def q(self): return self.block_squeeze(self._q)
	@getter
	def p(self): return self.block_squeeze(self._p)

	# Symplectic scheme
	def AdvanceQ(self):
		self.check()
		self._UpdateQ(self.shape_o,self.shape_i,(
			self._weights,self._offsets,self._firstorder,self._damping,
			self._q,self._p,self._tmp))
		self._q,self._tmp = self._tmp,self._q

	def AdvanceP(self):
		self.check()
		self._UpdateP((self.size_o,),(self.size_i,),(
			self._metric,self._damping,
			self._q,self._p,self._tmp))
		self._p,self._tmp = self._tmp,self._p

	def check(self):
		""" 
		Check that all arguments have the correct type, shape, are not None, 
		and are c-contiguous arrays
		"""
		if self._nocheck: return
		for arg in (self._weights,self._offsets,self._firstorder,self._damping,self._metric,
			self._q,self._p,self._tmp):
			assert arg.flags.c_contiguous
			assert args.shape[:self.vdim]==self.shape_o
			assert arg.shape[-self.vdim:]==self.shape_i
			assert arg.ndim in (self.vdim,self.vdim+1)
			assert arg.dtype == self.offsetpack_t if arg is self._offsets else self.float_t

		for arg in (self._dtP,self._dtQ,self._gridScales):
			assert arg is not None

	def Advance(self,dt,niter):
		"""
		niter steps of the Verlet self-adjoint scheme, starting with a half step of q.
		"""
		self.dtQ = dt/2
		self.dtP = dt
		self.AdvanceQ()
		self._nocheck=True
		self.dtQ = dt
		for i in range(niter-1):
			self.AdvanceP()
			self.AdvanceQ()
		self.AdvanceP()
		self.dtQ = dt/2
		self.AdvanceQ()
		self._nocheck=False





