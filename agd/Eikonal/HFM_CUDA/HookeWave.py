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
import os

from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from ... import Metrics
from ...Metrics.Seismic import Hooke
from . import cupy_module_helper
from .cupy_module_helper import SetModuleConstant
from .VoronoiDecomposition import VoronoiDecomposition

class HookeWave:
	"""
	Warning : accessing this object's properties has a significant memory and 
	computational cost, because all data is converted to a kernel friendly format.
	"""
	def __init__(self,shape,traits=None,periodic=False,vertical_thres=5e-3):
		self._shape = shape
		if self.vdim not in (2,3):
			raise ValueError("Only dimensions 2 and 3 are supported")

		traits_default = {
			'Scalar':np.float32,
			'Int':np.int32,
			'bypass_zeros_macro':True,
			'compact_scheme_macro':True,
			'fourth_order_macro':False,
			'isotropic_metric_macro':False,
			'vertical_macro':0,
			'shape_i': (8,8) if self.vdim==2 else (4,4,4),
			}
		if traits is None: traits = traits_default
		else: traits_default.update(traits); traits = traits_default

		self._offsetpack_t = np.int32

		# TODO ? Another block layer shape_j. Presumably (2,)*self.vdim
		assert len(self.shape_i) == self.vdim
		self._shape_o = tuple(fd.round_up_ratio(shape,self.shape_i))

		if periodic in (True,False): periodic=(periodic,)*self.vdim
		assert periodic is None or len(periodic)==self.vdim
		self._periodic = periodic

		# Voronoi decomposition of Hooke tensor
		self._weights = None
		self._offsets = None
		self._vertical_thres = vertical_thres # Use vertical approximation beyond

		self._metric = None
		self._dt = None
		self._gridScale = None
		self._q = None
		self._p = None
		self._tmp = self.block_expand(cp.zeros((self.vdim,)+self.shape,dtype=self.float_t))
		self._nocheck = False

		# Generate the cuda module
		traits.update({
			'ndim_macro':self.vdim,
			'periodic_macro':any(periodic),
			'periodic_axes':periodic,
		})

		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = cupy_module_helper.getmtime_max(cuda_path)
		source = cupy_module_helper.traits_header(traits,size_of_shape=True)
		self._traits = traits

		source += [
		'#include "HookeWave.h"',
		f"// Date cuda code last modified : {date_modified}"]
		cuoptions = ("-default-device", f"-I {cuda_path}") 

		source="\n".join(source)
		module = cupy_module_helper.GetModule(source,cuoptions)
		self._module = module
		self._AdvanceQ = module.get_function("AdvanceQ")
		self._AdvanceP = module.get_function("AdvanceP")

		self.SetCst('shape_o',self.shape_o,self.int_t)
		self.SetCst('shape_tot',self.shape,self.int_t)

	# Traits
	@property	
	def float_t(self):  return self._traits['Scalar']
	@property
	def int_t(self):    return self._int_t
	@property	
	def offsetpack_t(self): return self._offsetpack_t

	@property	
	def order(self):    return 4 if self._traits['fourth_order_macro'] else 2
	@property	
	def periodic(self): return self._periodic
	@property
	def isotropic_metric(self): return self._traits['isotropic_metric_macro']

	@property	
	def shape(self):    return self._shape
	@property	
	def shape_o(self):  return self._shape_o
	@property	
	def shape_i(self):  return self._traits['shape_i']	
	@property	
	def size_o(self):   return np.prod(self.shape_o)
	@property	
	def size_i(self):   return np.prod(self.shape_i)

	@property	
	def vdim(self):     return len(self.shape)
	def _triangular_number(n): return (n*(n+1))//2
	@property
	def symdim(self):   return self._triangular_number(self.vdim)
	@property
	def decompdim(self):return self._triangular_number(self.symdim)

	# Special handling of vertical geometry
	@property
	def vertical(self): return self._traits['vertical_macro']
	@property
	def vertical_kind(self): 
		return (None,'hexagonal','tetragonal','orthorombic2')[self.vertical]
	@property
	def vertical_thres(self): return self._vertical_thres
	
	
	@property
	def offsetnbits(self):
		if self.vdim==2: return 10
		elif self.vdim==3: return 5
		else: raise ValueError("Unsupported dimension")
	@property
	def voigt2lower(self):
		if self.vdim==2:   return (0,2,1)
		elif self.vdim==3: return (0,5,1,4,3,2)
		else: raise ValueError("Unsupported dimension")	
	
	def block_expand(self,value,constant_values=np.nan,**kwargs):
		"""
		Reshapes the array so as to factor shape_i. Also moves the geometry axis before
		shape_i, following the convention of HookeWave.h
		"""
		value = fd.block_expand(value,self.shape_i,constant_values=constant_values,**kwargs)
		if value.ndim == 2*self.vdim: pass
		elif value.ndim == 2*self.vdim+1: value = np.moveaxis(value,0,self.vdim)
		else: raise ValueError("Unsupported geometry depth")
		return cp.ascontiguousarray(value)
	def block_squeeze(self,value):
		"""Inverse operation to block_expand"""
		if value.ndim==2*self.vdim: pass
		elif value.ndim==2*self.vdim+1: value = np.moveaxis(value,self.vdim,0)
		else: raise ValueError("Unsupported geometry depth")
		return fd.block_squeeze(value,self.shape)

	# PDE parameters
	@property
	def weights(self): 
		if self.vertical: return np.moveaxis(self._full_weights,1,0)
		else: return self.block_squeeze(self._weights)
	@property	
	def offsets(self):
		if self.vertical: offsets2 = np.moveaxis(self._full_offsets,1,0)
		else: offsets2 = self.block_squeeze(self._offsets)
		# Uncompress
		offsets = cp.zeros((self.symdim,)+offsets2.shape,dtype=np.int8)
		order = self.voigt2lower
		nbits = self.offsetnbits
		for i,o in enumerate(order):
			offsets[o]=((offsets2//2**(i*nbits))% 2**nbits) - 2**(nbits-1)
		return offsets

	def _compress_offsets(self,offsets):
		offsets2 = cp.zeros(offsets.shape[1:],dtype=self.offsetpack_t)
		order = self.voigt2lower
		nbits = self.offsetnbits
		for i,o in enumerate(order):
			offsets2 += (offsets[o].astype(int)+2**(nbits-1)) * 2**(nbits*i)
		return offsets2

	@property	
	def hooke(self):
		weights,offsets = self.weights,self.offsets
		full_hooke = (weights * lp.outer_self(offsets)).sum(axis=2)
		if not self.vertical: return full_hooke
		

	@hooke.setter
	def hooke(self,value): 
		if value.ndim==2: self.set_hooke_cst(value)
		else: self.set_hooke(value)
	
	def set_hooke_cst(self,hooke):
		"""Sets a constant hooke tensor over the domain"""
		weights,offsets = VoronoiDecomposition(hooke,offset_t=np.int8)
		self._weights = self.block_expand(fd.as_field(weights,self.shape,depth=1))
		self._offsets = self.block_expand(fd.as_field(self._compress_offsets(offsets),
			self.shape,depth=1),constant_values=0)
		self._firstorder = self.block_expand(cp.zeros((self.vdim*self.symdim,)+self.shape,
			dtype=self.float_t))

	def set_hooke(self,hooke,div_hooke=None):
		hooke = fd.as_field(hooke,self.shape,depth=2)
		assert hooke.shape[:2]==(self.symdim,self.symdim)
		assert not self.vertical # __TODO__ : 
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
					[hooke[5],hooke[1],hooke[3]],
					[hooke[4],hooke[3],hooke[2]],
					])
			else: raise ValueError("Unsupported dimension")
			offsets = cp.eye(self.vdim,2+self.vdim,2,dtype=np.int32)
			# TODO : use one-sided finite differences instead on boundary
			div_hooke = sum([fd.DiffCentered(hk[i],offsets[i],gridScale=self.gridScale,
				padding = None if self.periodic[i] else np.nan) for i in range(self.vdim)])
			hk = None
			div_hooke[np.isnan(div_hooke)]=0 # Should not be too bad thanks to damping

		# Solve for the first order term. (self.vdim,self.symdim)+self.shape
		#reshape going around a cupy bug in linalg solve
		hooke2 = np.moveaxis(hooke.reshape((self.symdim,self.symdim,-1)),(0,1,2),(1,2,0))
		div_hooke = np.moveaxis(div_hooke.reshape((self.vdim,self.symdim,-1)),(0,1,2),(2,1,0))
		firstorder = np.linalg.solve(hooke2,div_hooke)
		hooke2,div_hooke = None,None
		firstorder = np.moveaxis(firstorder,(0,1,2),(2,1,0)).reshape((self.vdim,self.symdim)+self.shape)

		# reshape for kernel and save
		firstorder = np.reshape(firstorder,(-1,)+self.shape)
		self._firstorder = self.block_expand(firstorder)
		firstorder = None

		# Decompose the Hooke tensors using Voronoi's first reduction.
		weights,offsets = VoronoiDecomposition(hooke,offset_t=np.int8)
		self._weights = self.block_expand(weights)
		weights = None
		
		# Reorder the offset components, compress as integer
		self._offsets = self.block_expand(self._compress_offsets(offsets),constant_values=0)

	def _to_vertical(hooke):
		"""Coefficients of the vertical approximation."""
		kind = self.vertical_kind
		hk = Hooke(hooke)
		if   kind=='hexagonal':   return hk.to_hexagonal()
		elif kind=='tetragonal':  return hk.to_tetragonal()
		elif kind=='orthorombic2':return hk.to_orthorombic2()

	def _from_vertical(vertical):
		kind = self.vertical_kind
		if   kind=='hexagonal':   return Hooke.from_hexagonal(vertical)
		elif kind=='tetragonal':  return Hooke.from_tetragonal(vertical)
		elif kind=='orthorombic2':return Hooke.from_orthorombic2(vertical)

	def set_hooke_vertical(self,hooke,div_hooke=None):
		"""Sets a hooke tensor, using a vertical approximation where suitable."""
		# Project, reconstruct, and compute difference
		vertical = self._to_vertical(hooke)
		# Find where the vertical approximation is suitable
		reconstr = self._from_vertical(vertical)
		error2 = ((reconstr-hooke)**2).sum(axis=(0,1))
		reconstr=None
		norm2 = (hooke**2).sum(axis=(0,1))
		rel_error = np.sqrt(error2/norm2)
		norm2,error2=None,None
		self._vertical = self.block_expand(vertical) 
		vertical = None
		full = rel_error>self.vertical_thres # where to use the full hooke tensors
		rel_error = None
		full_ratio = full.sum()/np.prod(self.shape)
		if full_ratio>=0.3: print("Performance warning : proportion {full_ratio} of "
			"Hooke tensors cannot be handled by the vertical approximation")
		self._full_index = self.block_expand(np.nonzeros(full).astype(np.int32))
		full_hooke = hooke[:,:,full]
		full = None
		weights,offsets = VoronoiDecomposition(full_hooke,offset_t=np.int8)
		self._full_weights = cp.ascontiguousarray(np.moveaxis(weights,0,1))
		weights=None
		self._full_offsets = cp.ascontiguousarray(np.moveaxis(
			self._compress_offsets(offsets),0,1))
		offsets=None

		assert False # __TODO__ : compute div_hooke
		
	@property
	def metric(self):
		metric = self.block_squeeze(self._metric)
		if self.isotropic_metric: return metric
		else: return Metrics.misc.expand_symmetric_matrix(metric)
	
	@metric.setter
	def metric(self,value):
		if self.isotropic_metric:
			value = fd.as_field(value,self.shape,depth=0)
		else:
			value = fd.as_field(value,self.shape,depth=2)
			value = Metrics.misc.flatten_symmetric_matrix(value)
			assert len(value)==self.symdim
		self._metric = self.block_expand(value)

	@property
	def damping(self): return self.block_squeeze(self._damping)

	@damping.setter
	def damping(self,value):
		value = fd.as_field(value,self.shape,depth=0)
		self._damping = self.block_expand(value)

	@property
	def dt(self): return self._dt
	
	@dt.setter
	def dt(self,value):
		self.SetCst('dt',value,self.float_t)
		self._dt = value
	
	@property
	def gridScale(self): return self._gridScale

	@gridScale.setter
	def gridScale(self,value):
		assert np.ndim(value)==0
		self.SetCst('idx',1/value,self.float_t)
		self._gridScale = value
	
	def SetCst(self,name,value,dtype):
		SetModuleConstant(self._module,name,value,dtype)

	# Unknowns
	@property	
	def q(self): return self.block_squeeze(self._q)
	@property	
	def p(self): return self.block_squeeze(self._p)

	@q.setter
	def q(self,value): 
		assert len(q)==self.vdim
		value = fd.as_field(value,self.shape,depth=1)
		self._q = self.block_expand(value)
	@p.setter
	def p(self,value): 
		assert len(p)==self.vdim
		value = fd.as_field(value,self.shape,depth=1)
		self._p = self.block_expand(value)


	# Symplectic scheme
	def AdvanceP(self):
		self.check()

		if self.vertical: arg0 = (self._vertical,self._full_index,
			self._full_weights,self._full_offsets,self._full_firstorder)
		else: arg0 = (self._weights,self._offsets)

		self._AdvanceP(self.shape_o,self.shape_i,
			arg0+(self._damping,self._q,self._p,self._tmp))
		self._p,self._tmp = self._tmp,self._p

	def AdvanceQ(self):
		self.check()
		self._AdvanceQ((self.size_o,),(self.size_i,),(
			self._metric,self._damping,
			self._q,self._p,self._tmp))
		self._q,self._tmp = self._tmp,self._q

	def check(self):
		""" 
		Check that all arguments have the correct type, shape, are not None, 
		and are c-contiguous arrays
		"""
		assert not self.vertical # __TODO__ weights, offsets firstorder get special treatment, add geomindex and vertical
		if self._nocheck: return
		args0 = ((self._vertical,self._full_index) if self.vertical 
			else (self._weights,self._offsets,self._firstorder))
		for arg in args0+(self._damping,self._metric,self._q,self._p,self._tmp):
			assert arg.flags.c_contiguous
			assert arg.shape[:self.vdim]==self.shape_o
			assert arg.shape[-self.vdim:]==self.shape_i
			assert arg.ndim in (2*self.vdim,2*self.vdim+1)
			assert arg.dtype in (self.float_t,self.int_t,self.offsetpack_t)

		if self.vertical:
			for arg in (self._full_weights,self._full_offsets,self._full_firstorder):
				assert arg.flags.c_contiguous
				assert arg.shape[:self.vdim] == self.shape
				assert arg.ndim == 2
				assert arg.dtype == self.offsetpack_t if arg is self._offsets else self.float_t
		
		for arg in (self._dt,self._gridScale):
			assert arg is not None

	def Advance(self,dt,niter):
		"""
		niter steps of the Verlet scheme, starting with a half step of q.
		"""
		assert niter>=1
		self.dt = dt/2
		self.AdvanceQ()
		self._nocheck=True
		self.dt = dt
		for i in range(niter-1):
			self.AdvanceP()
			self.AdvanceQ()
		self.AdvanceP()
		self.dt = dt/2
		self.AdvanceQ()
		self._nocheck=False
		self._dt = None





