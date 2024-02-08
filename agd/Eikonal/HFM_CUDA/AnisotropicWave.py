# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This file implements the linear acoustic and elastic wave equations, using custom 
GPU kernels for efficiency. A reference implementation using sparse matrices is provided 
for completeness. 
"""

import numpy as np
import os

from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from ... import Metrics
from ... import Selling
from ...Metrics.Seismic import Hooke
from ...ODE.hamiltonian import QuadraticHamiltonian,QuadraticHamiltonianBase
rm_ad = ad.remove_ad

try: # These features require a GPU, but are optional.
	import cupy as cp
	from . import cupy_module_helper
	from .cupy_module_helper import SetModuleConstant
	from . import SellingAnisotropicDiffusion
	from .. import VoronoiDecomposition 

except ImportError:
	cp = None

# ------ Reference implementations using sparse matrices -------
# These implementations can be used to check the validity of the GPU kernels. 
# Additionally, they support automatic differentiation (ad.Dense.denseAD_Lin operators).

bc_to_padding = {'Periodic':None,'Neumann':np.nan,'Dirichlet':0}

def _mk_dt_max(dt_max_22, order_x):
	"""
	Helper for CFL condition. Includes CFL factors coming from the time and spatial 
	discretization orders in the considered wave equations.
	- dt_max_22 : CFL in the case where order_x == order_t == 2 
	"""
	dt_mult_x = {2:1/2, 4:1/np.sqrt(6)}
	dt_mult_t = {1:2, 2:2, 4:1.28596}
	return lambda order_t=2, order_x=order_x : dt_max_22 * dt_mult_x[order_x] * dt_mult_t[order_t]

def AcousticHamiltonian_Sparse(ρ,D,dx=1.,order_x=2,shape_dom=None,bc='Periodic',
	rev_ad=0,save_weights=False):
	"""
	Sparse matrix based implementation of the Hamiltonian of the acoustic wave equation.
	- bc : boundary conditions, see bc_to_padding.keys()
	- rev_ad (optional) : Implement reverse autodiff for the decomposition weights and inverse density
	"""
	padding = bc_to_padding[bc]
	if shape_dom is None: shape_dom = fd.common_shape((ρ,D),depths=(0,2))
	assert len(shape_dom)==len(D)
	iρ=1/ρ
	from .. import VoronoiDecomposition # CPU or GPU
	try: λ,e = VoronoiDecomposition(D)
	except FileNotFoundError: λ,e = Selling.Decomposition(D) # Fallback to python implem
	λ = fd.as_field(λ,shape_dom)
	def Hq(q,ad_channel=lambda x:x):
		dq = fd.DiffEll(q,e,dx,padding=padding,order=order_x) 
		if padding!=padding: dq[np.isnan(dq)]=0
		return 0.5 * ad_channel(λ)  * np.sum(dq**2,axis=0)
	def Hp(p,ad_channel=lambda x:x):
		return 0.5 * ad_channel(iρ) * p**2
	H = QuadraticHamiltonian(Hq,Hp) # Create then replace quadratic functions with sparse matrices
	if rev_ad>0: # Reverse autodiff support
		H.weights = ad.Dense.denseAD(λ, coef=np.zeros_like(λ, shape=(*λ.shape, rev_ad)))
		H.offsets = e
		H.iρ      = ad.Dense.denseAD(iρ,coef=np.zeros_like(iρ,shape=(*iρ.shape,rev_ad)))
		rev_ad = (H.weights.coef,H.iρ.coef)
	else:
		if save_weights: H.weights,H.M,H.iρ = λ,iρ,iρ 
		rev_ad = (None,None)
	H.set_spmat(np.zeros_like(rm_ad(D),shape=shape_dom),rev_ad=rev_ad) 
	H.dt_max = _mk_dt_max(dx * np.sqrt(np.min(rm_ad(ρ)/rm_ad(λ).sum(axis=0))), order_x)
	return H

def ElasticHamiltonian_Sparse(M,C,dx,order_x=2,S=None,shape_dom=None,bc='Periodic',
	rev_ad=0,save_weights=False):
	"""
	Sparse matrix based implementation of the Hamiltonian of the elastic wave equation, namely
	(1/2) int_X M(x,p(x),p(x)) + C(x,ε(x),ε(x)) dx
	where ε = grad(q) + grad(q)^T - S q is the stress tensor, and X is the domain.

	- M : (metric) array of positive definite matrices, shape (d,d,n1,...,nd),
		Also accepts (1,1,n1,...,nd) for isotropic metric. Ex: M = (1/ρ)[None,None]
	- C : (hooke tensor in voigt notation) array of positive definite matrices,
		shape (s,s,n1,...,nd) where s = d (d+1)/2
	- bc : boundary conditions, see bc_to_padding.keys()
	- rev_ad (optional) : Implement reverse autodiff for the decomposition weights and M.
	"""
	padding = bc_to_padding[bc]
	if shape_dom is None: shape_dom = fd.common_shape((M,C),depths=(2,2))
	vdim = len(shape_dom); assert len(C)==(vdim*(vdim+1))//2
	λ,E = Hooke(C).Selling()
	λ,E,S,M = fd.common_field( (λ,E,S,M), depths=(1,3,3,2), shape=shape_dom) # broadcasting
	if S is None: ES = (0,)*vdim
	else: ES = np.sum(E[:,:,None,:]*S[:,:,:,None],axis=(0,1))

	def Hq(q,ad_channel=lambda x:x): # Elastic energy
		λ_ = 2**-vdim * ad_channel(λ) # Normalize, extract relevant ad part
		dq0 = fd.DiffEll(q[0],E[0],dx,order=order_x,α=ES[0]*q[0],padding=padding)
		if padding!=padding: dq0[np.isnan(dq0)]=0
		if vdim==1: return λ_ * np.sum(dq0**2, axis=0)
		dq1 = fd.DiffEll(q[1],E[1],dx,order=order_x,α=ES[1]*q[1],padding=padding)
		if padding!=padding: dq1[np.isnan(dq1)]=0
		if vdim==2: return λ_ * np.sum((dq0+dq1)**2+(dq0+dq1[::-1])**2, axis=0)
		dq2 = fd.DiffEll(q[2],E[2],dx,order=order_x,α=ES[2]*q[2],padding=padding)
		if padding!=padding: dq2[np.isnan(dq2)]=0
		return λ_ * np.sum((dq0+dq1+dq2)**2+(dq0+dq1+dq2[::-1])**2
			+(dq0+dq1[::-1]+dq2)**2+(dq0+dq1[::-1]+dq2[::-1])**2, axis=0)

	def Hp(p,ad_channel=lambda x:x): # Kinetic energy
		if M.shape[:2]==(1,1): return 0.5*ad_channel(M[0,0])*np.sum(p**2,axis=0)# Typically M = 1/ρ
		return 0.5 * p[None,:]*ad_channel(M)*p[:,None] #lp.dot_VAV(p,ad_channel(M),p) changed for rev_ad

	H = QuadraticHamiltonian(Hq,Hp) # Create then replace quadratic functions with sparse matrices
	if rev_ad>0: # Reverse autodiff support
		H.weights  = ad.Dense.denseAD(λ,coef=np.zeros_like(λ,shape=(*λ.shape,rev_ad)))
		H.M = H.iρ = ad.Dense.denseAD(M,coef=np.zeros_like(M,shape=(*M.shape,rev_ad)))
		H.moffsets = E
		rev_ad = (H.weights.coef,H.M.coef)
	else: 
		if save_weights: H.weights,H.M,H.iρ = λ,M,M 
		rev_ad = (None,None)
	H.set_spmat(np.zeros_like(rm_ad(C),shape=(vdim,*shape_dom)),rev_ad=rev_ad)
	H.reshape = lambda x:x; H.unshape = lambda x:x # Dummy, for compatibility only with the GPU kernel

	ev = np.linalg.eigvalsh(np.moveaxis(rm_ad(M),(0,1),(-2,-1)))[...,-1] # Take largest eigvenvalue 
	H.dt_max = _mk_dt_max( dx/(np.sqrt(np.max(ev*rm_ad(λ).sum(axis=0)))*vdim), order_x)
	return H

def AcousticChgVar(q,p,ρ,D,ϕ,X):
	"""
	Change of variables in the acoustic wave equation.
	- q,p,ρ,D (callable) : problem data
	- ϕ : change of variables
	- X : points where to evaluate 
	returns
	- tq,tp,tρ,tD,ϕ(X) (arrays) : coordinate changed problem data, obtained as 
		q(ϕ), p(ϕ) J, ρ(ϕ) J, Φ^-1 D(ϕ) Φ^-t J
	"""
	X_ad = ad.Dense.identity(constant=X,shape_free=(len(X),))
	ϕ_ad = ϕ_fun(X_ad)
	ϕ = ϕ_ad.value
	dϕ = np.moveaxis(ϕ_ad.gradient(),1,0) # Gradient is differential transposed
	inv_dϕ = lp.inverse(dϕ)
	Jϕ = lp.det(dϕ)

	D_ = fd.as_field(D(ϕ),Jϕ.shape,depth=2)
	tD = lp.dot_AA(inv_dϕ,lp.dot_AA(D_,lp.transpose(inv_dϕ))) * Jϕ

	return q(ϕ), p(ϕ)*Jϕ, ρ(ϕ)*Jϕ, tD, ϕ

def ElasticChgVar(q,p,M,C,S,ϕ,X):
	"""
	Change of variables in the elastic wave equation.
	- q,p,M,C,S (callable) : problem data
	- ϕ (callable) : change of variables
	- X : points where to evaluate
	returns
	- tq,tp,tM,tC,tS,ϕ(X) (arrays) : coordinate changed problem data, obtained as 
	Φ^t q(ϕ), Φ^-1 p(ϕ) J, Φ^t M(ϕ) Φ / J, (Φ^t ε(ϕ) Φ,)
	∑_{i'j'k'l'} C_{i'j'k'l'}(ϕ) Ψ^{i'}_i Ψ^{j'}_j Ψ^{k'}_k Ψ^{l'}_l  J,
	∑_{i'j'} Φ^i_{i'} Φ^j_{j'} S^{i'j'}_{k'}(ϕ) Ψ_k^{k'} + ∑_{k'} ∂^{ij} ϕ_{k'} Ψ_k^{k'}
	"""
	X_ad = ad.Dense2.identity(constant=X,shape_free=(len(X),))
	ϕ_ad = ϕ_fun(X_ad)
	ϕ = ϕ_ad.value
	dϕ = np.moveaxis(ϕ_ad.gradient(),1,0) # Gradient is differential transposed
	inv_dϕ = lp.inverse(dϕ)
	Jϕ = lp.det(dϕ)
	d2ϕ = np.moveaxis(ϕ_ad.hessian(),2,0)

	tq = lp.dot_AV(lp.transpose(dϕ),q(ϕ))
	tp = lp.dot_AV(inv_dϕ,p(ϕ))*Jϕ

	M_ = fd.as_field(M(ϕ),Jϕ.shape,depth=2)
	tM = lp.dot_AA(lp.transpose(dϕ),lp.dot_AA(M_,dϕ))/Jϕ

	C_ = fd.as_field(C(ϕ),Jϕ.shape,depth=2)
	tC = Hooke(C_).rotate(inv_dϕ).hooke*Jϕ

	S_ = fd.as_field(S(ϕ),Jϕ.shape,depth=3)
	vdim = len(dϕ)
	S1 = sum(dϕ[ip,:,None,None]*dϕ[jp,None,:,None]*S_[ip,jp,kp]*inv_dϕ[:,kp]
		for ip in range(vdim) for jp in range(vdim) for kp in range(vdim))
	S2 = sum(d2ϕ[kp,:,:,None]*inv_dϕ[:,kp] for kp in range(vdim))
	tS = S1 + S2

	return tq,tp,tM,tC,tS,ϕ

# ------- Implementations based on GPU kernels -------

class AcousticHamiltonian_Kernel(QuadraticHamiltonianBase):
	"""
	The Hamiltonian of an anisotropic acoustic wave equation, implemented with GPU kernels,
	whose geometry is defined by a generic Riemannianian metric field.
	The Hamiltonian is a sum of squares of finite differences, via Selling's decomposition.

	The Mathematical expression of the Hamiltonian is 
	(1/2) int_X iρ(x) p(x)^2 + D(x,grad q(x), grad q(x)) dx
	where X is the domain, and D the is the (dual-)metric.
	"""

	def __init__(self,ρ,D,dx=1.,
		order_x=2,shape_dom=None,periodic=False,
		flattened=False,rev_ad=0,iρ=None,
		block_size=256,traits=None,**kwargs):
		if cp is None: raise ImportError("Cupy library needed for this class")
		super(AcousticHamiltonian_Kernel,self).__init__(**kwargs)

		if shape_dom is None: shape_dom = fd.common_shape((ρ,D),
			depths=(0,1 if flattened else 2))
		self._shape_dom = shape_dom
		self.shape_free = shape_dom
		size_dom = np.prod(shape_dom)
		self._sizes_oi = (int(np.ceil(size_dom/block_size)),),(block_size,)
		self.dx = dx

		fwd_ad = ρ.size_ad if ad.is_ad(ρ) else 0
		self._size_ad = max(rev_ad,fwd_ad)

		# Init the GPU kernel
		traits_default = {
			'fourth_order_macro':{2:False,4:True}[order_x],
			'Scalar':np.float32,
			'Int':self.int_t,
			'OffsetT':np.int8,
			'ndim_macro':self.ndim,
			'periodic_macro':periodic,
			'periodic_axes':(True,)*self.ndim,
			'fwd_macro':fwd_ad>0,
		}
		self._traits = traits_default if traits is None else {**traits_default,**traits}

		# Setup the problem data
		dtype32 = (self.float_t==np.float32)
		iρ = ad.cupy_generic.cupy_set(1/ρ if iρ is None else iρ, dtype32=dtype32) 
		ρ=None
		iρ = fd.as_field(iρ,shape_dom)
		self.iρ = ad.Base.ascontiguousarray(iρ)
		iρ=None

		D = ad.cupy_generic.cupy_set(D, dtype32=dtype32)
		λ,e = VoronoiDecomposition(D,offset_t=np.int8,flattened=flattened)
		D=None
		self._traits['decompdim'] = len(λ)
		self.dt_max = _mk_dt_max(dx/np.sqrt(
			np.max(rm_ad(self.iρ)*rm_ad(λ).sum(axis=0))), order_x)
		λ = fd.as_field(λ,shape_dom,depth=1)
		self.weights = ad.Base.ascontiguousarray(np.moveaxis(λ,0,-1))
		λ=None

		if self.way_ad<0: # Reverse autodiff
			self.weights  = ad.Dense.denseAD(self.weights)
			self.iρ = ad.Dense.denseAD(self.iρ)
		for arr in (self.weights,self.iρ): 
			self.check_ad(arr) if self.size_ad>0 else self.check(arr)

		# Generate the cuda module
		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = cupy_module_helper.getmtime_max(cuda_path)
		source = cupy_module_helper.traits_header(self.traits,size_of_shape=True)

		source += [
		'#include "Kernel_AcousticWave.h"',
		f"// Date cuda code last modified : {date_modified}"]
		cuoptions = ("-default-device", f"-I {cuda_path}") 

		source="\n".join(source)
		module = cupy_module_helper.GetModule(source,cuoptions)
		get_indices = module.get_function('get_indices')
		self._DpH_Kernel = module.get_function("DpH")
		self._DqH_Kernel = module.get_function("DqH")

		if self.size_ad:
			source_ad = f"#define size_ad_macro {self.size_ad}\n"+source
			module_ad = cupy_module_helper.GetModule(source_ad,cuoptions)
			self._DpH_Kernel_ad = module_ad.get_function("DpH")
			self._DqH_Kernel_ad = module_ad.get_function("DqH")
			self._modules = (module,module_ad)
		else: self._modules = (module,)

		for module in self._modules:
			SetModuleConstant(module,'shape_tot',self.shape_dom,self.int_t)
			SetModuleConstant(module,'size_tot',np.prod(self.shape_dom),self.int_t)

		# Get the indices of the scheme neighbors
		self._ineigh = cp.full((*self.weights.shape,order_x),-2**30,dtype=self.int_t)
		e = cp.ascontiguousarray(fd.as_field(e,shape_dom,depth=2))
		get_indices(*self._sizes_oi,(e,self._ineigh))
		e=None
		self.check(self._ineigh)

	@property
	def M(self): 
		"""Alias for the inverse density"""
		return self.iρ
	@property
	def size_ad(self): return self._size_ad
	@property
	def way_ad(self):
		"""0 : no AD. 1 : forward AD. -1 : reverse AD"""
		return 0 if self.size_ad==0 else 1 if self.traits['fwd_macro'] else -1
	def rev_reset(self):
		"""
		Reset the accumulators for reverse autodiff 
		Namely (self.metric.coef and self.weights.coef)
		"""
		self.weights.coef[:]=0
		self.iρ.coef[:]=0

	@property	
	def shape_dom(self):
		"""Shape of the PDE discretization domain."""
		return self._shape_dom
	@property	
	def ndim(self):
		"""Number of dimensions of the domain."""
		return len(self.shape_dom)
	@property	
	def decompdim(self):
		"""Length of quadratic form decomposition."""
		return self.weights.shape[-1]

	@property
	def traits(self): return self._traits
	@property
	def int_t(self): return np.int32
	@property
	def float_t(self): return self.traits['Scalar']
	@property
	def order_x(self): return 4 if self.traits['fourth_order_macro'] else 2

	def Expl_p(self,q,p,δ):
		"""
		Explicit time step for the impulsion p.
		Expects : q and p reshaped in GPU friendly format, using self.reshape
		"""
		mult = -δ/self.dx**2/{2:2,4:24}[self.order_x]
		if self.preserve_p: p = self._mk_copy(p)
		if ad.is_ad(q): # Use automatic differentiation, forward or reverse
			SetModuleConstant(self._modules[1],'DqH_mult',mult,self.float_t)
			self.check_ad(q); self.check_ad(p)
			self._DqH_Kernel_ad(*self._sizes_oi,(self.weights.value,self._ineigh,
				q.value,self.weights.coef,q.coef,p.coef,p.value))
		else: # No automatic differentiation
			SetModuleConstant(self._modules[0],'DqH_mult',mult,self.float_t)
			self.check(q); self.check(p)
			self._DqH_Kernel(*self._sizes_oi,
				(ad.remove_ad(self.weights),self._ineigh,q,p))
		return p

	def Expl_q(self,q,p,δ): 
		"""
		Explicit time step for the position q.
		Expects : q and p reshaped in GPU friendly format, using self.reshape
		"""
		if self.preserve_q: q = self._mk_copy(q)
		if ad.is_ad(p): # Use automatic differentiation, forward or reverse
			SetModuleConstant(self._modules[1],'DpH_mult',δ,self.float_t)
			self.check_ad(q); self.check_ad(p)
			self._DpH_Kernel_ad(*self._sizes_oi,(self.iρ.value,p.value,
				self.iρ.coef,p.coef,q.coef,q.value))
		else: # No automatic differentiation
			self.check(q); self.check(p)
			q += δ*ad.remove_ad(self.iρ)*p 
		return q
	
	def _DqH(self,q): return self.Expl_p(q,np.zeros_like(q),-1)
	def _DpH(self,p): return self.Expl_q(np.zeros_like(p),p, 1)

	def check_ad(self,x):
		"""
		Puts zero ad coefficients, with the correct c-contiguity, if those are empty.
		Basic additional checks of shapes, contiguity.
		"""
		assert self.way_ad!=0 # Either forward and reverse AD must be supported
		if x.size_ad==0: x.coef = np.zeros_like(x.value,shape=(*x.shape,self.size_ad))
		assert isinstance(x,ad.Dense.denseAD) and x.size_ad==self.size_ad
		assert x.coef.flags.c_contiguous 
		assert x.coef.dtype==self.float_t
		self.check(x.value)

	def check(self,x):
		""" Basic check of the types, shapes, contiguity of GPU inputs. """
		assert x.flags.c_contiguous
		assert x.shape[:self.ndim]==self.shape_dom
		assert x.dtype in (self.float_t,self.int_t)


class WaveHamiltonianBase(QuadraticHamiltonianBase):
	"""
	A base class for GPU implementations of Hamiltonians of wave equations.
	Warning : position and impulsion arrays are padded and reshaped in a GPU friendly format.
	__init__ arguments 
	- constant values : used as padding reshape function
	"""

	def __init__(self,shape_dom,traits=None,periodic=False,constant_values=0,**kwargs):
		super(WaveHamiltonianBase,self).__init__(**kwargs)
		self._shape_dom = shape_dom

		traits_default = {
			'Scalar':np.float32,
			'Int':np.int32,
			'bypass_zeros_macro':True,
			'fourth_order_macro':False,
			'shape_i': {1:(64,),2:(8,8),3:(4,4,4),4:(4,4,2,2)}[self.ndim],
			}
		if traits is not None: traits_default.update(traits)
		traits = traits_default

		if np.ndim(periodic)==0: periodic=(periodic,)*self.ndim
		assert periodic is None or len(periodic)==self.ndim
		self._periodic = periodic
		traits.update({
			'ndim_macro':self.ndim,
			'periodic_macro':any(periodic),
			'periodic_axes':periodic,
		})
		self._traits = traits
		self._check = True # Perform some safety input checks before calling GPU kernels

		assert len(self.shape_i) == self.ndim
		self._shape_o = tuple(fd.round_up_ratio(shape_dom,self.shape_i))
		self.constant_values = constant_values

	def _dot(self,q,p): 
		"""Duality bracket, ignoring NaNs used for padding."""
		return np.nansum(q*p) if np.isnan(self.constant_values) else np.sum(q*p)

	# Domain shape
	@property	
	def shape_dom(self):
		"""Shape of the PDE discretization domain."""
		return self._shape_dom
	@property	
	def shape_o(self):  
		"""Outer shape : number of blocks in each dimension (for GPU kernels)."""
		return self._shape_o
	@property	
	def shape_i(self):
		"""Inner shape : accessed by a block of threads (for GPU kernels)."""
		return self.traits['shape_i']
	@property	
	def size_o(self):   return np.prod(self.shape_o)
	@property	
	def size_i(self):   return np.prod(self.shape_i)

	@property	
	def ndim(self):
		"""Number of dimensions of the domain."""
		return len(self.shape_dom)
	@property
	def symdim(self):   
		"""DImension of the space of symmetric matrices."""
		return _triangular_number(self.ndim)

	# Traits
	@property
	def traits(self):
		"""Collection of traits, passed to GPU kernel."""
		return self._traits

	@property	
	def float_t(self):  
		"""Scalar type used by the GPU kernel. Defaults to float32."""
		return self.traits['Scalar']
	@property
	def int_t(self):
		"""Int type used by the GPU kernel. Defaults to int32."""
		return self.traits['Int']
	@property	
	def order_x(self):
		"""Consistency order of the finite differences scheme."""
		return 4 if self.traits['fourth_order_macro'] else 2
	@property	
	def periodic(self): 
		"""Wether to apply periodic boundary conditions, for each axis."""
		return self._periodic

	def SetCst(self,name,value,dtype):
		"""Set a constant un the cuda module"""
		for module in self._modules:
			SetModuleConstant(module,name,value,dtype)

	def reshape(self,x,constant_values=None,**kwargs):
		"""
		Reshapes and pads the array x in a kernel friendly format.
		Factors shape_i. Also moves the geometry axis before
		shape_i, following the convention of HookeWave.h
		- **kwargs : passed to fd.block_expand
		"""
		if constant_values is None: constant_values = self.constant_values
		x = fd.block_expand(x,self.shape_i,constant_values=constant_values,**kwargs)
		if x.ndim == 2*self.ndim: pass
		elif x.ndim == 2*self.ndim+1: x = np.moveaxis(x,0,self.ndim)
		else: raise ValueError("Unsupported geometry depth")
		#Cast to correct float and int types
		if x.dtype==np.float64: dtype=self.float_t
		elif x.dtype==np.int64: dtype=self.int_t
		else: dtype=x.dtype
		#dtype = {np.float64:self.float_t,np.int64:self.int_t}.get(value.dtype,value.dtype) fails
		if ad.is_ad(x):
			x.value = cp.ascontiguousarray(cp.asarray(x.value,dtype=dtype))
			# Setup a suitable contiguity of the AD variable coefficients for the GPU kernel
			x_coef = cp.ascontiguousarray(cp.asarray(np.moveaxis(x.coef,-1,self.ndim),dtype=dtype))
			x.coef = np.moveaxis(x_coef,self.ndim,-1)
			return x
		else: return cp.ascontiguousarray(cp.asarray(x,dtype=dtype))

		return cp.ascontiguousarray(cp.asarray(value,dtype=dtype))

	def unshape(self,value):
		"""Inverse operation to reshape"""
		if value.ndim==2*self.ndim: pass
		elif value.ndim==2*self.ndim+1: value = np.moveaxis(value,self.ndim,0)
		else: raise ValueError("Unsupported geometry depth")
		return fd.block_squeeze(value,self.shape_dom)

class ElasticHamiltonian_Kernel(WaveHamiltonianBase):
	"""
	The Hamiltonian of an anisotropic elastic wave equation, implemented with GPU kernels,
	whose geometry is defined by a generic Hooke tensor field.
	The Hamiltonian is a sum of squares of finite differences, via Voronoi's decomposition.
	Dirichlet boundary conditions are applied, see also optional damping layers.

	The Mathematical expression of the Hamiltonian is 
	(1/2) int_X M(x,p(x),p(x)) + C(x,ε(x),ε(x)) dx
	where ε = grad(q) + grad(q)^T - S q is the stress tensor, and X is the domain.

	- M : (metric) array of positive definite matrices, shape (d,d,n1,...,nd),
		Also accepts (1,1,n1,...,nd) for isotropic metric. Ex: M = (1/ρ)[None,None]
	- C : (hooke tensor in voigt notation) array of positive definite matrices,
		shape (s,s,n1,...,nd) where s = d (d+1)/2
		Reuse decomposition from previous run : C = H_prev.C_for_reuse

	Warning : accessing some of this object's properties has a significant memory and 
	computational cost, because all data is converted to a GPU kernel friendly format.
	"""
	def __init__(self,M,C,dx=1.,order_x=2,shape_dom=None,
		traits=None,rev_ad=0,**kwargs):
		if cp is None: raise ImportError("Cupy library needed for this class")
		fwd_ad = M.size_ad if ad.is_ad(M) else 0
		if fwd_ad>0 and rev_ad>0:
			raise ValueError("Please choose between forward and reverse differentiation")
		self._size_ad = max(fwd_ad,rev_ad)

		traits_default = {
			'isotropic_metric_macro':M.shape[0]==1,
			'fourth_order_macro':{2:False,4:True}[order_x],
			'fwd_macro': fwd_ad>0
			}
		if traits is not None: traits_default.update(traits)
		traits = traits_default

		# Flatten the symmetric matrix arrays, if necessary
		if (M.ndim==2 or M.shape[0] in (1,M.ndim-2)) and M.shape[0]==M.shape[1]: 
			M = Metrics.misc.flatten_symmetric_matrix(M)
		if isinstance(C,tuple): self._weights,self._offsets,shape_dom = C # Reuse decomposition
		elif (C.ndim==2 or C.shape[0]==_triangular_number(C.ndim-2)) and C.shape[0]==C.shape[1]: 
			C = Metrics.misc.flatten_symmetric_matrix(C)

		# Get the domain shape
		if shape_dom is None: shape_dom = M.shape[1:] if isinstance(C,tuple) else \
			fd.common_shape((M,C),depths=(1,1))
		super(ElasticHamiltonian_Kernel,self).__init__(shape_dom,traits,**kwargs)

		if self.ndim not in (1,2,3):
			raise ValueError("Only domains of dimension 1, 2 and 3 are supported")

		self._offsetpack_t = np.int32
		self.dx = dx

		# Voronoi decomposition of the Hooke tensor
		if not isinstance(C,tuple): # Decomposition was bypassed by feeding weights, offsets
			assert len(C) == self.decompdim
			from .. import VoronoiDecomposition 
			weights,offsets = VoronoiDecomposition(ad.cupy_generic.cupy_set(C),
				offset_t=np.int8,flattened=True)
			C = None
			# Broadcast if necessary
			weights = fd.as_field(weights,self.shape_dom,depth=1)
			offsets = fd.as_field(offsets,self.shape_dom,depth=2)
			self._weights = self.reshape(weights)
			weights = None
			# offsets = offsets.get() # Uncomment if desperate to save memory...
			self._offsets = self.reshape(self._compress_offsets(offsets),constant_values=0)
			offsets = None

		# Setup the metric
		assert len(M) == self.symdim or self.isotropic_metric
		self._metric = self.reshape(fd.as_field(M,self.shape_dom,depth=1))
		M = None

		if self.way_ad<0: # Reverse autodiff
			self._weights = ad.Dense.denseAD(self._weights)
			self._metric  = ad.Dense.denseAD(self._metric)
		if self.size_ad:   # Forward or reverse autodiff 
			self.check_ad(self._weights) 
			self.check_ad(self._metric)
		else:
			self.check(self._weights)
			self.check(self._metric)
		self.check(self._offsets)


		if self.isotropic_metric: # TODO : case where M is anisotropic, see Sparse variant
			self.dt_max = _mk_dt_max(dx/(self.ndim*np.sqrt(np.nanmax(
				ad.remove_ad(self._metric).squeeze(axis=self.ndim)
				*ad.remove_ad(self._weights).sum(axis=self.ndim)))),
			order_x=order_x)

		self.shape_free = (*self.shape_o,self.ndim,*self.shape_i)
		self._sizes_oi = (self.size_o,),(self.size_i,)

		# Generate the cuda module
		cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
		date_modified = cupy_module_helper.getmtime_max(cuda_path)
		source = cupy_module_helper.traits_header(self.traits,size_of_shape=True)

		source += [
		'#include "Kernel_ElasticWave.h"',
		f"// Date cuda code last modified : {date_modified}"]
		cuoptions = ("-default-device", f"-I {cuda_path}") 

		source="\n".join(source)
		module = cupy_module_helper.GetModule(source,cuoptions)
		self._DpH_Kernel = module.get_function("DpH")
		self._DqH_Kernel = module.get_function("DqH")

		if self.size_ad:
			source_ad = f"#define size_ad_macro {self.size_ad}\n"+source
			module_ad = cupy_module_helper.GetModule(source_ad,cuoptions)
			self._DpH_Kernel_ad = module_ad.get_function("DpH")
			self._DqH_Kernel_ad = module_ad.get_function("DqH")
			self._modules = (module,module_ad)
		else: self._modules = (module,)

		self.SetCst('shape_o',self.shape_o,self.int_t)
		self.SetCst('size_o',np.prod(self.shape_o),self.int_t)
		self.SetCst('shape_tot',self.shape_dom,self.int_t)

	# Traits
	@property
	def size_ad(self): return self._size_ad #return self._traits['size_ad_macro']
	@property
	def way_ad(self):
		"""0 : no AD. 1 : forward AD. -1 : reverse AD"""
		return 0 if self.size_ad==0 else 1 if self._traits['fwd_macro'] else -1
	def rev_reset(self):
		"""
		Reset the accumulators for reverse autodiff 
		Namely (self.metric.coef and self.weights.coef)
		"""
		assert self.way_ad<0
		self._metric.coef[:]=0
		self._weights.coef[:]=0

	@property
	def isotropic_metric(self): 
		"""Wether M has shape (1,1,...) or (d,d,...)"""
		return self._traits['isotropic_metric_macro']

	@property	
	def offsetpack_t(self): 
		"""Type used to store a matrix offset. Defaults to int32."""
		return self._offsetpack_t

	@property
	def decompdim(self):
		"""Number of terms in the decomposition of a generic Hooke tensor."""
		return _triangular_number(self.symdim)
	
	@property
	def offsetnbits(self):
		"""Number of bits for storing each integer coefficient of the matrix offsets"""
		return (None,10,10,5)[self.ndim]

	@property
	def voigt2lower(self):
		"""
		Correspondance between voigt notation and symmetric matrix indexing.
		d=2 : [0  ]	   d=3 : [0    ]
		      [2 1]          [5 1  ]
		                     [4 3 2]
		"""
		return {1:(0,), 2:(0,2,1), 3:(0,5,1,4,3,2)}[self.ndim]
	
	# PDE parameters
	@property
	def weights(self):
		"""Weights, obtained from Voronoi's decomposition of the Hooke tensors."""
		return self.unshape(self._weights)
	@property	
	def offsets(self):
		"""Offsets, obtained from Voronoi's decomposition of the Hooke tensors."""
		offsets2 = self.unshape(self._offsets)
		# Uncompress
		offsets = cp.zeros((self.symdim,)+offsets2.shape,dtype=np.int8)
		order = self.voigt2lower
		nbits = self.offsetnbits
		for i,o in enumerate(order):
			offsets[o]=((offsets2//2**(i*nbits))% 2**nbits) - 2**(nbits-1)
		return offsets

	@property
	def C_for_reuse(self):
		"""Avoid the initial tensor decomposition step."""
		return self._weights,self._offsets,self.shape_dom

	def _compress_offsets(self,offsets):
		"""Compress each matrix offset (symmetric, integer valued), into a single int_t"""
		offsets2 = np.zeros_like(offsets,shape=offsets.shape[1:],dtype=self.offsetpack_t)
		order = self.voigt2lower
		nbits = self.offsetnbits
		for i,o in enumerate(order):
			offsets2 += (offsets[o].astype(self.int_t)+2**(nbits-1)) * 2**(nbits*i)
		return offsets2

	@property	
	def hooke(self):
		"""The Hooke tensor, input 'C', defining the elasticity properties of the medium."""
		# The Hooke tensor is not stored, so as to save memory. We thus reconstruct it 
		# from Voronoi's decomposition.
		weights,offsets = self.weights,self.offsets
		full_hooke = (weights * lp.outer_self(offsets)).sum(axis=2)
		return full_hooke
				
	@property
	def metric(self):
		"""
		The metric tensor, input 'M'. Defines the norm for measuring momentum. 
		Usually metric = Id/ρ .
		"""
		return Metrics.misc.expand_symmetric_matrix(self.unshape(self._metric))

	@property
	def iρ(self):
		"""Inverse density. Alias for the metric M used to define the kinetic energy."""
		return self.metric

	def Expl_p(self,q,p,δ):
		"""
		Explicit time step for the impulsion p.
		Expects : q and p reshaped in GPU friendly format, using self.reshape
		"""
		mult = -δ/self.dx**2/{2:4,4:48}[self.order_x]
		if self.preserve_p: p = self._mk_copy(p)
		if ad.is_ad(q): # Use automatic differentiation, forward or reverse
			SetModuleConstant(self._modules[1],'DqH_mult',mult,self.float_t)
			self.check_ad(q); self.check_ad(p)
			self._DqH_Kernel_ad(self.shape_o,self.shape_i,(self._weights.value,self._offsets,
				q.value,self._weights.coef,q.coef,p.coef,p.value))
		else: # No automatic differentiation
			SetModuleConstant(self._modules[0],'DqH_mult',mult,self.float_t)
			self.check(q); self.check(p)
			self._DqH_Kernel(self.shape_o,self.shape_i,
				(ad.remove_ad(self._weights),self._offsets,q,p))
		return p

	def Expl_q(self,q,p,δ): 
		"""
		Explicit time step for the position q.
		Expects : q and p reshaped in GPU friendly format, using self.reshape
		"""
		if self.preserve_q: q = self._mk_copy(q)
		if ad.is_ad(p): # Use automatic differentiation, forward or reverse
			SetModuleConstant(self._modules[1],'DpH_mult',δ,self.float_t)
			self.check_ad(q); self.check_ad(p)
			self._DpH_Kernel_ad(*self._sizes_oi,(self._metric.value,p.value,
				self._metric.coef,p.coef,q.coef,q.value))
		else: # No automatic differentiation
			SetModuleConstant(self._modules[0],'DpH_mult',δ,self.float_t)
			self.check(q); self.check(p)
			self._DpH_Kernel(*self._sizes_oi,(ad.remove_ad(self._metric),p,q))
		return q

	def _DqH(self,q): return self.Expl_p(q,np.zeros_like(q),-1)
	def _DpH(self,p): return self.Expl_q(np.zeros_like(p),p, 1)

	def _mk_copy(self,x):
		"""Copies the variable x, with a specific contiguity for the AD coefficients"""
		if ad.is_ad(x):
			assert isinstance(x,ad.Dense.denseAD) and x.size_ad in (0,self.size_ad)
			return ad.Dense.denseAD(x.value.copy(),
				np.moveaxis(np.moveaxis(x.coef,-1,self.ndim).copy(),self.ndim,-1))
		return x.copy()

	def check_ad(self,x):
		"""
		Puts zero coefficients with the correct contiguity if those are empty.
		Checks that the AD variable x has the correct c-contiguity for the kernel.
		"""
		assert self.way_ad!=0 # Both forward and reverse
		if x.size_ad==0:
			x.coef = np.moveaxis(np.zeros_like(x.value,
			shape=(*x.shape[:self.ndim],self.size_ad,*x.shape[self.ndim:])),self.ndim,-1)
		assert isinstance(x,ad.Dense.denseAD) and x.size_ad==self.size_ad
		assert np.moveaxis(x.coef,-1,self.ndim).flags.c_contiguous
		assert x.coef.dtype==self.float_t
		self.check(x.value)

	def check(self,x):
		""" 
		Basic check of the types, shapes, contiguity of GPU inputs.
		"""
		assert x.flags.c_contiguous
		assert x.shape[:self.ndim]==self.shape_o and x.shape[-self.ndim:]==self.shape_i
		assert x.dtype in (self.float_t,self.int_t,self.offsetpack_t)
		assert x.ndim in (2*self.ndim,2*self.ndim+1)

# Utility functions
def _triangular_number(n): return (n*(n+1))//2