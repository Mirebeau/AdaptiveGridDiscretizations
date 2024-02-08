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
from .thomsen_data import ThomsenElasticMaterial, ThomsenGeometricMaterial


class TTI(ImplicitBase):
	"""
	A family of reduced models, known as Tilted Transversally Anisotropic,
	and arising in seismic tomography.

	The *dual* unit ball is defined by an equation of the form
	$$
	l(X^2+Y^2,Z^2) + (1/2)*q(X^2+Y^2,Z^2) = 1,
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

	def __init__(self,linear,quadratic,vdim=None,*args,**kwargs): #rotation_angles=None,
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
		#self.rotation_angles=rotation_angles

	@property
	def vdim(self): return self._vdim
	
	@property
	def shape(self): return self.linear.shape[1:]

	def _dual_level_root(self,v):
		"""Level set function defining the dual unit ball in the square root domain"""
		l,q = self._dual_params(v.shape[1:])
		return lp.dot_VV(l,v) + 0.5*lp.dot_VAV(v,q,v) - 1.

	def _dual_level(self,v,params=None,relax=0.):
		"""Level set function defining the dual unit ball."""
		l,q = self._dual_params(v.shape[1:]) if params is None else params
		v2 = v**2
		if self.vdim==3: v2 = ad.array([v2[:2].sum(axis=0),v2[2]])
		return lp.dot_VV(l,v2) + 0.5*np.exp(-relax)*lp.dot_VAV(v2,q,v2) - 1.
	
	def cost_bound(self):
		return self.Isotropic_approx()[1]
		# Ignoring the quadratic term for now.
		return self.Riemann_approx().cost_bound()

	def _dual_params(self,shape=None):
		return fd.common_field((self.linear,self.quadratic),depths=(1,2),shape=shape)

	def __iter__(self):
		yield self.linear
		yield self.quadratic
		yield self._vdim
#		yield self.rotation_angles
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

	def flatten(self,transposed_transformation=False,cp_get=False):
		if self.inverse_transformation is None:
			xp = ad.cupy_generic.get_array_module(self.linear)
			trans = fd.as_field(xp.eye(self.vdim,dtype=self.linear.dtype),self.shape,depth=2) 
		else: trans = self.inverse_transformation
		if transposed_transformation: trans = lp.transpose(lp.inverse(trans))

		if cp_get: # return a numpy array (memory optimization for GPU intensive)
			return np.concatenate((self.linear.get(),
				misc.flatten_symmetric_matrix(self.quadratic.get()),
				trans.get().reshape((self.vdim**2,)+self.shape)),axis=0)
		else: 
			return np.concatenate((self.linear,misc.flatten_symmetric_matrix(self.quadratic),
				trans.reshape((self.vdim**2,)+self.shape)),axis=0)

	@classmethod
	def expand(cls,arr):
		vdim = np.sqrt(len(arr)-(2+3))
		assert(vdim==int(vdim))
		vdim = int(vdim)
		shape = arr.shape[1:]

		linear = arr[0:2]
		quadratic = misc.expand_symmetric_matrix(arr[2:5])
		inv_trans = arr[5:].reshape((vdim,vdim)+shape)
		return cls(linear,quadratic,vdim=vdim,inverse_transformation=inv_trans)

	@classmethod
	def from_hexagonal(cls,c11,_,c13,c33,c44,vdim=3):
		linear = [c11+c44,c33+c44]
		mixed = c13**2-c11*c33+2.*c13*c44
		quadratic = [[-2.*c11*c44,mixed],[mixed,-2.*c33*c44]]
		return cls(linear,quadratic,vdim=vdim)

	@classmethod
	def from_ThomsenElastic(cls,tem,vdim=3):
		"""Produces a norm from the given Thomsem elasticity parameters."""
		if not isinstance(tem,ThomsenElasticMaterial): tem = ThomsenElasticMaterial(*tem)
		hex,ρ = tem.to_hexagonal()
		return cls.from_hexagonal(*hex,vdim),ρ

	@classmethod
	def from_ThomsenGeometric(cls,tgm,vdim=3,normalize_Vp=False):
		"""Produces a norm from the given Thomsen geometric paramters."""
		if not isinstance(tgm,ThomsenGeometricMaterial): tgm = ThomsenGeometricMaterial(*tgm)
		if normalize_Vp:
			Vp,Vs,ϵ,δ = tgm
			tgm = ThomsenGeometricMaterial(1.,Vs/Vp,ϵ,δ)
		c11,c13,c33,c44 = tgm.to_c()
		return cls.from_hexagonal(c11,None,c13,c33,c44,vdim=vdim)

	def α_bounds(self):
		"""
		The TTI norm can be written as an enveloppe of ellipses, with eigenvalues 
		(1-α,α) / μ(α) or (1-α,1-α,α)/μ(α), where α is within the bounds given by this function.
		Note : another way to obtain the envelope is to use the Riemann_envelope method.

		Returns : αa,αb,mix_is_min
		"""
		l,q = self.linear,self._q()
		a,b = _axes_intersections(l,q)
		z = np.zeros_like(a)
		ga = _grad_ratio(l,q,(a,z))
		gb = _grad_ratio(l,q,(z,b))
		αa = ga[1]/ga.sum(axis=0)
		αb = gb[1]/gb.sum(axis=0)
		return np.minimum(αa,αb),np.maximum(αa,αb),αa<αb

	def μ(self,α):
		""" See the method α_bounds. """
		a = ad.array((1-α,α))
		l = self.linear
		Q = self.quadratic
		δ = lp.det(Q)
		δRef = np.sum(Q**2,axis=(0,1)) # If δ/δRef is small, then degenerate case (two lines)
		R = ad.array([[Q[1,1],-Q[0,1]],[-Q[1,0],Q[0,0]]]) # Adjugate of Q
		Rl = lp.dot_AV(R,l)
		s = 2*δ+lp.dot_VV(l,Rl)
		ε = np.sign(s)
		aRa = lp.dot_VAV(a,R,a)

		# For generic parameters, the result is num/den. However, den = num = 0 is possible.
		# This arises in particular if Q = λ l l^T, in which case the conic consists of two lines.
		# We must also check wether Q is close to zero, in which case the conic degenerates
		# to a line. In both these cases, we approximate the conic piece witha line, and
		# we get μ by using one intersection along an axis : satisfy (1-α,α).(a0,0)/μ(α) = 1.
		# Another possible criterion, more costly but possibly better, would be to directly 
		# check the difference between self.α_bounds() - if they are close, then the conic piece
		# can be well approximated with a line. 
		tol = 5*np.finfo(Q.dtype).eps;
		degen = np.sum(Rl**2,axis=0)<=tol*np.sum(l**2,axis=0)**3 # conic piece degenerates to a line
#		print(np.sum(Rl**2,axis=0)/np.sum(l**2,axis=0)**3,tol)
#		tol = 100*np.finfo(Q.dtype).eps; degen = np.abs(δ)<=δRef*tol # Degenerate quadratic form => two lines
#		print(degen,l,R,Rl)
		sol_degen = (1-α) * _solve2_above(-2,l[0],Q[0,0],0.) # Use intersection along one axis
		
		num = lp.det([a,l])**2+2*aRa
		den = ε*np.sqrt(aRa*s)+lp.dot_VV(a,Rl)
#		print(δ/δRef,np.finfo(Q.dtype).eps)
#		print(self.α_bounds(),α)
#		print(degen,sol_degen,num,den)
		return np.where(degen,sol_degen, num/den)

	def Isotropic_approx(self):
		"""
		Isotropic approximation of the TTI norm.
		Returns : (cmin,cmax). These costs correspond to the interior and exterior approximations.
		Assumption : the linear transformation must be a rotation.
		"""
		αmin,αmax,mix_is_min = self.α_bounds()
		l,q = self.linear,self._q()
		a,b = _axes_intersections(l,q)
		# Costs corresponding to vertical and horizontal propagation for the Dual norm
		c0,c1=np.sqrt(1/a),np.sqrt(1/b)
		pos = (αmin<0.5) & (0.5<αmax) # If pos==True, one extremal speed is not on the axes.
		αmed = np.where(pos,0.5,(αmin+αmax)/2)
		# 0.5 corresponds to the isotropy. (αmin+αmax)/2 is a dummy value in [αmin,αmax]
		ch = np.where(pos,np.sqrt(0.5/self.μ(αmed)),(c0+c1)/2)
		c = np.sort(ad.asarray([c0,ch,c1]),axis=0)
		return 1/c[2],1/c[0] # Take inverses, to account for norm duality

	def Riemann_approx(self,avg=False,**kwargs):
		"""
		Riemannian approximations of the TTI norm, homothetic to each other, and expectedly good.
		Returns : 
		 - Mmin,Mmax. Some interior and exterior approximating Riemannian metrics.
		 - Mavg, if avg=True. A good approximating Riemannian metric, neither interior nor exterior.
		 - kwargs : passed to Riemann.dual (typical : avoid_np_linalg_inv=True, for speed)
		"""
		import time; top = time.time()
		l,q = self.linear,self._q()
		a,b = _axes_intersections(l,q)
		ai,bi = 1/a,1/b
		α = bi/(ai+bi)
		c = 1/(self.μ(α)*(ai+bi))
		cmin,cmax,cavg = np.minimum(1,c),np.maximum(1,c),((1+np.sqrt(c))/2)**2
		diag = (ai,bi) if self.vdim==2 else (ai,ai,bi)
		riem = Riemann.from_diagonal(diag)
		if self.inverse_transformation is not None: 
			riem = riem.inv_transform(self.inverse_transformation)
		m = riem.dual(**kwargs).m # Take inverse, to account for norm duality
		return m/cavg if avg else (m/cmax,m/cmin)

	def Riemann_envelope(self,nmix,gpu=False):
		"""
		Approximation of a TTI norm using an envelope of Riemannian norms.
		- nmix : number of ellipses used for the approximation.
		- gpu : same implementation as on the GPU (changes almost nothing)
		returns
		- riems : a list of nmix Riemannian norms.
		- mix_is_min : wether to take the minimum or the maximum of the Riemannian norms.
		"""
		# Set the interpolation times 
		if isinstance(nmix,int):
			if nmix==1: ts=[0.5]
			else: ts = np.linspace(0,1,nmix)
		else: ts = nmix # Hack to get specific interpolation times, must be sorted

		if gpu:
			l,q = self.linear,self._q()
			ab = _axes_intersections(l,q)
			diags = _diags(l,q,ts,ab)
			mix_is_min = lp.det([diags[0],diags[-1]])>0 if len(diags) else None
		else:
			αmin,αmax,mix_is_min = self.α_bounds()
			diags = [np.array([1-α,α])/self.μ(α) for α in np.linspace(αmin,αmax,nmix)]

		if self.vdim==3: diags = [(a,a,b) for a,b in diags]
		riems = [Riemann.from_diagonal(1./ad.array(diag)) for diag in diags]
		if self.inverse_transformation is not None:
			riems = [riem.inv_transform(self.inverse_transformation) for riem in riems]
		
		return mix_is_min, riems

	def _q(self):
		"""
		Quadratic part, in format compatible with the C routines adapted below
		"""
		return self.quadratic[((0,0,1),(0,1,1))]

# ----- The following code is adapted from agd/Eikonal/HFM_CUDA/cuda/TTI_.h -----
# It computes the approximation of the TTI norm with an envelope of riemannian norms
# The chosen collection of ellipses (very slightly) differs from a uniform sampling  
# within αmin,αmax given by self.α_bounds() 

def _solve2(a,b,c):
	"""
	Returns the two roots of a quadratic equation, a + 2 b t + c t^2.
	The discriminant must be non-negative, but aside from that the equation may be degenerate.
	"""
	sdelta = np.sqrt(b*b-a*c);
	u = -b + sdelta 
	v = -b - sdelta
	b0 = np.abs(c)>np.abs(a)
	xp = ad.cupy_generic.get_array_module(b0)
	b1 = xp.asarray(a!=0) # Needed with cupy
#	with np.errstate(divide='ignore'): # Does not silent much
	return xp.asarray( (np.where(b0,u/c,np.where(b1,a/u,0.)), 
		np.where(b0,v/c,np.where(b1,a/v,np.inf))) )

def _solve2_above(a, b, c, above):
	"""
	Returns the smallest root of the considered quadratic equation above the given threshold.
	Such a root is assumed to exist.
	"""
	r = np.sort(_solve2(a,b,c),axis=0)
	return np.where(r[0]>=above, r[0], r[1])

def _axes_intersections(l,q):
	"""
	Finds the intersections (a,0) and (0,b) of the curve f(x)=0
	with the axes (the closest intersections to the origin).
	"""
	return _solve2_above(-2,l[0],q[0],0.), _solve2_above(-2,l[1],q[2],0.)

def _grad_ratio(l,q,x):
	"""
	Returns g(x) := df(x)/<x,df(x)> where f(x):= C + 2 <l,x> + <qx,x> 
	Note that the curve tangent to the level line of f at x is 
	<y-x,df(x)> ≥ 0, equivalently <y,g(x)> ≥ 1
	"""
	hgrad = ad.array([q[0]*x[0]+q[1]*x[1]+l[0], q[1]*x[0]+q[2]*x[1]+l[1]]) #lp.dot_AV(q,x)+l # df(x)/2
	return hgrad/lp.dot_VV(x,hgrad)

def _diags(l, q, ts, axes_intersections):
	"""
	Samples the curve defined by {x≥0|f(x)=0}, 
	(the connected component closest to the origin)
	where f(x):= -2 + 2 <l,x> + <qx,x>,
	and returns diag(i) := grad f(x)/<x,grad f(x)>.
	"""
	a,b=axes_intersections
	zero = np.zeros_like(a)

	# Change of basis in f, with e0 = {1/2.,1/2.}, e1 = {1/2.,-1/2.}
	L = ad.array([l[0]+l[1], l[0]-l[1]])/2.
	Q = ad.array([q[0]+2*q[1]+q[2], q[0]-q[2], q[0]-2*q[1]+q[2]])/4.

	diags=[]
	for t in ts:
		if   t==0.: x=(a,zero)
		elif t==1.: x=(zero,b)
		else :
			v = (1.-t)*a - t*b
			# Solving f(u e0+ v e_1) = 0 w.r.t u
			u = _solve2_above(-2.+2.*L[1]*v+Q[2]*v*v, L[0]+Q[1]*v, Q[0], np.abs(v))
			# Inverse change of basis
			x = ad.array([u+v, u-v])/2.
		diags.append(_grad_ratio(l,q,x))

	return diags


# ---- Some instances of TTI metrics ----

# See Hooke.py file for reference
TTI.mica = TTI.from_hexagonal(178.,42.4,14.5,54.9,12.2), 2.79
# Stishovite is tetragonal, not hexagonal, but the P-Wave velocity, in the XZ plane,
# is equivalent to an hexagonal model.
TTI.stishovite2 = TTI.from_hexagonal(453,np.nan,203,776,252).extract_xz(), 4.29 



