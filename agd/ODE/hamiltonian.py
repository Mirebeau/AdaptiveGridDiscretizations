# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import scipy.sparse

from .. import AutomaticDifferentiation as ad
from .. import LinearParallel as lp
from .hamiltonian_base import HamiltonianBase,fixedpoint,damp_None,read_None,incr_None
import numpy as np
from copy import copy
from .backtrack import RecurseRewind


class MetricHamiltonian(HamiltonianBase):
	r"""
	Hamiltonian defined by an interpolated metric, which is dualized and interpolated.
	$$
		H(q,p) = \frac 1 2 F^*_q(p)^2
	$$
	__init__ arguments :
	- metric : dual defines the hamiltonian
	- **kwargs : passed to metric.dual().set_interpolation 
	"""

	def __init__(self,metric,**kwargs):
		super(MetricHamiltonian,self).__init__()
		self._dualmetric = metric.dual()
		self._dualmetric.set_interpolation(**kwargs)

	def H(self,q,p): return self._dualmetric.at(q).norm2(p)
	
	def DqH(self,q,p): 
		q_ad = ad.Dense.identity(constant=q,shape_free=(len(q),))
		return np.reshape(self._dualmetric.at(q_ad).norm2(p).gradient(),np.shape(p))

	def DpH(self,q,p): return self._dualmetric.at(q).gradient2(p)

class GenericHamiltonian(HamiltonianBase):
	r"""
	Hamiltonian defined by a arbitrary function $f$ of two variables, 
	the position $q$ and impulsion $p$, denoted
	$$
		H(q,p)
	$$

	__init__ arguments : 
	- H : the hamiltonian, must take two arguments.
	- shape_free (optional) : shape of position and momentum variables, used for autodiff
	- disassociate_ad (optional) : hide AD information when calling $f$. (Use to avoid 
	conflicts if the definition of $f$ itself involves automatic differentiation.)
	- **kwargs : passed to HamiltonianBase
	"""

	def __init__(self,H, shape_free=None, disassociate_ad=False):
		super(GenericHamiltonian,self).__init__()
		self.H = H
		self.shape_free = shape_free
		self.disassociate_ad = disassociate_ad

	def _identity_ad(self,x,noad=None):
		x_ad = ad.Dense.identity(constant=x,shape_free=self.shape_free) 
		if self.disassociate_ad: 
			x_dis = ad.disassociate(x_ad,shape_free=self.shape_free)
			if noad is None: return x_dis
			else:return (x_dis,ad.disassociate(type(x_ad)(noad),shape_free=self.shape_free)) 
		else: 
			return x_ad if noad is None else (x_ad,noad)

	def _gradient_ad(self,x):
		if self.disassociate_ad: x=ad.associate(x)
		return x.gradient()

	def DqH(self,q,p):
		q_ad,p = self._identity_ad(q,noad=p)
		# If the reshaping fails, then consider setting shape_free.
		return np.reshape(self._gradient_ad(self.H(q_ad,p)), np.shape(q))

	def DpH(self,q,p):
		p_ad,q = self._identity_ad(p,noad=q)
		# If the reshaping fails, then consider setting shape_free.
		return np.reshape(self._gradient_ad(self.H(q,p_ad)), np.shape(p))

# -------------------------------- Separable -------------------------------

class SeparableHamiltonianBase(HamiltonianBase):
	"""
	Base class for separable Hamiltonians, with generic form : H(q,p) = H_Q(q) + H_P(p).
	"""
	def __init__(self):
		super(SeparableHamiltonianBase,self).__init__()

	# Redirect to single argument version of the partial derivatives
	def DqH(self,q,_): return self._DqH(q)
	def DpH(self,_,p): return self._DpH(p)
	def _DqH(q): 
		"""Derivative of the Hamiltonian w.r.t position"""
		raise NotImplementedError
	def _DpH(p): 
		"""Derivative of the Hamiltonian w.r.t impulsion"""
		raise NotImplementedError

	def Impl_p(self,q,p,δ): 
		"""Time step for the impulsion p (implicit=explicit for a separable scheme"""
		return self.Expl_p(q,p,δ)

	def Impl_q(self,q,p,δ):
		"""Time step for the position q (implicit=explicit for a separable scheme)"""
		return self.Expl_q(q,p,δ)

	def Impl2_p(self,q,p,δ_before,δ_total,δ_after):
		"""
		Merge two implicit time steps for the impulsion p, with a damping step in between.
		"""
		# If there is no damping, then the two explicit time steps can be merged
		if (self._damp_p is damp_None and self._damp_q is damp_None and self.Impl2_p_merged 
		and self.read_p is read_None and self.incr_q is incr_None): # Could allow read_p easily
			self.read_q(self,q) # Read position before (non-existent) damp
			p = self.Expl_p(q,p,δ_before+δ_after)
			self.incr_p(self,p) # Modify momentum after (non-existent) damp
			return q,p

		return super().Impl2_p(q,p,δ_before,δ_total,δ_after)


class SeparableHamiltonian(SeparableHamiltonianBase):
	"""
	Separable Hamiltonian defined by a pair of functions, differentiated with AD.
	$$
		H(q,p) = H_Q(q) + H_P(p).
	$$
	__init__ arguments : 
	- Hq,Hp : the two functions $H_Q,H_P$, of a single argument, defining the hamiltonian
	- shape_free (optional) : shape of position and momentum variables, used for autodiff
	"""

	def __init__(self,Hq,Hp,shape_free=None):
		super(SeparableHamiltonian,self).__init__()
		self.Hq = Hq
		self.Hp = Hp
		self.shape_free = shape_free

	def H(self,q,p): return self.Hq(q) + self.Hp(p)

	def _DqH(self,q): 
		q_ad = ad.Dense.identity(constant=q,shape_free=self.shape_free)
		return np.reshape(self.Hq(q_ad).gradient(),np.shape(q))

	def _DpH(self,p): 
		p_ad = ad.Dense.identity(constant=p,shape_free=self.shape_free)
		return np.reshape(self.Hp(p_ad).gradient(),np.shape(p))

# -------------------------------- Quadratic -------------------------------

class QuadraticHamiltonianBase(SeparableHamiltonianBase):
	"""
	Base class for separable quadratic Hamiltonians. 
	Implements the perturbed Hamiltonians which are preserved by the symplectic schemes.
	"""
	def __init__(self,shape_free=None):
		super(QuadraticHamiltonianBase,self).__init__()
		self.shape_free = shape_free

	def flat(self,x):
		"""Flattens the vector x, for e.g. product with a sparse matrix"""
		if self.shape_free is None: return x.reshape(-1)
#		assert x.shape[:len(self.shape_free)] == self.shape_free
		return x.reshape((np.prod(self.shape_free),*x.shape[len(self.shape_free):])) 

	def _dot(self,q,p):
		"""Duality bracket between position and impulsion"""
		return np.sum(q*p, 
			axis=None if self.shape_free is None else tuple(range(len(self.shape_free))))

	def _ABC(self,δ):
		A = self._DpH
		B = self._DqH
		C = lambda q: δ**2 * A(B(q))
		return A,B,C,self._dot

	def HEuler_p(self,q,p,δ):
		"""Modified Hamiltonian, preserved by the symplectic Euler_p scheme"""
		A,B,_,dot = self._ABC(δ)
		Ap = A(p)
		return 0.5*dot(Ap,p) + 0.5*dot(B(q),q-δ*Ap)

	def HVerlet_p(self,q,p,δ):
		"""Modified Hamiltonian, preserved by the Verlet_p symplectic scheme"""
		A,B,C,dot = self._ABC(δ)
		return 0.5*dot(A(p),p) + 0.5*dot(B(q),q-0.25*C(q))

	def HRuth4_p(self,q,p,δ):
		"""Modified Hamiltonian, preserved by the Ruth4_p symplectic scheme"""
		A,B,C,dot = self._ABC(δ)
		Ap = A(p); a1,a2,a3,a4 = - 0.33349609375,-0.16400146484375,0.0319671630859375,0.009191930294036865
		b1,b2,b3,b4,b5 = - 0.33349609375, -0.087890625, 0.06305503845214844,0.0053994059562683105,0.0041955700144171715
		return 0.5*dot(p,Ap+C(a1*Ap+C(a2*Ap+C(a3*Ap)))) + 0.5*dot(B(q),q+C(b1*q+C(b2*q+C(b3*q+C(b4*q+C(b5*q))))))

	def H_p(self,q,p,δ,order):
		"""
		Modified Hamiltonian, preserved by the Euler_p, Verlet_p, or Ruth4_p 
		symplectic scheme, depending on the order parameter. (See method Sympl_p.)
		"""
		if order==1: return self.HEuler_p(q,p,δ)
		if order==2: return self.HVerlet_p(q,p,δ)
		if order==4: return self.HRuth4_p(q,p,δ)
		raise ValueError(f"Found {order=}, while expecting 1,2 or 4.")


	def Impl2_p(self,q,p,δ_before,δ_total,δ_after):
		"""Merge two implicit time steps for the impulsion p, with a damping step in between."""

		# We are computing α p - δ_after B β q - δ_before α B q, 
		# Where α = exp(-δ_total damp_p) and β = exp(-δ_total damp_q) 

		if (self._damp_q is damp_None and self._damp_p is not damp_None and self.Impl2_p_merged # β=1
		and self.incr_q is incr_None and self.read_p is read_None): 
			# Factorization : α p - (δ_after + δ_before α) B q
			self.read_q(self,q) # Read position before damp
			dp = self._DqH(q)
			α = np.exp(-δ_total*self._damp_p)
			p = α*p - (δ_after + δ_before*α) * dp
			self.incr_p(self,p)
			return q,p

		if (self._damp_p is damp_None and self._damp_q is not damp_None and self.Impl2_p_merged # α=1
		and self.read_p is read_None and self.incr_p is incr_None): 
			# Factorization : p - B (δ_after β + δ_before) q
			self.read_q(self,q)
			β = np.exp(-δ_total*self._damp_q)
			qnew = q*β
			self.incr_q(self,qnew)
			p = self.Expl_p(δ_after*qnew+δ_before*q,p,1) # Using Expl for reverse AD
			self.incr_p(self,p) 
			return qnew,p

		# read and incr : We need to see what is the typical usage, for now likely too restrictive.

		return super().Impl2_p(q,p,δ_before,δ_total,δ_after)

	def seismogram_with_backprop(self,q,p,δ,niter,order=2,qh_ind=None,ph_ind=None,**kwargs):
		"""
		Computes niter time steps of a symplectic scheme, collects the values at given indices along
		the way (the seismogram), and allows to backpropagate the results.
		
		Inputs : 
		- qh_ind,ph_ind : indices at which to collect the values of q and p, 
			in the flattened arrays. IMPORTANT : no duplicate values in either qh_ind or ph_ind.
		- kwargs : passed to Sympl_p

		Outputs : 
		- (qf,pf) : final values of q and p
		- (qh,ph) : history of q and p. The prescribed indices are extracted along the way, and 
			concatenated into a "simogram". Last iteration is not included, use qf,pf.
		- backprop : callable, which given (qf_coef,pf_coef) and (qh_coef,ph_coef), the gradient of
			some objective(s) functional w.r.t the outputs, backpropagates the results to obtain the
			gradients w.r.t the Hamiltonian parameters. 
			qf_coef.shape == (*qf.shape,size_ad), and likewise pf,qh,ph.
		"""

		H_fwd = copy(self); H_rev = copy(self)
		H_fwd.read_q = read_None; H_fwd.read_p = read_None; H_fwd.incr_q = incr_None; H_fwd.incr_p = incr_None
		H_rev.read_q = read_None; H_rev.read_p = read_None; H_rev.incr_q = incr_None; H_rev.incr_p = incr_None

		qh = []; ph = []; initial=True 
#		if qh_ind is not None: H_fwd.read_q = lambda _,q : qh.append(q.reshape(-1)[qh_ind])
#		if ph_ind is not None: H_fwd.read_p = lambda _,p : ph.append(p.reshape(-1)[ph_ind])

		# We construct a callable, which will extract the desired seismogram, and also collect
		# keypoint values of q and p.
		# The main objective of this routine is to preserve accuracy when damp_p and damp_q are set.
		# Otherwise, the iterations may be reversed by iterating the symplectic scheme 
		# in negative time, providing a result with similar accuracy. 

		# TODO : Define next(qp,niter) and modify RecurseRewind accordingly, so as to take advantage
		# of Impl2_p merged steps (at best, expecting a factor 2 computational cost reduction)
		def next(qp): # A single time step, including a preliminary damping
			q,p = H_fwd.Damp_qp(*qp,δ)
			q,p = H_fwd.Sympl_p(q,p,δ,order=order,**kwargs)
			if initial and qh_ind is not None: qh.append(q[*qh_ind].copy())
				#qh.append(q.reshape(-1)[qh_ind].copy())
			if initial and ph_ind is not None: ph.append(p[*ph_ind].copy())
				#ph.append(p.reshape(-1)[ph_ind].copy())
			return q,p
		# A single negative damping step should be fine...
		qph_eval = RecurseRewind(next,self.Damp_qp(q,p,-δ))

		# Perform forward propagation
		qf,pf = qph_eval(niter)
		# We drop the last element, for consistency with backprop. Recover it as qf[*q_ind] 
		qh = ad.array(qh[:-1]); ph = ad.array(ph[:-1]) 

		# Remove seismogram extraction
		initial = False; # H_fwd.read_q = read_None; H_fwd.read_p = read_None

		def backprop(qf_grad=None,pf_grad=None,qh_grad=None,ph_grad=None,
			check_val=False,check_ind=True):
			"""
			- qf_grad : gradient of objective functional(s) w.r.t qf, with shape (*qf.shape,size_ad)
			- pf_grad, qh_grad, ph_grad : gradients w.r.t ph, qh, ph, similar to qf_grad
			- check_val : check that the back propagation reconstructs values correctly
			- check_ind : check that the seismogram indices do not contain duplicates
			"""
			# Data checks
			for ind in (qh_ind,ph_ind):
				assert not check_ind or np.unique(np.ravel_multi_index(ind,qf.shape)).size==ind[0].size
#				assert np.unique(qh_ind).size==qh_ind.size and np.unique(ph_ind).size==ph_ind.size
			if ad.is_ad(qf) or ad.is_ad(pf) or ad.is_ad(qh) or ad.is_ad(ph): 
				raise ValueError("Please choose between forward and reverse autodiff")
			size_ad = max(x.size//y.size for x,y in 
				((qf_grad,qf),(pf_grad,pf),(qh_grad,qh),(ph_grad,ph)) if x is not None and x.size>0)
			for x_name,x_grad,x in (("qf",qf_grad,qf),("pf",pf_grad,pf),("qh",qh_grad,qh),("ph",ph_grad,ph)): 
				if x_grad is not None and x_grad.shape!=(*x.shape,size_ad):
					raise ValueError(f"Expecting shape {(*x.shape,size_ad)} for field {x_name}_grad, but found {x_grad.shape}")

			# Insert the gradients in the backpropagation
			if qf_grad is None: qf_grad = np.zeros_like(qf,shape=(*qf.shape,size_ad))
			if pf_grad is None: pf_grad = np.zeros_like(pf,shape=(*pf.shape,size_ad))
			qf_rev = ad.Dense.denseAD(qf, pf_grad) # Signs and exchange
			pf_rev = ad.Dense.denseAD(pf,-qf_grad) # account for symplectic formalism

			def incr_q(H,q):
				rev_iter = niter-1-H.current_iter
				q.coef[*ph_ind] += ph_grad[rev_iter-1] 
				assert not check_val or np.allclose(q.value * np.exp(-δ*(H._damp_p+H._damp_q)), qph_eval(rev_iter)[0])
				q.value = qph_eval(rev_iter)[0]
			def incr_p(H,p):
				rev_iter = niter-1-H.current_iter
				p.coef[*qh_ind] -= qh_grad[rev_iter-1] 
				assert not check_val or np.allclose(p.value * np.exp(-δ*(H._damp_p+H._damp_q)), qph_eval(rev_iter)[1])
				p.value = qph_eval(rev_iter)[1]

			if qh_ind is not None: H_rev.incr_p = incr_p
			if ph_ind is not None: H_rev.incr_q = incr_q

			# We do not reset the reverse AD accumulators, in case the user wants to accumulate 
			# more data (e.g. seismograms with varying initial impulsions and frequencies) 
			# H_rev.rev_reset() 

			# Setup for time-reversed propagation
			H_rev.damp_q,H_rev.damp_p = - H_fwd.damp_p,- H_fwd.damp_q
			q0_rev,p0_rev = H_rev.Sympl_p(qf_rev,pf_rev,-δ,niter,order,**kwargs)

			return -p0_rev.coef,q0_rev.coef

		return qf,pf,ad.asarray(qh),ad.asarray(ph),backprop

class QuadraticHamiltonian(QuadraticHamiltonianBase):
	r"""
	Quadratic Hamiltonian, defined by a pair of linear operators.
	(Expected to be symmetric semi-definite.)
	$$
		H(q,p) = \frac 1 2 (< q, M_Q q > + < p, M_P p >).
	$$

	__init__ arguments : 
	- Mq,Mp : positive semi-definite matrices $M_Q,M_P$, typically given in sparse form.
	 Alternatively, define Mq,Mp as functions, and use the set_spmat
	 to automatically generate the sparse matrices using automatic differentiation.
	"""

	def __init__(self,Mq,Mp,**kwargs):
		super(QuadraticHamiltonian,self).__init__(**kwargs)
		self.Mq = Mq
		self.Mp = Mp
		 # dMp, dMq are for reverse autodiff, see set_spmat
		self.dMp_has = False
		self.dMq_has = False
	
	def H(self,q,p):
		A,B,C,dot = self._ABC(1) # δ is irrelevant here
		return 0.5*dot(A(p),p) + 0.5*dot(B(q),q)

	def _D_H(self,x,M,dM__has):
		if ad.is_ad(x) and dM__has: raise ValueError("Cannot accumulate reverse AD without dt")
		return np.reshape(ad.apply_linear_mapping(M,self.flat(x)),np.shape(x))

	def _DqH(self,q): return self._D_H(q,self.Mq,self.dMq_has)
	def _DpH(self,p): return self._D_H(p,self.Mp,self.dMp_has)

	def rev_reset(self):
		if self.dMq_has: self.dMq_acc[:]=0
		if self.dMp_has: self.dMp_acc[:]=0

	def Expl_q(self,q,p,δ):
		"""Explicit time step for the position q."""
		if self.preserve_q: q = self._mk_copy(q)
		q += δ*self._D_H(p,self.Mp,False)
		if ad.is_ad(p) and self.dMp_has: self.dMp_acc += δ*self.dMp_op(p)
		return q 

	def Expl_p(self,q,p,δ):
		"""Explicit time step for the impulsion p."""
		if self.preserve_p: p = self._mk_copy(p)
		p -= δ*self._D_H(q,self.Mq,False)
		if ad.is_ad(q) and self.dMq_has: self.dMq_acc -= δ*self.dMq_op(q)
		return p 

	def set_spmat(self,x,rev_ad=(None,None),**kwargs):
		"""
		Replaces Mq,Mp with suitable sparse matrices, generated by spmat,
		if they are callables.

		- x : Correctly shaped input for calling Mq,Mp.
		- rev_ad (optional) : where to accumulate reverse autodiff.
			See Eikonal.HFM_CUDA.AnisotropicWave.AcousticHamiltonian_Sparse for an example
		"""
		if self.shape_free is None: self.shape_free = x.shape
		else: assert self.shape_free == x.shape
		spmat = ad.Sparse2.hessian_operator
		self.dMq_has, self.dMp_has = [y is not None for y in rev_ad]
		if callable(self.Mq): self.Mq = spmat(self.Mq,x,**kwargs,rev_ad=self.dMq_has)
		if callable(self.Mp): self.Mp = spmat(self.Mp,x,**kwargs,rev_ad=self.dMp_has)

		if self.dMq_has: self.Mq,self.dMq_op,self.dMq_acc = *self.Mq,rev_ad[0]
		if self.dMp_has: self.Mp,self.dMp_op,self.dMp_acc = *self.Mp,rev_ad[1]
