# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

r"""
This module implements some basic functionality for solving ODEs derived from a 
Hamiltonian in a manner compatible with automatic differentiation.
(Flow computation, symplectic schemes, etc)

Recall that Hamilton's equations read 
$$
\frac {dq}{dt} = \frac {\partial H}{\partial p},
\quad
\frac {dp}{dt} = - \frac {\partial H}{\partial q}.
$$
"""

import numpy as np
from copy import copy

from .. import AutomaticDifferentiation as ad

def fixedpoint(f,x,tol=1e-9,nitermax=100):
	"""
	Iterates the function f on the data x until a fixed point is found, 
	up to prescribed tolerance, or the maximum number of iterations is reached.
	"""
	norm_infinity = ad.Optimization.norm_infinity
	x_old = x
	for i in range(nitermax):
		x=f(x)
		if norm_infinity(x-x_old)<tol: break
		x_old = x
	return x

damp_None = 0 # Placeholder for the damping factors damp_q and damp_p of class HamiltonianBase
def read_None(H,x): pass # Placeholder for the data read functions read_q and read_p
def incr_None(H,x): pass # Placeholder for the data write functions incr_q and incr_p

class HamiltonianBase:
	"""
	Base class for Hamiltonians.

	Default initialized fields : 
	- impl_solver : Fixed point solver to be used for the implicit time steps
	- preserve_q, preserve_p : Wether updates modify q and p in place
	- damp_q, damp_p : damping factor between scheme iterations
	- read_q, read_p : called before each damping step, for reading q and p
	- incr_q, incr_p : called after each damping step, as a source term for q and p
	"""
	def __init__(self):
		self.impl_solver = fixedpoint 
		self.preserve_q = True; self.preserve_p = True 
		self.Impl2_p_merged = True 
		self.damp_q = damp_None; self.damp_p = damp_None 
		self.read_q = read_None; self.read_p = read_None
		self.incr_q = incr_None; self.incr_p = incr_None
		self.current_iter = None 

	@property
	def damp_q(self): return self._damp_q
	@property
	def damp_p(self): return self._damp_p
	@damp_q.setter
	def damp_q(self,value): self._damp_q = value
	@damp_p.setter
	def damp_p(self,value): self._damp_p = value

	def H(self,q,p):
		"""Evaluates the Hamiltonian, at a given position and impulsion."""
		raise NotImplementedError

	def DqH(q,p):
		"""Differentiates the Hamiltonian, w.r.t. position."""
		raise NotImplementedError

	def DpH(q,p):
		"""Differentiates the Hamiltonian, w.r.t. impulsion."""
		raise NotImplementedError

	def flow(self,q,p):
		"""
		Symplectic gradient of the Hamiltonian.
		"""
		return (self.DpH(q,p),-self.DqH(q,p))

	def flow_cat(self,qp,t=None):
		"""
		Symplectic gradient of the hamiltonian, intended for odeint. 

		Input : 
		 - qp : position q, impulsion p, flattened and concatenated.
		 - t : ignored parameter (compatibility with scipy.integrate.odeint)
		 - shape : how to reshape q and p for calling DpH and DqH, if needed

		Output :
		 - symplectic gradient, concatenated and flattened.
		"""
		d = len(qp)//2
		q = qp[:d]; p = qp[d:]
		if hasattr(self,'shape_free') and self.shape_free is not None:
			q = q.reshape((*self.shape_free,-1))
			p = p.reshape((*self.shape_free,-1))
		return np.concatenate(self.flow(q,p),axis=0).reshape(-1)

	def integrate(self,q,p,scheme,niter=None,dt=None,T=None,path=False):
		"""
		Solves Hamilton's equations by running the scheme niter times.

		Inputs : 
		 - q,p : Initial position and impulsion.
		 - scheme : ODE integration scheme. (string or callable)
		 - niter,dt,T : number of steps, time step, and total time
			  (exactly two among the three must be specified)
		 - path : wether to return the intermediate steps. 
			   (If a positive number, period of intermediate steps to return)
		
		Output : 
		 - q,p if path is False. 
		  Otherwise np.stack([q0,...,qn],axis=-1),np.stack([p0,...,pn],axis=-1),[t0,..tn], 
		  	with n=niter, tn=T.
		"""
		if isinstance(scheme,str):
			schemes = self.nonsymplectic_schemes()
			schemes.update(self.symplectic_schemes())
			scheme = schemes[scheme]

		if (niter is None) + (T is None) + (dt is None) != 1: 
			raise ValueError("Exactly two of niter, dt and T must be specified")
		if T is None:    T  = niter*dt
		elif dt is None: dt = T/niter
		elif niter is None: 
			niter = int(np.ceil(T/dt))
			dt = T/niter # slightly decreased time step

		q,p = copy(q),copy(p)
		if path: Q,P = [copy(q)],[copy(p)]

		for i in range(niter):
			q,p = scheme(q,p,dt)
			if path and not i%path: Q.append(copy(q)); P.append(copy(p))

		if path: return np.stack(Q,axis=-1),np.stack(P,axis=-1),np.linspace(0,T,niter+1)
		else: return q,p

	def nonsymplectic_schemes(self):
		"""
		Standard ODE integration schemes
		"""
		def Euler(q,p,dt):
			dq,dp = self.flow(q,p)
			return q+dt*dq, p+dt*dp

		def RK2(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			return q+dt*dq2, p+dt*dp2

		def RK4(q,p,dt):
			dq1,dp1 = self.flow(q, p)
			dq2,dp2 = self.flow(q+0.5*dt*dq1, p+0.5*dt*dp1)
			dq3,dp3 = self.flow(q+0.5*dt*dq2, p+0.5*dt*dp2)
			dq4,dp4 = self.flow(q+dt*dq3, p+dt*dp3)
			return q+dt*(dq1+2*dq2+2*dq3+dq4)/6., p+dt*(dp1+2*dp2+2*dp3+dp4)/6.

		return {"Euler":Euler,"Runge-Kutta-2":RK2,"Runge-Kutta-4":RK4}

	# -------------- sub-steps of schemes --------------

	def _mk_copy(self,x): 
		"""Returns a copy of variable x. Specialize if needed (for GPU c_contiguity...)."""
		return x.copy()

	def Expl_q(self,q,p,δ):
		"""Explicit time step for the position q."""
		if self.preserve_q: q = self._mk_copy(q)
		q += δ*self.DpH(q,p)
		return q 
	def Expl_p(self,q,p,δ):
		"""Explicit time step for the impulsion p."""
		if self.preserve_p: p = self._mk_copy(p)
		p -= δ*self.DqH(q,p)
		return p 

	def Damp_qp(self,q,p,δ):
		"""
		Optional damping step, interleaved in between Verlet and Ruth4 steps, equivalent to : 
		p *= exp(-δ*damp_p); q *= exp(-δ*damp_q)
		Please set damp_q and damp_p as appropriate.
		"""
		if self._damp_q is not damp_None:
			if self.preserve_q: q = self._mk_copy(q)
			q *= np.exp(-δ*self._damp_q) # Maybe we can avoid recomputing these exponentials
		if self._damp_p is not damp_None:
			if self.preserve_p: p = self._mk_copy(p)
			p *= np.exp(-δ*self._damp_p)
		return q,p

	def Impl_p(self,q,p,δ):
		"""
		Implicit time step for the impulsion p.
		"""
		return self.impl_solver(lambda p_:p-δ*self.DqH(q,p_), p)

	def Impl_q(self,q,p,δ):
		"""
		Implicit time step for the position q.
		"""
		return self.impl_solver(lambda q_:q-δ*self.DqH(q_,p), q)

	def Impl2_p(self,q,p,δ_before,δ_total,δ_after):
		"""
		Merge two implicit time steps for the impulsion p, with a damping step in between.
		"""
		#Default implementation, without optimization (i.e. merging of the Impl_p steps)
		p   = self.Impl_p(q,p,δ_before)
		self.read_q(self,q); self.read_p(self,p) # Read position and momentum before damp
		q,p = self.Damp_qp(q,p,δ_total)
		self.incr_q(self,q); self.incr_p(self,p) # Modify position and momentum after damp
		p   = self.Impl_p(q,p,δ_after)
		return q,p

	# ---------------- Symplectic schemes ---------------

	def Euler_p(self,q,p,δ,niter=1):
		"""niter time steps of the symplectic Euler scheme, starting with impulsion p update."""
		for i in range(niter):
			self.current_iter = i
			p=self.Impl_p(q,p,δ)
			q=self.Expl_q(q,p,δ)
			if i!=niter-1:
				self.read_q(self,q); self.read_p(self,p) 
				self.Damp_qp(q,p,δ) 
				self.incr_q(self,q); self.incr_p(self,p)
		return q,p

	def Euler_q(self,q,p,δ,niter=1):
		"""niter time steps of the symplectic Euler scheme, starting with position q update."""
		for i in range(niter):
			self.current_iter = i
			q=self.Impl_q(q,p,δ)
			p=self.Expl_p(q,p,δ)
			if i!=niter-1:
				self.read_q(self,q); self.read_p(self,p) 
				self.Damp_qp(q,p,δ) 
				self.incr_q(self,q); self.incr_p(self,p)
		return q,p

	def Verlet_p(self,q,p,δ,niter=1):
		"""
		niter time steps of the symplectic Verlet scheme.
		Optional damping steps interleaved.
		"""
		hδ=δ/2
		for i in range(niter):
			self.current_iter = i
			if i==0: p=self.Impl_p(q,p,hδ)
			q=self.Expl_q(q,p,δ)
			if i==niter-1: p=self.Impl_p(q,p,hδ)
			else: q,p=self.Impl2_p(q,p,hδ,δ,hδ)
		return q,p

	def Ruth4_p(self,q,p,δ,niter=1):
		"""
		niter time steps of the Ruth 1983 4th order symplectic scheme, as of Wikipedia.
		Optional damping steps interleaved.
		"""
		t = 2.**(1./3.)
		c1 = 1/(2*(2-t)); c2 = (1-t)*c1; 
		d1 = 2*c1; d2 = -t*d1;
		for i in range(niter):
			self.current_iter = i
			if i==0: p=self.Impl_p(q,p,c1*δ)
			q=self.Expl_q(q,p,d1*δ)
			p=self.Impl_p(q,p,c2*δ)
			q=self.Expl_q(q,p,d2*δ)
			p=self.Impl_p(q,p,c2*δ)
			q=self.Expl_q(q,p,d1*δ)
			if i==niter-1: p=self.Impl_p(q,p,c1*δ) 
			else: q,p=self.Impl2_p(q,p,c1*δ,δ,c1*δ)
		return q,p

	def Sympl_p(self,q,p,δ,niter=1,order=2):
		"""
		niter steps of the Euler_p, Verlet_p or Ruth4_p symplectic scheme, 
		depending on the order parameter.
		"""
		if order==1: return self.Euler_p(q,p,δ,niter)
		if order==2: return self.Verlet_p(q,p,δ,niter)
		if order==4: return self.Ruth4_p( q,p,δ,niter)
		raise ValueError(f"Found {order=}, while expecting 1,2 or 4.")

	def symplectic_schemes(self):
		return {'Euler-p':self.Euler_p,'Euler-q':self.Euler_q,'Verlet-p':self.Verlet_p,
		'Ruth4-p':self.Ruth4_p}

	# ------ Routines for backpropagation ------

	def _Sympl_p(self,q,p,δ,order=2):
		"""
		Returns a callable, which computes any given iteration of Sympl_p, and saves appropriate 
		keypoints for computational and memory efficiency.
		"""
		# TODO : remove since probably useless ? (Only use case is backprop in seismogram)

		# The main objective of this routine is to preserve accuracy when damp_p and damp_q are set.
		# Otherwise, the iterations may be reversed by iterating the symplectic scheme 
		# in negative time, providing a result with similar accuracy. 
		def next(qp): # A single time step, including a preliminary damping
			qp = self.Damp_qp(*qp,δ)
			return self.Sympl_p(*qp,δ,order=order)
		from .backtrack import RecurseRewind
		return RecurseRewind(next,self.Damp_qp(q,p,-δ)) # A single negative damping step should be fine...

	def seismogram(self,*args,qh_ind=None,ph_ind=None,**kwargs):
		"""
		Computes niter time steps of a symplectic scheme, collects the values at given indices along
		the way (the seismogram), and allows to backpropagate the results.
		- args,kwargs : passed to Sympl_p
		"""

		H_fwd = copy(self)
		H_fwd.read_q = H_fwd.read_p = read_None; H_fwd.incr_q = H_fwd.incr_p = incr_None
		qh = []; ph = []; initial=True 
		if qh_ind is not None: H_fwd.read_q = lambda _,q : qh.append(q[*qh_ind].copy())
		if ph_ind is not None: H_fwd.read_p = lambda _,p : ph.append(p[*ph_ind].copy())
		qf,pf = H_fwd.Sympl_p(*args,**kwargs)

		return qf,pf,ad.asarray(qh),ad.asarray(ph)

