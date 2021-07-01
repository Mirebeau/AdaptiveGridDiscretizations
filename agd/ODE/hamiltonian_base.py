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

class HamiltonianBase:
	"""
	Base class for Hamiltonians.
	
	__init__ arguments : 
	- shape_free (optional) : shape of the position and momentum variables
	- inv_inner (optional): inverse inner product, used for gradient normalization
	"""
	def __init__(self,is_separable=False,inv_inner=None,shape_free=None,vdim=-1):
		self.is_separable = is_separable
		self.inv_inner = inv_inner
		self.shape_free = shape_free
		if vdim!=-1: self.vdim=vdim

	def _igrad(self,grad):
		"""
		Applies the inverse inner product to a gradient, and reshapes it,
		yielding the 'intrinsic' gradient.
		"""
		if self.inv_inner is not None:  grad = ad.apply_linear_mapping(self.inv_inner,grad)
		return grad.reshape(self.shape_free+grad.shape[1:])

	@property
	def vdim(self):
		"""
		Dimension of space of positions. 
		(Also equals the dimension of the space of impulsions)
		Set to None for a Hamiltonian acting on scalars.
		"""
		if shape_free is None: raise ValueError("Unspecified space dimensions")
		elif len(self.shape_free)==0: return None
		elif len(self.shape_free)==1: return self.shape_free[0]
		else: raise ValueError(f"Hamiltonian acts on tensors of shape {shape_free}")

	@vdim.setter
	def vdim(self,vdim):
		if vdim is None: self.shape_free = tuple()
		else: self.shape_free = (vdim,)	

	@property
	def size_free(self): return np.prod(self.shape_free,dtype=int)
	@property
	def ndim_free(self): return len(self.shape_free)

	def _checkdim(self,x):
		"""
		Checks the dimension of the input variable agains shape_free.
		(Sets shape_free if it is undefined, assuming depth one of x)
		"""
		if self.shape_free is None: self.vdim = x.shape[0]
		assert self.shape_free==tuple() or x.shape[:self.ndim_free]==self.shape_free

	def H(self,q,p):
		"""Evaluates the Hamiltonian, at a given position and impulsion."""
		self._checkdim(q); self._checkdim(p) 
		return self._H(q,p)
	def DqH(self,q,p):
		"""Differentiates the Hamiltonian, w.r.t. position."""
		self._checkdim(q); self._checkdim(p)
		return self._igrad(self._DqH(q,p))
	def DpH(self,q,p):
		"""Differentiates the Hamiltonian, w.r.t. impulsion."""
		self._checkdim(q); self._checkdim(p)
		return self._igrad(self._DpH(q,p))

	def flow(self,q,p):
		"""
		Symplectic gradient of the Hamiltonian.
		"""
		return (self.DpH(q,p),-self.DqH(q,p))

	def flow_cat(self,qp,t=None):
		"""
		Symplectic gradient of the hamiltonian, intended for odeint. 

		Input : 
		 - qp : position q, impulsion p, concatenated and flattened.
		 - t : ignored parameter (compatibility with scipy.integrate.odeint)

		Output :
		 - symplectic gradient, concatenated and flattened.
		"""
		d = len(qp)//2
		q = qp[:d]
		p = qp[d:]
		if self.shape_free is not None:
			size_free = np.prod(self.shape_free,dtype=int)
			if size_free==d:
				q=q.reshape(self.shape_free)
				p=p.reshape(self.shape_free)
			else:
				size_bound = d//size_free
				q=q.reshape(self.shape_free+(size_bound,))
				p=p.reshape(self.shape_free+(size_bound,))
		dq,dp = self.flow(q,p)
		return np.concatenate((dq,dp),axis=0).flatten()

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
		  Otherwise [q0,...,qn],[p0,...,pn],[t0,..tn], with n=niter, tn=T.
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

		if path: 
			ndim_free = len(self.shape_free)
			return (np.stack(Q,axis=ndim_free),
				    np.stack(P,axis=ndim_free),
				    np.linspace(0,T,niter+1))
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

	def incomplete_schemes(self,solver=None):
		"""
		Incomplete schemes, updating only position or impulsion. 
		Inputs : 
			- solver (optional). Numerical solver used for the implicit steps.
			Defaults to a basic fixed point solver: "fixedpoint" in the same package.
		"""
		def Expl_q(q,p,dt):
			return q + dt*self.DpH(q,p)

		def Expl_p(q,p,dt):
			return p - dt*self.DqH(q,p)

		if self.is_separable:
			return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Expl_q,"Implicit-p":Expl_p}

		if solver is None:
			solver=fixedpoint

		def Impl_q(q,p,dt):
			def f(q_): return q+dt*self.DpH(q_,p)
			return solver(f,q)
		def Impl_p(q,p,dt):
			def f(p_): return p-dt*self.DqH(q,p_)
			return solver(f,p)
		
		return {"Explicit-q":Expl_q,"Explicit-p":Expl_p,
			"Implicit-q":Impl_q,"Implicit-p":Impl_p}

	def symplectic_schemes(self,**kwargs):
		"""
		Symplectic schemes, alternating updates of position and impulsion.
		The first updated variable is indicated with a suffix.

		In the non-separable cases, some of the updates are implicit, otherwise
		they are explicit.
		Inputs : 
			- kwargs. Passed to self.incomplete_schemes
		"""
		incomp = self.incomplete_schemes(**kwargs)
		if self.is_separable:
			Expl_q,Expl_p = tuple(incomp[s] for s in 
			("Explicit-q","Explicit-p"))

			def Euler_p(q,p,dt):
				p = Expl_p(q,p,dt)
				q = Expl_q(q,p,dt)
				return q,p
				
			def Euler_q(q,p,dt):
				q = Expl_q(q,p,dt)
				p = Expl_p(q,p,dt)
				return q,p 
				
			def Verlet_p(q,p,dt):
				p=Expl_p(q,p,dt/2)
				q=Expl_q(q,p,dt)
				p=Expl_p(q,p,dt/2)
				return q,p
				
			def Verlet_q(q,p,dt):
				q=Expl_q(q,p,dt/2)
				p=Expl_p(q,p,dt)
				q=Expl_q(q,p,dt/2)
				return q,p

			def Ruth3_p(q,p,dt):
				"""Ronald Ruth 1983, as of Wikipedia"""
				c1,c2,c2 = 1., -2./3., 2./3.
				d1,d2,d3 = -1./24, 3./4, 7./24
				p=Expl_p(q,p,c1*dt)
				q=Expl_q(q,p,d1*dt)
				p=Expl_p(q,p,c2*dt)
				q=Expl_q(q,p,d2*dt)
				p=Expl_p(q,p,c3*dt)
				q=Expl_q(q,p,d3*dt)
				return q,p

			def Ruth4_p(q,p,dt):
				"""Ronald Ruth 1983, as of Wikipedia"""
				t = 2.**(1./3.)
				c1 = 1/(2*(2-t)); c2 = (1-t)*c1; c3,c4 = c2,c1
				d1 = 2*c1; d2 = -t/d1; d3,d4 = d1,0
				p=Expl_p(q,p,c1*dt)
				q=Expl_q(q,p,d1*dt)
				p=Expl_p(q,p,c2*dt)
				q=Expl_q(q,p,d2*dt)
				p=Expl_p(q,p,c3*dt)
				q=Expl_q(q,p,d3*dt)
				p=Expl_p(q,p,c4*dt) 
				#q=Expl_q(q,p,d4*dt)
				return q,p


			return {"Euler-p":Euler_p,"Euler-q":Euler_q,
			"Verlet-p":Verlet_p,"Verlet-q":Verlet_q,
			"Ruth3-p":Ruth3_p,"Ruth4-p":Ruth4_p}


		else: # Non separable case
			Expl_q,Expl_p,Impl_q,Impl_p = tuple(incomp[s] for s in 
				("Explicit-q","Explicit-p","Implicit-q","Implicit-p"))

			def Euler_p(q,p,dt):
				p=Impl_p(q,p,dt)
				q=Expl_q(q,p,dt)
				return q,p

			def Euler_q(q,p,dt):
				q=Impl_q(q,p,dt)
				p=Expl_p(q,p,dt)
				return q,p

			def Verlet_p(q,p,dt):
				p=Impl_p(q,p,dt/2.)
				q=Expl_q(q,p,dt/2.)
				q=Impl_q(q,p,dt/2.)
				p=Expl_p(q,p,dt/2.)
				return q,p

			def Verlet_q(q,p,dt):
				q=Impl_q(q,p,dt/2.)
				p=Expl_p(q,p,dt/2.)
				p=Impl_p(q,p,dt/2.)
				q=Expl_q(q,p,dt/2.)
				return q,p
				
			return {"Euler-p":Euler_p,"Euler-q":Euler_q,
			"Verlet-p":Verlet_p,"Verlet-q":Verlet_q}


