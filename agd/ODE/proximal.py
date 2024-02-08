# Copyright 2022 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The proximal operator of a (usually convex) function $f$ is defined as 
prox_f(x0,τ) := argmin_x |x-x0|^2/2 + τ*f(x)
which is equivalent to an implicit time step of size τ for the ODE
dx/dt = - grad f(x).

When f is a characteristic function, only taking the values 0 and +infty, 
prox_f is the projection onto the domain of f, independently of the value of τ.

This file provides implementations of a few proximal operators, 
and of the ADMM algorithm. A classical reference for proximal operators is : 
1.Combettes, P. L. & Pesquet, J.-C. Proximal splitting methods in signal processing. 
in Fixed-point algorithms for inverse problems in science and engineering 185–212 
(Fixed-point algorithms for inverse problems in science and engineering, 2011).
"""

import numpy as np
from .. import AutomaticDifferentiation as ad
from scipy import fft
import time

def make_prox_dual(prox):
	"""
	The proximal operator for f^*(x) = sup_y <x,y> - f(x).
	(Moreau formula)
	"""
	return lambda x,τ: x - τ*prox(x/τ,1/τ)

def chambolle_pock(impl_f,impl_gs,τ_f,τ_gs,K,x,y=None,
	KT=None,E_rtol=1e-3,maxiter=1000,
	ρ_overrelax=None,cvx_f=0.,callback=None,verbosity=2):
	"""
	The chambolle_pock primal-dual proximal algorithm, for solving : 
	inf_x sup_y <K*x,y> + f(x) - g^*(y).
	This algorithm is equivalent to the (linearized) ADMM, but provides explicit 
	dual points, duality gap, and has a number of useful variants.
	Inputs : 
	- impl_f : possibilities
	  - a tuple (f,f_star,prox_f) implementing the function f, the 
	Legendre-Fenchel dual f_star, and the proximal operator prox_f. (f and f_star 
	are used to construct the primal and dual energies, which are involved in 
	the stopping criterion).
	  - an implementation of f, in which case E_rtol must provide E_primal_dual

	- impl_gs : similar to impl_f above, but for the function g^*.
	- K : possibilities
	  - a linear operator, called either as K(x) or K*x. 
	  - the string 'Id'

	- x : initial guess for the primal point
	- y (optional, default : K(x)) : intial guess for the dual point
	- KT (optional, default : K.T): transposed linear operator. 

	- E_rtol (optional) : possibilities
		- (positive float) : algorithm stops when (E_primal-E_dual) < E_rtol *abs(E_primal),
		which is checked every 10 iterations.
		- a tuple (callable, positive float) : the callable implements E_primal_dual(x,y) 
		(returns the pair of primal and dual energies). Same stopping criterion as above.
		- (callable) : algorithm stops when E_rtol(niter,x,y) is True

	- maxiter (optional) : maximum number of iterations
	- ρ_overrelax (optional, use value in [0,2], typically 1.8): over-relaxed variant 
	- cvx_f : coercivity constant of f (used in ALG2 variant)
	- callback (optional) : each iteration begins with callback(niter,x,y)
	"""
	if K=='Id': K = KT = lambda x:x 
	if KT is None: KT = K.T
	if not callable(K): # Handle both call and mutliplication syntax for the linear operator
		K_=K; KT_=KT
		K  = lambda x : K_*x
		KT = lambda y : KT_*y
	if y is None: y = K(x)

	if isinstance(E_rtol,tuple): # Directly provide the primal and dual energies
		E_primal_dual,E_rtol = E_rtol; E_primal,E_dual=None,None
		prox_f,prox_gs = impl_f,impl_gs
	else: # Construct the primal and dual energies
		f,fs,prox_f  = impl_f 
		gs,g,prox_gs = impl_gs
		def E_primal(x): return f(x)+g(K(x))
		def E_dual(y):   return -(fs(-KT(y)) + gs(y))
		def E_primal_dual(x,y): return E_primal(x),E_dual(y)
	primal_values=[]
	dual_values=[]
	τs_f=[]
	if callback is None: callback=lambda niter,x,y: None
	if callable(E_rtol): stopping_criterion=E_rtol
	else:
		def stopping_criterion(niter,x,y):
			if niter%10 != 0: return False # Check every 10 iterations
			e_p,e_d = E_primal_dual(x,y)
			primal_values.append(e_p); dual_values.append(e_d)
			return E_rtol>0 and (e_p-e_d)<E_rtol*abs(e_p)

	top = time.time()
	θ=1.; x_,xold,yold=None,None,None;
	for niter in range(maxiter):
		callback(niter,x,y)
		if stopping_criterion(niter,x,y): break
		xold,yold = x,y
		prox_f_arg = x-τ_f*KT(y)
# 		If one introduces a smooth term, then a different stopping criterion is needed.
#-dfsmooth (optionnal) : gradient of an additional smooth term fsmooth(x) in the objective
#		if dfsmooth is not None: prox_f_arg -= τ_f*dfmooth(x) 
		x = prox_f(prox_f_arg,τ_f)

		x_ = 2*x - xold if cvx_f==0 else x+θ*(x-xold)
		y = prox_gs(y+τ_gs*K(x_),τ_gs)
		if ρ_overrelax is not None: 
			x = (1-ρ_overrelax)*xold+ρ_overrelax*x
			y = (1-ρ_overrelax)*yold+ρ_overrelax*y
		if cvx_f>0:
			τs_f.append(τ_f)
			θ = 1/np.sqrt(1+cvx_f*τ_f)
			τ_f  *= θ
			τ_gs /= θ
	else:
		if E_rtol>0 and verbosity>=1: 
			print("Warning : duality gap not reduced to target within iteration budget")
	if verbosity>=2: 
		print(f"Primal-dual solver completed {niter+1} steps in {time.time()-top} seconds")

	primal_values = np.array(primal_values); dual_values = np.array(dual_values)
	return {'x':x,'y':y,'niter':niter+1,
	'primal_values':primal_values,'dual_values':dual_values,
#	'rgap':2*(primal_values-dual_values)/(np.abs(primal_values)+np.abs(dual_values)),
	'ops':{'K':K,'KT':KT,'E_primal':E_primal,'E_dual':E_dual,'E_primal_dual':E_primal_dual},
	'tmp':{'x_':x_,'xold':xold,'yold':yold,'θ':θ,'τ_f':τ_f,'τ_gs':τ_gs,'τs_f':τs_f}} 
	

# ------------------------- Helpers for proximal operators --------------------------

def make_prox_multivar(prox):
	"""
	Defines Prox((x1,...,xn),τ) := prox(x1,...,xn,τ), 
	or  Prox((x1,...,xn),τ) := (prox1(x1,τ),...,proxn(xn,τ)).
	Result is cast to np.array(dtype=object)
	"""
	if isinstance(prox,tuple): 
		return lambda x,τ: np.array( tuple(proxi(xi,τ) for (proxi,xi) in zip(prox,x)), dtype=object)
	return lambda x,τ: np.array( prox(*x,τ), dtype=object)	


def impl_inmult(impl,λ):
	"""
	Implements the Lengendre-Fenchel dual and proximal operator of F(x) := f(λ*x)
	Input : 
	- impl : f,fs,prox_f
	- λ : a scalar
	Output : 
	- F, Fs, prox_F where F(x) = f(λ x)
	"""
	primal0,dual0,prox0 = impl
	iλ = 1/λ; λ2 = λ**2
	def primal(x): return primal0(λ*x)
	def dual(x):   return dual0(iλ*x)
	def prox(x,τ): return iλ*prox0(λ*x,λ2*τ)
	return primal,dual,prox

def impl_exmult(impl,λ):
	"""
	Implements the Lengendre-Fenchel dual and proximal operator of F(x) := λ*f(x)
	Input : 
	- impl : f,fs,prox_f
	- λ : a scalar
	Output : 
	- F, Fs, prox_F where F(x) = λ f(x)
	"""
	primal0,dual0,prox0 = impl
	iλ = 1/λ
	def primal(x): return λ*primal0(x)
	def dual(x):   return λ*dual0(iλ*x)
	def prox(x,τ): return prox0(x,λ*τ)
	return primal,dual,prox


def impl_sub(impl,x0):
	"""
	Implements the Legendre-Fenchel dual and proximal operator of F(x) := F(x-x0)
	Input : 
	- impl : f,fs,prox_f
	- x0 : the shift parameter
	Output : 
	- F, Fs, prox_F where F(x) = f(x-x0)
	"""
	primal0,dual0,prox0 = impl
	def primal(x): return primal0(x-x0)
	def dual(x):   return np.sum(x*x0)+dual0(x)
	def prox(x,τ): return x0+prox0(λ*τ,λ2*x)
	return primal,dual,prox


# def make_prox_addlin(prox,w):
# 	"""The proximal operator for F(x) := f(x) + w.x"""
# 	return lambda x,τ: prox(x-τ*w,τ)

# def make_prox_addquad(prox,a):
# 	"""The proximal operator for F(x) := f(x)+a x^2, where a>=0"""
# 	b = 1./(τ*a+1)
# 	return lambda x,τ: prox(b*x,b*τ)

# def norm_multivar(arrs): 
# 	"""Euclidean norm of an np.array(dtype=object)"""
# 	return np.sqrt(sum(np.sum(arr**2) for arr in arrs))
