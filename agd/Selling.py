# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

r"""
This file implements Selling's algorithm in dimension $d \in \\{2,3\\}$, which decomposes a 
symmetric positive definite matrix $D$, of dimension $d\leq 3$, in the form 
$$
	D = \sum_{0\leq i < I} a_i e_i e_i^\top,
$$
where $a_i \geq 0$ and $e_i\in Z^d$ is a vector with integer coordinates, 
and where $I = d(d+2)/2$.

Selling's decomposition is a central tool in the design of adaptive discretization schemes
for anisotropic partial differential equations, on Cartesian grids.
"""

import numpy as np
from itertools import cycle
from .LinearParallel import dot_VV, dot_AV, perp, cross
from . import AutomaticDifferentiation as ad
cps = ad.cupy_support
samesize_int_t = ad.cupy_generic.samesize_int_t

iterMax2 = 100 
iterMax3 = 100 

# -------- Dimension based dispatch -----

def ObtuseSuperbase(m,sb=None):
	"""
	Input : 
	- m : symmetric positive definite matrix, defined as an
	 array of shape $(d,d, n_1,...,n_k)$.
	- sb (optional) : initial guess for the obtuse superbase.

	Ouput : an m-obtuse superbase 
	"""
	dim = len(m)
	if sb is None: sb = CanonicalSuperbase(m)
	if   dim==1: return _ObtuseSuperbase1(m,sb)	
	elif dim==2: return _ObtuseSuperbase2(m,sb)
	elif dim==3: return _ObtuseSuperbase3(m,sb)
	else: raise ValueError("Selling's decomposition only applies in dimension <=3.") 	

def Decomposition(m,sb=None):
	r"""
	Use Selling's algorithm to decompose a tensor

	Input : 
	- m : symmetric positive definite matrix, defined as an
	 array of shape $(d,d, n_1,...,n_k)$ where $d\leq 3$.
	- sb (optional) : superbase to use for the decomposition,
	 array of shape $(d,d+1, n_1,...,n_k)$.
	Output : the coefficients and offsets of the decomposition.
	"""
	if ad.cupy_generic.from_cupy(m):
		from . import Eikonal
		return Eikonal.VoronoiDecomposition(m,mode='gpu') # Faster and safer

	dim = len(m)
	if sb is None: sb = ObtuseSuperbase(m,sb)
	if   dim==1: return _Decomposition1(m,sb)	
	elif dim==2: return _Decomposition2(m,sb)
	elif dim==3: return _Decomposition3(m,sb)
	else: raise ValueError("Selling's decomposition only applies in dimension <=3.") 


def GatherByOffset(T,Coefs,Offsets):
	"""
	Get the coefficient of each offset.
	This function is essentially used to make nice plots of how the superbase coefficients
	and offsets vary as the decomposed tensor varies. 
	"""
	Coefs,Offsets = map(ad.cupy_generic.cupy_get,(Coefs,Offsets))
	TimeCoef = {};
	for (i,t) in enumerate(T):
		coefs = Coefs[:,i]
		offsets = Offsets[:,:,i]
		for (j,c) in enumerate(coefs):
			offset = tuple(offsets[:,j].astype(int))
			offset_m = tuple(-offsets[:,j].astype(int))
			if offset<offset_m:
				offset=offset_m
			if offset in TimeCoef:
				TimeCoef[offset][0].append(t)
				TimeCoef[offset][1].append(c)
			else:
				TimeCoef[offset] = ([t],[c])
	return TimeCoef


def CanonicalSuperbase(m):
	"""
	Returns a superbase with the same dimensions and array type as m.

	Output : 
	 - m : array of shape $(d,d, n_1,...,n_k)$
	"""
	d=len(m); assert m.shape[1]==d
	shape=m.shape[2:]

	sb=cps.zeros_like(m,shape=(d,d+1,*shape))
	sb[:,0]=-1
	for i in range(d):
		sb[i,i+1]=1
	return sb

def SuperbasesForConditioning(cond,dim=2):
	"""
	Returns a family of superbases. 
	For any positive definite matrix $M$ with condition number below the given bound,
	one of these superbases will be $M$-obtuse.
	(Condition number is the ratio of the largest to the smallest eigenvalue.)

	Input : 
	 - cond (scalar) : the bound on the condition number.
	"""
	if   dim==1: return _SuperbasesForConditioning1(cond)
	elif dim==2: return _SuperbasesForConditioning2(cond)
	elif dim==3: return _SuperbasesForConditioning3(cond)
	else: raise ValueError("Selling's decomposition only applies in dimension <=3.") 


# ------- One dimensional variant (trivial) ------

def _ObtuseSuperbase1(m,sb=None):
	return CanonicalSuperbase(m)

def _Decomposition1(m,sb):
	shape = m.shape[2:]
	offsets = cps.ones_like(m,shape=(1,1,*shape),dtype=samesize_int_t(m.dtype))
	coefs = m.reshape((1,*shape))
	return coefs,offsets

def _SuperbasesForConditioning1(cond):
	sb = CanonicalSuperbase(np.eye(1))
	return sb.reshape(sb.shape+(1,))

# ------- Two dimensional variant ------

# We do everyone in parallel, without selection or early abort
def _ObtuseSuperbase2(m,sb):
	"""
		Use Selling's algorithm to compute an obtuse superbase.

		input : symmetric positive definite matrix m, dim=2
		input/output : superbase b (must be valid at startup)
		
		module variable : iterMax2, max number of iterations

		output : wether the algorithm succeeded
	"""
	iterReduced = 0
	for iter,(i,j,k) in zip(range(iterMax2), cycle([(0,1,2),(1,2,0),(2,0,1)]) ):
		# Test for a positive angle, and modify superbase if necessary
		acute = dot_VV(sb[:,i],dot_AV(m,sb[:,j])) > 0
#		print(f"nacute : {np.sum(acute)}, {np.max(dot_VV(sb[:,i],dot_AV(m,sb[:,j])))}")
		if np.any(acute):
			try:
				sb[:,k,acute] = sb[:,i,acute]-sb[:,j,acute]
				sb[:,i,acute] = -sb[:,i,acute]
			except IndexError: # Some cupy versions require simpler indexing
				# Works with numpy, but often fails on cupy at the time of testing
				# -> Bypass using Eikonal.VoronoiDecomposition
				sb[:,k][:,acute] = sb[:,i][:,acute]-sb[:,j][:,acute]
				sb[:,i][:,acute] = -sb[:,i][:,acute]
			iterReduced=0
		elif iterReduced<3: iterReduced+=1
		else: return sb

	raise ValueError(f"Selling's algorithm did not terminate in iterMax2={iterMax2} iterations")
	
# Produce the matrix decomposition
def _Decomposition2(m,sb):
	"""
		Use Selling's algorithm to decompose a tensor

		input : symmetric positive definite tensor 
		output : coefficients, offsets
	"""
	shape = m.shape[2:]
	coef  = cps.zeros_like(m,shape=(3,*shape))
	for (i,j,k) in [(0,1,2),(1,2,0),(2,0,1)]:
		coef[i] = -dot_VV(sb[:,j], dot_AV(m, sb[:,k]) )
	
	return coef,perp(sb).astype(samesize_int_t(coef.dtype))

def _SuperbasesForConditioning2(cond):
	"""
	Implementation is based on exploring the Stern-Brocot tree, 
	with a stopping criterion based on the angle between consecutive vectors.
	"""

	mu = np.sqrt(cond)
	theta = np.pi/2. - np.arccos( 2/(mu+1./mu))

	u=np.array( (1,0) )
	l = [np.array( (-1,0) ),np.array( (0,1) )]
	m = []
	superbases =[]

	def angle(u,v): return np.arctan2(u[0]*v[1]-u[1]*v[0], u[0]*v[0]+u[1]*v[1])

	while l:
		v=l[-1]
		if angle(u,v)<=theta:
			m.append(u)
			u=v
			l.pop()
		else:
			l.append(u+v)
			superbases.append((u,v,-u-v))


	return np.array(superbases).transpose((2,1,0))

# ------- Three dimensional variant -------

# We do everyone in parallel, without selection or early abort
def _ObtuseSuperbase3(m,sb):
	"""
		Use Selling's algorithm to compute an obtuse superbase.

		input : symmetric positive definite matrix m, dim=3
		input/output : superbase b (must be valid at startup)
		
		module variable : iterMax3, max number of iterations

		output : wether the algorithm succeeded
	"""
	iterReduced = 0
	sigma = cycle([(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)])
	for iter,(i,j,k,l) in zip(range(iterMax3), sigma):		
		# Test for a positive angle, and modify superbase if necessary
		acute = dot_VV(sb[:,i],dot_AV(m,sb[:,j])) > 0
		if np.any(acute):
			try:
				sb[:,k,acute] += sb[:,i,acute]
				sb[:,l,acute] += sb[:,i,acute]
				sb[:,i,acute] = -sb[:,i,acute]
			except IndexError: # Some cupy versions require simpler indexing
				sb[:,k][:,acute] += sb[:,i][:,acute]
				sb[:,l][:,acute] += sb[:,i][:,acute]
				sb[:,i][:,acute] = -sb[:,i][:,acute]
			iterReduced=0
		elif iterReduced<6: iterReduced+=1
		else: return sb
	raise ValueError(f"Selling's algorithm did not terminate in iterMax3={iterMax3} iterations")
	
def _Decomposition3(m,sb):
	"""
		Use Selling's algorithm to decompose a tensor

		input : symmetric positive definite tensor, d=3
		output : coefficients, offsets
	"""
	shape = m.shape[2:]
	
	coef   = cps.zeros_like(m,shape=(6,*shape))
	offset = cps.zeros_like(m,shape=(3,6,*shape))
	for iter,(i,j,k,l) in enumerate(
		[(0,1,2,3),(0,2,1,3),(0,3,1,2),(1,2,0,3),(1,3,0,2),(2,3,0,1)]):
		coef[iter] = -dot_VV(sb[:,i], dot_AV(m, sb[:,j]) )
		offset[:,iter] = cross(sb[:,k], sb[:,l])
		
	return coef,offset.astype(samesize_int_t(coef.dtype))

def _SuperbasesForConditioning3(cond):
	raise ValueError("Sorry, _SuperbasesForConditioning3 is not implemented yet")