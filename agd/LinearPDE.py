# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module is DEPRECATED, but kept for compatibility.
"""

import numpy as np
import scipy.sparse as sp

from . import Selling
from . import LinearParallel as LP
from .AutomaticDifferentiation.cupy_generic import get_array_module

def OperatorMatrix(diff,omega=None,mult=None, \
		gridScale=1,
		boundaryConditions='Periodic', 
		divergenceForm=False,
		intrinsicDrift=False):
	r"""
	Constructs a linear operator sparse matrix, given as input 
	- an array `diff` of symmetric positive definite matrices, 
		with shape $(d,d,n_1,...,n_d)$ where $d$ is the domain dimension.
	- an array `omega` of vectors (optionnal), with shape $(d,n_1,...,n_d)$.
	- an array of scalars (optionnal), with shape $(n_1,...,n_d)$.

	additional parameters
	- a grid scale 
	- boundary conditions, possibly axis by axis 
		('Periodic', 'Reflected', 'Neumann', 'Dirichlet') 
	- divergence form or not

	The discretized operator is
	$$
	 {-} \mathrm{div}(D \nabla u) + < \\omega, \nabla u> + mult*u,
	$$
	denoting $D:=$`diff` and $\\omega:=$`omega`.

	Replace the first term with $\mathrm{Tr}(D \nabla^2 u)$ in the 
	non-divergence form case.
	
	Returns : a list of triplets, for building a coo matrix
	"""
	# ----- Get the domain shape -----
	bounds = diff.shape[2:]
	dim = len(bounds)
	xp = get_array_module(diff)
	if isinstance(boundaryConditions,str):
		boundaryConditions = np.full( (dim,2), boundaryConditions)
	elif len(boundaryConditions)!=dim:
		raise ValueError("""OperatorMatrix error : 
		inconsistent boundary conditions""")
	
	if diff.shape[:2]!=(dim,dim):
		raise ValueError("OperatorMatrix error : inconsistent matrix dimensions")

	# -------- Decompose the tensors --------
	coef,offset = Selling.Decomposition(diff)
	nCoef = coef.shape[0]
	
	# ------ Check bounds or apply periodic boundary conditions -------
	grid = np.mgrid[tuple(slice(0,n) for n in bounds)]
	grid = grid[:,None,...]
	grid = xp.asarray(grid,dtype=offset.dtype)

	neighPos = grid + offset
	neighNeg = grid - offset
	neumannPos = np.full_like(coef,False,dtype=bool)
	neumannNeg = np.full_like(coef,False,dtype=bool)
	dirichletPos = np.full_like(coef,False,dtype=bool)
	dirichletNeg = np.full_like(coef,False,dtype=bool)
	
	for neigh_,neumann,dirichlet in zip( (neighPos,neighNeg), (neumannPos,neumannNeg), (dirichletPos,dirichletNeg) ): 
		for neigh,cond_,bound in zip(neigh_,boundaryConditions,bounds): # Component by component
			for out,cond in zip( (neigh<0,neigh>=bound), cond_):
				if cond=='Periodic':
   				 	neigh[out] %= bound 
				elif cond=='Neumann':
					neumann[out] = True
				elif cond=='Dirichlet':
					dirichlet[out] = True
	
	# ------- Get the neighbor indices --------
	# Cumulative product in reverse order, omitting last term, beginning with 1
	cum = tuple(list(reversed(np.cumprod(list(reversed(bounds+(1,)))))))[1:]
	bCum = np.broadcast_to( np.reshape(cum, (dim,)+(1,)*(dim+1)), offset.shape)
	bCum = xp.asarray(bCum,dtype=grid.dtype)

	index = (grid*bCum).sum(0)
	indexPos = (neighPos*bCum).sum(0)
	indexNeg = (neighNeg*bCum).sum(0)
	index = np.broadcast_to(index,indexPos.shape)
	
	# ------- Get the coefficients for the first order term -----
	if omega is not None:
		if intrinsicDrift:
			eta=omega
		else:
			eta = LP.dot_AV(LP.inverse(diff),omega)
			
		scalEta = LP.dot_VV(offset.astype(float), 
			np.broadcast_to(np.reshape(eta,(dim,1,)+bounds),offset.shape)) 
		coefOmega = coef*scalEta

	# ------- Create the triplets ------
	
	# Second order part
	# Nemann : remove all differences which are not inside (a.k.a multiply coef by inside)
	# TODO : Dirichlet : set to zero the coef only for the outside part
	
	coef = coef.reshape(-1)/ (gridScale**2) # Take grid scale into account

	index = index.reshape(-1)
	indexPos = indexPos.reshape(-1)
	indexNeg = indexNeg.reshape(-1)

	nff = lambda t : np.logical_not(t).astype(float).reshape(-1)
	IP = nff(np.logical_or(neumannPos,dirichletPos))
	IN = nff(np.logical_or(neumannNeg,dirichletNeg))
	iP = nff(neumannPos)
	iN = nff(neumannNeg)

	if divergenceForm:
		row = np.concatenate((index, indexPos, index, indexPos))
		col = np.concatenate((index, index, indexPos, indexPos))
		data = np.concatenate((iP*coef/2, -IP*coef/2, -IP*coef/2, IP*coef/2))
		
		row  = np.concatenate(( row, index, indexNeg, index, indexNeg))
		col  = np.concatenate(( col, index, index, indexNeg, indexNeg))
		data = np.concatenate((data, iN*coef/2, -IN*coef/2, -IN*coef/2, IN*coef/2))
		
	else:
		row = np.concatenate( (index, index,	index))
		col = np.concatenate( (index, indexPos, indexNeg))
		data = np.concatenate((iP*coef+iN*coef, -IP*coef, -IN*coef))
	

	# First order part, using centered finite differences
	if omega is not None:	   
		coefOmega = coefOmega.flatten() / gridScale # Take grid scale in
		row = np.concatenate((row, index,	index))
		col = np.concatenate((col, indexPos, indexNeg))
		data= np.concatenate((data,IP*iN*coefOmega/2,-IN*iP*coefOmega/2))
	
	if mult is not None:
		# TODO Non periodic boundary conditions
		size=np.prod(bounds)
		row = np.concatenate((row, range(size)))
		col = np.concatenate((col, range(size)))
		data= np.concatenate((data,mult.flatten()))

	nz = data!=0
	return data[nz],(row[nz],col[nz])
	
	
	
	
	
	