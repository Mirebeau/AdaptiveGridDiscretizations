# Code automatically exported from notebook NonlinearMonotoneFirst2D.ipynb in directory Notebooks_NonDiv
# Do not modify
from ... import Selling
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import Domain

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg;
import itertools

def Gradient(u,A,bc,decomp=None):
    """
    Approximates grad u(x), using finite differences along the axes of A.
    """
    coefs,offsets = Selling.Decomposition(A) if decomp is None else decomp
    du = bc.DiffCentered(u,offsets) 
    AGrad = lp.dot_AV(offsets.astype(float),(coefs*du)) # Approximates A * grad u
    return lp.solve_AV(A,AGrad) # Approximates A^{-1} (A * grad u) = grad u

def SchemeCentered(u,A,F,rhs,bc):
    """
    Discretization of - Tr(A(x) hess u(x)) + F(grad u(x)) - rhs,
    with Dirichlet boundary conditions. The scheme is second order,
    and degenerate elliptic under suitable assumptions.
    """
    # Compute the tensor decomposition
    coefs,offsets = Selling.Decomposition(A)
    A,coefs,offsets = (bc.as_field(e) for e in (A,coefs,offsets))
    
    # Obtain the first and second order finite differences
    grad = Gradient(u,A,bc,decomp=(coefs,offsets))
    d2u = bc.Diff2(u,offsets)    
    
    # Numerical scheme in interior    
    residue = -lp.dot_VV(coefs,d2u) + F(grad) - rhs
    
    # Placeholders outside domain
    return np.where(bc.interior,residue,u-bc.grid_values)

# Specialization for the quadratic non-linearity
def SchemeCentered_Quad(u,A,omega,D,rhs,bc):
    omega,D = (bc.as_field(e) for e in (omega,D))
    def F(g): return lp.dot_VAV(g-omega,D,g-omega)
    return SchemeCentered(u,A,F,rhs,bc)

def SchemeUpwind(u,A,omega,D,rhs,bc):
    """
    Discretization of -Tr(A(x) hess u(x)) + \| grad u(x) - omega(x) \|_D(x)^2 - rhs,
    with Dirichlet boundary conditions, using upwind finite differences for the first order part.
    The scheme is degenerate elliptic if A and D are positive definite. 
    """
    # Compute the decompositions (here offset_e = offset_f)
    nothing = (np.full((0,),0.), np.full((2,0),0)) # empty coefs and offsets
    mu,offset_e = nothing if A is None else Selling.Decomposition(A) 
    nu,offset_f = nothing if D is None else Selling.Decomposition(D)
    omega_f = lp.dot_VA(omega,offset_f.astype(float))

    # First and second order finite differences
    maxi = np.maximum
    mu,nu,omega_f = (bc.as_field(e) for e in (mu,nu,omega_f))

    dup = bc.DiffUpwind(u, offset_f)
    dum = bc.DiffUpwind(u,-offset_f)
    dup[...,bc.not_interior]=0. # Placeholder values to silence NaN warnings
    dum[...,bc.not_interior]=0.
    
    d2u = bc.Diff2(u,offset_e)
        
    # Scheme in the interior
    du = maxi(0.,maxi( omega_f - dup, -omega_f - dum) )
    residue = - lp.dot_VV(mu,d2u) + lp.dot_VV(nu,du**2) - rhs

    # Placeholders outside domain
    return np.where(bc.interior,residue,u-bc.grid_values)

