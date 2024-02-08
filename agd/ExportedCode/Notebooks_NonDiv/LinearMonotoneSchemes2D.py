# Code automatically exported from notebook LinearMonotoneSchemes2D.ipynb in directory Notebooks_NonDiv
# Do not modify
from ... import Selling
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import Domain

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg 

norm = ad.Optimization.norm
    
def streamplot_ij(X,Y,VX,VY,subsampling=1,*varargs,**kwargs):
    def f(array): return array[::subsampling,::subsampling].T
    return plt.streamplot(f(X),f(Y),f(VX),f(VY),*varargs,**kwargs) # Transpose everything

def SchemeCentered(u,cst,mult,omega,diff,bc,ret_hmax=False):
    """Discretization of a linear non-divergence form second order PDE
        cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
        Second order accurate, centered yet monotone finite differences are used for <omega,grad u>
        - bc : boundary conditions. 
        - ret_hmax : return the largest grid scale for which monotony holds
    """
    # Decompose the tensor field
    coefs2,offsets = Selling.Decomposition(diff)
    
    # Decompose the vector field
    scals = lp.dot_VA(lp.solve_AV(diff,omega), offsets.astype(float))
    coefs1 = coefs2*scals
    if ret_hmax: return 2./norm(scals,ord=np.inf)
    
    # Compute the first and second order finite differences    
    du  = bc.DiffCentered(u,offsets)
    d2u = bc.Diff2(u,offsets)
    
    # In interior : cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
    coefs1,coefs2 = (bc.as_field(e) for e in (coefs1,coefs2))    
    residue = cst + mult*u +lp.dot_VV(coefs1,du) - lp.dot_VV(coefs2,d2u)
    
    # On boundary : u-bc = 0
    return np.where(bc.interior,residue,u-bc.grid_values)

def SchemeUpwind(u,cst,mult,omega,diff,bc):
    """Discretization of a linear non-divergence form second order PDE
        cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
        First order accurate, upwind finite differences are used for <omega,grad u>
        - bc : boundary conditions. 
    """
    # Decompose the tensor field
    coefs2,offsets2 = Selling.Decomposition(diff)
    omega,coefs2 = (bc.as_field(e) for e in (omega,coefs2))    

    # Decompose the vector field
    coefs1 = -np.abs(omega)
    basis = bc.as_field(np.eye(len(omega)))
    offsets1 = -np.sign(omega)*basis
    
    # Compute the first and second order finite differences    
    du  = bc.DiffUpwind(u,offsets1.astype(int))
    d2u = bc.Diff2(u,offsets2)
    
    # In interior : cst + mult u + <omega,grad u>- tr(diff hess(u)) = 0
    residue = cst + mult*u +lp.dot_VV(coefs1,du) - lp.dot_VV(coefs2,d2u)
    
    # On boundary : u-bc = 0
    return np.where(bc.interior,residue,u-bc.grid_values)

