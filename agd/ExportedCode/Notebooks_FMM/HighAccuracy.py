# Code automatically exported from notebook HighAccuracy.ipynb in directory Notebooks_FMM
# Do not modify
from ... import Eikonal
from ... import Metrics
from agd.Metrics.Seismic import Hooke
from ... import FiniteDifferences as fd
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from agd.Plotting import savefig; #savefig.dirName = 'Figures/HighAccuracy'

import numpy as np
import matplotlib.pyplot as plt

def PoincareCost(q):
    """
    Cost function defining the Poincare half plane model of the hyperbolic plane.
    """
    return 1/q[1]

def PoincareDistance(p,q):
    """
    Distance between two points of the half plane model of the hyperbolic plane.
    """
    a = p[0]-q[0]
    b = p[1]-q[1]
    c = p[1]+q[1]
    d = np.sqrt(a**2+b**2)
    e = np.sqrt(a**2+c**2)
    return np.log((e+d)/(e-d))

diagCoef = (0.5**2,1) # Diagonal coefficients of M

def diff(x,y,α=0.5): return ad.array([x,y+α*np.sin(np.pi*x)]) # Diffeomorphism f

def RiemannMetric(diag,diff,x,y): 
    X_ad = ad.Dense.identity(constant=(x,y),shape_free=(2,))
    Jac = np.moveaxis(diff(*X_ad).gradient(),0,1)
    return Metrics.Riemann.from_diagonal(diag).inv_transform(Jac)

def RiemannExact(diag,diff,x,y):
    a,b = diag
    fx,fy = diff(x,y)
    return np.sqrt(a*fx**2+b*fy**2)

M=((1.25,0.5),(0.5,2.))

def v(x,y,γ): return γ*np.sin(np.pi*x)*np.sin(np.pi*y)/np.pi

def RanderMetric(x,y,γ=0.8):
    X_ad = ad.Dense.identity(constant=(x,y),shape_free=(2,))
    omega = v(*X_ad,γ).gradient()
    return Metrics.Rander(M,omega)

def RanderSolution(x,y,γ=0.8):
    return Metrics.Riemann(M).norm((x,y)) + v(x,y,γ)

def ConformalMap(x):
    """
    Implements the mapping x -> (1/2) * x^2, where x is seen as a complex variable.
    """
    return ad.array([0.5*(x[0]**2-x[1]**2), x[0]*x[1]])

def ConformalApply(norm,f,x,decomp=True):
    """
    Applies a conformal change of coordinates to a norm.
    decomp : decompose the Jacobian into a scaling and rotation.
    """
    x_ad = ad.Dense.identity(constant=x,shape_free=(2,))
    Jac = np.moveaxis(f(x_ad).gradient(),0,1)
    if not decomp: return norm.inv_transform(Jac)
    
    # Assuming Jac is a homothety-rotation
    α = np.power(lp.det(Jac), 1/norm.vdim)
    R = Jac/α
    return norm.with_cost(α).rotate(lp.transpose(R))
    

def MappedNormValues(norm,f,x,seed):
    seed = fd.as_field(seed,x.shape[1:])
    return norm.norm(f(x)-f(seed))

