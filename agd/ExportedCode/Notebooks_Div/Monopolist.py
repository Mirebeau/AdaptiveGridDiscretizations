# Code automatically exported from notebook Monopolist.ipynb in directory Notebooks_Div
# Do not modify
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
norm_infinity = ad.Optimization.norm_infinity

import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import scipy
import scipy.optimize as sciopt
from matplotlib import pyplot as plt

from agd.ExportedCode.Notebooks_Algo.Meissner import LinearConstraint_AD,QuadraticObjective_AD

def square_X(n):
    """A regular sampling of the unit square, approx n^2 points."""
    aX = np.linspace(0,1,n+1)
    X = np.array(np.meshgrid(aX,aX,indexing='ij')).reshape(2,-1)
    tol = 0.5/n
    return X, (X[0]>1-tol,X[1]>1-tol,X[0]<tol,X[1]<tol)

def triangle_X(n):
    """A regular sampling of the equailateral triangle of vertices 
    (1,0), (-1/2,sqrt 3/2),(-1/2,-sqrt 3/2), approx n^2/2 points"""
    X = np.array([ (i/n,j/n) for i in range(n+1) for j in range(n-i+1)]).T
    tol = 0.5/n
    def aff(X): return np.array([1-(X[0]+X[1])*3/2, (X[1]-X[0])*np.sqrt(3)/2])
    return aff(X), (X[0]+X[1]>1-tol,X[0]<tol,X[1]<tol)

def disk_X(n):
    """A regular sampling of the unit disc, approx π n^2 points"""
    τ = 2*np.pi
    X = np.array([ (i*np.cos(θ),i*np.sin(θ)) for i in range(n+1) 
        for θ in np.linspace(0,τ,np.maximum(1,int(τ*i)),endpoint=False)]).T/n
    return X,tuple() # No facet : the disk is uniformly convex

def Delaunay(X): return scipy.spatial.Delaunay(X.T).simplices.T

def show_X(X,DX):
    plt.triplot(*X,Delaunay(X).T)
    for F in DX: plt.scatter(*X[:,F])
    plt.axis('equal')

def gradientFE(u,X,T):
    """Gradient on each simplex of the piecewise affine function u"""
    A = [ X[:,T[1]]-X[:,T[0]], X[:,T[2]]-X[:,T[0]] ]
    return lp.solve_AV(A,u[T[1:]]-u[T[0]])

def meanFE(u,T):
    """Mean on each simplex of piecewise affine function u"""
    return np.mean(u[...,T],axis=-2)

def cellmeasFE(X,T):
    """Area (or volume) of each simplex in the triangulation"""
    return lp.det(X[:,T[1:]]-X[:,None,T[0]])/np.math.factorial(len(X))

def monopolist_objective(u,X,T,ρ=1.,quadratic=True):
    g = gradientFE(u,X,T)
    integrand = meanFE(u,T) - lp.dot_VV(meanFE(X,T),g) # u(z) - <grad u(z),z>
    if quadratic: integrand += lp.dot_VV(g,g)/2 # + |grad u(z)|^2 /2
    return np.sum(integrand*cellmeasFE(X,T)*ρ)

