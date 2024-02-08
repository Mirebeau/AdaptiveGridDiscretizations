# Code automatically exported from notebook Meissner.ipynb in directory Notebooks_Algo
# Do not modify
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import Plotting
norm_infinity = ad.Optimization.norm_infinity

import numpy as np
from matplotlib import pyplot as plt
import scipy
import scipy.optimize as sciopt

def LinearConstraint_AD(f):
    """
    Takes a linear constraint f>=0, 
    encoded as an ad.Sparse variable, 
    and turns it into a scipy compatible constraint.
    """
    return sciopt.LinearConstraint(f.tangent_operator(), lb = -f.value, ub=np.inf)

def QuadraticObjective_AD(f):
    """
    Takes a quadratic objective function, 
    encoded as an ad.Sparse2 variable,
    and returns fun, jac and hessian methods.
    """
    val0 = f.value
    grad0 = f.to_first().to_dense().gradient()
    n = len(grad0)
    hess0 = f.hessian_operator(shape=(n,n))
    def fun(x):  return val0+np.dot(grad0,x)+np.dot(x,hess0*x)/2
    def grad(x): return grad0+hess0*x
    def hess(x): return hess0
    return {'fun':fun,'jac':grad,'hess':hess}

def NonlinearObjective(f,fargs):
    """Returns methods computing the value, gradient and hessian 
    of a given objective function f"""
    def fun(x):
        return f(x,*fargs)
    def grad(x): 
        return f(ad.Sparse.identity(constant=x),*fargs).to_dense().gradient()
    def hess(x): 
        return f(ad.Sparse2.identity(constant=x),*fargs).hessian_operator()
    return {'fun':fun,'jac':grad,'hess':hess}

def NonlinearConstraint(f,fargs):
    """
    Represents the constraint : np.bincount(indices,weights) >= 0,
    where (indices, weights) = f(x,*fargs)
    (Indices may be repeated, and the associated values must be summed.)
    """
    def fun(x):
        ind,wei = f(x,*fargs); 
        return np.bincount(ind,wei)
    def grad(x): 
        ind,wei = f(ad.Sparse.identity(constant=x),*fargs)
        triplets = (wei.coef.reshape(-1),(ind.repeat(wei.size_ad),wei.index.reshape(-1)))
        return scipy.sparse.coo_matrix(triplets).tocsr()
    def hess(x,v): # v is a set of weights, provided by the optimizer
        ind,wei = f(ad.Sparse2.identity(constant=x),*fargs)
        return np.sum(v[ind]*wei).hessian_operator()
    return sciopt.NonlinearConstraint(fun,0.,np.inf,jac=grad,hess=hess,keep_feasible=True)

def geodesate(points,triangles,n,nosym=False,tol=1e-6):
    """
    Generate a point set on a sphere, by projecting a regularly refined triangulation.
    (This implementation is stupid and inefficient, but will be enough for here.)
    Input : 
    - points, triangles : a triangulation of the sphere.
    - nosym : keep only a single representative among opposite points
    Output : the points of a refined triangulation.
    """
    out = np.zeros((3,1))
    def norm2(p): return (p**2).sum(axis=0)
    for i,j,k in triangles:
        pi,pj,pk = points[i],points[j],points[k]
        for u in range(n+1):
            for v in range(n+1-u):
                p = (u*pi+v*pj+(n-u-v)*pk)/n
                p /= np.sqrt(norm2(p))
                p = p[:,None]
                dist2 = np.min( norm2(out-p) )
                if nosym: dist2 = min(dist2, np.min( norm2(out+p) ) )
                if dist2<tol: continue 
                out = np.concatenate((out,p),axis=1)
    if nosym: out *= np.where(out[2]>=0,1,-1)
    return out[:,1:]

ico_points = np.array([[0., 0., -0.9510565162951536], [0., 0., 0.9510565162951536], [-0.85065080835204, 0., -0.42532540417601994], 
  [0.85065080835204, 0., 0.42532540417601994], [0.6881909602355868, -0.5, -0.42532540417601994], 
  [0.6881909602355868, 0.5, -0.42532540417601994], [-0.6881909602355868, -0.5, 0.42532540417601994], 
  [-0.6881909602355868, 0.5, 0.42532540417601994], [-0.2628655560595668, -0.8090169943749475, -0.42532540417601994], 
  [-0.2628655560595668, 0.8090169943749475, -0.42532540417601994], [0.2628655560595668, -0.8090169943749475, 0.42532540417601994], 
  [0.2628655560595668, 0.8090169943749475, 0.42532540417601994]])
ico_triangles = np.array([[2, 12, 8], [2, 8, 7], [2, 7, 11], [2, 11, 4], [2, 4, 12], [5, 9, 1], [6, 5, 1], [10, 6, 1], [3, 10, 1], [9, 3, 1], 
   [12, 10, 8], [8, 3, 7], [7, 9, 11], [11, 5, 4], [4, 6, 12], [5, 11, 9], [6, 4, 5], [10, 12, 6], [3, 8, 10], [9, 7, 3]])-1

def sphere_sampling(n,nosym=False):
    """A rather regular sapling of the sphere, obtained by refining an icosahedron mesh."""
    return geodesate(ico_points,ico_triangles,n,nosym)

def to_mathematica(Z): return str(Z.T.tolist()).replace("[","{").replace("]","}")

