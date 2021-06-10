# Code automatically exported from notebook TensorSelling.ipynb in directory Notebooks_Algo
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ... import Selling
from agd.Plotting import savefig; #savefig.dirName = 'Figures/TensorSelling'
from ... import AutomaticDifferentiation as ad

import numpy as np; xp = np; allclose = np.allclose
import matplotlib.pyplot as plt

def MakeRandomTensor(dim,shape = tuple(),relax=0.01):
    identity = fd.as_field(np.eye(dim),shape,depth=2)
    A = np.random.standard_normal( (dim,dim) + shape ) 
    M = lp.dot_AA(lp.transpose(A),A) + relax*identity
    return xp.asarray(M) # Convert to GPU array if needed

def Reconstruct(coefs,offsets):
     return lp.mult(coefs,lp.outer_self(offsets)).sum(2)

def offsets_prereduced(b1,b2):
    """Offsets of the decomposition associated to a prereduced basis. Assumes |det(b1,b2)|=1."""
    e1,e2 = -lp.perp(b2),lp.perp(b1) 
    #e1,e2 = np.moveaxis(lp.inverse([b1,b2]),0,1) # Equivalent
    return np.stack([e1,e2,e1+e2,e1-e2],axis=1).astype(int)

def coefs_prereduced(D,b1,b2):
    """Coefficients of the decomposition associated to a prereduced basis."""
    b11,b12,b22 = [lp.dot_VAV(bi,D,bj) for (bi,bj) in ((b1,b1),(b1,b2),(b2,b2))]
    ρ = ad.asarray([(b11-b12)*(b11+b12),(b22-b12)*(b22+b12),(b11+b12)*(b22+b12)/2,(b11-b12)*(b22-b12)/2])
    return ρ/(b11+b22)

def coefs_Selling(D,b0,b1,b2):
    """Coefficients of Selling's decomposition."""
    return ad.array([-lp.dot_VAV(b0,D,b1),-lp.dot_VAV(b2,D,b0),np.zeros_like(b0[0]),-lp.dot_VAV(b1,D,b2)])

def sigmoid(x): 
    """-1 -> -1, 1 -> 1, constant outside, smooth inside"""
    return np.where(x<=-1,-1,np.where(x>=1, 1, (3*x-x**3)/2))
def sigmoid2(x):
    """1/2 -> 0, 1 -> 1, constant outside, smooth inside"""
    return (1+sigmoid(4*x-3))/2

def SmoothDecomposition(D):
    """A smooth interpolation between Selling's decomposition, 
    and the one associated to a well chosen prereduced basis."""
    b0,b1,b2 = np.moveaxis(Selling.ObtuseSuperbase(D),0,1)
    ρ = ad.array([-lp.dot_VAV(bi,D,bj) for (bi,bj) in ((b1,b2),(b2,b0),(b0,b1))])
    
    # We want ρ0 to be the smallest
    ai = np.argsort(ρ,axis=0)
    ρ0,ρ1,ρ2 = np.take_along_axis(ρ,ai,axis=0)
    b0,b1,b2 = np.take_along_axis(ad.array([b0,b1,b2]),ai[:,None],axis=0)
    
    # Interpolate between the two decompositions
    σ = sigmoid2( ρ1*ρ2/(ρ0*ρ1+ρ1*ρ2+ρ2*ρ0) )
    return (1-σ)*coefs_Selling(D,b0,b1,b2)+σ*coefs_prereduced(D,b1,b2), offsets_prereduced(b1,b2)

