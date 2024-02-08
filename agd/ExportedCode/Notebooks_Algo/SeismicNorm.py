# Code automatically exported from notebook SeismicNorm.ipynb in directory Notebooks_Algo
# Do not modify
from agd.Metrics.Seismic import Hooke,Thomsen,TTI
from agd.Metrics import Riemann
from ... import Sphere
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from agd.Plotting import SetTitle3D

import numpy as np
import copy
from matplotlib import pyplot as plt

def proj_orthorombic(c):
    """Project onto the vector space of Hooke tensors corresponding to orthorombic materials in their frame of reference"""
    to_orthorombic = (c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2],c[3,3],c[4,4],c[5,5]) # Seismologists start at 1 ...
    return Hooke.from_orthorombic(*to_orthorombic).hooke

def frame_score(c,proj,r):
    """Score for wether c coincides with its projection in the frame defined by r"""
    c = c.rotate(lp.transpose(r)) # Put in specified frame
    c.hooke -= proj(c.hooke) # Substract projection 
    return (c.to_Mandel()**2).sum(axis=(0,1)) # Return frobenius norm squared

def proj_hooke2(c):
    """Project onto the vector space of Hooke tensors with (2,1) block diagonal structure"""
    z = np.zeros_like(c[0,0])
    return ad.array([
        [c[0,0],c[0,1],     z],
        [c[1,0],c[1,1],     z],
        [     z,     z,c[2,2]] ])

def proj_tetragonal(c):
    """Tetragonal Hooke tensors share the coefficients c11=c22, c13=c23, c44=c55"""
    c11,c12,c13,c22,c23,c33,c44,c55,c66 = ( # Seismologists start at 1 ...
        c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2],c[3,3],c[4,4],c[5,5])
    α=(c11+c22)/2
    γ=(c13+c23)/2
    δ=(c44+c55)/2
    return Hooke.from_orthorombic(α,c12,γ,α,γ,c33,δ,δ,c66).hooke

def proj_hexagonal(c):
    """Hexagonal Hooke tensors are tetragonal, with the additional property that c66=(c11-c12)/2"""
    c11,c12,c13,c22,c23,c33,c44,c55,c66 = ( # Seismologists start at 1 ...
        c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2],c[3,3],c[4,4],c[5,5])
    α=(3*(c11+c22)+2*c12+4*c66)/8
    β=(c11+c22+6*c12-4*c66)/8
    γ=(c13+c23)/2
    δ=(c44+c55)/2
    return Hooke.from_orthorombic(α,β,γ,α,γ,c33,δ,δ,(α-β)/2).hooke

def rotation_from_ball(x):
    """
    Produces a rotation matrix from an element of the unit ball
    B_3 -> S_3 (lower half) -> SO_3, via quaternions
    B_1 -> SO_2, via rotation of angle pi*x
    """
    if   len(x)==3: return Sphere.rotation3_from_ball3(x)[0] 
    elif len(x)==1: return lp.rotation(np.pi*x[0])
    else: raise ValueError("Unsupported dimension")

def ball_samples(vdim,dens):
    """
    Produce samples of the unit ball of dimension vdim.
    Approx c(vdim) * dens^vdim elements.
    """
    aB,h = np.linspace(-1,1,dens,retstep=True,endpoint=False)
    B = np.array(np.meshgrid(*(aB,)*vdim,indexing='ij'))
    B += h*np.random.rand(*B.shape) # Add random perturbations
    B = B[:,np.linalg.norm(B,axis=0)<=1]
    return B.reshape(vdim,-1)

def newton_extremize(f,x,niter,max_cond=1e10):
    """
    Runs niter steps of Newton's method for extremizing f. 
    (A.k.a, solve grad(f) = 0)
    """
    x_ad = ad.Dense2.identity(constant=x,shape_free=(len(x),))
    for i in range(niter):
        f_ad = f(x_ad)
#        print(np.linalg.eigvalsh(f_ad.coef2[0]))
        x_ad.value -= lp.solve_AV(f_ad.hessian(),f_ad.gradient())
    return x_ad.value,f_ad.value

def hooke_frame(c,proj,dens=8,niter=8):
    """
    Optimize the frame score, via a Newton method, with a large number of initial seeds.
    Return the best rotation and associated score.
    """
    def f(x): return frame_score(c,proj,rotation_from_ball(x))
    x = ball_samples({2:1,3:3}[c.vdim],dens) 
    x,f_x = newton_extremize(f,x,niter)
    argmin = np.nanargmin(f_x)
    return lp.transpose(rotation_from_ball(x[:,argmin])),f_x[argmin]

