# Code automatically exported from notebook TensorSelling.ipynb in directory Notebooks_Algo
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ... import Selling
from agd.Plotting import savefig; #savefig.dirName = 'Figures/TensorSelling'
from ... import AutomaticDifferentiation as ad
from ... import Interpolation
from agd.Metrics import Riemann

import numpy as np; xp = np; allclose = np.allclose; π = np.pi
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=2000)

def MakeRandomTensor(dim,shape = tuple(),relax=0.01):
    identity = fd.as_field(np.eye(dim),shape,depth=2)
    A = np.random.standard_normal( (dim,dim) + shape ) 
    M = lp.dot_AA(lp.transpose(A),A) + relax*identity
    return xp.asarray(M) # Convert to GPU array if needed

def Reconstruct(coefs,offsets):
     return (coefs*lp.outer_self(offsets)).sum(2)

def sabs(x,order=1):
    """
    Regularized absolute value. Guarantee : 0 <= result-|x| <= 1/2.
    - order (0, 1, 2 or 3) : order of the last continuous derivative. 
    """
    x = np.abs(x) # Actually useless for this specific application
    if order==0: return x # Continuous value
    elif order==1: return np.where(x<=1, 0.5*(1+x**2), x) # Continuous derivative
    elif order==2: return np.where(x<=1, (3+6*x**2-x**4)/8,x) # Continuous second order derivative
    elif order==3: return np.where(x<=1, (5+15*x**2-5*x**4+x**6)/16,x) # Continuous second order derivative
    else: raise ValueError(f"Unsupported {order=} in sabs")

def smed(ρ0,ρ1,ρ2):
    """Regularized median (a.k.a. ρ1) assuming ρ0<=ρ1<=ρ2.
    Guarantee : ρ1/(2*sqrt(2)) <= result < ρ1"""
    s,q = ρ0*ρ1+ρ1*ρ2+ρ2*ρ0, (ρ2-ρ1)**2 # Invariant quantities under Selling superbase flip 
    return 0.5*s/np.sqrt(q+2*s) 

def smooth_decomp(D,order=1):
    """Smooth variant of Selling's two dimensional decomposition"""
    v = np.moveaxis(Selling.ObtuseSuperbase(D),1,0)
    ρ = np.array([-lp.dot_VAV(v[1],D,v[2]),-lp.dot_VAV(v[0],D,v[2]),-lp.dot_VAV(v[0],D,v[1])])
    ord = np.argsort(ρ,axis=0) 
    v = np.take_along_axis(v,ord[:,None],axis=0); ρ=np.take_along_axis(ρ,ord,axis=0)

    m = smed(*ρ)
    w = np.maximum(0,m*sabs(ρ[0]/m,order)-ρ[0]) # Positive up to roundoff error (max for safety)
    
    return (ad.array([ρ[0]+w/2, ρ[1]-w, ρ[2]-w, w/2]),
         lp.perp(np.moveaxis(ad.array([v[0],v[1],v[2],v[1]-v[2]]),0,1)).astype(int) )

def linear_decomp_offsets(d):
    """The stencil for this linear decomposition is fixed and has $d^2$ elements"""
    e = xp.eye(d).astype(int)
    return np.concatenate(
        (e,xp.array([e[i]+s*e[j] for i in range(d) for j in range(i) for s in (-1,1)])),axis=0).T

def linear_decomp_coefs(D):
    """These weights depend linearly on the matrix $D$"""
    d=len(D)
    α=(d+1)/(2*d); β=-1/(2*d); γ=1/(4*d); δ=1/4; # ε=(d-1)/(d+1)
    t = sum(D[i,i] for i in range(d))
    return np.concatenate((ad.array([(α-β)*D[i,i]+β*t for i in range(d)]),
        ad.array([γ*(D[i,i]+D[j,j])+s*δ*(D[i,j]+D[j,i]) for i in range(d) 
        for j in range(i) for s in (-1,1)]) ),axis=0)

def linear_decomp(D): return linear_decomp_coefs(D),linear_decomp_offsets(len(D))

def conv_circulant(circ,val):
    from numpy.fft import fft,ifft
    """Product of a circulant matrix with an array of values, computed using the FFT."""
    return np.moveaxis(ifft(fft(np.moveaxis(val,0,-1)) * fft(circ)).real,-1,0)

def conv_decomp(D,order=5,periodic=True,rtol=1e-10):
    """Deconvolve the matrix field D, apply Selling's decomposition, then convolve the coefficients.
    - D : positive definite matrix field, of shape (d,d,n1,...,nk). Convolution is applied to axes n1,...,nk.
    - order (either 3, 5 or 7): spline interpolation order
    - periodic : choose between periodic and reflected boundary conditions
    - rtol : relative tolerance for eliminating small coefficients.
    """
    D = Interpolation.spline_coefs(D,depth=2,order=order,periodic=periodic) # Deconvolution
    tol = rtol * lp.trace(D) # Compute absolute tolerance
    λ,e = Selling.Decomposition(D) # Selling decomposition
    vdim = len(e)
    shape = λ[0].shape
    pos = np.argmax(e!=0,axis=0) # position of the first non-zero coefficient
    e *= np.sign(np.take_along_axis(e,pos[None],0)) # Normalized offset, starts > 0
    emax = np.max(np.abs(e))
    r = 2*emax+1 # Base exponent for conversion to integer
    ie = sum(e[i] * r**(vdim-i-1) for i in range(vdim)) # Convert offsets to integers
    Λ = ad.Sparse.spAD(np.zeros_like(λ[0]),np.moveaxis(λ,0,-1),np.moveaxis(ie,0,-1))
    Λ = Λ.tangent_operator().todense() # Possible improvement : optimization opportunities here
    Λ = np.moveaxis(np.asarray(Λ).reshape((*shape,-1)),-1,0)
    Λ = Interpolation.spline_coefs(Λ,depth=1,solver=conv_circulant,order=order,periodic=periodic)
    Λ = np.roll(Λ,order-1,axis=tuple(range(1,Λ.ndim)))
    Λ[np.abs(Λ)<=tol]=0 # Remove almost zero coefficients
    index = fd.as_field(np.arange(len(Λ)),shape,depth=1)
    x = ad.Sparse.spAD(np.zeros_like(λ[0]),np.moveaxis(Λ,0,-1),np.moveaxis(index,0,-1))
    x.simplify_ad() # Remove null coefficients.
    λ = np.moveaxis(x.coef,-1,0) # The new coefficients
    ie = np.moveaxis(x.index,-1,0) # The new offsets, for now as integers
    e = []
    for i in range(vdim):
        mod = ie%r
        pos = mod>emax
        ie = (ie//r)+pos
        e.append(np.where(pos,mod-r,mod))
    return λ,np.array(e[::-1])

