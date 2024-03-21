# Code automatically exported from notebook HighOrderWaves.ipynb in directory Notebooks_Div
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from agd.Metrics import Riemann
from agd.Metrics.Seismic import Hooke
from agd.ODE.hamiltonian import QuadraticHamiltonian
from ... import AutomaticDifferentiation as ad
from ... import Selling
mica,_ = Hooke.mica # Note: ρ is in g/cm^3, which is inconsistent with the Hooke tensor SI units 
norm = ad.Optimization.norm

import copy
import numpy as np; xp=np; allclose=np.allclose; π = np.pi
import matplotlib.pyplot as plt

def MakeRandomTensor(dim, shape=tuple(), relax=0.05):
    """
    Some random symmetric positive definite matrices. 
    (relax>0 ensures definiteness)
    """
    A = np.random.standard_normal( (dim,dim) + shape )
    D = lp.dot_AA(lp.transpose(A),A)
    identity = np.eye(dim).reshape((dim,dim)+(1,)*len(shape))
    return D+lp.trace(D)*relax*identity

# Helper functions for the dispersion relation
def _γ(x,order): 
    if order_x==2: return 4*np.sin(x/2)**2 / x**2
    if order_x==4: return (np.cos(2*x)-16*np.cos(x)+15)/6 / x**2
    if order_x==6: return (49/18-3*np.cos(x)+3/10*np.cos(2*x)-1/45*np.cos(3*x)) / x**2

_cs = (6+5*np.power(2,1/3)+4*np.power(2,2/3))/144
_css = (4 + 4*np.power(2,1/3) + 3*np.power(2,2/3))/144
def _γs(x,order): return 1 if order==2 else 1-x/12-_cs*x**2
def _γss(x,order):return 1 if order==2 else 1-x/6-_css*x**2

def _γ2inv(y,dt): 
    r"""Inverse function to $x^2 γ_2(x) = 4 \sin^2(x/2)."""
    return 2*np.arcsin(dt*np.sqrt(y)/2)/dt

def _dispersion(b,ρ,dt,order_t):
    ω = _γ2inv(b * _γs(b*dt**2,order_t), dt)
    r = - ρ*np.sin(ω*dt)/(dt*_γss(b*dt**2,order_t))
    return ω,r
    
def dispersion1a(k,ρ,D,dx,dt,order_x=2,order_t=2):
    """
    Returns the frequency corresponding to the given wavenumber and model coefficients, for the discrete 1D model.
    Also returns the multiplier, to obtain $p$ from $q$ (in addition to shifting by a period).
    """
    b = D/ρ * k**2 * _γ(k*dx,order_x)
    return _dispersion(b,ρ,dt,order_t)

def make_domain(N,ndim):
    """Returns X, a regular sampling of [0,1]^d with N^d points, and dx the grid scale."""
    aX,dx = xp.linspace(0,1,N,endpoint=False,retstep=True)
    return xp.asarray(xp.meshgrid(*(aX+dx/2,)*ndim,indexing="ij")),dx

def eig(A,sort=False,moveaxis=True):
    """Wrapper around np.linalg.eig, which allows sorting, geometry first, import gpu arrays."""
    # cupy has no eig function, hence we do the computation on the CPU.
    if moveaxis: A = np.moveaxis(A,(0,1),(-2,-1))
    λ,v = np.linalg.eig(A) if xp is np else map(ad.cupy_generic.cupy_set, np.linalg.eig(ad.cupy_generic.cupy_get(A)))
    # sort (by increasing eigenvalues), assuming real eigenvalues
    if sort: order = np.argsort(λ,axis=-1); λ = np.take_along_axis(λ,order,-1); v = np.take_along_axis(v,order[...,None,:],-1)
    if moveaxis: λ = np.moveaxis(λ,-1,0); v = np.moveaxis(v,(-2,-1),(0,1))
    return λ,v

def elastic_modes(M,C,k):
    """
    Returns the propagation modes for the elastic wave equation
    with constant coefficients and a given wavenumber, sorted 
    from slowest to fastest.
    """
    # This code is a slight generalization of the Hooke.waves method
    Ck = Hooke(C).contract(k)
    λ,v = eig(lp.dot_AA(M,Ck),sort=True)
    ω = np.sqrt(λ)
    return ω,v

def sin_eet(e,dx,order_x=2):
    """Approximates of e e^T as dx->0, arises from FFT of scheme."""
    def ondiag(s): # Fourier transform of finite difference approximation of second derivative, see _γ
        if order_x==2: return 4*np.sin(s/2)**2
        if order_x==4: return (np.cos(2*s)-16*np.cos(s)+15)/6
        if order_x==6: return 49/18-3*np.cos(s)+3/10*np.cos(2*s)-1/45*np.cos(3*s)
    def offdiag(s,t): # Fourier transform of finite difference approximation of cross derivative
        if order_x==2: return np.sin(s)*np.sin(t)
        if order_x==4: return (8*np.sin(s)*np.sin(t) + (4*np.sin(s)-np.sin(2*s))*(4*np.sin(t)-np.sin(2*t)))/12
        if order_x==6: return (-9*np.sin(2*s)*(13*np.sin(t)-5*np.sin(2*t)+np.sin(3*t))+9*np.sin(s)*(50*np.sin(t)-13*np.sin(2*t)+2*np.sin(3*t))+np.sin(3*s)*(18*np.sin(t)-9*np.sin(2*t)+2*np.sin(3*t)))/180        
    return xp.array([[ondiag(e[i]*dx) if i==j else offdiag(e[i]*dx,e[j]*dx)
             for i in range(len(e))] for j in range(len(e))])/dx**2
    
def sin_contract(C,k,dx,order_x):
    """Approximates Hooke(C).contract(k) as dx->0, arises from FFT of scheme."""
    λ,σ = Hooke(C).Selling()
    σk = lp.dot_AV(σ,k[:,None])
    return np.sum(λ*sin_eet(σk,dx,order_x),axis=2)
    
def dispersion_e(k,ρ,C,dx,dt,order_x=2,order_t=2):
    """Return all discrete propagation modes."""
    # For now, we assume that M = Id/ρ. Anisotropic M seem doable, but would require taking care
    # of the non-commutativity of a number of matrices in the scheme.
    Ck = sin_contract(C,k,dx,order_x)
    b,vq = eig(Ck/ρ,sort=True)
    ω,r = _dispersion(b,ρ,dt,order_t)
    return ω,vq,r*vq

def mk_planewave_e(k,ω,vq,vp):
    """Make an elastic planewave with the specified parameters."""
    def brdcst(v,x): return fd.as_field(v,x[0].shape)
    def expi(t,x): return np.exp(1j*(lp.dot_VV(brdcst(k,x),x) - ω*t)) 
    def q_exact(t,x): return  expi(t,x).real * brdcst(vq,x)
    def p_exact(t,x): return -expi(t,x).imag * brdcst(vp,x)
    return q_exact,p_exact

