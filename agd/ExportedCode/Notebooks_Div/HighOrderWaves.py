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

def KineticEnergy_a(p,ρ):
    """
    Kinetic energy (x2), for the acoustic wave equation.
    p : momentum. ρ : density.
    """
    return 0.5 * p**2 / ρ # Return energy density, summation will be implicit

def deriv(f,s):
    """Evaluate the derivative f'(s) of a univariate function f"""
    s_ad = ad.Dense.identity(constant=s,shape_free=tuple())
    return f(s_ad).gradient(0) # Automatic differentiation

# Helper functions for the dispersion relation
def _γ(x,order):
    tol=[None,1e-6,3e-3,1e-2][order//2]; x=np.where(np.abs(x)<=tol,tol,x) # Avoid zero divide
    if order==2: return 4*np.sin(x/2)**2 / x**2
    if order==4: return (np.cos(2*x)-16*np.cos(x)+15)/6 / x**2
    if order==6: return (49/18-3*np.cos(x)+3/10*np.cos(2*x)-1/45*np.cos(3*x)) / x**2

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

def PotentialEnergy_a(q,D,dx,order=2):
    """Acoustic potential energy density (x2)"""
    λ,e = D if isinstance(D,tuple) else Selling.Decomposition(D) # Coefficients and offsets of Selling's decomposition
    λ = fd.as_field(λ,q.shape,depth=1) # Broadcasting coefficients if necessary
    diff = fd.DiffEll(q,e,dx,order,padding=None) # padding=None means periodic
    return 0.5 * λ*diff**2 # Summation is implicit

def Hamiltonian_a(ρ,D,X,dx,order=2):
    """Discretized Hamiltonian of the acoustic wave equation."""
    Hq = lambda q: PotentialEnergy_a(q,D,dx,order=order) 
    Hp = lambda p: KineticEnergy_a(p,ρ)
    H = QuadraticHamiltonian(Hq,Hp)
    H.set_spmat(xp.zeros_like(X[0])) # Replaces quadratic functions with sparse matrices
    return H

def make_domain(N,ndim):
    """Returns X, a regular sampling of [0,1]^d with N^d points, and dx the grid scale."""
    aX,dx = xp.linspace(0,1,N,endpoint=False,retstep=True)
    return xp.asarray(xp.meshgrid(*(aX+dx/2,)*ndim,indexing="ij")),dx

def ExactSol_a(ρ,D,ks,fs):
    """
    Produce an exact solution for the acoustic wave equation, 
    with constant density ρ and dual metric D.
    - ks : list of wave numbers.
    - fs : list of functions.
    Choose ks with integer coordinates, and fs 1-periodic, to obtain a solution on (R/Z)^2
    """
    vdim = len(D); ks = xp.array(ks)
    ωs = [np.sqrt(lp.dot_VAV(k,D,k)/ρ) for k in ks]
    ks = ks.reshape((-1,vdim)+(1,)*vdim)
    def q(t,x): return sum( f(lp.dot_VV(k,x)-ω*t) for (k,ω,f) in zip(ks,ωs,fs) )
    def p(t,x): return sum( -ρ*ω*deriv(f,lp.dot_VV(k,x)-ω*t) for (k,ω,f) in zip(ks,ωs,fs) )
    return q,p

def tq_a(q,ϕ): return q(ϕ)
def tp_a(p,ϕ,Jϕ): return p(ϕ)*Jϕ
def tρ_a(ρ,ϕ,Jϕ): return ρ(ϕ)*Jϕ
def tD_a(D,ϕ,inv_dϕ,Jϕ): 
    D_ = fd.as_field(D(ϕ),Jϕ.shape,depth=2)
    return lp.dot_AA(inv_dϕ,lp.dot_AA(D_,lp.transpose(inv_dϕ))) * Jϕ

def differentiate(ϕ_fun,X):
    """
    Returns ϕ, dϕ, (dϕ)^-1, J=det(dϕ), and d^2ϕ.
    """
    X_ad = ad.Dense2.identity(constant=X,shape_free=(len(X),))
    ϕ_ad = ϕ_fun(X_ad)
    dϕ = np.moveaxis(ϕ_ad.gradient(),1,0) # Gradient is differential transposed
    d2ϕ = np.moveaxis(ϕ_ad.hessian(),2,0)
    return ϕ_ad.value, dϕ, lp.inverse(dϕ), lp.det(dϕ), d2ϕ

def dispersion_a(k,ρ,D,dx,dt,order_x=2,order_t=2):
    """
    Returns the frequency corresponding to the given wavenumber and model coefficients, for the discrete 1D model.
    Also returns the multiplier, to obtain $p$ from $q$ (in addition to shifting by a period).
    """
    λ,e = D if isinstance(D,tuple) else Selling.Decomposition(D)
    ke = lp.dot_VV(k[:,None],e)
    b = np.sum(λ * ke**2 * _γ(ke*dx,order_x),axis=0) / ρ
    return _dispersion(b,ρ,dt,order_t)

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

def ExactSol_e(M,C,ks,fs,imodes):
    """
    Produce an exact solution of the elastic wave equation, 
    with constant metric M and Hooke tensor C (and S=0).
    - ks : list of wave numbers.
    - fs : list of functions.
    - imodes : list of integers in [0,...,d-1], the propagation mode number.
    Choose ks with integer coordinates, and fs 1-periodic, to obtain a solution on (R/Z)^2
    """
    vdim = len(M); ks = xp.array(ks)
    modes = [elastic_modes(M,C,k) for k in ks]
    ωs = [ω[i] for (ω,_),i in zip(modes,imodes)]
    vs = [v[:,i] for (_,v),i in zip(modes,imodes)]
    ivs = [lp.solve_AV(M,v) for v in vs]    

    def q(t,x): 
        x_,vs_,ks_ = fd.common_field((x,vs,ks),depths=(1,2,2))
        return sum( v*f(lp.dot_VV(k,x_)-ω*t) for (v,k,ω,f) in zip(vs_,ks_,ωs,fs) )
    
    def p(t,x): 
        x_,ivs_,ks_ = fd.common_field((x,ivs,ks),depths=(1,2,2))
        return sum( -iv*ω*deriv(f,lp.dot_VV(k,x_)-ω*t) for (iv,k,ω,f) in zip(ivs_,ks_,ωs,fs) )

    return q,p

def tq_e(q,ϕ,dϕ): return lp.dot_AV(lp.transpose(dϕ),q(ϕ))
def tp_e(p,ϕ,inv_dϕ,Jϕ): return lp.dot_AV(inv_dϕ,p(ϕ))*Jϕ

def tM_e(M,ϕ,dϕ,Jϕ): 
    M_ = fd.as_field(M(ϕ),Jϕ.shape,depth=2)
    return lp.dot_AA(lp.transpose(dϕ),lp.dot_AA(M_,dϕ))/Jϕ

def tC_e(C,ϕ,inv_dϕ,Jϕ): 
    C_ = fd.as_field(C(ϕ),Jϕ.shape,depth=2)
    return Hooke(C_).rotate(inv_dϕ).hooke*Jϕ

def tS_e(S,ϕ,dϕ,inv_dϕ,d2ϕ): 
    S_ = fd.as_field(S(ϕ),ϕ[0].shape,depth=3)
    vdim = len(dϕ)
    
    S1 = sum(dϕ[ip,:,None,None]*dϕ[jp,None,:,None]*S_[ip,jp,kp]*inv_dϕ[:,kp]
          for ip in range(vdim) for jp in range(vdim) for kp in range(vdim))
    S2 = sum(d2ϕ[kp,:,:,None]*inv_dϕ[:,kp] for kp in range(vdim))
    return S1 + S2

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
    λ,σ = C if isinstance(C,tuple) else Hooke(C).Selling()
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

def MakeRandomTensor(dim, shape=tuple(), relax=0.05):
    """
    Some random symmetric positive definite matrices. 
    (relax>0 ensures definiteness)
    """
    A = np.random.standard_normal( (dim,dim) + shape )
    D = lp.dot_AA(lp.transpose(A),A)
    identity = np.eye(dim).reshape((dim,dim)+(1,)*len(shape))
    return D+lp.trace(D)*relax*identity

