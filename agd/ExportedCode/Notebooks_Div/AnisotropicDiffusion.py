# Code automatically exported from notebook AnisotropicDiffusion.ipynb in directory Notebooks_Div
# Do not modify
from ... import AutomaticDifferentiation as ad
from ... import LinearParallel as lp
from ... import Selling
from ... import FiniteDifferences as fd

import numpy as np; xp=np
from scipy import ndimage

def DirichletEnergy(u,D,dx):
    """
    Dirichlet energy associated with an anisotropic tensor field.
    """
    # Decompose the tensors
    λ,e = Selling.Decomposition(D)
    
    # Compute finite differences in the appropriate directions
    dup = fd.DiffUpwind(u, e,dx) # (u(x+dx*e)-u(x))/dx
    dum = fd.DiffUpwind(u,-e,dx) # (u(x-dx*e)-u(x))/dx
    
    # Apply Neumann-like boundary conditions
    dup[np.isnan(dup)]=0
    dum[np.isnan(dum)]=0
    
    # Return energy density
    return ( 0.25 * λ * (dup**2+dum**2) ).sum(axis=0)  # Sum over i but not over x

def DiffusionOperator(D,dx):
    """
    Anisotropic diffusion linear operator.
    - D : diffusion tensors
    - dx : grid scale
    """
    # Differentiate the Dirichlet energy
    u_ad = ad.Sparse2.identity(constant=np.zeros_like(D[0,0]))
    E_ad = DirichletEnergy(u_ad,D,dx)
        
    # Return hessian of this quadratic form
    return E_ad.hessian_operator() # Summation is implicit

def CFL(D,dx):
    """Returns a time step for which the explicit scheme is guaranteed to be stable"""
    return dx**2 / (2*np.max(lp.trace(D)))

def evolve(u0,dt,ndt,A):
    u = u0.flatten() # Makes an implicit copy
    for i in range(ndt):
        u -= dt * A*u
    return u.reshape(u0.shape)

def StructureTensor(u,σ=2.,ρ=5.,dx=1.,gaussian_filter=None):
    """
    Computes the structure tensor of u,
    - σ : noise scale, measured in pixels
    - ρ : feature scale, measured in pixels.
    - dx (optional) : grid scale, for rescaling the gradient
    - gaussian_filter (optional) : provided an implementation of gaussian filter
    """
    if gaussian_filter is None: gaussian_filter = ndimage.gaussian_filter
    # Compute grad uσ
    d = u.ndim 
    eye = np.eye(d).astype(int)
    duσ = [gaussian_filter(u,σ,order=e)/dx for e in eye]
    
    # Self outer product and averaging 
    S = lp.outer_self(duσ)
    for i in range(d): 
        for j in range(d):
            S[i,j] = gaussian_filter(S[i,j],ρ)
    
    return S 

def coherence_enhancing_diffusion_eigvals(μ,α,γ,
    cond_max=10**2,cond_amplification_threshold=2):
    """
    - μ : array of shape (n1,...,nk,d) where d=2,3, the structure tensor eigenvalues. 
        Assumes μ is increasing along last axis.
    - α : same as in CoherenceEnhancing parameters
    - γ : Related to λ = sqrt(γ/3.314)
    - cond_max : bound on the condition number of the generated tensors
    - cond_amplification_threshold : real in [1,infty[. Values >1 may help avoid creating structure in noise
    (Geometry last convention)
    returns : - the ced tensor eigenvalues, decreasing along the last axis. 
    """
    # Assumes increasing μ, the structure tensor eigenvalues
    # Returns decreasing λ, the diffusion tensor eigenvalues
    ε=1e-6 # Only to avoid zero divide
    λ = np.empty_like(μ)
    μ0 = μ[...,0]; μ1 = μ[...,1]; λ0 = λ[...,0]; λ1 = λ[...,1]
    if μ.shape[-1]==2:
        μdiff = np.maximum(ε,μ1-cond_amplification_threshold*μ0)
        λ0[:] = α+(1.-α)*np.exp(-γ/μdiff**2)
        λ1[:] = np.maximum(α,λ0/cond_max)
    elif μ.shape[-1]==3:
        μ2 = μ[...,2]; λ2 = λ[...,2]
        μdiff = np.maximum(ε,μ2-cond_amplification_threshold*μ0)
        λ0[:] = α+(1.-α)*np.exp(-γ/μdiff**2)
        λmin = λ0/cond_max # Minimum diffusivity
        μdiff = np.maximum(ε,μ2-cond_amplification_threshold*μ1)
        λ1[:] = np.maximum(α+(1.-α)*np.exp(-γ/μdiff**2),λmin)
        λ2[:] = np.maximum(α,λmin)
    else: raise ValueError("Unsupported dimension")
    return λ

def struct2ced(S,**kwargs):
    """
    Structure tensor to coherence enhancing diffusion tensor (Geometry last)
    - S : array of shape (n1,...,nk,d,d) where d = 2,3
    - kwargs : passed to coherence_enhancing_diffusion_eigvals
    returns : the ced tensor, array of shape (n1,...,nk,d,d)
    """
    μ,eVec = np.linalg.eigh(S)
    λ = coherence_enhancing_diffusion_eigvals(μ,**kwargs)
    D = eVec @ (λ[...,None] * np.swapaxes(eVec,-1,-2))
    # The above matrix is mathematically expected to be symmetric. 
    # However, this is only up to roundoff errors, which can cause issues,
    # hence we symmetrize the result
    return (D+np.swapaxes(D,-1,-2))/2.

