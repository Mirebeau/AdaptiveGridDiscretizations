# Code automatically exported from notebook NonlinearMonotoneSecond2D.ipynb in directory Notebooks_NonDiv
# Do not modify
from ... import Selling
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from ... import Domain
from agd.Plotting import savefig; #savefig.dirName = "Figures/NonlinearMonotoneSecond2D"

import numpy as np
import matplotlib.pyplot as plt

def SchemeNonMonotone(u,α,β,bc,sqrt_relax=1e-6):
    # Compute the hessian matrix of u
    uxx = bc.Diff2(u,(1,0))
    uyy = bc.Diff2(u,(0,1))
    uxy = 0.25*(bc.Diff2(u,(1,1)) - bc.Diff2(u,(1,-1)))
    
    # Compute the eigenvalues
    # The relaxation is here to tame the non-differentiability of the square root.
    htr = (uxx+uyy)/2. # Half trace
    Δ = ((uxx-uyy)/2.)**2 + uxy**2 # Discriminant of characteristic polynomial
    sΔ = np.sqrt( np.maximum( Δ, sqrt_relax) )

    λ_max = htr+sΔ
    λ_min = htr-sΔ
    
    # Numerical scheme
    residue = β - α*λ_max - λ_min
    
    # Boundary conditions
    return np.where(bc.interior,residue,u-bc.grid_values)

def SchemeSampling(u,diffs,β,bc):
    # Tensor decomposition 
    μ,e = Selling.Decomposition(diffs)
    
    # Numerical scheme 
    μ = bc.as_field(μ)
    residue = β - (μ*bc.Diff2(u,e)).sum(0).min(0)
    
    # Boundary conditions
    return np.where(bc.interior,residue,u-bc.grid_values)

def Diff(α,θ):
    e0 = np.array(( np.cos(θ),np.sin(θ)))
    e1 = np.array((-np.sin(θ),np.cos(θ)))
    if isinstance(α,np.ndarray): 
        e0,e1 = (as_field(e,α.shape) for e in (e0,e1))
    return α*lp.outer_self(e0) + lp.outer_self(e1)

def SchemeSampling_OptInner(u,diffs,bc,oracle=None):
    # Select the active tensors, if they are known
    if not(oracle is None):
        diffs = np.take_along_axis(diffs, np.broadcast_to(oracle,diffs.shape[:2]+(1,)+oracle.shape),axis=2)
    
    print("Has AD information :", ad.is_ad(u), ". Number active tensors per point :", diffs.shape[2])
    
    # Tensor decomposition 
    coefs,offsets = Selling.Decomposition(diffs)
    
    # Return the minimal value, and the minimizing index
    return ad.min_argmin( lp.dot_VV(coefs,bc.Diff2(u,offsets)), axis=0)

def SchemeSampling_Opt(u,diffs,β,bc):
    # Evaluate the operator using the envelope theorem
    result,_ = ad.apply(SchemeSampling_OptInner, u,bc.as_field(diffs),bc, envelope=True)
        
    # Boundary conditions
    return np.where(bc.interior, β-result, u-bc.grid_values)

def MakeD(α):
    return np.moveaxis(0.5*np.array([
        (α+1)*np.array([[1,0],[0,1]]),
        (α-1)*np.array([[1,0],[0,-1]]),
        (α-1)*np.array([[0,1],[1,0]])
    ]), 0,-1)

def NextAngleAndSuperbase(θ,sb,D):
    pairs = np.stack([(1,2), (2,0), (0,1)],axis=1)
    scals = lp.dot_VAV(np.expand_dims(sb[:,pairs[0]],axis=1), 
                       np.expand_dims(D,axis=-1), np.expand_dims(sb[:,pairs[1]],axis=1))
    ϕ = np.arctan2(scals[2],scals[1])
    cst = -scals[0]/np.sqrt(scals[1]**2+scals[2]**2)
    θ_max = np.pi*np.ones(3)
    mask = cst<1
    θ_max[mask] = (ϕ[mask]-np.arccos(cst[mask]))/2
    θ_max[θ_max<=0] += np.pi
    θ_max[θ_max<=θ] = np.pi
    k = np.argmin(θ_max)
    i,j = (k+1)%3,(k+2)%3
    return (θ_max[k],np.stack([sb[:,i],-sb[:,j],sb[:,j]-sb[:,i]],axis=1))

def AnglesAndSuperbases(D,maxiter=200):
    sb = Selling.CanonicalSuperbase(np.eye(2)).astype(int)
    θs=[]
    superbases=[]
    θ=0
    for i in range(maxiter):
        θs.append(θ)
        if(θ>=np.pi): break
        superbases.append(sb)
        θ,sb = NextAngleAndSuperbase(θ,sb,D)
    return np.array(θs), np.stack(superbases,axis=2)

def MinimizeTrace(u,α,bc,sqrt_relax=1e-16):
    # Compute the tensor decompositions
    D=MakeD(α)
    θ,sb = AnglesAndSuperbases(D)
    θ = np.array([θ[:-1],θ[1:]])
    
    # Compute the second order differences in the direction orthogonal to the superbase
    sb_rotated = np.array([-sb[1],sb[0]])
    d2u = bc.Diff2(u,sb_rotated)
    d2u[...,bc.not_interior]=0. # Placeholder values to silent NaNs
    
    # Compute the coefficients of the tensor decompositions
    sb1,sb2 = np.roll(sb,1,axis=1), np.roll(sb,2,axis=1)
    sb1,sb2 = (e.reshape( (2,3,1)+sb.shape[2:]) for e in (sb1,sb2))
    D = D.reshape((2,2,1,3,1)+D.shape[3:])
    # Axes of D are space,space,index of superbase element, index of D, index of superbase, and possibly shape of u
    scals = lp.dot_VAV(sb1,D,sb2)

    # Compute the coefficients of the trigonometric polynomial
    scals,θ = (bc.as_field(e) for e in (scals,θ))
    coefs = -lp.dot_VV(scals, np.expand_dims(d2u,axis=1))
    
    # Optimality condition for the trigonometric polynomial in the interior
    value = coefs[0] - np.sqrt(np.maximum(coefs[1]**2+coefs[2]**2,sqrt_relax))
    coefs_ = ad.remove_ad(coefs) # removed AD information
    angle = np.arctan2(-coefs_[2],-coefs_[1])/2.
    angle[angle<0]+=np.pi
    
    # Boundary conditions for the trigonometric polynomial minimization
    mask = np.logical_not(np.logical_and(θ[0]<=angle,angle<=θ[1]))
    t,c = θ[:,mask],coefs[:,mask]
    value[mask],amin_t = ad.min_argmin(c[0]+c[1]*np.cos(2*t)+c[2]*np.sin(2*t),axis=0)
        
    # Minimize over superbases
    value,amin_sb = ad.min_argmin(value,axis=0)
    
    # Record the optimal angles for future use
    angle[mask]=np.take_along_axis(t,np.expand_dims(amin_t,axis=0),axis=0).squeeze(axis=0) # Min over bc
    angle = np.take_along_axis(angle,np.expand_dims(amin_sb,axis=0),axis=0) # Min over superbases

    return value,angle

def SchemeConsistent(u,α,β,bc):
    value,_ = MinimizeTrace(u,α,bc)
    residue = β - value
    return np.where(bc.interior,residue,u-bc.grid_values)

def MinimizeTrace_Opt(u,α,bc,oracle=None):
    if oracle is None:  return MinimizeTrace(u,α,bc)
    
    # The oracle contains the optimal angles
    diffs=Diff(α,oracle.squeeze(axis=0))
    coefs,sb = Selling.Decomposition(diffs)
    value = lp.dot_VV(coefs,bc.Diff2(u,sb))
    return value,oracle
    

def SchemeConsistent_Opt(u,α,β,bc):
    value,_ = ad.apply(MinimizeTrace_Opt,u,α,bc,envelope=True)
    residue = β - value
    return np.where(bc.interior,residue,u-bc.grid_values)

def Pucci_ad(u,α,x):
    """
    Computes alpha*lambda_max(D^2 u) + lambda_min(D^2 u), 
    at the given set of points, by automatic differentiation.
    """
    x_ad = ad.Dense2.identity(constant=x,shape_free=(2,))
    hessian = u(x_ad).hessian()
    
    Δ = ((hessian[0,0]-hessian[1,1])/2.)**2 + hessian[0,1]**2
    sΔ = np.sqrt(Δ)
    mean = (hessian[0,0]+hessian[1,1])/2.
    λMin,λMax = mean-sΔ,mean+sΔ
    
    return λMin+α*λMax

