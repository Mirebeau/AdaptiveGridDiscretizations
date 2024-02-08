# Code automatically exported from notebook MongeAmpere.ipynb in directory Notebooks_NonDiv
# Do not modify
from ... import Selling
from ... import Domain
from ... import LinearParallel as lp
from ... import AutomaticDifferentiation as ad
from agd.Plotting import savefig; #savefig.dirName = "Figures/MongeAmpere"

import numpy as np
from matplotlib import pyplot as plt

newton_root = ad.Optimization.newton_root
stop    = ad.Optimization.stop_default
damping = ad.Optimization.damping_default 
norm = ad.Optimization.norm

def SchemeNonMonotone(u,f,bc):
    # Compute the hessian matrix of u
    uxx = bc.Diff2(u,(1,0))
    uyy = bc.Diff2(u,(0,1))
    uxy = 0.25*(bc.Diff2(u,(1,1)) - bc.Diff2(u,(1,-1)))
    
    # Numerical scheme
    det = uxx*uyy-uxy**2
    residue = f - det
    
    # Boundary conditions
    return np.where(bc.interior,residue,u-bc.grid_values)

def MALBR_H(d2u):
    a,b,c = np.sort(np.maximum(0.,d2u), axis=0)

    # General formula, handling infinite values separately
    A,B,C = (np.where(e==np.inf,0.,e) for e in (a,b,c))
    result = 0.5*(A*B+B*C+C*A)-0.25*(A**2+B**2+C**2)
    
    pos_inf = np.logical_or.reduce(d2u==np.inf)    
    result[pos_inf]=np.inf
    
    pos_ineq = a+b<c
    result[pos_ineq] = (A*B)[pos_ineq]
        
    return result
    
def SchemeMALBR(u,SB,f,bc):
    # Compute the finite differences along the superbase directions
    d2u = bc.Diff2(u,SB)
    d2u[...,bc.not_interior] = 0. # Replace NaNs with arbitrary values to silence warnings
    
    # Numerical scheme
    residue = f-MALBR_H(d2u).min(axis=0)
    
    # Boundary conditions
    return np.where(bc.interior,residue,u-bc.grid_values)

def InvalidMALBR(u,SB,f,bc):
    residue = SchemeMALBR(u,SB,f,bc)
    return np.any(residue[bc.interior]>=f/2)

def SchemeMALBR_OptInner(u,SB,bc,oracle=None):
    # If the active superbases are known, then take only these
    if not(oracle is None):
        SB = np.take_along_axis(SB,np.broadcast_to(oracle,SB.shape[:2]+(1,)+oracle.shape),axis=2)
                
    d2u = bc.Diff2(u,SB)
    d2u[...,bc.not_interior] = 0. # Placeholder value to silent NaN warnings
    # Evaluate the complex non-linear function using dense - sparse composition
    result = ad.apply(MALBR_H,d2u,shape_bound=u.shape)
    
    return ad.min_argmin(result,axis=0)

def SchemeMALBR_Opt(u,SB,f,bc):
    
    # Evaluate using the envelope theorem
    result,_ = ad.apply(SchemeMALBR_OptInner, u,bc.as_field(SB),bc, envelope=True)
        
    # Boundary conditions
    return np.where(bc.interior, f - result, u-bc.grid_values)

def ConstrainedMaximize(Q,l,m):
    dim = l.shape[0]
    if dim==1:
        return (l[0]+np.sqrt(Q[0,0]))/m[0]
    
    # Discard infinite values, handled afterwards
    pos_bad = l.min(axis=0)==-np.inf
    L = l.copy(); L[:,pos_bad]=0
    
    # Solve the quadratic equation
    A = lp.inverse(Q)    
    lAl = lp.dot_VAV(L,A,L)
    lAm = lp.dot_VAV(L,A,m)
    mAm = lp.dot_VAV(m,A,m)
    
    delta = lAm**2 - (lAl-1.)*mAm
    pos_bad = np.logical_or(pos_bad,delta<=0)
    delta[pos_bad] = 1.
    
    mu = (lAm + np.sqrt(delta))/mAm
    
    # Check the positivity
#    v = dot_AV(A,mu*m-L)
    rm_ad = np.array
    v = lp.dot_AV(rm_ad(A),rm_ad(mu)*rm_ad(m)-rm_ad(L))
    pos_bad = np.logical_or(pos_bad,np.any(v<0,axis=0))
    
    result = mu
    result[pos_bad] = -np.inf
    
    # Solve the lower dimensional sub-problems
    # We could restrict to the bad positions, and avoid repeating computations
    for i in range(dim):             
        axes = np.full((dim),True); axes[i]=False
        res = ConstrainedMaximize(Q[axes][:,axes],l[axes],m[axes])
        result = np.maximum(result,res)
    return result

def SchemeUniform(u,SB,f,bc):
    # Compute the finite differences along the superbase directions
    d2u = bc.Diff2(u,SB) 
    d2u[...,bc.not_interior] = 0. # Placeholder value to silent NaN warnings
    
    # Generate the parameters for the low dimensional optimization problem
    Q = 0.5*np.array([[0,1,1],[1,0,1],[1,1,0]])
    l = -d2u
    m = lp.dot_VV(SB,SB)
    
    # Evaluate the numerical scheme
    m = bc.as_field(m)
    from agd.FiniteDifferences import as_field
    Q = as_field(Q,m.shape[1:])
    
    dim = 2
    alpha = dim * f**(1/dim)
    mask= (alpha==0)

    Q = Q* np.where(mask,1.,alpha**2)
    residue = ConstrainedMaximize(Q,l,m).max(axis=0)
    residue[mask] = np.max(l/m,axis=0).max(axis=0)[mask]
    
    # Boundary conditions
    return np.where(bc.interior,residue,u-bc.grid_values)

def SchemeUniform_OptInner(u,SB,f,bc,oracle=None):
    # Use the oracle, if available, to select the active superbases only
    if not(oracle is None):
        SB = np.take_along_axis(SB,np.broadcast_to(oracle,SB.shape[:2]+(1,)+oracle.shape),axis=2)

    d2u = bc.Diff2(u,SB) 
    d2u[...,bc.not_interior] = 0. # Placeholder value to silent NaN warnings
    
    # Generate the parameters for the low dimensional optimization problem
    Q = 0.5*np.array([[0,1,1],[1,0,1],[1,1,0]])
    dim = 2
    l = -d2u
    m = lp.dot_VV(SB,SB)
    
    m = bc.as_field(m)
    from agd.FiniteDifferences import as_field
    Q = as_field(Q,m.shape[1:])
    
    dim = 2
    alpha = dim * f**(1/dim)
    mask= (alpha==0)

    Q = Q* np.where(mask,1.,alpha**2)
    # Evaluate the non-linear functional using dense-sparse composition
    residue = ad.apply(ConstrainedMaximize,Q,l,m,shape_bound=u.shape).copy()
    residue[:,mask] = np.max(l/m,axis=0)[:,mask]
    
    return ad.max_argmax(residue,axis=0)

def SchemeUniform_Opt(u,SB,f,bc):
    
    # Evaluate the maximum over the superbases using the envelope theorem
    residue,_ = ad.apply(SchemeUniform_OptInner, u,bc.as_field(SB),f,bc, envelope=True)
    
    return np.where(bc.interior,residue,u-bc.grid_values)

def Hessian_ad(u,x):
    x_ad = ad.Dense2.identity(constant=x,shape_free=(2,))
    return u(x_ad).hessian()
def MongeAmpere_ad(u,x):
    return lp.det(Hessian_ad(u,x))

