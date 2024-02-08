# Code automatically exported from notebook Prox_MinCut.ipynb in directory Notebooks_Div
# Do not modify
from ... import AutomaticDifferentiation as ad
import numpy as np; xp=np
from agd.Eikonal.HFM_CUDA import MinCut
from agd.ODE import proximal
from scipy.ndimage import gaussian_filter

def make_Gs(metric):
    """Implementation of the constraint G(η) the characteristic function of the constraint
    F_x^*(η(x)) <= 1 for all x, and its dual and prox"""
    proj = metric.make_proj_dual()
    def Gs(η): return 0. # Gs:=G^* will only be evaluated inside its domain
    def G(σ):  return metric.norm(σ).sum()
    def prox_Gs(η,τ=1.): return proj(η)
    return Gs,G,prox_Gs

def make_F(g):
    def F(ϕ):  return np.sum(g*ϕ) # Only evaluated inside its domain
    def Fs(ψ): return np.sum(np.abs(ψ-g))
    def prox_F(ϕ,τ): return np.maximum(-1.,np.minimum(1.,ϕ-τ*g))
    return F,Fs,prox_F

def norm2_grad(dx,gradname):
    """Squared norm of the discrete gradient operator"""
    assert np.ndim(dx)==1 # One grid scale per axis
    res = 4*(dx**-2).sum()
    return (res/2) if gradname=='grad2' else res

def drop(arr,n,axis=0): 
    """
    Drop the first or last elements of an array, along the chosen axis.
    - n : number of elements to drop. If n<0 then drop from the end.
    - axis : index of the axis along which to drop.
    """
    sl = slice(n,None) if n>=0 else slice(0,n)
    if axis<0: axis += arr.ndim
    return arr[(slice(None),)*axis+(sl,)]

def grad1(ϕ,dx): return (drop(ϕ,1)-drop(ϕ,-1))[None]/dx

def gradb(ϕ,dx): return ad.array([drop(drop(ϕ, 1,i)-drop(ϕ,-1,i),-1,1-i)/dxi for i,dxi in enumerate(dx)])
def gradt(ϕ,dx): return ad.array([drop(drop(ϕ, 1,i)-drop(ϕ,-1,i), 1,1-i)/dxi for i,dxi in enumerate(dx)])

def gradc(ϕ,dx): return 0.5*(gradb(ϕ,dx)+gradt(ϕ,dx))

def grad2(ϕ,dx): return 0.5*np.stack([gradb(ϕ,dx), gradt(ϕ,dx)],axis=1)

def grad_ndiv_operators(grad,dx,shape):
    """
    Gradient and negative divergence operators, sparse matrix based implementation.
    (Turns a programmatic implementation of a linear function into a sparse matrix implementation,
    which respects the shapes of the inputs and outputs. Also transposes.)
    Input : 
    - grad : gradient implementation, usually finite differences based.
    - dx : grid scale.
    - shape : shape of the potential domain.
    Output : (grad, ndiv) functions
    """
    ϕ_ad = ad.Sparse.identity(shape)
    grad_ad = ad.array(grad(ϕ_ad,dx))
    shape_grad = grad_ad.shape
    grad_ = grad_ad.tangent_operator(ϕ_ad.size)
    ndiv_ = grad_ad.adjoint_operator(ϕ_ad.size)
    def grad(ϕ): return (grad_*ϕ.reshape(-1)).reshape(shape_grad)
    def ndiv(η): return (ndiv_*η.reshape(-1)).reshape(shape)
    grad.T=ndiv; ndiv.T=grad
    return grad, ndiv

def mincut_cpu(g,metric,dx,gradname='gradb',τ_F=0.2,ρ_overrelax=1.8,**kwargs):
    if len(dx)==1: gradname='grad1'
    grad,_ = grad_ndiv_operators({'grad1':grad1,'gradb':gradb,'gradt':gradt,'gradc':gradc,'grad2':grad2}[gradname],dx,g.shape)
    ϕ0 = np.zeros(g.shape)
    τ_Gs = 1./(τ_F*norm2_grad(dx,gradname))    
    res = proximal.chambolle_pock(make_F(g),make_Gs(metric),τ_F,τ_Gs,grad,ϕ0,ρ_overrelax=ρ_overrelax,**kwargs)
    return {**res,'ϕ':res['x'],'η':res['y']}

def mincut_gpu(*args,**kwargs): return ad.cupy_generic.cupy_get(MinCut.mincut(*args,**kwargs),iterables=(dict,))

def structure_tensor(g,noise_scale=3.,feature_scale=4.): # scales in pixels
    # Interpolate on the staggered grid, used for geometric data
    g = (g[:-1,:-1]+g[1:,:-1]+g[:-1,1:]+g[1:,1:])/4.
    
    # Smoothed gradient
    gx = gaussian_filter(g,noise_scale,[1,0])
    gy = gaussian_filter(g,noise_scale,[0,1])
    
    # Structure tensor
    gxx = gaussian_filter(gx**2,feature_scale)
    gxy = gaussian_filter(gx*gy,feature_scale)
    gyy = gaussian_filter(gy**2,feature_scale)
    
    # Similarly smoothed gradient (for Randers metrics)
    gx = gaussian_filter(gx,feature_scale)
    gy = gaussian_filter(gy,feature_scale)

    return (gx,gy),(gxx,gxy,gyy)

