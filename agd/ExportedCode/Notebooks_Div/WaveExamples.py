# Code automatically exported from notebook WaveExamples.ipynb in directory Notebooks_Div
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ... import Domain
from agd.Metrics import Riemann
from agd.Metrics.Seismic import Hooke, Thomsen
from agd.ODE.hamiltonian import QuadraticHamiltonian
from ... import AutomaticDifferentiation as ad
from agd.Eikonal.HFM_CUDA import AnisotropicWave as aw
from agd.Interpolation import UniformGridInterpolation
from agd.Plotting import savefig,quiver,Tissot; #savefig.dirName = 'Images/ElasticityDirichlet'
norm = ad.Optimization.norm
mica,_ = Hooke.mica 

#import copy
import numpy as np; xp=np; allclose=np.allclose; π = np.pi
from matplotlib import pyplot as plt
from matplotlib.colors import AsinhNorm
from scipy.ndimage import gaussian_filter
from numpy.random import rand

if xp is np: # Use CPU sparse matrix solvers
    AcousticHamiltonian = aw.AcousticHamiltonian_Sparse
    ElasticHamiltonian = aw.ElasticHamiltonian_Sparse
else: # Use GPU kernels
    AcousticHamiltonian = aw.AcousticHamiltonian_Kernel
    ElasticHamiltonian = aw.ElasticHamiltonian_Kernel

def heaviside(x):
    """Smoothed heaviside function, with a transition over [-1,1]"""
    # Primitives of (1-x^2)^k : x - x**3/3, x - (2*x**3)/3 + x**5/5, x - x**3 + (3*x**5)/5 - x**7/7
    def p(x): return x - (2*x**3)/3 + x**5/5
    return 0.5+0.5*np.where(x<=-1,-1,np.where(x>=1,1,p(x)/p(1)))
    
def bump(x,r,δ):
    """A bump over the interval [-r,r], with a transition over [r-δ,r+δ]"""
    return heaviside((r-np.abs(x))/δ)

def make_domain(radius,vdim):
    """Produces the periodic domain [-radius,radius]^2, with 25 pixels per unit"""
    aX,dx = xp.linspace(-radius,radius,50*radius,endpoint=False,retstep=True)
    X=ad.array(np.meshgrid(*(aX,)*vdim,indexing='ij'))
    dom = Domain.MockDirichlet(X.shape[1:],dx,padding=None) #Periodic domain (wrap instead of pad)
    return dom,X,dx

def gaussian(X,σ=0.1,x0=None): 
    if x0 is not None: X = X - fd.as_field(x0,X.shape[1:],depth=1)
    return np.exp(-lp.dot_VV(X,X)/(2*σ**2)) # Non normalized gaussian with prescribed variance
def churp(X,*args,**kwargs):
    """
    Laplacian of Gaussian, often used as a source in acoustic wave equation.
    *args, **kwargs : passed to gaussian
    """
    vdim = len(X)
    X_ad = ad.Dense2.identity(constant=X,shape_free=(vdim,))
    g_ad = gaussian(X_ad,*args,**kwargs)
    return -sum(g_ad.hessian(i,i) for i in range(vdim))
def explosion(X,*args,**kwargs):
    """
    Triggers a pressure wave in all directions, emulating an explosion.
    *args, **kwargs : passed to gaussian
    """
    X_ad = ad.Dense.identity(constant=X,shape_free=(len(X),))
    return -gaussian(X_ad,*args,**kwargs).gradient()
def torsion(X,*args,**kwargs):
    """
    Triggers a torsion wave in all directions, using a torsion-like initial momentum (2D)
    *args, **kwargs : passed to gaussian
    """
    e0,e1 = explosion(X,*args,**kwargs) 
    return ad.array([-e1,e0]) # Perpendicular vector

def gram(x,y):
    assert x.ndim==y.ndim
    return np.sum(x[...,None,:]*y[...,:,None],axis=tuple(range(x.ndim-1)))
    
def fwd_grad(qf_fwd, pf_fwd, qh_fwd, ph_fwd,
             qf_grad,pf_grad,qh_grad,ph_grad):
    """Gradient of the objective functional obtained via forward AD"""
    return gram(qf_grad,qf_fwd.coef) + gram(pf_grad,pf_fwd.coef) + gram(qh_grad,qh_fwd.coef) + gram(ph_grad,ph_fwd.coef)

def rev_grad(q0_fwd, p0_fwd, H_fwd,
             q0_grad,p0_grad,H_rev):
    """Gradient of the objective functional obtained via reversed AD"""
    return gram(q0_grad,q0_fwd.coef) + gram(p0_grad,p0_fwd.coef) \
          + gram(H_rev.iρ.coef,H_fwd.iρ.coef) - gram(H_rev.weights.coef,H_fwd.weights.coef)

def sensitivity_to_ρ(H_rev): 
    return -(H_rev.iρ*H_rev.iρ.value**2).coef

def sensitivity_to_D(H_rev,offsets=None):
    from agd.Metrics import misc # import flatten_symmetric_matrix as fltsym
    if offsets is None:offsets = H_rev.offsets
    eet_flat = misc.flatten_symmetric_matrix(lp.outer_self(offsets)).astype(H_rev.weights.dtype)
    # Could be made more efficient and accurate, since the inverses are essentially known
    res = lp.solve_AV(lp.transpose(eet_flat)[...,None],H_rev.weights.coef)
    res = misc.expand_symmetric_matrix(res)/2 
    for i in range(len(res)): res[i,i]*=2 # Normalization accounts for duplication of off-diagonal coefficients
    return res

def sensitivity_to_M(H_rev): 
    return H_rev.M.coef
    
def sensitivity_to_C(H_rev): 
    m = H_rev.moffsets 
    vdim = len(m)
    # Take care of Voigt notation
    if vdim==1:   return sensitivity_to_D(H_rev,m[0])
    elif vdim==2: return sensitivity_to_D(H_rev,(m[0,0],m[1,1],m[0,1]))
    elif vdim==3: return sensitivity_to_D(H_rev,(m[0,0],m[1,1],m[2,2],m[1,2],m[0,2],m[0,1]))

def MakeRandomTensor(dim, shape=tuple(), relax=0.05):
    A = np.random.standard_normal( (dim,dim) + shape )
    D = lp.dot_AA(lp.transpose(A),A)
    identity = np.eye(dim).reshape((dim,dim)+(1,)*len(shape))
    return D+lp.trace(D)*relax*identity

def rand_ad(x,size_ad,sym=False):
    x_coef = rand(*x.shape,size_ad)
    if sym: x_coef = (x_coef+lp.transpose(x_coef))/2
    return ad.Dense.denseAD(x,x_coef)
    
def check_ad(vdim,nX,wavetype,order_x=2,order_t=2,bc='Neumann',niter=4,size_fwd=2,size_rev=2):
    """
    - size_fwd : number of independent symbolic perturbations of the inputs
    - size_rev : number of objective functions at the target
    """
    print(f"Testing {vdim=}, {nX=}, {wavetype=}, {order_x=}, {order_t=}, {niter=}, {bc=}, {size_fwd=}, {size_rev=}, ",end='')
    np.random.seed(42)
    size_ad = 2
    shape = (nX,)*vdim
    dt = 0.1
    dx = 0.5
    if wavetype=='Acoustic':
        Hamiltonian = aw.AcousticHamiltonian_Sparse
        ρ = 0.2+rand(*shape); ρ /= np.max(ρ)
        D = MakeRandomTensor(vdim,shape,0.2); D/=np.max(D)
	#	D[:] = np.eye(vdim).reshape((vdim,vdim)+(1,)*vdim); print("Alternative : D = Id")
        params = (ρ,D)
        q0,p0 = rand(*shape), rand(*shape)
    elif wavetype=='Elastic': 
        Hamiltonian = aw.ElasticHamiltonian_Sparse
	#	ρ = 0.2+rand(*shape); ρ /= np.max(ρ); M = ρ[None,None]
        M = MakeRandomTensor(vdim,shape,0.3); M /= np.max(M)
        symdim = (vdim*(vdim+1))//2
        C = MakeRandomTensor(symdim,shape,0.2); C/=np.max(C)
        params = (M,C)
        q0,p0 = rand(vdim,*shape),rand(vdim,*shape)
    else: raise ValueError("Unrecognized wave type")

    params_fwd = [rand_ad(x,size_ad,sym) for x,sym in zip(params,(False,True,True))]
    q0_fwd = rand_ad(q0,size_fwd); p0_fwd = rand_ad(p0,size_fwd)

    qh_ind = np.random.choice(q0.size,8,replace=False) # Fails if there are duplicates
    ph_ind = np.random.choice(p0.size,7,replace=False)
    
#    q0_fwd.coef*=0
#    p0_fwd.coef*=0
#    params_fwd[0].coef*=0
#    params_fwd[1].coef*=0
    
    H_fwd = Hamiltonian(*params_fwd,dx,bc=bc,order_x=order_x,save_weights=True)
    H_rev = Hamiltonian(*params    ,dx,bc=bc,order_x=order_x,rev_ad = size_rev)

    H_fwd.damp_p = rand(*shape); H_fwd.damp_q = rand(*shape)
    H_rev.damp_p = H_fwd.damp_p; H_rev.damp_q = H_fwd.damp_q
    
    qf_fwd,pf_fwd,qh_fwd,ph_fwd = H_fwd.seismogram(q0_fwd,p0_fwd,dt,niter,order_t,qh_ind,ph_ind)
    qf,pf,qh,ph,backprop = H_rev.seismogram_with_backprop(q0,p0,dt,niter,order_t,qh_ind,ph_ind)

    assert np.allclose(qf_fwd.value,qf)
    assert np.allclose(pf_fwd.value,pf)
    assert np.allclose(ad.remove_ad(qh_fwd),qh)
    assert np.allclose(ad.remove_ad(ph_fwd),ph)

    qf_grad,pf_grad,qh_grad,ph_grad = [rand(*x.shape,size_rev) for x in (qf,pf,qh,ph)]
#    qf_grad*=0
#    pf_grad*=0
#    qh_grad*=0
#    ph_grad*=0
    
    q0_grad,p0_grad = backprop(qf_grad,pf_grad,qh_grad,ph_grad)

    grad1 = fwd_grad(qf_fwd, pf_fwd, qh_fwd, ph_fwd, qf_grad, pf_grad, qh_grad, ph_grad)
    grad2 = rev_grad(q0_fwd, p0_fwd, H_fwd, q0_grad, p0_grad, H_rev)
    assert np.allclose(grad1,grad2)

    if wavetype=='Acoustic':
        ρ_ξ = sensitivity_to_ρ(H_rev)
        D_ξ = sensitivity_to_D(H_rev)
        ρ_fwd,D_fwd = params_fwd
        assert np.allclose(gram(ρ_ξ,ρ_fwd.coef), gram(H_rev.iρ.coef,H_fwd.iρ.coef))
        assert np.allclose(gram(D_ξ,D_fwd.coef), gram(H_rev.weights.coef,H_fwd.weights.coef))
    elif wavetype=='Elastic':
        M_ξ = sensitivity_to_M(H_rev)
        C_ξ = sensitivity_to_C(H_rev)
        M_fwd,C_fwd = params_fwd
        assert np.allclose(gram(M_ξ,M_fwd.coef), gram(H_rev.M.coef,H_fwd.M.coef))
        assert np.allclose(gram(C_ξ,C_fwd.coef), gram(H_rev.weights.coef,H_fwd.weights.coef))

    print("passed.")

    

def some_check_ad():
    check_ad(1,10,'Acoustic')
    check_ad(2,7,'Acoustic',order_t=4,bc='Neumann')
    check_ad(2,6,'Acoustic',order_t=4,bc='Dirichlet',niter=10)
    check_ad(3,5,'Acoustic',order_x=4)
    check_ad(1,9,'Elastic',order_t=4)
    check_ad(2,6,'Elastic',order_x=4,bc='Dirichlet')
    check_ad(2,8,'Elastic',order_x=2)
    check_ad(3,5,'Elastic',bc='Neumann') # Uses 6D Voronoi decomposition (may not be compiled)

def layer(l,X):
    """
    A smooth layer of equivalent width l (total width (3/2) l) 
    on the boundary of X, with values from 0 to 1."""
    w = 1.5*l
    dist2 = sum( np.maximum(0,w+np.min(x)-x)**2 + np.maximum(0,w+x-np.max(x))**2 for x in X)
    return bump(w-np.sqrt(dist2),l,l/2)

# Interpolation data
layer_heights = xp.array([ 
    [0.8,0.7,0.7,0.75,0.8],  # Height of top layer
    [0.5,0.5,0.55,0.6,0.6],  # Height of middle layer
    [0.3,0.2,0.35,0.45,0.35] # Height of bottom layer
])
layer_heights_X = np.linspace(-1,1,layer_heights.shape[1],endpoint=True)

# Thickness of the transition
layer_δ = 0.05

# Materials for the acoustic wave
layer_ρ = np.array((1.5,1,2,2.5))
layer_D = np.array(([[1**2,0],[0,1**2]],[[2**2,0],[0,1**2]],[[3**2,0],[0,2**2]],[[2**2,0],[0,1.5**2]]))

# Materials for the elastic wave
materials = [Hooke.from_ThomsenElastic(Thomsen.ThomsenData[key]) for key in 
             ("Pierre shale - 1", "Mesaverde (4946) immature sandstone", 
              "Mesaverde (5501) clayshale", "Mesaverde sandstone (1958)") ]
layer_C = [hk.extract_xz().hooke for (hk,ρ) in materials]
layer_C = np.array(layer_C)/np.max(layer_C) # normalization

def LayeredMedium(X,heights,δ, ρs,Ds):
    shape = X.shape[1:]
    heights_X = np.linspace(-1,1,heights.shape[1],endpoint=True)
    height_fun = UniformGridInterpolation(heights_X[None],heights,order=2)
    height_val = height_fun(X[0].reshape((-1,*shape))).reshape((len(heights),*shape))
    heav = heaviside( (X[1,None]-height_val)/δ )
    partition = ad.array((heav[0],*[(1-heav[i])*heav[i+1] for i in range(len(heav)-1)],1-heav[-1]))
    partition /= partition.sum(axis=0) # Ensure partition of unity
    ρ_ = sum(ϕ * np.expand_dims(ρ,axis=(-2,-1)) for ϕ,ρ in zip(partition,ρs))
    D_ = sum(ϕ * np.expand_dims(D,axis=(-2,-1)) for ϕ,D in zip(partition,Ds))
    return ρ_,D_

def decomp_OS(M):
    """
    Compute the orthogonal x symmetric decomposition 
    of a 2x2 matrix, in a way that is compatible with AD.
    """
    # Appears to be quite slow...
    S2 = lp.dot_AA(lp.transpose(M),M)
    a,b,c = S2[0,0],S2[0,1],S2[1,1]
    h,d = (a+c)/2, np.sqrt(((a-c)/2)**2+b**2)
    λ=h+d; μ = h-d # Eigenvalues of S^2
    eye = np.eye(2).reshape((2,2)+(1,)*a.ndim)
    S = (S2+np.sqrt(λ*μ)*eye)/(np.sqrt(λ)+np.sqrt(μ)) # Square root of matrix
    iS = lp.inverse(S,avoid_np_linalg_inv=True) # np.linalg.inv is incredibly slow
    return lp.dot_AA(M,iS),S

def grad(arr,X):
    """Numerical gradient. Assumes X is regularly spaced."""
    # There is a similar functionality in numpy, but it does not work with AD classes
    vdim = len(X)
    g = []
    for i in range(vdim):
        dx = X[(i,*(1,)*vdim)]-X[(i,*(0,)*vdim)]
        a = np.moveaxis(arr,i-vdim,0)
        da = np.concatenate(((a[1]-a[0])[None],(a[2:]-a[:-2])/2,(a[-1]-a[-2])[None]),axis=0)/dx
        g.append(np.moveaxis(da,0,i-vdim))
    return ad.array(g)                 

def DeformedLayeredMedium(ϕ,X,*args,grad_ad=True,**kwargs):
    """
    Transforms a layered medium according to provided diffeomorphism ϕ.
    (Material is rotated, but not stretched.)
    """
    vdim = len(X)
    if grad_ad: # Use automatic differentiation to compute grad ϕ
        X_ad = ad.Dense.identity(constant=X,shape_free=(vdim,))
        ϕ_ad = ϕ(X_ad)
        ϕ_val = ϕ_ad.value
        ϕ_grad = ϕ_ad.gradient()
    else: # 
        ϕ_val = ϕ(X)
        ϕ_grad = grad(ϕ_val,X)
        
    O,_ = decomp_OS(lp.transpose(ϕ_grad))
    ρ,C = LayeredMedium(ϕ_val,*args,**kwargs)
    if len(C)==vdim: C = Riemann(C).inv_transform(O).m
    else: C = Hooke(C).inv_transform(O).hooke
    return ρ,C

def TopographicTransform(heights):
    """Vertical shift, interpolated according to data"""
    heights_X = np.linspace(-1,1,len(heights),endpoint=True)
    height_fun = UniformGridInterpolation(heights_X[None],heights,order=2)
    def ϕ(X): 
        X = ad.array(X)
        return ad.array((X[0],X[1]-height_fun(X[None,0],interior=True)))
    return ϕ

topo_heights = 0.7*np.array([0, 0.3, 0.1, 0, -0.2, -0.1]) # vertical shifts to interpolate

def inclusion(support0,medium0,medium1):
    """
    Weighted average of two media,
    using support0 as the weight function of medium0
    """
    ρ0,D0 = medium0
    ρ1,D1 = medium1
    support1 = 1-support0
    ρ0,D0,ρ1,D1 = [fd.as_field(e,support0.shape) for e in (ρ0,D0,ρ1,D1)]
    return ρ0*support0+ρ1*support1, D0*support0+D1*support1   

inc_ρ = 1.2
inc_D = np.eye(2)
inc_radius = 0.25
inc_center = [-0.5,0.3]

shift_θ = π/3
shift_amplitude = 0.25
shift_origin    = np.array([0.4,0.6])

def make_medium(X,
    # Layered medium
    layer_heights = layer_heights,
    layer_ρM = layer_ρ, # Acoustic : ρ. Elastic : M.
    layer_DC = layer_D, # Acoustic D. Elastic C. 
    layer_δ = layer_δ,

    # Deformation
    topo_heights = topo_heights,
    
    # Inclusion
    inc_ρM = inc_ρ,
    inc_DC = inc_D,
    inc_radius = inc_radius,
    inc_center = inc_center,
    
    # Shift
    shift_θ = shift_θ,
    shift_amplitude = shift_amplitude,
    shift_origin = shift_origin,
    grad_ad=True
):
    shape = X.shape[1:]
    ϕ = TopographicTransform(topo_heights)
    layer_medium = DeformedLayeredMedium(ϕ,X,layer_heights,layer_δ,layer_ρM,layer_DC,grad_ad=grad_ad)
    
    inc_support = heaviside( (inc_radius - norm( X-fd.as_field(inc_center,shape),axis=0))/layer_δ)
    inc_layer_medium = inclusion(inc_support,(inc_ρM,inc_DC),layer_medium)

    s_o,s_d = [fd.as_field(e,shape) for e in (shift_origin,(np.cos(shift_θ),np.sin(shift_θ)))]
    shift_medium = DeformedLayeredMedium(ϕ,X-s_d*shift_amplitude,layer_heights,
                                         layer_δ,layer_ρM,layer_DC,grad_ad=grad_ad)
    shift_support = heaviside(lp.det((X-s_o,s_d))/layer_δ)
    
    final_medium = inclusion(shift_support,shift_medium,inc_layer_medium)
    return final_medium

def damping_layer(X,k=None,ω=None,α=2,r=500,c=None,l=None):
    """
    Produce a damping (absorbing) layer on the boundary of X.
    - k : wave number. (Alternatively, provide the wavelength l.)
    - ω : pulsation of the signal (Alternatively, provide the speed c.)
    - α : relative thickness of the boundary layer transition, typical 1.5 <= α <= 3
    - r : desired amplitude reduction
    """
    if k is None: k=2*π/l # Wave number, inferred from wavelength
    if ω is None: ω = c*k # Pulsation, inferred from speed and wave number
    
    l_damp = α*np.log(r)/k # Damping width
    ω_damp = ω/α # Damping strength
    return ω_damp * layer(l_damp,X)

def mk_obj(qf_ref,pf_ref,qh_ref,ph_ref,  qf_w,pf_w,qh_w,ph_w,  σ):
    """Construct a basic objective functional and its gradient"""
    def obj(qf,pf,qh,ph): 
        return sum([w* gf2(x-x_ref,σ) for w,x,x_ref in 
              ((qf_w,qf,qf_ref), (pf_w,pf,pf_ref), (qh_w,qh,qh_ref), (ph_w,ph,ph_ref))])
        
    def grad(qf,pf,qh,ph):
        # We add one final singleton dimension. Backprop allows several gradients to be concatenated along the last axis.
        return [w* gf(x-x_ref,σ)[...,None] for w,x,x_ref in 
            ((qf_w,qf,qf_ref), (pf_w,pf,pf_ref), (qh_w,qh,qh_ref), (ph_w,ph,ph_ref))]
    return obj,grad

inc_C = Hooke.from_Lame(1,1).with_speed(1/2).hooke

