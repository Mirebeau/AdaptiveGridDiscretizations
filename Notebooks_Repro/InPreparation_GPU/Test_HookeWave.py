import sys; sys.path.insert(0,"../..")

""" 
This test reproduces the contents of notebook ElasticWave, replacing the sparse matrix 
approach with custom cuda kernels and a slightly different discretization, intended to 
be less memory intensive e.g. for three dimensional fourth order test cases.
"""

from agd.Eikonal.HFM_CUDA.HookeWave import HookeWave
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd.Metrics.Seismic import Hooke
from agd.ODE.hamiltonian import QuadraticHamiltonian
from agd import AutomaticDifferentiation as ad
from agd import Domain
from agd.Plotting import savefig,quiver; #savefig.dirName = 'Images/ElasticityDirichlet'
norm_infinity = ad.Optimization.norm_infinity

from agd.ExportedCode.Notebooks_Div.ElasticEnergy import ElasticEnergy
import numpy as np; 
import numpy as np; xp=np; allclose=np.allclose
import matplotlib.pyplot as plt

xp,plt,quiver,allclose = map(ad.cupy_friendly,(xp,plt,quiver,allclose))

vdim = 2
if vdim==2:
    mica,ρ = Hooke.mica 
    crystal_material = (mica.extract_xz().rotate_by(0.5),ρ)
    isotropic_material = (Hooke.from_Lame(1.,1.), 1.)
else:
    assert False
    
crystal_material,isotropic_material = [ad.cupy_generic.cupy_set(e,iterables=(tuple,Hooke))
    for e in (crystal_material,isotropic_material)]

material = crystal_material

from agd.Eikonal.HFM_CUDA.VoronoiDecomposition import VoronoiDecomposition
hooke = material[0].hooke
print("type of hooke ", type(hooke))
coefs,offsets = VoronoiDecomposition(hooke,offset_t=np.int8)
print("Decomposition")
print(coefs)
print(offsets)


def CFL(dx,hooke,ρ,order=1):
    """Largest time step guaranteed to be stable for the elastic wave equation"""
    tr = lp.trace(hooke.to_Mandel())
    L = (0,16,78)[order] # See elastic energy notebook
    return 2*dx / np.sqrt(L * ρ * tr)

def make_domain(radius):
    """Produces the periodic domain [-radius,radius]^2, with 25 pixels per unit"""
    aX,dx = xp.linspace(-radius,radius,50*radius,endpoint=False,retstep=True)
    X=ad.array(np.meshgrid(aX,aX,indexing='ij'))
    dom = Domain.MockDirichlet(X.shape[1:],dx,padding=None) #Periodic domain (wrap instead of pad)
    return dom,X,dx

def gaussian(X,σ): return np.exp(-lp.dot_VV(X,X)/(2*σ**2)) # Non normalized gaussian with prescribed variance
def explosion(X):
    """Triggers a pressure wave in all directions, emulating an explosion"""
    X_ad = ad.Dense.identity(constant=X,shape_free=(2,))
    return -gaussian(X_ad,0.1).gradient()
def torsion(X):
    """Triggers a torsion wave in all directions, using a torsion-like initial momentum"""
    e0,e1 = explosion(X) 
    return ad.array([-e1,e0]) # Perpendicular vector



dom,X,dx = make_domain(1)

fourth_order=False

hw = HookeWave(X.shape[1:],periodic=True,
    traits={
    'compact_scheme_macro':False,
    'fourth_order_macro':fourth_order,
    })

# Initial conditions
q0,p0 = explosion(X),np.zeros_like(X)
#q0,p0 = np.zeros_like(X),torsion(X)
hw.q,hw.p = q0,p0

# PDE marameters
hooke,ρ = material
hw.gridScale = dx
hw.metric = xp.eye(vdim)/ρ
hw.hooke = hooke.hooke
hw.damping=0.

hw.dt = CFL(dx,hooke,ρ)
print("shape_o",hw.shape_o,type(hw.shape_o))
print("Self check : ",hw.check())

print(hw.weights[:,0,0])
print(hw.offsets[:,:,0,0])
assert allclose(hw.hooke[:,:,0,0],hooke.hooke)
assert allclose(hw._firstorder[0,0,:,0,0],0.)
assert allclose(hw.metric[:,:,0,0], xp.eye(vdim)/ρ)
assert allclose(hw.q,q0)
assert allclose(hw.p,p0)

# --------- Reproducing the sparse matrix based implementation, for comparison --------
from agd.ExportedCode.Notebooks_Div.ElasticEnergy import ElasticEnergy

def KineticEnergy(m,ρ):
    """Squared norm of the momentum, divided by the density, 
    which is (twice) the kinetic energy density."""
    return (m**2).sum(axis=0) / ρ

def WaveHamiltonian(hooke,ρ,dom,order=1):
    """Returns the Hamiltonian of the linear elastic wave equation."""
    # Summation is implicit, and purposedly not done here (for simplify_ad)
    h = dom.gridscale
    Hq = lambda q: 0.5 * ElasticEnergy(q,hooke,dom,order=order) * h**2 
    Hp = lambda p: 0.5 * KineticEnergy(p,ρ) * h**2
    H = QuadraticHamiltonian(Hq,Hp,inv_inner=h**-2)
    z = xp.zeros((dom.vdim,*dom.shape)) # Correctly shaped placeholder for position or impulsion
    H.set_spmat(z) # Replaces quadratic functions with sparse matrices
    return H

WaveH = WaveHamiltonian(*material,dom,
    order=2 if fourth_order else 1)
incomp = WaveH.incomplete_schemes()
AdvanceQ_sp = incomp['Explicit-q']
AdvanceP_sp = incomp['Explicit-p']

print("Initial hamiltonian", WaveH.H(q0,p0))
flowQ,flowP = WaveH.flow(q0,p0)
print(np.sum(np.isnan(flowQ)))
print(np.sum(np.isnan(flowP)))
print(hw.dt)

# --------------- Checking AdvanceQ ---------------
print("Advancing Q")
hw.q,hw.p = q0,p0
q_sp = AdvanceQ_sp(q0,p0,hw.dt)
print(norm_infinity(q_sp))
print(np.sum(np.isnan(q_sp)))
hw.AdvanceQ()
q_ker = hw.q
print(norm_infinity(q_ker))
print(np.sum(np.isnan(q_ker)))
print(dx**2)

assert allclose(q_ker,q_sp)


# --------------- Checking AdvanceP --------------
print("Advancing P")
hw.q,hw.p = q0,p0
p_sp = AdvanceP_sp(q0,p0,hw.dt)
hw.AdvanceP()
p_ker = hw.p

print(p_ker[:,0,0])
print(p_sp[:,0,0])

plt.figure(figsize=(12,6)); 
plt.subplot(1,2,1)
plt.title('p_ker'); plt.axis('equal')
quiver(*X,*p_ker,subsampling=(2,2),scale=1000.)

plt.subplot(1,2,2)
plt.title('p_sp'); plt.axis('equal')
quiver(*X,*p_sp,subsampling=(2,2),scale=1000.)
plt.show()

print("p_sp norm:", norm_infinity(p_sp))
print("p_ker norm:",norm_infinity(p_ker))
print("diff norm:", norm_infinity(p_sp-p_ker))
assert allclose(p_ker,p_sp,atol=1e-4)


