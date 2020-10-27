import sys; sys.path.insert(0,"../..")

"""
This test attempts to run the seismic wave equation in 3D.
"""

from agd.Eikonal.HFM_CUDA.HookeWave import HookeWave
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd.Metrics.Seismic import Hooke
from agd import AutomaticDifferentiation as ad
from agd import Domain
from agd.Plotting import savefig,quiver; #savefig.dirName = 'Images/ElasticityDirichlet'
norm_infinity = ad.Optimization.norm_infinity

from agd.ExportedCode.Notebooks_Div.ElasticEnergy import ElasticEnergy
import numpy as np; 
import numpy as np; xp=np; allclose=np.allclose
import matplotlib.pyplot as plt
import time

xp,plt,quiver,allclose = map(ad.cupy_friendly,(xp,plt,quiver,allclose))

vdim = 3
mica,ρ = Hooke.mica 
crystal_material = (mica.rotate_by(0.5,(1,2,3)),ρ)
isotropic_material = (Hooke.from_Lame(1.,1.,vdim=vdim), 1.)
    
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
    aX,dx = xp.linspace(-radius,radius,int(50*radius),endpoint=False,retstep=True)
    X=ad.array(np.meshgrid(aX,aX,aX,indexing='ij'))
    dom = Domain.MockDirichlet(X.shape[1:],dx,padding=None) #Periodic domain (wrap instead of pad)
    return dom,X,dx

def gaussian(X,σ): return np.exp(-lp.dot_VV(X,X)/(2*σ**2)) # Non normalized gaussian with prescribed variance
def explosion(X):
    """Triggers a pressure wave in all directions, emulating an explosion"""
    X_ad = ad.Dense.identity(constant=X,shape_free=(3,))
    return -gaussian(X_ad,0.1).gradient()
def torsion(X):
    """Triggers a torsion wave in all directions, using a torsion-like initial momentum"""
    e0,e1,e2 = explosion(X) 
    return ad.array([-e1,e0,np.zeros_like(e2)]) # Some perpendicular vector

dom,X,dx = make_domain(0.5)

fourth_order=True

hw = HookeWave(X.shape[1:],periodic=True,
    traits={
    'compact_scheme_macro':False,
    'fourth_order_macro':fourth_order,
    })


# Initial conditions
#q0,p0 = explosion(X),np.zeros_like(X)
#q0,p0 = np.zeros_like(X),torsion(X)
q0,p0 = explosion(X),torsion(X)
hw.q,hw.p = q0,p0

# PDE marameters
hooke,ρ = material
hw.gridScale = dx
hw.metric = xp.eye(vdim)/ρ
print("starting hooke decomp..."); start = time.time()
hw.hooke = hooke.hooke
print("elapsed",time.time()-start)
hw.damping=0.

dt = CFL(dx,hooke,ρ)
hw.dt = dt
print("shape_o",hw.shape_o,type(hw.shape_o))
print("Self check : ",hw.check())

print(hw.weights[:,0,0,0])
print(hw.offsets[:,:,0,0,0])
#print(norm_infinity(hw.hooke[:,:,0,0,0]-hooke.hooke))
assert allclose(hw.hooke[:,:,0,0,0],hooke.hooke,atol=1e-3)
assert allclose(hw._firstorder[0,0,0,:,0,0,0],0.)
assert allclose(hw.metric[:,:,0,0,0], xp.eye(vdim)/ρ)
assert allclose(hw.q,q0)
assert allclose(hw.p,p0)

hw.AdvanceQ()
hw.AdvanceP()
hw.dt = -dt
hw.AdvanceP()
hw.AdvanceQ()

print(norm_infinity(hw.q-q0),norm_infinity(q0))
print(norm_infinity(hw.p-p0),norm_infinity(p0))
assert allclose(hw.q,q0,atol=1e-4)
assert allclose(hw.p,p0,atol=1e-4)

print(1/dt)
print("starting evolution"); start = time.time()
hw.Advance(dt,1000)
print(hw.q[:,0,0,0])
print("elapsed",time.time()-start)