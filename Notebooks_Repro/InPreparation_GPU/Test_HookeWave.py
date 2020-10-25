import sys; sys.path.insert(0,"../..")

""" 
This test reproduces the contents of notebook ElasticWave, replacing the sparse matrix 
approach with custom cuda kernels and a slightly different discretization, intended to 
be less memory intensive e.g. for three dimensional fourth order test cases.
"""

from agd.Eikonal.HFM_CUDA import HookeWave
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

xp,plt,quiver,allclose = map(ad.cupy_friendly,(xp,plt,quiver,allclose))

mica,ρ = Hooke.mica 
crystal_material = (mica.extract_xz().rotate_by(0.5),ρ)
isotropic_material = (Hooke.from_Lame(1.,1.), 1.)
vdim = 2

crystal_material,isotropic_material = map(ad.cupy_friendly,(crystal_material,isotropic_material))

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

hw = HookeWave(X.shape[1:])

# Initial conditions
hw.q = explosion(X)
hw.p = np.zeros_like(X)

# PDE marameters
hooke,ρ = isotropic_material
hw.metric = ρ*xp.eye(vdim)
hw.hooke = hooke.hooke
hw.periodic = (True,)*vdim
hw.damping=0.

hw.dt = CFL(dx,hooke,ρ)
