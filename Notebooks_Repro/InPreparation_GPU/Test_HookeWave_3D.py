import sys; sys.path.insert(0,"../..")

"""
This test attempts to run the seismic wave equation in 3D.
"""

from agd.Eikonal.HFM_CUDA.HookeWave import HookeWave
from agd import LinearParallel as lp
from agd import FiniteDifferences as fd
from agd.Metrics.Seismic import Hooke
from agd import AutomaticDifferentiation as ad
from agd.Plotting import savefig,quiver; #savefig.dirName = 'Images/ElasticityDirichlet'
norm_infinity = ad.Optimization.norm_infinity

from agd.ExportedCode.Notebooks_Div.ElasticEnergy import ElasticEnergy
import numpy as np; 
import numpy as np; xp=np; allclose=np.allclose
import matplotlib.pyplot as plt

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
    aX,dx = xp.linspace(-radius,radius,50*radius,endpoint=False,retstep=True)
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

