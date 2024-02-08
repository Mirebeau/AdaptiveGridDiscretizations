# Code automatically exported from notebook ElasticEnergy.ipynb in directory Notebooks_Div
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from agd.Metrics.Seismic import Hooke
from ... import AutomaticDifferentiation as ad
from ... import Domain
from agd.Plotting import savefig; #savefig.dirName = 'Images/ElasticityDirichlet'

norm_infinity = ad.Optimization.norm_infinity
norm_average = ad.Optimization.norm_average
mica,_ = Hooke.mica # Hooke tensor associated to this crystal

def ElasticEnergy(v,hooke,dom,order=1):
    """
    Finite differences approximation of c(ϵ,ϵ), where c is the Hooke tensor and ϵ the stress tensor,
    which is twice the (linearized) elastic energy density.
    """
    assert len(v)==2
    coefs,moffsets = hooke.Selling()
    dvp = tuple( dom.DiffUpwind(v[i], moffsets[i], order=order) for i in range(2))
    dvm = tuple(-dom.DiffUpwind(v[i],-moffsets[i], order=order) for i in range(2))
    
    # Consistent approximations of Tr(moffset*grad(v))
    dv  = ad.array((dvp[0]+dvp[1], dvp[0]+dvm[1], dvm[0]+dvp[1], dvm[0]+dvm[1]))
    dv2 = 0.25* (dv**2).sum(axis=0)
    
    coefs = fd.as_field(coefs,v.shape[1:])
    return (coefs*dv2).sum(axis=0) 

