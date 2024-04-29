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

import numpy as np; xp=np; allclose=np.allclose
import matplotlib.pyplot as plt
from copy import copy
import scipy.sparse; import scipy.sparse.linalg
import itertools

def ElasticEnergy(v,hooke,dom):
    """
    Finite differences approximation of c(ϵ,ϵ), where c is the Hooke tensor and ϵ the strain tensor,
    which is twice the (linearized) elastic energy density.
    """
    d=len(v)
    coefs,moffsets = hooke.Selling()
    dvp = tuple( dom.DiffUpwind(v[i], moffsets[i]) for i in range(d))
    dvm = tuple(-dom.DiffUpwind(v[i],-moffsets[i]) for i in range(d))
    
    # Consistent approximations of Tr(moffset*grad(v))
    dv = ad.array([sum(s) for s in itertools.product(*zip(dvp,dvm))])
    dv2 = (dv**2).sum(axis=0)
    
    coefs = fd.as_field(coefs,v.shape[1:]) * 2**(-d)
    return (coefs*dv2).sum(axis=0) 

