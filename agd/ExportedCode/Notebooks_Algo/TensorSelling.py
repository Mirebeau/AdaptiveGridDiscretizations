# Code automatically exported from notebook TensorSelling.ipynb in directory Notebooks_Algo
# Do not modify
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from ... import Selling
from agd.Plotting import savefig; #savefig.dirName = 'Figures/TensorSelling'
from ... import AutomaticDifferentiation as ad

import numpy as np; xp = np; allclose = np.allclose
import matplotlib.pyplot as plt

def MakeRandomTensor(dim,shape = tuple(),relax=0.01):
    identity = fd.as_field(np.eye(dim),shape,depth=2)
    A = np.random.standard_normal( (dim,dim) + shape ) 
    M = lp.dot_AA(lp.transpose(A),A) + relax*identity
    return xp.asarray(M) # Convert to GPU array if needed

def Reconstruct(coefs,offsets):
     return (coefs*lp.outer_self(offsets)).sum(2)

