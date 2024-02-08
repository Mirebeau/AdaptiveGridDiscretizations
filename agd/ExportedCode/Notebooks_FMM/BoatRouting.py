# Code automatically exported from notebook BoatRouting.ipynb in directory Notebooks_FMM
# Do not modify
import sys; sys.path.insert(0,"..") # Allow import of agd from parent directory (useless if conda package installed)
#from Miscellaneous import TocTools; print(TocTools.displayTOC('BoatRouting','FMM'))

from ... import Eikonal
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
from agd.Metrics import Rander,Riemann
from ... import AutomaticDifferentiation as ad
from agd.Plotting import savefig,quiver; #savefig.dirName = 'Images/BoatRouting'

import numpy as np; xp = np
from copy import copy
import matplotlib.pyplot as plt

def route_min(z,params):
    z,μ,ω,M = fd.common_field((z,)+params, depths=(1,0,1,2))
    z_norm = np.sqrt(lp.dot_VAV(z,M,z))
    μω_norm = np.sqrt( 2*μ +lp.dot_VAV(ω,M,ω) )
    cost = z_norm*μω_norm - lp.dot_VAV(z,M,ω)
    time = z_norm / μω_norm
    fuel = cost/time
    rvel = z/time - ω
    return {
        'cost':cost, # minimal cost for this travel
        'time':time, # optimal travel time 
        'fuel':fuel, # instantaneous fuel consumption
        'rvel':rvel, # relative velocity, w.r.t current
    }

def metric(params):
    μ,ω,M = fd.common_field(params,depths=(0,1,2))
    return Rander( M*(2*μ + lp.dot_VAV(ω,M,ω)), -lp.dot_AV(M,ω))

def Spherical(θ,ϕ): 
    """Spherical embedding: θ is longitude, ϕ is latitude from equator toward pole"""
    return (np.cos(θ)*np.cos(ϕ), np.sin(θ)*np.cos(ϕ), np.sin(ϕ))

def IntrinsicMetric(Embedding,*X):
    """Riemannian metric for a manifold embedded in Euclidean space"""
    X_ad = ad.Dense.identity(constant=X,shape_free=(2,)) # First order dense AD variable
    Embed_ad = ad.asarray(Embedding(*X_ad)) # Differentiate the embedding
    Embed_grad = Embed_ad.gradient()
    Embed_M = lp.dot_AA(Embed_grad,lp.transpose(Embed_grad)) # Riemannian metric
    return Embed_M

def bump(x,y): 
    """Gaussian-like bump (not normalized)"""
    return np.exp(-(x**2+y**2)/2)

def Currents(θ,ϕ):
    """Some arbitrary vector field (water currents)"""
    bump0 = bump(θ+1,(ϕ+0.3)*2); ω0=(0,1) # intensity and direction of the currents
    bump1 = 2*bump(2*(θ-0.7),ϕ-0.2); ω1=(1,-1)
    bump0,ω0,bump1,ω1 = fd.common_field( (bump0,ω0,bump1,ω1), depths=(0,1,0,1))
    return bump0*ω0+bump1*ω1

def ArrivalTime(hfmIn,params):
    hfmIn = copy(hfmIn)
#    if hfmIn.xp is not np: hfmIn['solver']='AGSI' #TODO : why needed ?
    hfmIn['metric'] = metric(params)
    hfmIn['exportGeodesicFlow']=1
    cache = Eikonal.Cache(needsflow=True)
    hfmOut = hfmIn.Run(cache=cache)
    
    flow = hfmOut['flow']
    no_flow = np.all(flow==0,axis=0)
    flow[:,no_flow]=np.nan  # No flow at the seed point, avoid zero divide    
    route = route_min(flow,params)
    costVariation = route['time']
    costVariation[no_flow] = 0
    hfmIn['costVariation'] = np.expand_dims(costVariation,axis=-1)
    
    hfmOut2 = hfmIn.Run(cache=cache) # cache avoids some recomputations
    time = hfmOut2['values'].gradient(0)
    return time,hfmOut

