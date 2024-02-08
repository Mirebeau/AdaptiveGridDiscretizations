# Code automatically exported from notebook ElasticaVariants.ipynb in directory Notebooks_FMM
# Do not modify
from ... import Eikonal 
from ... import AutomaticDifferentiation as ad
from ... import Plotting
from ... import FiniteDifferences as fd
from agd.ExportedCode.Notebooks_FMM.DubinsState import control_source

import numpy as np
from matplotlib import pyplot as plt
π=np.pi

def elastica_control(ϕ): 
    """
    Control for the elastica model (two dimensional projection).
    Boundary point of the disk of radius 1/2 and center (1/2,0).
    """
    return np.cos(ϕ)*np.array((np.cos(ϕ),np.sin(ϕ)))

def elastica_controls(I,ϕmin=-π/2,ϕmax=π/2,pad=False):
    """
    Controls for the elastica model, regularly sampled.
    - I : number of controls
    - ϕmin, ϕmax : smallest and largest control
    - pad : add a zero control.
    """
    if ϕmin==-π/2: ϕmin = -π/2+π/I
    if ϕmax== π/2: ϕmax =  π/2-π/I
    ϕ=np.linspace(ϕmin,ϕmax,I)
    if pad: ϕ = np.pad(ϕ,(1,1),constant_values=π/2)
    return elastica_control(ϕ)

def with_prior(c,ξ=1,κ=0):
    """
    Adjusts the control vectors to account for the following priors : 
    - ξ the typical turning radius, 
    - κ the reference curvature.
    """
    ds,dθ = c
    return np.array((ds,(dθ/ξ+κ*ds)))

def embed(c,θ):
    """
    Embed a two dimensional control (α,β) in R^2xR as (α n(θ),β)
    where n(θ) = (cos θ, sin θ)
    """
    ds,dθ = c
    return np.array((ds*np.cos(θ),ds*np.sin(θ),dθ*np.ones_like(θ)))

fejer_weights_ = [
    [],[2.],[1.,1.],[0.444444, 1.11111, 0.444444],[0.264298, 0.735702, 0.735702, 0.264298],
    [0.167781, 0.525552, 0.613333, 0.525552, 0.167781],[.118661, 0.377778, 0.503561, 0.503561, 0.377778, 0.118661],
    [.0867162, 0.287831, 0.398242, 0.454422, 0.398242, 0.287831, 0.0867162],
    [.0669829, 0.222988, 0.324153, 0.385877, 0.385877, 0.324153, 0.222988, 0.0669829],
    [.0527366, 0.179189, 0.264037, 0.330845, 0.346384, 0.330845, 0.264037, 0.179189, 0.0527366]
]
def fejer_weights(I):
    """
    Returns the Fejer weights, for computing a cosine weighted integral.
    If I>=10, returns an approximation based on the midpoint rule.
    """
    if I<len(fejer_weights_): return fejer_weights_[I]
    x=np.linspace(0,π,I,endpoint=False)
    return np.cos(x)-np.cos(x+π/I)

def elastica_terms(I,ϕmin=-π/2,ϕmax=π/2):
    midpoints = Eikonal.CenteredLinspace(ϕmin,ϕmax,I)
    weights = np.array(fejer_weights(I))
    return np.array((np.cos(midpoints),np.sin(midpoints)))*np.sqrt((3/4)*weights)

def in_place_rotation_controls(): return np.array(((0,0,1),(0,0,-1)))
state_transition_base_cost = np.array(((0,1),(1,0))) # discrete cost of jumping from one state to another

def embed4(c,θ,κ):
    """
    Embed a two dimensional control (α,β) in R^2 x R x R as (α n(θ),α κ, β)
    where n(θ) = (cos θ, sin θ)
    """
    ds,dθ = c
    uθ,uκ = np.ones_like(θ),np.ones_like(κ)
    return np.array((ds*np.cos(θ)*uκ,ds*np.sin(θ)*uκ,ds*uθ*κ,dθ*uθ*uκ))

