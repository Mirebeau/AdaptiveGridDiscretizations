# Code automatically exported from notebook DubinsState.ipynb in directory Notebooks_FMM
# Do not modify
import numpy as np; xp = np
from matplotlib import pyplot as plt

def control_source(activeNeighs,ncontrols,control_default=np.nan,source_default=np.nan):
    """
    Returns the optimal control, or the source of the optimal jump, at each point reached 
    by the front.
    Input : 
     - activeNeighs : produced by the eikonal solver, with option 'exportActiveNeighs':True
     - ncontrols : number of controls of the model (needed in case of several states)
     - control default : when no control is used (jump to another state, or stationnary point)
     - jump default : when no jump is done (following a control vector, or stationnary point)
    Output : 
     - control : the index of the control used
     - source (only if several states) : the index of the source state of the jump
    """
    nstates = activeNeighs.shape[-1]
    ndim = activeNeighs.ndim
    decompdim = (ndim*(ndim-1))//2
    active = activeNeighs%(2**decompdim)
    stationnary = (active==0) # No control used. Seeds and non reachable points (e.g. inside walls)
    control = np.where(stationnary,control_default,activeNeighs//(2**decompdim)) 
    assert np.all(control[~stationnary]<ncontrols+(nstates>1))
    if nstates==1: return control # Model with a single state
    jump = (~stationnary) & (control==ncontrols) # Points where the optimal option is to jump to a different state
    source = np.log2(active).round().astype(int) # source of the jump
    source = np.where(jump,source + (source>=xp.arange(nstates)), source_default)
    return np.where(control==ncontrols,control_default,control),source

