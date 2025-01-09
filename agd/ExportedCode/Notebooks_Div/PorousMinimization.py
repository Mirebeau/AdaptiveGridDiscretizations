# Code automatically exported from notebook PorousMinimization.ipynb in directory Notebooks_Div
# Do not modify
from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd
from agd.ODE import proximal
norminf = ad.Optimization.norm_infinity

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import lsqr as sparse_lsqr
from numpy import fft
π = np.pi

try: 
    import cupy as cp
    from cupy import get_array_module
except ImportError: 
    cp = None
    get_array_module = lambda _:np

def perspective(x):
    η = x[0]; y = x[1:]; Ny2 = np.sum(y**2,axis=0)
    return np.where(η<=0,np.where(Ny2==0,0,np.inf),Ny2/(2*η))

def prox_perspective(τ,x,niter=12,verb=1,cupy_kernel=True):
    xp = get_array_module(x)
    if xp is not np and cupy_kernel: return prox_perspective_cupy(τ,x,niter) # GPU kernel
    η = x[0]; y = x[1:]
    Ny=np.linalg.norm(y,axis=0)
    to_origin = τ*η+Ny**2/2<=0 # Wether prox is attained at the origin
    s = np.maximum( (2*Ny/τ)**(1/3), np.maximum(0,-2*(η+τ)/τ)**(1/2) ) # Over estimate the root
    for i in range(niter): # Fixed number of Newton iterations, without damping
        s -= (τ*s**3 + 2*(η+τ)*s - 2*Ny)/(3*τ*s**2+2*(η+τ)) 
    sol = xp.concatenate(((η+τ*s**2/2)[None],np.where(Ny==0,0,1-τ*s/Ny)*y),axis=0)
    residue = np.where(to_origin,0,τ*s**3 + 2*(η+τ)*s - 2*Ny)
    if verb>=0 and norminf(residue)>1e-4:
        print("Should be zero if Newton perspective converged :",norminf(residue),"at",np.argmax(np.abs(residue))) 
    return np.where(to_origin,np.zeros_like(x),sol)

if cp is not None:
    prox_perspective_kernel = cp.RawKernel(r'''
extern "C" __global__
void kernel(float tau, const float * eta_in, float * y_in, 
float * eta_out, float * y_out, unsigned int size, unsigned int niter){
const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
if(tid>=size){return;}
const float eta = eta_in[tid], y = y_in[tid];
if(tau*eta+y*y/2<=0){eta_out[tid]=0; y_out[tid]=0; return;} // to_origin
const float et2 = 2*(eta+tau), Ny = abs(y);
float s = max( pow(2*Ny/tau,float(1./3)), sqrt(max(float(0),-et2/tau)) ); 
for(int i=0; i<niter; ++i){ // Using a fixed number of iterations
    const float s2=s*s, s3=s*s2;
    s -= (tau*s3 + et2*s - 2*Ny) / (3*tau*s2+et2);
}
eta_out[tid] = eta+tau*s*s/2;
y_out[tid] = y==0 ? 0 : (1-tau*s/Ny)*y;
}''',"kernel")

def prox_perspective_cupy(τ,x_in,niter=12):
    assert x_in.flags['C_CONTIGUOUS'] # Never forget these two lines ! Will save hours of debugging.
    assert x_in.dtype==np.float32 
    
    x_out = np.empty_like(x_in)
    size = x_in[0].size
    blocksize = 1024
    gridsize = np.ceil(size/blocksize).astype(int)
    prox_perspective_kernel((gridsize,),(blocksize,),(np.float32(τ),x_in[0],x_in[1],x_out[0],x_out[1],size,niter))
    return x_out

