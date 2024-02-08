# Code automatically exported from notebook Isotropic_GPU.ipynb in directory Notebooks_Repro
# Do not modify
from ... import AutomaticDifferentiation as ad
if ad.cupy_generic.cp is None: raise ad.DeliberateNotebookError('Cupy module required')
from ... import Eikonal
from ... import LinearParallel as lp
from ... import FiniteDifferences as fd
import agd.AutomaticDifferentiation.cupy_generic as cugen
norm_infinity = ad.Optimization.norm_infinity
Eikonal.dictIn.default_mode = 'gpu'

import cupy as cp
import numpy as np
import itertools
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%5.3g" % x))

cp = ad.functional.decorate_module_functions(cp,cugen.set_output_dtype32) # Use float32 and int32 types in place of float64 and int64
plt = ad.functional.decorate_module_functions(plt,cugen.cupy_get_args)

variants_basic = (
    [{},{"seedRadius":2.}], # Spread seed information ?
    [{},{'multiprecision':True}] # Reduce floating point roundoff errors
)

variants_ext = (
    [{},{"seedRadius":2.},{"factoringRadius":10.,'factoringPointChoice':'Key'}], # source factorization ?
    [{},{'multiprecision':True}] # Reduce floating point roundoff errors
)

variants_ext2 = (
    [{},{'order':2}], # second order scheme ?
    [{},{"seedRadius":2.},{"factoringRadius":10.,'factoringPointChoice':'Key'}], # source factorization ?
    [{},{'multiprecision':True}] # Reduce floating point roundoff errors
)

def RunCompare(gpuIn,check=True,check_ratio=0,variants=None,**kwargs):
    # Dispatch the common variants if requested
    if isinstance(variants,str): variants = {'basic':variants_basic,'ext':variants_ext,'ext2':variants_ext2}[variants]
    if variants:
        for variant in variants[0]:
            RunCompare(gpuIn,check=check,check_ratio=check_ratio,variants=variants[1:],**kwargs,**variant)
        return

    if kwargs: print("\n",f"--- Variant {kwargs} ---")

    # Run the CPU and GPU solvers
    gpuIn = gpuIn.copy(); gpuIn.update(kwargs)
    gpuOut = gpuIn.Run()
    if gpuIn.get('verbosity',1):  print(f"--- gpu done, turning to cpu ---")
    cpuOut = gpuIn.Run(join={'mode':'cpu_transfer'})
    
    # Print performance info
    fmTime = cpuOut['FMCPUTime']; stencilTime = cpuOut['StencilCPUTime']; 
    cpuTime = fmTime+stencilTime; gpuTime = gpuOut['solverGPUTime'];
    print(f"Solver time (s). GPU : {gpuTime}, CPU : {cpuTime}. Device acceleration : {cpuTime/gpuTime}")
    
    # Check consistency 
    cpuVals = cpuOut['values'].copy(); gpuVals = gpuOut['values'].get()
    # Inf is a legitimate value in the presence of e.g. obstacles
    commonInfs = np.logical_and(np.isinf(cpuVals),np.isinf(gpuVals)) 
    cpuVals[commonInfs]=0; gpuVals[commonInfs]=0
    print("Max |gpuValues-cpuValues| : ", norm_infinity(gpuVals-cpuVals))
    
    if check is True: assert np.allclose(gpuVals,cpuVals,atol=1e-5,rtol=1e-4)
    elif check is False: pass
    else: assert np.sum(np.abs(gpuVals-cpuVals)>check)<=check_ratio*gpuVals.size

    return gpuOut,cpuOut

