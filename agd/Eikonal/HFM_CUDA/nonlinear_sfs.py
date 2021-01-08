import cupy as cp
import numpy as np
from . import cupy_module_helper

def ShapeFromShading(rhs,mask,u0,params,niter=100,traits=None):
	float_t = np.float32
	int_t = np.int32
	boolatom_t = np.uint8
	assert len(params)==4

	# Reshape data
	rhs = cp.asarray(rhs,dtype=float_t)
	mask = cp.asarray(mask,dtype=boolatom_t)
	u0 = cp.asarray(u0,dtype=float_t)
	shape = rhs.shape
	assert mask.shape==shape and u0.shape==shape

	traits_default = {'side_i':8,'niter_i':8}
	if traits is None: traits = traits_default
	else: traits = traits_default.update(traits)

	shape_i = (traits['side_i'],)*2
	rhs,mask,u0 = [fd.block_expand(e,shape_i) for e in (rhs,mask,u0)]

	# Find active blocks
	shape_o = rhs.shape[:2]
	update_o = np.nonzeros(np.any(mask,axis=(-2,-1))).astype(int_t)

	# Setup the cuda module
	cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(cuda_path)
	source = cupy_module_helper.traits_header(traits)

	source += ['#include "nonlinear_sfs.h"',
	f"// Date cuda code last modified : {date_modified}"]
	cuoptions = ("-default-device", f"-I {cuda_path}") 
	source = "\n".join(source)
	module = cupy_module_helper.GetModule(source,cuoptions)

	def SetCst(name,var,dtype):cupy_module_helper.SetModuleConstant(module,name,var,dtype)
	SetCst('params',params,float_t)
	SetCst('shape_o',shape_o,int_t)
	SetCst('shape_tot',shape,int_t)

	sfs = module.GetFunction('sfs')

	# Call the kernel
	rhs,mask,u,active_o = [cp.ascontiguousarray(e) for e in (rhs,mask,u0,active_o)]
	for i in range(niter):
		sfs((len(active_o),),shape_i,u,rhs,mask,update_o)

	return fd.block_squeeze(u,shape)

	