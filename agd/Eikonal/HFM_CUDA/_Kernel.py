# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
import os
from collections import OrderedDict
import copy
import itertools


from . import kernel_traits
from . import misc
from .cupy_module_helper import SetModuleConstant,GetModule
from . import cupy_module_helper
from ... import AutomaticDifferentiation as ad

"""
This file implements some member functions of the Interface class, related with the 
eikonal cuda kernel.
"""

def SetKernelTraits(self):
	"""
	Set the traits of the eikonal kernel.
	"""
	if self.verbosity>=1: print("Setting the kernel traits.")
	eikonal = self.kernel_data['eikonal']
	policy = eikonal.policy

	traits = kernel_traits.default_traits(self)
	traits.update(self.GetValue('traits',default=traits,
		help="Optional trait parameters for the eikonal kernel."))
	eikonal.traits = traits

	policy.values_float64 = self.GetValue('values_float64',default=False,
		help="Export values using the float64 data type")

	policy.multiprecision = policy.values_float64 or self.GetValue('multiprecision',
		default=False, help="Use multiprecision arithmetic, to improve accuracy") 
	
	traits['multiprecision_macro']=policy.multiprecision
	if policy.multiprecision: 
		traits['strict_iter_o_macro']=True
		traits['strict_iter_i_macro']=True

	self.factoringRadius = self.GetValue('factoringRadius',default=0,
		help="Use source factorization, to improve accuracy")
	if self.factoringRadius: traits['factor_macro']=True

	order = self.GetValue('order',default=1,
		help="Use second order scheme to improve accuracy")
	if order not in {1,2}: raise ValueError(f"Unsupported scheme order {order}")
	if order==2: traits['order2_macro']=True
	self.order=order

	traits['ndim_macro'] = self.ndim
	if self.model_ == 'Rander':
		traits['drift_macro']=1
	if self.model == 'ReedsSheppGPU3':
		traits['forward_macro'] = self.GetValue('forward',default=False,
			help="Use the Reeds-Shepp forward model")

	policy.bound_active_blocks = self.GetValue('bound_active_blocks',default=False,
		help="Limit the number of active blocks in the front. " 
		"Admissible values : (False,True, or positive integer)")
	if policy.bound_active_blocks:
		traits['minChg_freeze_macro']=True
		traits['pruning_macro']=True

	if self.HasValue('fim_front_width') or self.fim_front_width_default:policy.solver='FIM'
	else: policy.solver = 'AGSI'
	policy.solver = self.GetValue('solver',policy.solver,
		help="Choice of fixed point solver (AGSI, global_iteration)")
	solverAltNames={'AGSI':'adaptive_gauss_siedel_iteration','FIM':'fast_iterative_method'}
	policy.solver = solverAltNames.get(policy.solver,policy.solver)

	if policy.solver=='global_iteration' and traits.get('pruning_macro',False):
		raise ValueError("Incompatible options found for global_iteration solver "
			"(bound_active_blocks, pruning)")
	if policy.solver=='fast_iterative_method': traits['fim_macro']=True

	policy.strict_iter_o = traits.get('strict_iter_o_macro',0)
	self.float_t  = np.dtype(traits['Scalar'] ).type
	self.int_t    = np.dtype(traits['Int']    ).type
	self.offset_t = np.dtype(traits['OffsetT']).type
	self.shape_i = traits['shape_i']
	self.size_i = np.prod(self.shape_i)
	self.caster = lambda x : cp.asarray(x,dtype=self.float_t)
	self.nscheme = kernel_traits.nscheme(self)
#	assert self.float_t == self.hfmIn.float_t # Not satisfied with gpu_transfer
	self.hasChart = self.HasValue('chart_mapping')

def SetKernel(self):
	"""
	Setup the eikonal kernel, and (partly) the flow kernel
	"""
	if self.verbosity>=1: print("Preparing the GPU kernel")
	eikonal,geodesic,flow,scheme = [self.kernel_data[key] 
		for key in ('eikonal','geodesic','flow','scheme')]

	# ---- Produce a first kernel, for solving the eikonal equation ----
	# Set a few last traits
	policy = eikonal.policy
	traits = eikonal.traits
	traits.update({
		'import_scheme_macro':self.precompute_scheme,
		'local_i_macro':True, # threads work on a common block of solution
		'periodic_macro': bool(np.any(self.periodic)),
		'isotropic_macro': self.model_=='Isotropic', # Isotropic/diagonal switch
		'walls_macro': 'wallDist' in eikonal.args,
		})
	if traits['periodic_macro']: traits['periodic_axes']=self.periodic
	policy.count_updates = self.GetValue('count_updates',default=False,
		help='Count the number of times each block is updated')

	integral_max = policy.multiprecision # Int_Max needed for multiprecision to avoid overflow
	eikonal.source = cupy_module_helper.traits_header(traits,
		join=True,size_of_shape=True,log2_size=True,integral_max=integral_max)+"\n"

	if self.isCurvature: 
		model_source = f'#include "{self.model}.h"\n'
	else: 
		model = self.model_ # Dimension generic
		if model == 'Diagonal': model = 'Isotropic' # Same file handles both
		elif   model in ('Rander','SubRiemann'): model = 'Riemann' # Rander = Riemann + drift
		model_source = f'#include "{model}_.h"\n' 
	
	self.cuda_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"cuda")
	date_modified = cupy_module_helper.getmtime_max(self.cuda_path)
	self.cuda_date_modified = f"// Date cuda code last modified : {date_modified}\n"
	self.cuoptions = ("-default-device", f"-I {self.cuda_path}",
		) + self.GetValue('cuoptions',default=tuple(),
		help="Options passed via cupy.RawKernel to the cuda compiler")

	eikonal.source += model_source+self.cuda_date_modified
	eikonal.module = GetModule(eikonal.source,self.cuoptions)

	nofront_traits = { # For the other kernels, disables the front related options
		**eikonal.traits,
		'pruning_macro':False,
		'fim_macro':False,
		'minChg_freeze_macro':False,
		'niter_i':1,
	}

	# ---- Produce a kernel for computing the geodesics ----
	if self.hasTips:
		geodesic.policy.online_flow=self.GetValue('geodesic_online_flow',default=False,
			help="Compute the flow online when extracting geodesics (saves memory)")
		online_flow = geodesic.policy.online_flow

		# Get step, and related stopping criteria (compile time specified these are array lengths)
		geodesic.policy.step = self.GetValue('geodesic_step',default=0.25,
			help='Step size, in pixels, for the geodesic ODE solver')
		eucl_delay = int(self.GetValue('geodesic_PastSeed_delay',
			default=np.sqrt(self.ndim)/geodesic.policy.step,
			help="Delay, in iterations, for the 'PastSeed' stopping criterion of the "
			"geodesic ODE solver")) # Likely in curvature penalized models
		nymin_delay = int(self.GetValue('geodesic_Stationnary_delay',
			default=8.*np.sqrt(self.ndim)/geodesic.policy.step,
			help="Delay, in iterations, for the 'Stationnary' stopping criterion of the "
			"geodesic ODE solver")) # Rather unlikely

		geodesic.policy.eucl_t = np.uint8
		geodesic.policy.eucl_integral,geodesic.policy.eucl_max,geodesic.policy.eucl_chart \
			= misc.integral_largest_nextlargest(geodesic.policy.eucl_t)


		geodesic.traits = { # Suggested defaults
			**nofront_traits,
			'eucl_delay':eucl_delay,
			'nymin_delay':nymin_delay,
			'EuclT':geodesic.policy.eucl_t,
			'EuclT_Chart':geodesic.policy.eucl_chart,
			'online_flow_macro':online_flow,
			'chart_macro':self.hasChart,
			'flow_vector_macro':True,
			'local_i_macro':False, # thread block does Not work on common solution block
			}
		geodesic.traits.update(self.GetValue('geodesic_traits',default=geodesic.traits,
			help='Traits for the geodesic backtracking kernel') )

		if self.hasChart: 
			mapping = self.kernel_data['chart'].args['mapping']
			geodesic.traits['ndim_s']=mapping.ndim-1
			chart_jump_deviation = self.GetValue('chart_jump_deviation',
				default=np.inf,array_float=tuple(),
				help="Do not interpolate the jump coordinates, among pixel corners, "
				" if their (adimensionized) standard deviation exceeds this threshold. "
				"(Use if chart_mapping is discontinuous. Typical value : 5.) ")
			if chart_jump_deviation is True: chart_jump_deviation=5
			geodesic.policy.chart_jump_variance = chart_jump_deviation**2
			geodesic.traits['chart_jump_variance_macro'] = chart_jump_deviation<np.inf

		geodesic.source = cupy_module_helper.traits_header(geodesic.traits,
			join=True,size_of_shape=True,log2_size=True,integral_max=True) + "\n"
		if online_flow: geodesic.source += model_source
		geodesic.source += '#include "GeodesicODE.h"\n'+self.cuda_date_modified
		geodesic.module = cupy_module_helper.GetModule(geodesic.source,self.cuoptions)
		if self.hasChart: 
			SetModuleConstant(geodesic.module,'size_s',mapping.size/len(mapping),self.int_t)
			if geodesic.traits['chart_jump_variance_macro']: 
				SetModuleConstant(geodesic.module,'chart_jump_variance',chart_jump_variance,self.float_t)

	else: # No geodesic tips
		online_flow=True # Dummy value

	# ---- Produce a kernel for computing the geodesic flow ---
	self.flow_needed = (self.forwardAD or self.reverseAD or self.exportGeodesicFlow 
		or not online_flow)
	geodesic_outofline = None if online_flow else geodesic

	if self.flow_needed:
		flow.traits = {
			**nofront_traits
		}
		flow.policy = copy.copy(eikonal.policy) 
		flow.policy.nitermax_o = 1
		flow.policy.solver = 'global_iteration'

		if self.forwardAD or self.reverseAD:
			for key in ('flow_weights','flow_weightsum','flow_indices'): 
				flow.traits[key+"_macro"]=True
		if self.hasTips: 
			for key in ('flow_vector','flow_weightsum'): 
				flow.traits[key+"_macro"]=True
		if self.exportGeodesicFlow: flow.traits['flow_vector_macro']=True
		if self.model_=='Rander' and (self.forwardAD or self.reverseAD): 
			flow.traits['flow_vector_macro']=True

		flow.source = cupy_module_helper.traits_header(flow.traits,
			join=True,size_of_shape=True,log2_size=True,integral_max=True) + "\n"
		flow.source += model_source+self.cuda_date_modified
		flow.module = GetModule(flow.source,self.cuoptions)


	# ---- Produce a kernel for precomputing the stencils (if requested) ----
	if self.precompute_scheme:
		scheme.traits = {
			**nofront_traits,
			'import_scheme_macro':False,
			'export_scheme_macro':True,
			}
		for key in ('strict_iter_o_macro','multiprecision_macro',
			'walls_macro','minChg_freeze_macro'):
			scheme.traits.pop(key,None)

		scheme.source = cupy_module_helper.traits_header(scheme.traits,
		join=True,size_of_shape=True,log2_size=True,integral_max=integral_max) + "\n"
		scheme.source += model_source+self.cuda_date_modified
		scheme.module = GetModule(scheme.source,self.cuoptions)

	# ------- Set the constants of the cuda modules -------
	def SetCst(*args,exclude=None,include=None):
		datas = [eikonal,flow,scheme]
		if online_flow: datas.append(geodesic) # included by default iff online flow
		if exclude is not None:  datas.remove(exclude)
		if include is not None and include not in datas: datas.append(include)
		datas = [data for data in datas if data.module is not None]
		for kernel_data in datas: SetModuleConstant(kernel_data.module,*args)

	float_t,int_t = self.float_t,self.int_t

	self.size_o = np.prod(self.shape_o)
	SetCst('shape_o',self.shape_o,int_t, include=geodesic)
	SetCst('size_o', self.size_o, int_t)

	self.size_tot = self.size_o * np.prod(self.shape_i)
	SetCst('shape_tot',self.shape,   int_t, include=geodesic) # Used for periodicity
	SetCst('size_tot', self.size_tot,int_t, include=geodesic) # Used for geom indexing

	shape_geom_i,shape_geom_o = [s[self.geom_indep:] for s in (self.shape_i,self.shape_o)]
	if self.geom_indep: # Geometry only depends on a subset of coordinates
		size_geom_i,size_geom_o = [np.prod(s,dtype=int) for s in (shape_geom_i,shape_geom_o)]
		for key,value in [('size_geom_i',size_geom_i),('size_geom_o',size_geom_o),
			('size_geom_tot',size_geom_i*size_geom_o)]: 
			SetCst(key,value,int_t)
	else: SetCst('size_geom_tot', self.size_tot,int_t)

	if policy.multiprecision:
		SetCst('multip_step',self.multip_step, float_t,exclude=scheme) 
		SetCst('multip_max', self.multip_max, float_t, exclude=scheme)

	if self.factoringRadius: # Single seed only
		SetCst('factor_origin', self.seed,              float_t) 
		SetCst('factor_radius2',self.factoringRadius**2,float_t)
		factor_metric = ad.remove_ad(self.factor_metric.to_HFM())
		# The drift part of a Rander metric can be ignored for factorization purposes 
		if self.model_=='Rander': factor_metric = factor_metric[:-self.ndim]
		elif self.model_ in ('Isotropic','Diagonal'): factor_metric = factor_metric**2 
		SetCst('factor_metric',factor_metric,float_t)

	if self.order==2:
		order2_threshold = self.GetValue('order2_threshold',0.3,
			help="Relative threshold on second order differences / first order difference,"
			"beyond which the second order scheme deactivates")
		SetCst('order2_threshold',order2_threshold,float_t)		
	
	if self.model_ =='Isotropic':
		SetCst('weights', self.h**-2, float_t)
	if self.isCurvature:
		eps = self.GetValue('eps',default=0.1,array_float=tuple(),
			help='Relaxation parameter for the curvature penalized models')
		SetCst('decomp_v_relax',eps**2,float_t)

		if self.ndim_phys==2:
			nTheta = self.shape[2]
			theta = self.hfmIn.Axes()[2]

			if traits['xi_var_macro']==0:    SetCst('ixi',  self.ixi,  float_t) # ixi = 1/xi
			if traits['kappa_var_macro']==0: SetCst('kappa',self.kappa,float_t)
			if traits['theta_var_macro']==0: 
				SetCst('cosTheta_s',np.cos(theta),float_t)
				SetCst('sinTheta_s',np.sin(theta),float_t)
				
		elif self.ndim_phys==3:
			SetCst('sphere_proj_h',self.h_per,float_t)
			SetCst('sphere_proj_r',self.sphere_radius,float_t)
			if traits['sphere_macro']: 
				SetCst('sphere_proj_sep_r',self.separation_radius,float_t)

	if self.precompute_scheme:
		nactx = self.nscheme['nactx']
		# Convention 'geometry last turns' out to be much faster than the contrary.
		weights=cp.zeros((*shape_geom_o,*shape_geom_i,nactx),float_t)
		offsets=cp.zeros((*shape_geom_o,*shape_geom_i,nactx,self.ndim),self.offset_t)

		updateList_o = cp.arange(np.prod(shape_geom_o,dtype=int_t),dtype=int_t)
		dummy = cp.array(0,dtype=float_t) #; weights[0,0]=1; offsets[0,0,0]=2
		scheme.kernel = scheme.module.get_function("Update")
		# args : u_t,geom_t,seeds_t,rhs_t,..,..,..,updateNext_o
		args=(dummy,eikonal.args['geom'],dummy,dummy,weights,offsets,updateList_o,dummy)
		scheme.kernel((updateList_o.size,),(self.size_i,),args)

		eikonal.args['weights']=weights
		eikonal.args['offsets']=offsets

	# Set the kernel arguments
	policy.nitermax_o = self.GetValue('nitermax_o',default=2000,
		help="Maximum number of iterations of the solver")
	self.raiseOnNonConvergence = self.GetValue('raiseOnNonConvergence',default=True,
		help="Raise an exception if a solver fails to converge")
	if eikonal.policy.solver == 'fast_iterative_method':
		fim_front_width = self.GetValue('fim_front_width',
			default=self.fim_front_width_default,
			help="Dictates the max front width in the FIM variant.\n"
			"(original FIM : 2. Must be >=2.)")
		if fim_front_width is None: 
			raise ValueError("Please specify an fim_front_width (integer >=2)")
		SetModuleConstant(eikonal.module,'fim_front_width',fim_front_width,np.uint8)

	# Sort the kernel arguments
	args = eikonal.args
	argnames = ('values','valuesq','valuesNext','valuesqNext',
		'geom','seedTags','rhs','wallDist','weights','offsets')
	eikonal.args = OrderedDict([(key,args[key]) for key in argnames if key in args])
	flow.args = eikonal.args.copy() # Further arguments added later