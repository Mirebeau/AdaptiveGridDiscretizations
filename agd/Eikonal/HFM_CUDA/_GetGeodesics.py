import numpy as np
import cupy as cp
from . import cupy_module_helper
from . import inf_convolution
from ...AutomaticDifferentiation import cupy_support as cps
from ... import AutomaticDifferentiation as ad
from ... import FiniteDifferences as fd


def GetGeodesics(self):
	if not self.hasTips: return
	if self.tips is not None: tips = self.hfmIn.PointFromIndex(self.tips,to=True)
	
	if self.isCurvature and self.tips_Unoriented is not None:
		assert self.ndim_phys==2 # TODO : ReedsShepp3
		tipsU = self.tips_Unoriented
		tipsU = self.hfmIn.OrientedPoints(tipsU)
		tipsU = self.hfmIn.PointFromIndex(tipsU,to=True)
		tipIndicesU = np.round(tipsU).astype(int)
		values = ad.remove_ad(self.values).astype(self.float_t)
		valuesU = values[tuple(np.moveaxis(tipIndicesU,-1,0))]
		amin = np.argmin(valuesU,axis=0) # Select most favorable value
		amin = amin.reshape((1,*amin.shape,1))
		amin = np.broadcast_to(amin,(*amin.shape[:-1],3))
		tipsU = np.squeeze(cps.take_along_axis(tipIndicesU,amin,axis=0),axis=0)
		tips = np.concatenate((tips,tipsU)) if self.tips is not None else tipsU

	geodesic = self.kernel_data['geodesic'] 
	eikonal = self.kernel_data['eikonal']

	# Set the module constants (Other constants were set in SetKernel.)
	def SetCst(*args):
		cupy_module_helper.SetModuleConstant(geodesic.module,*args)
#	SetCst('shape_i',self.shape_i,self.int_t)
#	SetCst('size_i', self.size_i, self.int_t)
	shape_tot = self.shape
	SetCst('geodesicStep',geodesic.policy.step,self.float_t)
	typical_len = int(max(40,0.5*np.max(shape_tot)/geodesic.policy.step))
	typical_len = self.GetValue('geodesic_typical_length',default=typical_len,
		help="Typical expected length of geodesics (number of points).")
	# Typical geodesic length is max_len for the GPU solver, which computes just a part
	SetCst('max_len', typical_len, self.int_t) 
	causalityTolerance = self.GetValue('geodesic_causalityTolerance',default=4.,
		help="Used in criterion for rejecting points in flow interpolation")
	SetCst('causalityTolerance', causalityTolerance, self.float_t)
	nGeodesics=len(tips)

	# Prepare the euclidean distance to seed estimate (for stopping criterion)
	eucl_t,eucl_integral,eucl_max,eucl_chart = geodesic.policy.eucl_t, \
		geodesic.policy.eucl_integral, geodesic.policy.eucl_max,geodesic.policy.eucl_chart
	eucl_bound_default = 12 if self.isCurvature else 6
	eucl_bound = self.GetValue('geodesic_targetTolerance',default=eucl_bound_default,
		help="Tolerance, in pixels, for declaring a seed as reached.")
	# Note: self.seedTags includes the walls, which we do not want here, hence trigger
	seeds = eikonal.trigger
	eucl = np.full_like(seeds,eucl_max,dtype=eucl_t)
	eucl[seeds] = 0
	eucl_mult = 5 if eucl_integral else 1
	eucl_kernel = inf_convolution.distance_kernel(radius=1,ndim=self.ndim,
		dtype=eucl_t,mult=eucl_mult)
	eucl = inf_convolution.inf_convolution(eucl,eucl_kernel,periodic=self.periodic,
		upper_saturation=eucl_max,overwrite=True,niter=int(np.ceil(eucl_bound)))
	eucl[eucl>eucl_mult*eucl_bound] = eucl_max
	if self.hasChart: 
		eucl[eucl==eucl_chart]=eucl_max
		chart_jump = self.GetValue('chart_jump',help="Where the geodesics should jump "
			"to another local chart of the manifold")
		chart_jump = np.broadcast_to(chart_jump,eucl.shape)
		eucl[chart_jump] = eucl_chart # Set special key for jump 
		
	eucl = fd.block_expand(eucl,self.shape_i,mode='constant',constant_values=eucl_max)
	eucl = cp.ascontiguousarray(eucl)
	geodesic.args['eucl'] = eucl

	# Run the geodesic ODE solver
	stopping_criterion = list(("Stopping criterion",)*nGeodesics)
	corresp = list(range(nGeodesics))
	geodesics = [ [tip.reshape(1,-1)] for tip in tips]

	block_size=self.GetValue('geodesic_block_size',default=32,
		help="Block size for the GPU based geodesic solver")
	geodesic_termination_codes = [
		'Continue', 'AtSeed', 'InWall', 'Stationnary', 'PastSeed', 'VanishingFlow']

	max_len = int(max(40,20*np.max(shape_tot)/geodesic.policy.step))
	max_len = self.GetValue("geodesic_max_length",default=max_len,
		help="Maximum allowed length of geodesics.")
	
	# Prepare the cuda kernel and arguments
	if geodesic.policy.online_flow: flow_argnames = [(key,'eikonal') 
		for key in ('geom','seedTags','rhs','wallDist','weights','offsets')]
	else: flow_argnames = [('flow_vector','flow'),('flow_weightsum','flow')]

	argnames = [('values','eikonal'),('valuesq','eikonal')] + flow_argnames + \
		[('eucl','geodesic'),('mapping','chart')]
	args = []
	for key,ker in argnames:
		ker_args = self.kernel_data[ker].args 
		if key in ker_args: args.append(ker_args[key])

	args = tuple(cp.ascontiguousarray(arg) for arg in args)
	kernel = geodesic.module.get_function('GeodesicODE')
	SetCst('pastseed_delay',self.GetValue('pastseed_delay',default=0,
		help="Number of geodesic backtracking steps before checking "
		"the PastSeed criterion (use if seed and tip are close)"),self.int_t)

	geoIt=0; geoMaxIt = int(np.ceil(max_len/typical_len))
	while len(corresp)>0:
		if geoIt>=geoMaxIt: 
			self.Warn("Geodesic solver failed to converge, or geodesic has too many points"
				" (in latter case, try setting 'geodesic_max_len':np.inf)")
			break
		geoIt+=1
		nGeo = len(corresp)
		x_s = cp.full( (nGeo,typical_len,self.ndim), np.nan, self.float_t)
		x_s[:,0,:] = np.stack([geodesics[i][-1][-1,:] for i in corresp], axis=0)
		len_s = cp.full((nGeo,),-1,self.int_t)
		stop_s = cp.full((nGeo,),-1,np.int8)

		nBlocks = int(np.ceil(nGeo/block_size))

		SetCst('nGeodesics', nGeo, self.int_t)
		kernel( (nBlocks,),(block_size,),args + (x_s,len_s,stop_s))
		corresp_next = []
		for i,x,l,stop in zip(corresp,x_s,len_s,stop_s): 
			geodesics[i].append(x[1:int(l)])
			if stop!=0: stopping_criterion[i] = geodesic_termination_codes[int(stop)]
			else: corresp_next.append(i)
		corresp=corresp_next
		SetCst('pastseed_delay',0,self.int_t)

	geodesics_cat = [np.concatenate(geo,axis=0) for geo in geodesics]
	geodesics = [self.hfmIn.PointFromIndex(geo).T for geo in geodesics_cat]
	if self.tips is not None: 
		self.hfmOut['geodesics']=geodesics[:len(self.tips)]
	if self.isCurvature and self.tips_Unoriented is not None:
		self.hfmOut['geodesics_Unoriented']=geodesics[-len(self.tips_Unoriented):]
	self.hfmOut['geodesic_stopping_criteria'] = stopping_criterion
