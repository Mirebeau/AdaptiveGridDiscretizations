# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def default_traits(self):
	"""
	Default traits of the GPU implementation of an HFM model.
	(self is an instance of the class Interface from file interface.py)
	Side effect : sets the default FIM front width. (None : AGSI is fine.)
	"""
	traits = {
	'Scalar': np.float32,
	'Int':    np.int32,
	'OffsetT':np.int32,
#	'multiprecision_macro':False,
	'pruning_macro':False,
	'geom_first_macro':True,
	}

	ndim = self.ndim
	model = self.model

	if   model == 'ReedsShepp2': 
		traits.update({'shape_i':(4,4,4),'niter_i':6})
		fim_front_width = None
	elif model == 'ReedsSheppForward2':
		traits.update({'shape_i':(4,4,4),'niter_i':6})
		fim_front_width = 5
	elif model == 'Elastica2':
		# Small shape, single iteration, since stencils are too wide anyway
		traits.update({'shape_i':(4,4,2),'niter_i':1,
			'merge_sort_macro':True,'nFejer_macro':5})
		fim_front_width = 4
	elif model == 'Dubins2':
		traits.update({'shape_i':(4,4,2),'niter_i':1})
		#traits.update({'shape_i':(4,4,4),'niter_i':2}) # Similar, often slightly better
		fim_front_width = None				

	elif model == 'AsymmetricQuadratic2':
		traits.update({'shape_i':(8,8),'niter_i':10})
		fim_front_width = 6
	elif model == 'AsymmetricQuadratic3':
		traits.update({'shape_i':(4,4,4),'niter_i':4})
		fim_front_width = 5

	elif model == 'TTI2':
		traits.update({'shape_i':(8,8),'niter_i':10,'nmix_macro':7})
		fim_front_width = 6
	elif model == 'TTI3':
		traits.update({'shape_i':(4,4,4),'niter_i':3,'nmix_macro':7})
		fim_front_width = 6

	elif model == 'Rander2':
		traits.update({'shape_i':(8,8),'niter_i':12})
		fim_front_width = 8
	elif model == 'Rander3':
		traits.update({'shape_i':(4,4,4),'niter_i':8})
		fim_front_width = 8

	elif model == 'Riemann2':
		traits.update({'shape_i':(8,8),'niter_i':8})
		fim_front_width = 4
	elif model == 'Riemann3':
		traits.update({'shape_i':(4,4,4),'niter_i':4})
		fim_front_width = 6
	elif model in ('Riemann4','Rander4'):
		traits.update({'shape_i':(4,4,4,2),'niter_i':3})
		fim_front_width = None
	elif model in ('Riemann5','Rander5'):
		traits.update({'shape_i':(2,2,2,2,2),'niter_i':3}) # Untested
		fim_front_width = None
	elif model == ('Riemann6','Rander6'):
		traits.update({'shape_i':(2,2,2,2,2,2),'niter_i':2}) # Untested
		fim_front_width = None
		
	elif model in ('Isotropic2','Diagonal2'):
		#Alternative : Large shape, many iterations, to take advantage of block based causality 
#		traits.update({'shape_i':(16,16),'niter_i':32,})
#		traits.update({'shape_i':(24,24),'niter_i':48,}) # Slightly faster, but may fail with forwardAD : CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
#		fim_front_width = None
		# Alternative, more standard and reasonable shape
		#traits.update({'shape_i':(8,8),'niter_i':16,})
		traits.update({'shape_i':(16,16),'niter_i':16,})
		fim_front_width = 4
	elif model in ('Isotropic3','Diagonal3'):
		traits.update({'shape_i':(4,4,4),'niter_i':8,})
		fim_front_width = 4
	elif model in ('Isotropic4','Diagonal4'):
		traits.update({'shape_i':(4,4,4,4),'niter_i':8,}) # Untested
		fim_front_width = None
	elif model in ('Isotropic5','Diagonal5'):
		traits.update({'shape_i':(2,2,2,2,2),'niter_i':4,}) # Untested
		fim_front_width = None
	elif model in ('Isotropic6','Diagonal6'):
		traits.update({'shape_i':(2,2,2,2,2,2),'niter_i':4,}) # Untested
		fim_front_width = None

	# traits below have not been optimized
	elif model == 'DubinsState2':
		traits.update({'shape_i':(32,2),'niter_i':32}) 
		fim_front_width = 4
	elif model == 'DubinsState3':
		traits.update({'shape_i':(8,8,2),'niter_i':8})
		fim_front_width = 4
	elif model == 'DubinsState4':
		traits.update({'shape_i':(4,4,2,1),'niter_i':1})
		fim_front_width = 4

	elif model in ('Forward1','Custom1'):
		traits.update({'shape_i':(32,),'niter_i':32})
		fim_front_width = 4
	elif model in ('Forward2','Custom2'):
		traits.update({'shape_i':(8,8),'niter_i':8})
		fim_front_width = 4
	elif model in ('Forward3','Custom3'):
		traits.update({'shape_i':(4,4,2),'niter_i':2})
		fim_front_width = 4
	elif model in ('Forward4','Custom4'):
		traits.update({'shape_i':(4,2,2,2),'niter_i':2})
		fim_front_width = 4
	elif model in ('Forward5','Custom5'):
		traits.update({'shape_i':(2,2,2,2,2),'niter_i':2})
		fim_front_width = 4

	else:
		raise ValueError("Unsupported model")

	if model in ('ReedsSheppForward2','Elastica2','Dubins2'):
		traits['convex_curvature_macro']=self.convex
	if self.model_ == 'DubinsState': traits['nstates_macro']=self.hfmIn.shape[-1]
	if self.model_ in ('DubinsState','Custom'):
		traits['ncontrols_macro']=len(self.hfmIn['controls'])
		traits['controls_max_macro']=True

	self.fim_front_width_default = fim_front_width
	return traits

def voronoi_decompdim(ndim): 
	"""
	Number of offsets in Voronoi's decomposition of a symmetric positive definite matrix.
	"""
	return 12 if ndim==4 else (ndim*(ndim+1))//2 

def nscheme(self):
	"""
	Provides the structure of the finite difference scheme used.
	(number of symmmetric offsets, foward offsets, max or min of a number of schemes)
	"""
	ndim = self.ndim
	model = self.model_
	traits = self.kernel_data['eikonal'].traits
	decompdim = voronoi_decompdim(ndim)

	nsym=0 # Number of symmetric offsets
	nfwd=0 # Number of forward offsets
	nmix=1 # maximum or minimum of nmix schemes
	if model in ('Isotropic','Diagonal'):   nsym = ndim
	elif model in ('Riemann','Rander'):     nsym = decompdim

	elif model=='ReedsShepp':               nsym = decompdim
	elif model=='ReedsSheppForward':    
		if traits['convex_curvature_macro']:nfwd = 1+decompdim;
		else:                               nsym = 1; nfwd = decompdim
	elif model=='Dubins':                   nfwd = decompdim; nmix = 2
	elif model=='Elastica':                 nfwd = traits['nFejer_macro']*decompdim
	elif self.model=='ReedsSheppGPU3':
		if traits['forward_macro']:         nsym=2; nfwd=6
		else: nsym=2+6

	elif model=='AsymmetricQuadratic':  nsym = decompdim; nmix = 3
	elif model=='TTI':					nsym = decompdim; nmix = traits['nmix_macro']

	elif model=='DubinsState':
		if traits['controls_max_macro']:
			nfwd = voronoi_decompdim(ndim-1)  # One dimension is for the state
			nmix = traits['ncontrols_macro']+int(traits['nstates_macro']>1)
		else:
			nfwd = (voronoi_decompdim(ndim-1) * traits['ncontrols_macro'] 
				+ traits['nstates_macro']-1)
	elif model=='Custom':
		if traits['controls_max_macro']: nfwd = decompdim; nmix = traits['ncontrols_macro']
		else: nfwd = decompdim * traits['ncontrols_macro']
	elif model=="Forward":
		nfwd = voronoi_decompdim(ndim)

	else: raise ValueError('Unsupported model')

	nact = nsym+nfwd # max number of active offsets
	ntot = 2*nsym+nfwd
	nactx = nact*nmix
	ntotx = ntot*nmix



	return {'nsym':nsym,'nfwd':nfwd,'nmix':nmix,
	'nact':nact,'ntot':ntot,'nactx':nactx,'ntotx':ntotx}


