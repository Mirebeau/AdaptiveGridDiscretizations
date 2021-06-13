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
	elif model == 'Riemann4':
		traits.update({'shape_i':(4,4,4,2),'niter_i':3})
		fim_front_width = None
	elif model == 'Riemann5':
		traits.update({'shape_i':(2,2,2,2,2),'niter_i':3}) # Untested
		fim_front_width = None
		
	elif model in ('Isotropic2','Diagonal2'):
		#Alternative : Large shape, many iterations, to take advantage of block based causality 
		traits.update({'shape_i':(16,16),'niter_i':32,})
#		traits.update({'shape_i':(24,24),'niter_i':48,}) # Slightly faster, but may fail with forwardAD : CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES
		fim_front_width = None
		# Alternative, more standard and reasonable shape
		#traits.update({'shape_i':(8,8),'niter_i':16,})
	elif model in ('Isotropic3','Diagonal3'):
		traits.update({'shape_i':(4,4,4),'niter_i':8,})
		fim_front_width = 4
	elif model in ('Isotropic4','Diagonal4'):
		traits.update({'shape_i':(4,4,4,4),'niter_i':8,}) # Untested
		fim_front_width = None
	elif model in ('Isotropic5','Diagonal5'):
		traits.update({'shape_i':(2,2,2,2,2),'niter_i':10,}) # Untested
		fim_front_width = None
	else:
		raise ValueError("Unsupported model")

	if model in ('ReedsSheppForward2','Elastica2','Dubins2'):
		traits['convex_curvature_macro']=False

	self.fim_front_width_default = fim_front_width
	return traits

def nscheme(self):
	"""
	Provides the structure of the finite difference scheme used.
	(number of symmmetric offsets, foward offsets, max or min of a number of schemes)
	"""
	ndim = self.ndim
	# Voronoi decomposition of a symmetric positive definite matrix
	decompdim = 12 if ndim==4 else (ndim*(ndim+1))//2 
	model = self.model_
	traits = self.kernel_data['eikonal'].traits

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

	else: raise ValueError('Unsupported model')

	nact = nsym+nfwd # max number of active offsets
	ntot = 2*nsym+nfwd
	nactx = nact*nmix
	ntotx = ntot*nmix



	return {'nsym':nsym,'nfwd':nfwd,'nmix':nmix,
	'nact':nact,'ntot':ntot,'nactx':nactx,'ntotx':ntotx}


