# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import cupy as cp
from collections import OrderedDict
from types import SimpleNamespace
import numbers

# Deferred implementation of Interface member functions
from . import _Kernel
from . import _solvers 
from . import _GetGeodesics
from . import _PostProcess
from . import _SetGeometry
from . import _SetArgs
from . import _StopCriterion

from ... import AutomaticDifferentiation as ad
from ... import Metrics
from ... import LinearParallel as lp

class Interface(object):
	"""
	This class carries out the RunGPU function work. n
	It should not be used directly.
	"""
	def __init__(self,hfmIn):

		self.hfmIn = hfmIn
		if hfmIn['arrayOrdering'] != 'RowMajor':
			raise ValueError("Only RowMajor indexing supported")

		# Needed for GetValue
		self.hfmOut = {'keys':{
		'used':['origin','arrayOrdering','dims','mode','projective'],
		'default':OrderedDict(),
		'visited':[],
		'help':OrderedDict(),
		'kernelStats':OrderedDict(),
		} }

		self.verbosity = 1
		self.clear_hfmIn = False

		self.verbosity = self.GetValue('verbosity',default=1,
			help="Choose the amount of detail displayed on the run")
		self.clear_hfmIn = self.GetValue('clear_hfmIn',default=False,
			help="Delete hfmIn member fields. (May save memory.)")
		
		self.model = self.GetValue('model',help='Minimal path model to be solved.')
		# Unified treatment of standard and extended curvature models
		if self.model=='ElasticaExt2_5': self.model='ElasticaExt2'
		if self.model.endswith("Ext2"): self.model=self.model[:-4]+"2"

		self.ndim = len(hfmIn['dims'])
		if self.isCurvature: 
			self.ndim_phys = (self.ndim+1)//2
			self.ndim_ang = self.ndim-self.ndim_phys
		self.kernel_data = {key:SimpleNamespace()
			for key in ('eikonal','flow','scheme','geodesic','forwardAD','reverseAD','chart')}
		for value in self.kernel_data.values(): 
			value.__dict__.update({'args':dict(),'policy':SimpleNamespace(),
				'stats':dict(),'module':None})
		# ['traits','source','policy','module','kernel','args','trigger','stats']


	@property # Dimension agnostic model
	def model_(self): return self.model[:-1]
		
	def HasValue(self,key):
		self.hfmOut['keys']['visited'].append(key)
		return key in self.hfmIn

	def GetValue(self,key,default="_None",verbosity=2,array_float=False,
		help="Sorry : no help for this key"):
		"""
		Get a value from a dictionnary, printing some help if requested.
		"""
		# We only import arguments once, otherwise risks of issues with multiple defaults
		# Also, this allows to delete arguments and potentially save space
		assert key not in self.hfmOut['keys']['help']

		self.hfmOut['keys']['help'][key] = help
		self.hfmOut['keys']['default'][key] = default

		if key in self.hfmIn:
			self.hfmOut['keys']['used'].append(key)
			value = self.hfmIn[key]
			if self.clear_hfmIn:
				if isinstance(value,Metrics.Base) or ad.isndarray(value) and value.size>100:
					self.hfmIn.store[key] = (None,"Deleted in GetValue")
			if array_float is False: return value
			value = self.caster(value)
			# Check shape
			if isinstance(array_float,tuple):
				shapeRef,shape = array_float,value.shape
				if len(shapeRef)!=len(shape):
					raise ValueError(f"Field {key} has incorrect number of dimensions. "
						f"Expected shape {shapeRef}, found {shape}")
				for sRef,s in zip(shapeRef,shape):
					if sRef not in (-1,s): 
						raise ValueError(f"Field {key} has incorrect dimensions. "
							f"Expected shape {shapeRef}, found {shape}")
			return value
		elif isinstance(default,str) and default == "_None":
			raise ValueError(f"Missing value for key {key}")
		else:
			if verbosity<=self.verbosity:
				if isinstance(default,str) and default=="_Dummy":
					print(f"see out['keys']['default'][{key}] for default")
				else:print(f"key {key} defaults to {default}")
			if isinstance(default, numbers.Number) and array_float is not False:
				default = self.caster(default)
			return default

	def Warn(self,msg):
		if self.verbosity>=-1:
			print("---- Warning ----\n",msg,"\n-----------------\n")

	def Run(self):
		self.SetKernelTraits()
		self.SetGeometry()
		self.SetArgs()
		self.SetChart()
		self.SetKernel()
		self.Solve('eikonal')
		self.PostProcess()
		self.SolveAD()
		self.GetGeodesics()
		self.FinalCheck()

		if self.extractValues or self.retself:
			retval = [self.hfmOut]
			if self.extractValues: retval.insert(0,self.values)
			if self.retself: retval.append(self)
			return retval
		else:
			return self.hfmOut

	SetKernelTraits = _Kernel.SetKernelTraits
	SetGeometry = _SetGeometry.SetGeometry
	SetArgs = _SetArgs.SetArgs
	SetKernel = _Kernel.SetKernel
	InitStop = _StopCriterion.InitStop
	SetChart = _StopCriterion.SetChart
	Solve = _solvers.Solve
	PostProcess = _PostProcess.PostProcess
	SolveAD = _PostProcess.SolveAD
	GetGeodesics = _GetGeodesics.GetGeodesics

	SetRHS = _SetArgs.SetRHS
	SolveLinear = _PostProcess.SolveLinear
	
	def FinalCheck(self):
		if self.GetValue('exportValues',False,help="Return the solution numerical values"):
			self.hfmOut['values'] = self.values
		self.extractValues = self.GetValue('extractValues',False,
			help="Return the solution numerical values separately from other data")
		self.retself = self.GetValue('retself',False,
			help="Return the class instance that did the work")
		self.hfmOut['stats'] = {key:value.stats for key,value in self.kernel_data.items()}
		self.hfmOut['solverGPUTime'] = self.kernel_data['eikonal'].stats['time']
		self.hfmOut['keys']['unused'] = list(set(self.hfmIn.keys()) 
			-set(self.hfmOut['keys']['used']) ) # Used by interface
		if self.verbosity>=1 and self.hfmOut['keys']['unused']:
			print(f"!! Warning !! Unused keys from user : {self.hfmOut['keys']['unused']}")

# --- Specific model properties ---

	@property
	def isCurvature(self):
		return self.model in ['ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2',
		'ReedsSheppGPU3']

	@property
	def drift_model(self):
		return self.model_ in ('Rander','AsymmetricQuadratic')

# ---- Metrics ---

	@property
	def metric(self):
		"""Adimensionized metric"""
		if self._metric is None: 
			self._metric = self._dualMetric.dual()
			if self._metric_delete_dual: self._dualMetric = (None,"Deleted in metric")
		return self._metric

	def CostMetric(self,x):
		"""Adimensionized interpolated metric""" 
		if self._CostMetric is None:
			self._CostMetric = self.metric.with_cost(self.cost)
			# TODO : remove. No need to create this grid for our interpolation
			grid = ad.array(np.meshgrid(*(cp.arange(s,dtype=self.float_t) 
				for s in self.shape), indexing='ij')) # Adimensionized coordinates
			self._CostMetric.set_interpolation(grid,periodic=self.periodic) # First order interpolation
			if self._CostMetric_delete_dual: self._metric = (None,"Deleted in CostMetric")
		return self._CostMetric.at(x) 

	@property
	def dualMetric(self):
		"""Adimensionized dual metric"""
		if self._dualMetric is None: self._dualMetric = self._metric.dual()
		return self._dualMetric

# ----- Array manipulation -----

	def as_field(self,e,name,depth=0):
		shape = self.hfmIn.shape
		oshape,ishape = e.shape[:depth],e.shape[depth:]
		if ishape==shape: # Already a field
			return e
		elif ishape==tuple(): # Constant field
			return np.broadcast_to(e.reshape(oshape+(1,)*self.ndim),oshape+shape)
		elif self.isCurvature:
			if ishape==shape[self.ndim_phys:]:  # Angular field
				return np.broadcast_to(e.reshape(oshape+(1,)*self.ndim_phys+ishape),oshape+shape)
			elif ishape==shape[:self.ndim_phys]: # Physical field
				return np.broadcast_to(e.reshape(oshape+ishape+(1,)*self.ndim_ang), oshape+shape)
		raise ValueError(f"Field {name} has incorrect dimensions. Found {e.shape}, "
			f"whereas domain has shape {shape}")

	def print_big_arrays(self,data):
		for name,value in ad.Base.array_members(data,
			iterables=(type(self),type(self.hfmIn),Metrics.Base,
				tuple,list,dict,SimpleNamespace)):
			ratio = value.nbytes/self.hfmIn.size
			if ratio>0.1: print(name,f"{ratio:.2f}")

	@property
	def values_expand(self): return _PostProcess.values_expand(self)
	@property
	def values(self): return _PostProcess.values(self)



























