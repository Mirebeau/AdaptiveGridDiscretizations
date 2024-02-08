# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
from .LibraryCall import RunDispatch,GetBinaryDir
from .DictIn_detail import factoringPointChoice
from .. import Metrics
from .. import AutomaticDifferentiation as ad

def GetGeodesics(output,suffix=''): 
	if suffix != '' and not suffix.startswith('_'): suffix='_'+suffix
	geodesics = np.vsplit(output['geodesicPoints'+suffix],
					 output['geodesicLengths'+suffix].cumsum()[:-1].astype(int))
	return [geo.T for geo in geodesics]

class Cache(object):
	def __init__(self,needsflow=False):
		self.contents = dict()
		self.verbosity = None
		self.requested = None
		self.needsflow = needsflow
		self.dummy = False
	
	def empty(self):
		"""Wether the cache lacks data needed to bypass computation"""
		return not self.contents 

#	def full(self): 
#		return self.contents and ('geodesicFlow' in self.contents or not needsflow)
	
	def PreProcess(self,hfmIn_raw):
		if self.dummy: return
		self.verbosity = hfmIn_raw.get('verbosity',1)
		self.requested = []
		if hfmIn_raw.get('exportValues',False): 		self.requested.append('values')
		if hfmIn_raw.get('exportActiveNeighs',False):	self.requested.append('activeNeighs')
		if hfmIn_raw.get('exportGeodesicFlow',False):	self.requested.append('geodesicFlow')
		if self.empty():
			if self.verbosity>=1: print("Requesting cacheable data")
			hfmIn_raw['exportValues']=True
			hfmIn_raw['exportActiveNeighs']=True
			if self.needsflow: hfmIn_raw['exportGeodesicFlow']=True
		else:
			if self.verbosity>=1: print("Providing cached data")
			for key in ('values','activeNeighs'):
				setkey_safe(hfmIn_raw,key,self.contents[key])
			# Note : all points set as accepted in HFM algorithm
			hfmIn_raw['exportValues']=False
			hfmIn_raw['exportActiveNeighs']=False
			hfmIn_raw['exportGeodesicFlow']=self.needsflow and ('geodesicFlow' not in self.contents)

	def PostProcess(self,hfmOut_raw):
		if self.dummy: return
		if self.empty():
			if self.verbosity>=1 : print("Filling cache data")
			for key in ('values','activeNeighs'):
				self.contents[key] = hfmOut_raw[key]
			if self.needsflow:
				self.contents['geodesicFlow'] = hfmOut_raw['geodesicFlow']
		else:
			for key in self.requested:
				if key not in hfmOut_raw:
					setkey_safe(hfmOut_raw,key,self.contents[key])
				if self.needsflow and 'geodesicFlow' not in self.contents:
					self.contents['geodesicFlow'] = hfmOut_raw['geodesicFlow']

	def geodesicFlow(self,hfmIn=None):
		if 'geodesicFlow' in self.contents:
			return self.contents['geodesicFlow']
		elif hfmIn is None:
			raise ValueError("geodesicFlow not cached and lacking hfm input data")
		else:
			self.dummy = False
			self.needsflow = True

			hfmIn_ = {key:ad.remove_ad(value,iterables=(Metrics.Base,)) for key,value in hfmIn.items()
			if key not in [ # Keys useless for evaluating the geodesic flow
			'tips','tips_Unoriented',
			'costVariation','seedValueVariation','inspectSensitivity',
			'exportActiveNeighs','exportActiveOffsets']}

			hfmOut_raw = RunSmart(hfmIn_,cache=self,returns='out_raw')
			if self.verbosity: 
				print("--- HFM call triggered above to compute geodesic flow ---")
			self.PostProcess(hfmOut_raw)
			return self.contents['geodesicFlow']


def RunRaw(hfmIn):
	"""Raw call to the HFM library"""
	rawIn = hfmIn.copy()
	mode = rawIn.pop('mode',None); assert mode is None or mode=='cpu_raw'
	float_t = rawIn.pop('float_t',None); assert float_t is None or float_t==np.float64 
	return RunDispatch(rawIn,GetBinaryDir("FileHFM","HFMpy"))


def RunSmart(hfmIn,co_output=None,cache=None,returns="out"):
	"""
	Calls the HFM library, a CPU eikonal solver written in C++,
	with pre-processing and post-processing of data.

	Main input : 
	- hfmIn, a dictionary like structure, containing the eikonal solver data.

	The C++ library embeds some help information, which can be accessed using the 
	following key:value pairs in hfmIn.
	* 'verbosity' : set to 1 or 2 to display information on run, including the defaulted keys.
	 set to 0 to silence the run.
	* 'keyHelp' : set as a string of keys separated by spaces, to print help on these.
	 example 'keyHelp':"seeds tips"

	Optional input:
	- co_output (optional) : used for reverse automatic differentiation.
	- cache (optional) : store some intermediate results to bypass computations at a later stage. 
	- returns (optional) : early aborts the run (debug purposes). ('in_raw','out_raw','out')
	"""
	assert returns in ('in_raw','out_raw','out')

	if factoringPointChoice(hfmIn) == 'Both':
		hfmIn = hfmIn.copy()
		hfmIn.SetFactor()

	hfmIn_raw = {}

	if cache is None:
		cache = Cache()
		cache.dummy = True

	# Pre-process usual arguments
	for key,value in hfmIn.items():
		PreProcess(key,value,hfmIn,hfmIn_raw,cache)

	# Reverse automatic differentiation
	if co_output is not None:
		assert ad.misc.reverse_mode(co_output)=="Reverse"
		[co_hfmOut,co_value],_ = co_output
		assert hfmIn.get('extractValues',False) and co_hfmOut is None
		indices = np.nonzero(co_value)
		positions = hfmIn.PointFromIndex(np.asarray(indices).T)
		weights = co_value[indices]
		setkey_safe(hfmIn_raw,'inspectSensitivity',positions)
		setkey_safe(hfmIn_raw,'inspectSensitivityWeights',weights)
		setkey_safe(hfmIn_raw,'inspectSensitivityLengths',[len(weights)])
		if 'metric' in hfmIn or 'dualMetric' in hfmIn:
			cache.needsflow = True
			cache.dummy = False
	
	# Dealing with cached data
	cache.PreProcess(hfmIn_raw)

	# Call to the HFM library
	if returns=='in_raw': return hfmIn_raw
	hfmOut_raw = RunDispatch(hfmIn_raw,GetBinaryDir("FileHFM","HFMpy"))
	if returns=='out_raw': return hfmOut_raw
	
	# Dealing with cached data
	cache.PostProcess(hfmOut_raw)

	# Post process
	hfmOut = {}
	for key,val in hfmOut_raw.items():
		PostProcess(key,val,hfmOut_raw,hfmOut)

	# Reverse automatic differentiation
	if co_output is not None:
		result=[]
		for key,value in hfmIn.items(): 
			if key in ('cost','speed'):
				if isinstance(value,Metrics.Isotropic):
					value = value.to_HFM()
				result.append((value,
					hfmOut['costSensitivity_0'] if key=='cost' else (hfmOut['costSensitivity_0']/value**2)))
			if key in ('metric','dualMetric'):
				shape_bound = value.shape
				considered = co_output.second
				value_ad = ad.Dense.register(value,iterables=(Metrics.Base,),shape_bound=shape_bound,considered=considered)
				metric_ad = value_ad if key=='metric' else value_ad.dual()

				costSensitivity = np.moveaxis(flow_variation(cache.geodesicFlow(hfmIn),metric_ad),-1,0)*hfmOut['costSensitivity_0']
				shift = 0
				size_bound = np.prod(shape_bound,dtype=int)
				for x in value:
					if any(x is val for val in considered):
						xsize_free = x.size//size_bound
						result.append((x,costSensitivity[shift:(shift+xsize_free)].reshape(x.shape)) )
						shift+=xsize_free
			elif key=='seedValues':
				sens = hfmOut['seedSensitivity_0']
				# Match the seeds with their approx given in sensitivity
				corresp = np.argmin(ad.Optimization.norm(np.expand_dims(hfmIn_raw['seeds'],axis=2)
					-np.expand_dims(sens[:,:2].T,axis=1),ord=2,axis=0),axis=0)
				sens_ordered = np.zeros_like(value)
				sens_ordered[corresp]=sens[:,2]
				result.append((value,sens_ordered))
			# TODO : speed
		return result

	return (hfmOut,hfmOut.pop('values')) if hfmIn.get('extractValues',False) else hfmOut	

def setkey_safe(dico,key,value):
	if key in dico:
		if value is not dico[key]:
			raise ValueError("Multiple values for key ",key)
	else:
		dico[key]=value

def flow_variation(flow,metric):
	flow = np.moveaxis(flow,-1,0)
	zeros = np.all(flow==0.,axis=0)
	flow.__setitem__((slice(None),zeros),np.nan)
	norm = metric.norm(flow)
	variation = norm.coef/np.expand_dims(norm.value,axis=-1)
	variation.__setitem__((zeros,slice(None)),0.)
	return variation

# ----------------- Preprocessing ---------------
def PreProcess(key,value,refined_in,raw_out,cache):
	"""
	copies key,val from refined to raw, with adequate treatment
	"""

	verbosity = refined_in.get('verbosity',1)
	if   key=='mode': assert value in ('cpu','cpu_transfer'); return
	elif key=='float_t': assert value==np.float64; return
	elif key in ('cost','speed'):
		if isinstance(value,Metrics.Isotropic):
			value = value.to_HFM()
		if isinstance(value,ad.Dense.denseAD):
			setkey_safe(raw_out,'costVariation',
				value.coef if key=='cost' else (1/value).coef)
			value = ad.remove_ad(value)
		setkey_safe(raw_out,key,value)
	elif key in ('metric','dualMetric'):
		if refined_in['model'].startswith('Isotropic') or refined_in['model'].startswith('Diagonal'):
			assert isinstance(value,(Metrics.Isotropic,Metrics.Diagonal))
			return PreProcess('cost' if key=='metric' else 'speed',value.to_HFM(),refined_in,raw_out,cache)
		if isinstance(value,Metrics.Base): 
			if ad.is_ad(value,iterables=(Metrics.Base,)):
				metric_ad = value if key=='metric' else value.dual()
				setkey_safe(raw_out,'costVariation',flow_variation(cache.geodesicFlow(refined_in),metric_ad))
			value = ad.remove_ad(value.to_HFM())
		setkey_safe(raw_out,key,value)

	elif key=='seedValues':
		if ad.is_ad(value):
			setkey_safe(raw_out,'seedValueVariation',value.gradient())
			value=ad.remove_ad(value)
		setkey_safe(raw_out,key,value)

	elif key=='extractValues':
		setkey_safe(raw_out,'exportValues',value)
	else:
		setkey_safe(raw_out,key,value)

#---------- Post processing ------------
def PostProcess(key,value,raw_in,refined_out):
	"""
	copies key,val from raw to refined, with adequate treatment
	"""
	if key.startswith('geodesicPoints'):
		suffix = key[len('geodesicPoints'):]
		geodesics = GetGeodesics(raw_in,suffix=suffix)
		setkey_safe(refined_out,"geodesics"+suffix,geodesics)
	elif key.startswith('geodesicLengths'):
		pass

	elif key=='values':
		if 'valueVariation' in raw_in:
			value = ad.Dense.denseAD(value,raw_in['valueVariation'])
		setkey_safe(refined_out,key,value)
	elif key=='valueVariation':
		pass
	
	elif key=='geodesicFlow':
		setkey_safe(refined_out,'flow',np.moveaxis(value,-1,0))
	elif key=='factoringGradients':
		setkey_safe(refined_out,'factoringGradients',np.moveaxis(value,-1,0))
	elif key=='activeOffsets':
		setkey_safe(refined_out,'offsets',CastOffsets(value))
	else:
		setkey_safe(refined_out,key,value)

def CastOffsets(raw_offsets):
	"""Offsets are exported with their coefficients bundled together in a double.
	This function unpacks them."""
	raw_shape = raw_offsets.shape
	d = len(raw_shape)-1
	nOffsets = raw_shape[-1]
	refined_shape = (d,nOffsets)+raw_shape[:-1]
	refined_offsets = np.zeros(refined_shape,dtype=int)
	def coef(x,i):
		x = x//256**(d-i-1)
		x = x%256
		return np.where(x<128,x,x-256)

	for j in range(nOffsets):
		raw = raw_offsets[...,j].astype(np.int64)
		for i in range(d):
			refined_offsets[i,j] = coef(raw,i)
	return refined_offsets


