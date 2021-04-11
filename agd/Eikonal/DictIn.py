# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The Eikonal.dictIn class is used to hold the input parameters to the eikonal solvers of 
the HFM library, CPU based or GPU based.
"""

import numpy as np
from collections.abc import MutableMapping
from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd
from .. import LinearParallel as lp
from .. import Metrics
from . import run_detail
from . import DictIn_detail

_array_float_fields = {
	'origin','dims','gridScale','gridScales','values',
	'seeds','seeds_Unoriented','tips','tips_Unoriented',
	'seedValues','seedValues_Unoriented','seedValueVariation','seedFlags',
	'cost','speed','costVariation',
	'inspectSensitivity','inspectSensitivityWeights','inspectSensitivityLengths',
	'exportVoronoiFlags','factoringIndexShift',
	'chart_mapping',
}

# Alternative key for setting or getting a single element
_singleIn = {
	'seed':'seeds','seedValue':'seedValues',
	'seed_Unoriented':'seeds_Unoriented','seedValue_Unoriented':'seedValues_Unoriented',
	'tip':'tips','tip_Unoriented':'tips_Unoriented',
	'seedFlag':'seedFlags','seedFlag_Unoriented':'seedFlags_Unoriented',
}

_readonlyIn = {
	'float_t'
}

_array_module = {
	'cpu':'numpy','cpu_raw':'numpy','gpu_transfer':'numpy',
	'gpu':'cupy','cpu_transfer':'cupy',
}

_singleOut = {
	'geodesic':'geodesics','geodesic_Unoriented':'geodesics_Unoriented',
	'geodesic_euclideanLength':'geodesics_euclideanLength',
}

SEModels = {'ReedsShepp2','ReedsSheppForward2','Elastica2','Dubins2',
'ReedsSheppExt2','ReedsSheppForwardExt2','ElasticaExt2','ElasticaExt2_5','DubinsExt2',
'ReedsShepp3','ReedsSheppForward3'}

# These models do not follow the usual dimension naming convention (physical dimension last)
dimModels = {'ElasticaExt2_5':3,'Riemann3_Periodic':3,'ReedsSheppGPU3':5} 

class dictOut(MutableMapping):
	"""
	A dictionnary like structure used as output of the Eikonal solvers. 
	"""

	def __init__(self,store=None):
		self.store=store

	def __copy__(self): return dictOut(self.store.copy())
	def copy(self): return self.__copy__()

	def __repr__(self):
		return f"dictOut({self.store})"

	def __setitem__(self, key, value):
		if key in _singleOut: key = _singleOut[key]; value = [value]
		self.store[key] = value

	def __getitem__(self, key): 
		if key in _singleOut:
			values = self.store[_singleOut[key]]
			if len(values)!=1: 
				raise ValueError(f"Found {len(values)} values for key {key}")
			return values[0]
		return self.store[key]

	def __delitem__(self, key): 
		key = _singleOut.get(key,key)
		del self.store[key]

	def __iter__(self): return iter(self.store)
	def __len__(self):  return len(self.store)
	def keys(self): 
		"""
		The keys of the dictionary-like structure.
		"""
		return self.store.keys()

def CenteredLinspace(a,b,n):
	"""
	Returns a linspace shifted by half a node length.
	Inputs : 
	 - a,b : interval endpoints
	 - n : number of points
	"""
	n_=int(n); assert(n==n_) #Allow floats for convenience
	r,dr=np.linspace(a,b,n_,endpoint=False,retstep=True)
	if np.any(np.isnan(dr)): assert n==1; dr=b-a #Cupy 8.6 bug
	return r+dr/2


class dictIn(MutableMapping):
	"""
	A dictionary like structure used as input of the Eikonal solvers.

	__init__ arguments: 
	- store : a dictionary, used for initialization.

	See dictIn().RunHelp() for details on the eikonal solver inputs.
	"""

	default_mode = 'cpu' # class attribute

	def __init__(self, store=None):
		if store is None: store=dict()
		self.store = {'arrayOrdering':'RowMajor'}
		if 'mode' in store:
			mode = store['mode']
			self.store['mode']=mode
		else:
			mode = dictIn.default_mode
		assert mode in ('cpu','cpu_raw','cpu_transfer','gpu','gpu_transfer')
		self._mode = mode
		if self.mode in ('gpu','cpu_transfer'):
			import cupy as cp
			self.xp = cp
			caster = lambda x : cp.asarray(x,dtype=float_t)
			self.array_float_caster = lambda x : ad.asarray(x,caster=caster) #cupy >= 8.6
		else: 
			self.xp = np
			self.array_float_caster = lambda x : np.asarray(x,dtype=float_t)
		
		if self.mode in ('gpu','cpu_transfer','gpu_transfer'): float_t=np.float32
		else: float_t=np.float64

		if 'float_t' in store:
			float_t = store['float_t']
			self.store['float_t']=float_t

		self._float_t = float_t
		if store: self.update(store)

	def __copy__(self):     return dictIn(self.store.copy())
	def __deepcopy__(self): return dictIn(self.store.deepcopy())
	def copy(self): 
		"""
		Returns a shallow copy of the structure.
		"""
		return self.__copy__()

	@property
	def mode(self): 
		"""
		The running mode of the eikonal solver, see the Run method.
		The input data must be provided using a compatible array module: 
		numpy in 'cpu' mode, cupy in 'gpu' mode.

		Supported running mode for the eikonal solver : 
		- 'cpu' : Run algorithm on host, store data on host
		- 'cpu_transfer' : Run algorithm on host, store data on device
		- 'cpu_raw' : Raw call to the HFM CPU library (debug purposes)
		- 'gpu' : Run algorithm on device, store data on device
		- 'gpu_transfer' : Run algorithm on device, store data on host
		"""
		return self._mode
	@property
	def float_t(self):
		"""
		The floating point type of the data arrays. Typically np.float64 in 'cpu' mode, 
		and np.float32 in 'gpu' mode.
		"""
		return self._float_t
	
	def __repr__(self): 
		return f"dictIn({self.store})"

	def __setitem__(self, key, value):
		"""
		Set a key:value pair in the dictionary like structure.

		Special treatment of keys and values:
		- The values associated to some keys are converted, if needed, to numeric 
		arrays of floats for the numpy or cupy module. (key in _array_float_fields) 
		- The key value pair 'seed':x is converted into 'seeds':[x], (as for setting 
		 multiple seeds, but their number is just one). (key in _singleIn)
		- Some keys are readonly, changes will be rejected (key in _readonlyIn.)
		"""
		if key=='mode':
			if _array_module[value]!=_array_module[self.mode]:
				raise ValueError('Switching between modes with distinct array storage')
			else: self._mode = value
		if key in _readonlyIn and self.store[key]!=value: 
			raise ValueError(f"Key {key} is readonly (set at init)") 
		if key in _singleIn: 
			key = _singleIn[key]; value = [value]
		if key in _array_float_fields and not ad.isndarray(value):
			value = self.array_float_caster(value)
		self.store[key] = value

	def __getitem__(self, key): 
		"""
		Get a value associated to a given key, in the dictionary like structure.

		Special treatment of keys and values:
		- self['seed'] = self['seeds'][0], if self['seeds'] has length one, 
		and fails otherwise. (key in _singleIn)
		"""
		if key in _singleIn:
			values = self.store[_singleIn[key]]
			if len(values)!=1: 
				raise ValueError(f"Found {len(values)} values for key {key}")
			return values[0]
		return self.store[key]

	def __delitem__(self, key): 
		key = _singleIn.get(key,key)
		del self.store[key]

	def __iter__(self): return iter(self.store)
	
	def __len__(self):  return len(self.store)

	def keys(self): 
		"""
		The keys of this dictionary structure.
		"""
		return self.store.keys()

	def RunHelp(self,mode=None):
		"""
		Help on the eikonal solver, depending on the running mode.
		"""
		if mode is None: mode = self.mode
		if mode in ('cpu','cpu_raw','cpu_transfer'): 
			help(run_detail.RunSmart)
		else:
			from . import HFM_CUDA
			help(HFM_CUDA.RunGPU)

	def Run(self,join=None,**kwargs):
		"""
		Calls the HFM library, prints log and returns output.
		Inputs : 
		- join (optional) : join the dictionary with these additional entries before running.
		- **kwargs (optional) : passed to the run_detail.RunSmart or HFM_CUDA.RunGPU methods.
		
		See dictIn().RunHelp() for additional details, depending on the running mode.
		"""
		if join is not None:
			other = self.copy()
			other.update(join)
			return other.Run(**kwargs)

		if self['arrayOrdering']!='RowMajor': 
			raise ValueError("Unsupported array ordering")
		def to_dictOut(out):
			if isinstance(out,tuple): return (dictOut(out[0]),) + out[1:]
			else: return dictOut(out)
			
		if   self.mode=='cpu': return to_dictOut(run_detail.RunSmart(self,**kwargs))
		elif self.mode=='cpu_raw': return to_dictOut(run_detail.RunRaw(self.store,**kwargs))
		elif self.mode=='cpu_transfer':
			cpuIn = ad.cupy_generic.cupy_get(self,dtype64=True,iterables=(dictIn,Metrics.Base))
			cpuIn.xp = np; cpuIn._mode = 'cpu'; cpuIn['mode'] = 'cpu'
			for key in list(cpuIn.keys()): 
				if key.startswith('traits'): cpuIn.pop(key)
			return to_dictOut(run_detail.RunSmart(cpuIn,**kwargs))
		
		from . import HFM_CUDA
		if   self.mode=='gpu': return to_dictOut(HFM_CUDA.RunGPU(self,**kwargs))
		elif self.mode=='gpu_transfer':
			gpuStoreIn = ad.cupy_generic.cupy_set(self.store, # host->device
				dtype32=True, iterables=(dict,Metrics.Base))
			gpuIn = dictIn({**gpuStoreIn,'mode':'gpu'})
			gpuOut = HFM_CUDA.RunGPU(gpuIn)
			cpuOut = ad.cupy_generic.cupy_get(gpuOut,iterables=(dict,list))
			return to_dictOut(cpuOut) # device->host

	# ------- Grid related functions ------

	@property
	def shape(self):
		"""
		The shape of the discretization grid.
		"""
		dims = self['dims']
		if ad.cupy_generic.from_cupy(dims): dims = dims.get()
		return tuple(dims.astype(int))

	@property
	def size(self): 
		"""
		The number of points in the discretization grid.
		"""
		return np.prod(self.shape)
	
	@property
	def SE(self):
		"""
		Wether the model is based on the Special Euclidean group.
		True for curvature penalized models.
		"""
		return self['model'] in SEModels	
	
	@property
	def vdim(self):
		"""
		The dimension of the ambient vector space.
		"""
		model = self['model']
		if model in dimModels: return dimModels[model]
		dim = int(model[-1])
		return (2*dim-1) if self.SE else dim

	@property
	def nTheta(self):
		"""
		Number of points for discretizing the interval [0,2 pi], in the angular space 
		discretization, for the SE models (a.k.a. curvature penalized models).
		"""
		if not self.SE: raise ValueError("Not an SE model")
		shape = self.shape
		if self.vdim!=len(self.shape): raise ValueError("Angular resolution not set")
		n = shape[-1]
		return (2*n) if self.get('projective',False) and self.vdim==3 else n

	@nTheta.setter
	def nTheta(self,value):
		if not self.SE: raise ValueError("Not an SE model")
		shape = self.shape
		vdim = self.vdim
		projective = self.get('projective',False)
		if vdim==len(shape): shape=shape[:int((vdim+1)/2)] #raise ValueError("Angular resolution already set")
		if   vdim==3: self['dims'] = (*shape, value/2 if projective else value) 
		elif vdim==5: self['dims'] = (*shape, value/4 if projective  else value/2, value) 

	@property
	def gridScales(self):
		"""
		The discretization grid scale along each axis.
		"""
		if self.SE:
			h = self['gridScale']
			hTheta = 2.*np.pi / self.nTheta
			if self.vdim==3: return self.array_float_caster( (h,h,hTheta) )
			else: return self.array_float_caster( (h,h,h,hTheta,hTheta) )
		elif 'gridScales' in self: return self['gridScales']
		else: return self.array_float_caster((self['gridScale'],)*self.vdim)	
		
	@property
	def corners(self):
		"""
		Returns the extreme points grid[:,0,...,0] and grid[:,-1,...,-1] of the 
		discretization grid.
		"""
		dims = self['dims']
		origin = self.get('origin',self.xp.zeros_like(dims))
		gridScales = self.gridScales
		if self.SE: 
			tail = self.array_float_caster((-gridScales[-1]/2,)*(len(dims)-len(origin)))
			origin = np.concatenate((origin,tail))
		return (origin,origin+gridScales*dims)

	def Axes(self,dims=None):
		"""
		The discretization points used along each coordinate axis.
		"""
		bottom,top = self.corners
		if dims is None: dims=self['dims']
		return [self.array_float_caster(CenteredLinspace(b,t,d)) 
			for b,t,d in zip(bottom,top,dims)]

	def Grid(self,dims=None):
		"""
		Returns a grid of coordinates, containing all the discretization points of the domain.
		Similar to np.meshgrid(*self.Axes(),indexing='ij')
		Inputs : 
		- dims(optional) : use a different sampling of the domain
		"""
		axes = self.Axes(dims);
		ordering = self['arrayOrdering']
		if ordering=='RowMajor': return ad.array(np.meshgrid(*axes,indexing='ij',copy=False))
		elif ordering=='YXZ_RowMajor': return ad.array(np.meshgrid(*axes,copy=False))
		else: raise ValueError('Unsupported arrayOrdering : '+ordering)

	def SetUniformTips(self,dims):
		"""
		Place regularly spaced tip points all over the domain, 
		from which to backtrack minimal geodesics.
		Inputs : 
		- dims : number of tips to use along each dimension.
		"""
		self['tips'] = self.Grid(dims).reshape(self.vdim,-1).T

	def SetRect(self,sides,sampleBoundary=False,gridScale=None,gridScales=None,
		dimx=None,dims=None):
		"""
		Defines a box domain, for the HFM library.
		Inputs:
		- sides, e.g. ((a,b),(c,d),(e,f)) for the domain [a,b]x[c,d]x[e,f]
		- sampleBoundary : switch between sampling at the pixel centers, and sampling including the boundary
		- gridScale, gridScales : side h>0 of each pixel (alt : axis dependent)
		- dimx, dims : number of points along the first axis (alt : along all axes)
		"""
		# Ok to set a new domain, or completely replace the domain
		domain_count = sum(e in self for e in ('gridScale','gridScales','dims','origin'))
		if domain_count not in (0,3): raise ValueError("Domain already partially set")
		
		caster = self.array_float_caster
		corner0,corner1 = caster(sides).T
		dim = len(corner0)
		sb=float(sampleBoundary)
		width = corner1-corner0
		if gridScale is not None: 
			gridScales=[gridScale]*dim; self['gridScale']=gridScale
		elif gridScales is not None:
			self['gridScales']=gridScales
		elif dimx is not None:
			gridScale=width[0]/(dimx-sb); gridScales=[gridScale]*dim; self['gridScale']=gridScale
		elif dims is not None:
			gridScales=width/(caster(dims)-sb); self['gridScales']=gridScales
		else: 
			raise ValueError('Missing argument gridScale, gridScales, dimx, or dims')

		h=caster(gridScales)
		ratios = (corner1-corner0)/h + sb
		dims = np.round(ratios)
		assert(np.min(dims)>0)
		origin = corner0 + (ratios-dims-sb)*h/2
		self['dims']   = dims
		self['origin'] = origin

	def PointFromIndex(self,index,to=False):
		"""
		Turns an index into a point.
		Optional argument to: if true, inverse transformation, turning a point into a continuous index
		"""
		bottom,_ = self.corners
		scale = self.gridScales
		start = bottom +0.5*scale
		index = self.array_float_caster(index)
		assert index.shape[-1]==self.vdim
		if not to: return start+scale*index
		else: return (index-start)/scale

	def IndexFromPoint(self,point):
		"""
		Returns the index that yields the position closest to a point, and the error.
		"""
		point = self.array_float_caster(point)
		continuousIndex = self.PointFromIndex(point,to=True)
		index = np.round(continuousIndex)
		return index.astype(int),(continuousIndex-index)

	def OrientedPoints(self,pointU):
		"""
		Appends all possible orientations to the point coordinates.
		"""
		pointU = self.array_float_caster(pointU)
		if self['model'] not in SEModels: 
			raise ValueError("OrientedPoints only makes sense for models SE space.")
		if self.vdim!=3:
			raise ValueError("Sorry, oriented point not implemented for SE(3) models.")
		pdim = int((self.vdim+1)/2) # Number of physical dimensions
		if pointU.shape[-1]!=pdim:
			raise ValueError(f"Unoriented points expected to have {pdim} dimensions, "
				f"found {pointU.shape[-1]}.")
		theta = self.Axes()[2]
		point = self.xp.full((len(theta),*pointU.shape[:-1],self.vdim),np.nan)
		for i,t in enumerate(theta):
			point[i,...,:pdim]=pointU
			point[i,...,pdim:]=t
		return point

	def VectorFromOffset(self,offset,to=False):
		"""
		Turns a finite difference offset into a vector, by multiplying by the gridScale.
		Inputs : 
		- offset : the offset to convert.
		- to (optional) : if True, produces an offset from a vector (reverse operation).
		"""
		offset = self.array_float_caster(offset)
		if not to: return offset*self.gridScales
		else: return offset/self.gridScales  

	def GridNeighbors(self,point,gridRadius):
		"""
		Returns the neighbors around a point on the grid. 
		Geometry last convention
		Inputs: 
		- point (array): geometry last
		- gridRadius (scalar): given in pixels
		"""
		xp = self.xp
		point = self.array_float_caster(point)
		point_cindex = self.PointFromIndex(point,to=True)
		aX = [xp.arange(int(np.floor(ci-gridRadius)),int(np.ceil(ci+gridRadius)+1),
			dtype=self.float_t) for ci in point_cindex]
		neigh_index =  xp.stack(xp.meshgrid( *aX, indexing='ij'),axis=-1)
		neigh_index = neigh_index.reshape(-1,neigh_index.shape[-1])

		# Check which neighbors are close enough
		offset = neigh_index-point_cindex
		close = np.sum(offset**2,axis=-1) < gridRadius**2

		# Check which neighbors are in the domain (periodicity omitted)
		neigh = self.PointFromIndex(neigh_index)
		bottom,top = self.corners
		inRange = np.all(np.logical_and(bottom<neigh,neigh<top),axis=-1)

		return neigh[np.logical_and(close,inRange)]

	SetFactor = DictIn_detail.SetFactor
	SetSphere = DictIn_detail.SetSphere
	@property
	def factoringPointChoice(self): return DictIn_detail.factoringPointChoice(self)




