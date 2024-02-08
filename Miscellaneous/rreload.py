import types
import sys
import importlib
from os.path import realpath

# # Reloading with configuration of the eikonal default_mode and ad.array.caster
#def ReloadPackages():
#    from Miscellaneous.rreload import rreload
#    global Eikonal,Metrics
#    Eikonal,Metrics = rreload([Eikonal,Metrics],rootdir="../..")
#    Eikonal.dictIn.default_mode = eikonal_mode
#    ad.array.caster = lambda x:cupy.asarray(x,dtype=np.float32)

def rreload(objects,rootdir,verbose=False):
	"""
	Reloads recursively some objects (typically modules, classes, or functions).
	Circular dependencies within submodules are not accepted.
	Inputs :
		- objects : the objects to be reloaded
		- rootdir : root directory, containing all the files to be reloaded
		- verbose 
	Outputs : 
		- list of the reloaded objects


	Loosely inspired from https://stackoverflow.com/a/58201660/12508258
	but with significant modifications to allow reloading class hierarchies 
	defined in multiple files. See also IPython deepreload

	Recommended usage, within a jupyter notebook:

	import A
	import B
	from C import D

	def reload_packages():
		from Miscellaneous.rreload import rreload
		global A,B,D
		A,B,D = rreload([A,B,D],rootdir="my/module/directory")
	"""
	def noreload(mod): 
		return not (hasattr(mod,'__file__') 
			and realpath(mod.__file__).startswith(realpath(rootdir)))
	reloaded = {}
	reloading = []
	return [_rreload(x,noreload,reloaded,reloading,verbose) for x in objects]


def _rreload(obj,noreload,reloaded,reloading,verbose):
	"""
	Reloads recursively an object
	Inputs : 
		- obj : object to be reloaded (typically module, class, or function)
		- noreload : predicated for not reloading an object
		- reloaded : dictionnary of reloaded modules, indexed by filenames
		- reloading : stack of modules being recursively reloaded
		- verbose
	Output : 
		- reloaded object
	"""

	# --- Reloading a non-module ---
	if not isinstance(obj,types.ModuleType):
		# If possible, find the module, reload it, return object from it 
		if hasattr(obj,'__module__') and hasattr(obj,'__name__'):
			mod=_rreload(sys.modules[obj.__module__],noreload,reloaded,reloading,verbose)
			return mod.__dict__[obj.__name__]
		return obj

	# --- Reloading a module ----
	if noreload(obj): return obj
	filename = obj.__file__
	if filename in reloaded:
		return reloaded[filename]
	if filename in reloading:
		raise ValueError(f"Circular dependency in modules : {filename} in {reloading}")
	reloading.append(filename)

	if verbose: print(" "*len(reloading),f"reloading module {obj.__name__}")

	for key,value in obj.__dict__.items():
		if isinstance(value,types.ModuleType):
			obj.__dict__[key] = _rreload(value,noreload,reloaded,reloading,verbose)
		if hasattr(value,'__module__') and hasattr(value,'__name__'):
			if value.__module__ != obj.__name__: 
				obj.__dict__[key] = _rreload(value,noreload,reloaded,reloading,verbose) 
			
	obj=importlib.reload(obj)
	reloaded[obj.__file__] = obj
	reloading.pop()

	return obj