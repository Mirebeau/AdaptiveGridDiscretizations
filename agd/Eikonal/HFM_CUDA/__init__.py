# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This package is implementation detail for a GPU accelerated eikonal solver. It should 
not be used directly, but through the parent package. (Run method of the dictIn class,
with mode='gpu'.)
"""

def RunGPU(hfmIn,*args,cache=None,**kwargs):
	"""
	Runs the GPU eikonal solver.

	Main input:
	- hfmIn : a dictionary like-structure

	The solver embeds some help information, which can be accessed as follows.
	* set hfmIn['verbosity'] to 1 or 2 to display information on run, including the defaulted keys.
	 set to 0 to silence the run.
	* look at hfmOut['keys']['help'] to see a basic help regarding each key,
	where hfmOut is the output of this function.
	"""
	if cache is not None: print(f"Warning : gpu eikonal solver does not support caching")
	from . import _Interface
	return _Interface.Interface(hfmIn).Run(*args,**kwargs)

	# Arguments for glued domains : glue, far, niter, nitergeo
	# Run niter times, while pasting the values
	# Extract geodesics. Stop them when too far.
	# -> reintroduce suitable seed
	# -> stop when leave domain. Use InWall stopping criterion, 
	# by setting the distance to infinity when far (?)
	# In that case, we do not really need a predicate...
	# -> Present geodesics as successions of pieces




class EikonalGPU_NotImplementedError(Exception):
	def __init__(self,message):
		super(EikonalGPU_NotImplementedError,self).__init__(message)