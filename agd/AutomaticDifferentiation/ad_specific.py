# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import itertools
import numpy as np
from . import ad_generic
from . import misc
from . import Dense
from . import Sparse
from . import Dense2
from . import Sparse2

def simplify_ad(a,*args,**kwargs):
	"""
	Simplifies, if possible, the sparsity pattern of a sparse AD variable.
	See Sparse.spAD.simplify_ad for detailed help.
	"""
	if type(a) in (Sparse.spAD,Sparse2.spAD2): 
		return a.simplify_ad(*args,**kwargs)
	return a

def apply(f,*args,**kwargs):
	"""
	Applies the function to the given arguments, with special treatment if the following 
	keywords : 
	- envelope : take advantage of the envelope theorem, to differentiate a min or max.
	  The function is called twice, first without AD, then with AD and the oracle parameter.
	- shape_bound : take advantage of dense-sparse (or dense-dense) AD composition to 
	 differentiate the function efficiently. The function is called with dense AD, and 
	 the dimensions in shape_bound are regarded as a simple scalar.
	- reverse_history : use the provided reverse AD trace.
	"""
	envelope,shape_bound,reverse_history = (kwargs.pop(s,None) 
		for s in ('envelope','shape_bound','reverse_history'))
	if not any(ad_generic.is_ad(a) for a in itertools.chain(args,kwargs.values())):
		return f(*args,**kwargs)
	if envelope:
		def to_np(a): return a.value if ad_generic.is_ad(a) else a
		_,oracle = f(*[to_np(a) for a in args],**{key:to_np(val) for key,val in kwargs.items()})
		result,_ = apply(f,*args,**kwargs,oracle=oracle,envelope=False,shape_bound=shape_bound)
		return result,oracle
	if shape_bound is not None:
		size_factor = np.prod(shape_bound,dtype=int)
		t = tuple(b.reshape((b.size//size_factor,)+shape_bound) 
			for b in itertools.chain(args,kwargs.values()) if ad_generic.is_ad(b)) # Tuple containing the original AD vars
		lens = tuple(len(b) for b in t)
		def to_dense(b):
			if not ad_generic.is_ad(b): return b
			nonlocal i
			shift = (sum(lens[:i]),sum(lens[(i+1):]))
			i+=1
			if type(b) in (Sparse.spAD,Dense.denseAD): 
				return Dense.identity(constant=b.value,shape_bound=shape_bound,shift=shift)
			elif type(b) in (Sparse2.spAD2,Dense2.denseAD2):
				return Dense2.identity(constant=b.value,shape_bound=shape_bound,shift=shift)
		i=0
		args2 = [to_dense(b) for b in args]
		kwargs2 = {key:to_dense(val) for key,val in kwargs.items()}
		result2 = f(*args2,**kwargs2)
		return compose(result2,t,shape_bound=shape_bound)
	if reverse_history:
		return reverse_history.apply(f,*args,**kwargs)
	return f(*args,**kwargs)

def compose(a,t,shape_bound):
	"""Compose ad types, mostly intended for dense a and sparse b"""
	if not isinstance(t,tuple): t=(t,)
	if isinstance(a,tuple):
		return tuple(compose(ai,t,shape_bound) for ai in a)
	if not(type(a) in (Dense.denseAD,Dense2.denseAD2)) or len(t)==0:
		return a
	return type(t[0]).compose(a,t)