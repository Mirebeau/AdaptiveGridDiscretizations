# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

from .. import AutomaticDifferentiation as ad
from .. import FiniteDifferences as fd
import numpy as np

def flatten_symmetric_matrix(m):
	"""
	Input : a square (symmetric) matrix.
	Output : a vector containing the lower triangular entries
	"""
	d=m.shape[0]
	assert d==m.shape[1]
	return np.concatenate([m[i,:(i+1)] for i in range(d)],axis=0)

def expand_symmetric_matrix(arr,d=None,extra_length=False):
	if d is None:
		d=0
		while (d*(d+1))//2 < len(arr):
			d+=1
	assert (extra_length or len(arr)==(d*(d+1))//2)
	
	def index(i,j):
		i,j = max(i,j),min(i,j)
		return (i*(i+1))//2+j
	return ad.asarray([ [ arr[index(i,j)] for i in range(d)] for j in range(d) ])



