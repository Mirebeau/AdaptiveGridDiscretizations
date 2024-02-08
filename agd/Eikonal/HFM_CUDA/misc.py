# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np

def integral_largest_nextlargest(dtype):
	dtype = np.dtype(dtype)
	integral = dtype.kind in ('i','u') # signed or unsigned integer
	largest = np.iinfo(dtype).max if integral else np.inf
	nextlargest = largest-1 if integral else np.finfo(dtype).max
	return integral,largest,nextlargest