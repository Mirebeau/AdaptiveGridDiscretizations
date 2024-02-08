# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This package implements automatic differentiation (AD) methods, in the following flavors:
- Dense, Sparse, and Reverse (experimental) modes
- First and second order differentiation
- CPU and GPU support, using numpy and cupy.
The AD types implement numpy's overloading mechanisms, and one should be able to use them
as drop in replacement for numpy arrays in many contexts.

Main submodules:
- Dense : first order, forward AD with dense storage.
- Dense2 : second order, forward AD with dense storage.
- Sparse : first order, forward AD with sparse storage.
- Sparse2 : second order, forward AD with sparse storage.
- Reverse, Reverse2 (experimental) : first and second order, reverse AD.
- Optimization : basic Newton method implemented using AD

Main functions:
- asarray, array: turn a list/tuple of arrays into a larger array.
- is_ad : test whether a variable embeds AD information.
- remove_ad : remove AD information
- simplify_ad : compress the AD information, of Sparse and Sparse2 types.
- apply : apply a function to some arguments, using specified AD tricks.
- isndarray : returns true for numpy, cupy, and AD types.
- cupy_friendly : helper function for CPU/GPU generic programming.
"""


from . import functional
#from . import Base # No need to import, but this level in hierarchy
from . import cupy_support
from . import cupy_generic
from . import ad_generic
from . import misc
from . import Dense
from . import Sparse
from . import Reverse
from . import Dense2
from . import Sparse2
from . import Reverse2
from . import Optimization
from . import ad_specific

from .ad_generic import array,asarray,is_ad,remove_ad,common_cast,min_argmin, \
	max_argmax,disassociate,associate,apply_linear_mapping,apply_linear_inverse,precision

from .ad_specific import simplify_ad,apply,compose
from .cupy_generic import isndarray,cupy_friendly

class DeliberateNotebookError(Exception):
	def __init__(self,message):
		super(DeliberateNotebookError,self).__init__(message)