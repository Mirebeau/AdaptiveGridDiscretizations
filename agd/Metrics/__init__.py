# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The Metrics package defines a variety of norms on R^d, defined by appropriate parameters,
possibly non-symmetric. The data of a such a norm, at each point of a domain, is referred 
to as a metric, for instance a Riemannian metric, and defines a path-length distance 
over the domain.

Metrics are the fundamental input to (generalized) eikonal equations, which can be solved
using the provided Eikonal module.

Main norm/metric classes:
- Isotropic : a multiple of the Euclidean norm on $R^d$.
- Diagonal : axis-dependent multiple of the Euclidean norm.
- Riemann : anisotropic norm on $R^d$ defined by a symmetric positice definite matrix of 
 shape $(d,d)$. Used to define a Riemannian metric.
- Rander : non-symmetric anisotropic norm, defined as the sum of a Riemann norm and 
 of a drift term.
- AsymQuad : non-symmetric anisotropic norm, defined by gluing two Riemann norms along 
 a hyperplane.

Additional norm/metric classes are defined in the Seismic subpackage.
"""


from .base 	    import Base
from .isotropic import Isotropic
from .diagonal  import Diagonal 
from .riemann 	import Riemann
from .rander    import Rander
from .asym_quad import AsymQuad
