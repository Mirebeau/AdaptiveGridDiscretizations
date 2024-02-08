# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The seismic package defines norms/metrics that characterize the first arrival time of
elastic waves in anisotropic materials. It also provides helper functions for solving 
the elastic wave equation itself.

Main metric/norm classes:
- Hooke : norm defined by a Hooke tensor, corresponding to the fastest velocity.
- TTI : tilted transversally isotropic norm.
"""

#from .implicit_base import ImplicitBase
from .hooke   import Hooke
from .tti import TTI
from . import thomsen_data as Thomsen

