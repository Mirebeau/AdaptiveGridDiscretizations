# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
The ODE package provides basic functionality for generating ODEs, and solving them.

Main submodules:
- hamiltonian : defines classes of hamiltonians with various mathematical structures,
 (separable, quadratic, associated with a metric, ...)
- hamiltonian_base : contains the base class for the hamiltonians, and some symplectic 
 ODE solvers.
- backtrack : tools intended for path backtracking in time dependent optimal control
"""