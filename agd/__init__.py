# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0


"""
Adaptive Grid Discretizations (agd) package.

This package is intented as a toolbox for discretizing and solving partial differential
equations (PDEs), especially in the following contexts : 
- A Cartesian grid is used for the domain discretization.
- The PDE embeds geometric information, possibly strongly anisotropic.
- One puts a strong emphasis on preserving the structure of the PDE 
 (monotony, causality, degenerate ellipticity, ...) at the discrete level.
- Generic CPU/GPU programming.

This package comes with an extensive suite of notebooks, which serve simultaneously the
purposes of documentation, mathematical description, and testing. Please see 
https://github.com/Mirebeau/AdaptiveGridDiscretizations

The AGD package is architectured around the following main components:
- AutomaticDifferentiation : automatically compute gradients, hessians, jacobians, in 
 dense or sparse format, using operator and function overloading.

- Eikonal : a ready to use solver of (generalized, anisotropic) eikonal equations. Those
 are partial differential equations which characterize minimal distances w.r.t. Riemannian
 or other classes of metrics.

- Metrics : helper classes for classical and less classical objects 
 (Riemannian metrics, Hooke elasticity tensors, etc) used to encode geometric information.

- FiniteDifferences, Domain, Interpolation : helper classes for handling function values
 stored in arrays and designing numerical schemes.

- LinearParallel : basic linear algebra operations, following an axes ordering convention
somewhat opposite to numpy's (geometry first).

- Selling : a decomposition method for symmetric positive definite matrices, which is a 
 central tool in our designs of anisotropic PDE discretizations.

.. include:: ../README.md
"""
__docformat__ = "restructuredtext"