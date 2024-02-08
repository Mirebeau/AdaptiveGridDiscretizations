# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

"""
This module provides basic conversion utilities to manipulate low-dimensional spheres, 
and related objects : rotations, quaternions, Pauli matrices, etc
"""

from . import AutomaticDifferentiation as ad
from . import LinearParallel as lp

import numpy as np

# -----------------------
# Equatorial projection
# -----------------------

def sphere_from_plane(e):
    """Produces a point in the unit sphere by projecting a point in the equator plane."""
    e = ad.asarray(e)
    e2 = lp.dot_VV(e,e)
    return ad.array([1.-e2,*(2*e)])/(1.+e2)
def plane_from_sphere(q):
    """Produces a point in the equator plane from a point in the unit sphere."""
    e2 = (1-q[0])/(1+q[0])
    return q[1:]*((1+e2)/2.)

# ------------------------
# Rotations, dimension 3
# ------------------------

# Two or three dimensional rotation, defined by angle, and axis in dimension three.
rotation = lp.rotation

def rotation3_from_sphere3(q):
    """Produces the rotation associated with a unit quaternion."""
    qr,qi,qj,qk = q
    return 2*ad.array([
        [0.5-(qj**2+qk**2), qi*qj-qk*qr, qi*qk+qj*qr],
        [qi*qj+qk*qr, 0.5-(qi**2+qk**2), qj*qk-qi*qr],
        [qi*qk-qj*qr, qj*qk+qi*qr, 0.5-(qi**2+qj**2)]])

def sphere3_from_rotation3(r):
    """Produces the unit quaternion, with positive real part, associated with a rotation."""
    qr = np.sqrt(lp.trace(r)+1.)/2.
    qi = (r[2,1]-r[1,2])/(4*qr)
    qj = (r[0,2]-r[2,0])/(4*qr)
    qk = (r[1,0]-r[0,1])/(4*qr)
    return ad.array([qr,qi,qj,qk])

def ball3_from_rotation3(r,qRef=None):
    """Produces an euclidean point from a rotation, 
    selecting in the intermediate step the quaternion 
    in the same hemisphere as qRef. (Defaults to southern.)"""
    q = sphere3_from_rotation3(r)
    if qRef is not None: q[:,lp.dot_VV(q,qRef)<0] *= -1
    return plane_from_sphere(q)

def rotation3_from_ball3(e): 
    """Produces a rotation from an euclidean point. 
    Also returns the intermediate quaternion."""
    q = sphere_from_plane(e)
    return rotation3_from_sphere3(q),q

# -----------------------
# Rotations, dimension 2
# -----------------------

# For now, this is skipped, since it is not very useful, and not very consistent 
# with the three dimensional case (which involves a double cover). 

# def rotation_from_sphere(q):
# 	"""
# 	- (dimension 3) Produces the rotation associated with a unit quaternion.
# 	- (dimension 2) Produces the rotation whose first column is the given unit vector.
# 	"""
# 	if   np.ndim(q)==2: return rotation2_from_sphere1(q)
# 	elif np.ndim(q)==4: return rotation3_from_sphere3(q)
# 	else: raise ValueError("Unsupported dimension")

def rotation2_from_sphere1(q):
	"""Produces the rotation whose first column is the given unit vector."""
	c,s = q
	return ad.array([[c,-s],[s,c]])

def sphere1_from_rotation2(r):
	"""Produces the unit vector which is the first column of the given rotation"""
	return ad.array([r[0,0],r[1,0]])

# -----------------------
# Pauli matrices
# -----------------------

def pauli(a,b,c,d=None):
	"""
	Pauli matrix. Symmetric if d is None, Hermitian otherwise.
	Determinant is $a^2-b^2-c^2-d^2$
	"""
	if d is None: return ad.array([[a+b,c],[c,a-b]])
	else: return ad.array([[a+b,c+1j*d],[c-1j*d,a-b]])
