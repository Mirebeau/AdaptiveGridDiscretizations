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

def make_metric(m=None,w=None,a=None,w_rander=None,geometry_last=False):
    """
    Defines the metric F(x) = sqrt(a^2 x.m.x + sign(a)(w.x)_+^2) + w_rander.x,
    with expected defauts for m,w,a,w_rander.
    """
    from .. import Metrics
    if geometry_last:
        if m is not None: m = np.moveaxis(m,(-2,-1),(0,1))
        if w is not None: w = np.moveaxis(w,-1,0)
        if w_rander is not None: w_rander = np.moveaxis(w_rander,-1,0)
    if m is None:
        if w is None: return Isotropic(np.abs(a))
        vdim = len(w)
        m = np.broadcast_to(np.eye(vdim,like=w),(vdim,*w.shape))
    if a is not None: 
        if not overwrite_m: 
            m = m.copy()
            if w is not None: w = w.copy()
        m *= a**2
        if w is not None:
            pos = a<0
            w[pos] *= -1
            #m[pos] -= np.outer(w[pos],w[pos]) # Some issues with empty shapes
            m -= pos*np.outer(w,w)
    if w is None: 
        if w_rander is None: return Riemann(m)
        else: return Metrics.Rander(m,w_rander)
    if w_rander is None: return AsymQuad(m,w)
    else:
        from .asym_rander import AsymRander
        return AsymRander(m,w,None,w_rander)
