#pragma once
// Copyright 2024 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
 This file implements a smooth variant of Selling's two-dimensional decomposition, published in
 Bonnans, F., Bonnet, G. & Mirebeau, J.-M. Monotone discretization of anisotropic differential operators using Voronoi’s first reduction. Constructive Approximation 1–61 (2023).
 */

#include "NetworkSort.h"
#include "Geometry2.h"
const int smooth2_decomp_order = 2; // TUNING PARAMETER. Adjust smoothness.
namespace smooth {
const int decompdim = 4; // Less sparse than Selling

/**Regularized absolute value. Guarantee : 0 <= result-|x| <= 1/2.
 - sabs_order (0, 1, 2 or 3) : order of the last continuous derivative. */
Scalar sabs(Scalar x){
    const int order = smooth2_decomp_order;
    x=abs(x);
    if(x>=1 || order==0) return abs(x);
    const Scalar x2 = x*x;   if(order==1) return Scalar(1./2)*(1+x2);
    const Scalar x4 = x2*x2; if(order==2) return Scalar(1./8)*(3+6*x2-x4);
    const Scalar x6 = x2*x4; if(order==3) return Scalar(1./16)*(5+15*x2-5*x4+x6);
}

/**Regularized median (a.k.a. ρ1) assuming p0<=p1<=p2.
 Guarantee : p1/(2*sqrt(2)) <= result < p1 */
Scalar smed(Scalar p0, Scalar p1, Scalar p2){
    const Scalar s = p0*p1+p1*p2+p2*p0;
    const Scalar p12 = p2-p1, q = p12*p12; // Invariant quantities under Selling superbase flip
    return Scalar(0.5)*s/sqrt(q+2*s);
}

/** Smooth variant of Selling's two dimensional decomposition */
void decomp_m(const Scalar m[symdim], Scalar weights[__restrict__ decompdim], OffsetT offsets[__restrict__ decompdim][ndim]){
    // Compute the standard Selling decomposition
    OffsetT v[ndim+1][ndim];
    obtusesuperbase_m(m,v);
    const Scalar rho_[ndim+1] = {-scal_vmv(v[1],m,v[2]),-scal_vmv(v[0],m,v[2]),-scal_vmv(v[0],m,v[1])};
    
    // Sort the weights, compute auxiliary quantities
    Int order[ndim+1] = {0,1,2};
    fixed_length_sort<ndim+1>(rho_,order);
    const Scalar rho[ndim+1] = {rho_[order[0]],rho_[order[1]],rho_[order[2]]};
    const Scalar median = smed(rho[0],rho[1],rho[2]);
    // The maximum with zero is in theory useless (up to roundoff error)
    const Scalar w = max(Scalar(0),median*sabs(rho[0]/median)-rho[0]);
    
    // Fill the weights and offsets of the modified Selling decomposition
    weights[0] = rho[0]+w/2;
    weights[1] = rho[1]-w;
    weights[2] = rho[2]-w;
    weights[3] = w/2;
    
    perp_v(v[order[0]],offsets[0]);
    perp_v(v[order[1]],offsets[1]);
    perp_v(v[order[2]],offsets[2]);
    sub_vv(offsets[1], offsets[2], offsets[3]);
} // decomp_m

} // smooth2
