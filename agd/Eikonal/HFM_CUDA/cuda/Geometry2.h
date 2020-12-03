#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=2;
#include "Geometry_.h"

/// Perpendicular vector, in dimension two. Caution : assume x and out are distinct.
template<typename T>
void perp_v(const T x[2], T out[__restrict__ 2]){ 
	out[0]=-x[1];
	out[1]= x[0];
}

// Computation of an obtuse superbase of a positive definite matrix, by Selling's algorithm
#ifndef Selling_maxiter_macro
const Int Selling_maxiter = 50;
#endif

void obtusesuperbase_m(const Scalar m[symdim], Int sb[ndim+1][ndim]){
	canonicalsuperbase(sb);
	const Int iterReducedMax = 3;
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int i=iter%3, j=(iter+1)%3,k=(iter+2)%3;
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			sub_vv(sb[i],sb[j],sb[k]);
			neg_V(sb[i]);
			iterReduced=0;
		}
	}
}

// Selling decomposition of a symmetric positive definite matrix
// Note : 3=symdim=ndim+1=decompdim
const Int decompdim = symdim;
void decomp_m(const Scalar m[symdim], Scalar weights[symdim], Int offsets[symdim][ndim]){
	Int sb[ndim+1][ndim];
	obtusesuperbase_m(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int i=r, j = (r+1)%3, k=(r+2)%3;
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]));
		perp_v(sb[k],offsets[r]);
	}
}
