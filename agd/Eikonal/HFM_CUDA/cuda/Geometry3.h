#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=3;
#include "Geometry_.h"

#ifndef Selling_maxiter_macro
const Int Selling_maxiter = 100;
#endif

/// Cross product, in dimension three. 
template<typename Tx, typename Ty, typename Tout=Tx>
void cross_vv(const Tx x[3], const Ty y[3], Tout out[__restrict__ 3]){
	for(Int i=0; i<3; ++i){
		const Int j=(i+1)%3, k=(i+2)%3;
		out[i]=x[j]*y[k]-x[k]*y[j];
	}
}

// the first two elements of these permutations range among all possible pairs
const Int iterReducedMax = 6;
const Int Selling_permutations[iterReducedMax][ndim+1] = { 
	{0,1,2,3},{0,2,1,3},{0,3,1,2},{1,2,0,3},{1,3,0,2},{2,3,0,1}};

// Computation of an obtuse superbase of a positive definite matrix, by Selling's algorithm
void obtusesuperbase_m(const Scalar m[symdim], OffsetT sb[ndim+1][ndim]){
	canonicalsuperbase(sb);
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int it = iter%6; 
		const Int * perm = Selling_permutations[it];
		const Int i = perm[0], j=perm[1];
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			const Int k=perm[2], l=perm[3];
			add_vV(sb[i],sb[k]);
			add_vV(sb[i],sb[l]);
			neg_V(sb[i]);
			iterReduced=0;
		}
	}
}

// Selling decomposition of a positive definite matrix
const Int decompdim = symdim;
void decomp_m(const Scalar m[symdim], Scalar weights[symdim], OffsetT offsets[symdim][ndim]){
	OffsetT sb[ndim+1][ndim];
	obtusesuperbase_m(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int * perm = Selling_permutations[r];
		const Int i=perm[0],j=perm[1],k=perm[2],l=perm[3];
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]) );
		cross_vv(sb[k],sb[l],offsets[r]);
	}
}
