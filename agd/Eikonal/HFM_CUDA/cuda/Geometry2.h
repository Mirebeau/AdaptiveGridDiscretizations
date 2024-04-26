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
const Int Selling_maxiter = 100;
#endif

void obtusesuperbase_m(const Scalar m[symdim], OffsetT sb[__restrict__ ndim+1][ndim]){
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
void decomp_m(const Scalar m[symdim], 
	Scalar weights[__restrict__ symdim], OffsetT offsets[__restrict__ symdim][ndim]){
	OffsetT sb[ndim+1][ndim];
	obtusesuperbase_m(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int i=r, j = (r+1)%3, k=(r+2)%3;
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]));
		perp_v(sb[k],offsets[r]);
	}
}

Scalar det_m(const Scalar m[symdim]){return m[0]*m[2]-m[1]*m[1];}
Scalar trace_m(const Scalar m[symdim]){return m[0]+m[2];}

/// Compute the eigenvalues of a symmetric matrix, in ascending order
void eigvalsh(const Scalar m[symdim], Scalar lambda[__restrict__ ndim]){
	const Scalar htr = trace_m(m)/2;
	const Scalar diff = (m[0]-m[2])/2;
	const Scalar delta = sqrt(diff*diff + m[1]*m[1]);
	lambda[0] = htr-delta;
	lambda[1] = htr+delta;
}

/** Compute the eigenvalues, and the first eigenvector, of a symmetric matrix
The cuda routines are likely more accurate, but they are slow and memory intensive.
 */
void eigh_first(const Scalar m[symdim], 
	Scalar lambda[__restrict__ ndim], Scalar v[__restrict__ ndim]){
	eigvalsh(m,lambda);
	// Any column of m-lambda1*Id is an eigenvector for lambda0
	Scalar mp[symdim];
	copy_mM(m,mp);
	mp[0]-=lambda[1]; mp[2]-=lambda[1];
	if(abs(mp[0]) >= abs(mp[1])){v[0] = mp[0]; v[1] = mp[1];}
	else {v[0] = mp[1]; v[1] = mp[2];}
	// Normalize the eigenvector
	const Scalar norm2 = norm2_v(v);
	if(norm2 > 0) mul_kV(Scalar(1)/sqrt(norm2), v);
	else v[0]=1; // m is proportional to the identity, return any unit vector
}

/// Compute the eigenvalues and eigenvectors of a symmetric matrix
void eigh(const Scalar m[symdim], 
	Scalar lambda[__restrict__ ndim], Scalar v[__restrict__ ndim][ndim]){
	eigh_first(m,lambda,v[0]);
	perp_v(v[0],v[1]); // The second eigenvector is obtained by rotating the first
	trans_A(v); // For compatibility with numpy
}

