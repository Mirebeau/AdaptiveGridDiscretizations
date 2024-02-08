#pragma once
// Copyright 2022 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** 
Compute the eigendecomposition of a symmetric matrix.
The cuda routines are likely more accurate, but they are often slow and memory intensive.
*/

/** The following need to be defined in including file (example)
typedef int Int;
typedef float Scalar;
#define ndim_macro 3
#define quaternion_macro False
*/

#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#else
STATIC_ASSERT(false,Unsupported_dimension);
#endif 

__constant__ Int size_tot;

extern "C" {

__global__ void kernel_eigvalsh(
	const Scalar * __restrict__ m_t, // Decomposed matrix
	Scalar * __restrict__ lambda_t // Eigenvalues
){
const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
if(n_t>=size_tot) {return;}

Scalar m[symdim]; 
for(Int i=0; i<symdim; ++i) m[i] = m_t[n_t*symdim+i];
Scalar lambda[ndim];
eigvalsh(m,lambda);
for(Int i=0; i<ndim; ++i) lambda_t[n_t*ndim+i] = lambda[i];
}

__global__ void kernel_eigh(
	const Scalar * __restrict__ m_t, // Decomposed matrix
	Scalar * __restrict__ lambda_t, // Eigenvalues
	Scalar * __restrict__ v_t // Eigenvectors
){
const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
if(n_t>=size_tot) {return;}

Scalar m[symdim]; 
for(Int i=0; i<symdim; ++i) m[i] = m_t[n_t*symdim+i];

Scalar lambda[ndim];
#if quaternion_macro && ndim_macro==2
Scalar v[ndim];
eigh_first(m,lambda,v);
for(Int i=0; i<ndim; ++i) v_t[n_t*ndim+i] = v[i];
#else
Scalar v[ndim][ndim];
eigh(m,lambda,v);
#if quaternion_macro
Scalar q[4];
sphere3_from_rotation3(v,q);
for(Int i=0; i<4; ++i) v_t[n_t*4+i] = q[i];
#else
for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) v_t[(n_t*ndim+i)*ndim+j] = v[i][j];}
#endif
#endif

for(Int i=0; i<ndim; ++i) lambda_t[n_t*ndim+i] = lambda[i];
}

} // extern "C"
