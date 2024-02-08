#pragma once 
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define adaptive_offsets_macro false

#if isotropic_macro // We're using the same file for the anisotropic diagonal model
#define adaptive_weights_macro false
#define geom_macro false // No geometry for the isotropic metric (only a cost function)
#endif

#include "TypeTraits.h"

// ndim_macro must be defined
const Int ndim = ndim_macro;
const Int nsym = ndim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int factor_size = ndim; // Size of the metric used for factorization (diagonal)

#if   (ndim_macro == 2)
const Int offsets[ndim][ndim] = {{1,0},{0,1}};
#elif (ndim_macro == 3)
const Int offsets[ndim][ndim] = {{1,0,0},{0,1,0},{0,0,1}};
#elif (ndim_macro == 4)
const Int offsets[ndim][ndim] = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
#elif (ndim_macro == 5)
const Int offsets[ndim][ndim] = {{1,0,0,0,0},{0,1,0,0,0},{0,0,1,0,0},{0,0,0,1,0},{0,0,0,0,1}};
#endif

#include "Geometry_.h"
#include "Constants.h" // Needed for factor, and cannot put safely inside FACTOR(...) Macro

#if isotropic_macro
__constant__ Scalar weights[ndim];
void scheme(const Scalar weights[ndim], const Int offsets[ndim][ndim]){}
#else // Isotropic model
const Int geom_size = ndim;
void scheme(const Scalar dual_costs2[ndim], 
	Scalar weights[ndim], const Int offsets[ndim][ndim]){
	copy_vV(dual_costs2,weights);}
#endif

#include "EuclideanFactor.h"
FACTOR(
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products
	Scalar * d = factor_metric;
	const Scalar xx=scal_vdv(x,d,x), xe=scal_vdv(x,d,e), ee=scal_vdv(e,d,e);
	euclidean_factor_sym(xx,xe,ee,fact ORDER2(,fact2));
}
) // FACTOR

#include "Update.h"