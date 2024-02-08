#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

// ndim_macro must be defined
#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 5)
#include "Geometry5.h"
#endif

const Int nsym = decompdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int geom_size = symdim DRIFT(+ndim); // DRIFT -> Rander case
const Int factor_size = symdim;

#include "Constants.h"

void scheme(const Scalar geom[geom_size], 
	Scalar weights[nactx], OffsetT offsets[nactx][ndim] DRIFT(,Scalar drift[1][ndim]) ){
	STATIC_ASSERT(nactx==decompdim,inconsistent_scheme_parameters)
	const Scalar * dual_metric = geom; // dual_metric[symdim]
	decomp_m(dual_metric,weights,offsets);
	DRIFT(copy_vV(geom+symdim,drift[0]);)
}

FACTOR(
#include "EuclideanFactor.h"
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products and norms
	const Scalar * m = factor_metric;
	const Scalar xx=scal_vmv(x,m,x), xe=scal_vmv(x,m,e), ee=scal_vmv(e,m,e);
	euclidean_factor_sym(xx,xe,ee,fact ORDER2(, fact2));
}
) 

#include "Update.h"