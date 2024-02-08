#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define nsym_macro 0 // Only forward offsets
#define nmix_macro 2 // Maximum of a family of two schemes. 
// Maximum of a family of two schemes. -> take the minimal update among the two.
const bool mix_is_min = true; 

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = decompdim; // Number of forward offsets

#include "Constants.h"
#include "Decomp_v_.h"

#if !precomputed_scheme_macro
bool scheme(GEOM(const Scalar geom[geom_size],) const Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==2*nfwd, inconsistent_scheme_structure)

	XI_VAR(Scalar ixi;) KAPPA_VAR(Scalar kappa;) 
	Scalar cT, sT; // cos(theta), sin(theta)
	get_ixi_kappa_theta(GEOM(geom,) x, XI_VAR(ixi,) KAPPA_VAR(kappa,) cT,sT);

	const Scalar
	vL[ndim]={cT,sT,kappa+ixi},
	#if convex_curvature_macro // Model variant where the vehicle always turns left
	vR[ndim]={cT,sT,kappa};
	#else
	vR[ndim]={cT,sT,kappa-ixi};
	#endif

	decomp_v(vL,  weights,        offsets);
	decomp_v(vR, &weights[nfwd], &offsets[nfwd]);
}
#endif
#include "Update.h"