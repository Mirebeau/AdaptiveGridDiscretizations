#pragma once 
// Copyright 2023 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/*
This file implements an eikonal model with a custom hamiltonian, either in :
- max form, Hamiltonian is max_i <p,omega_i>_+^2
- sum form, Hamiltonian is sum_i <p,omega_i>_+^2
It is similar to the DubinsState_ model, except with no states.
*/

// ncontrols must be defined 

#if controls_max_macro && (ncontrols_macro>=2) // max form
#define nmix_macro ncontrols_macro
const bool mix_is_min = true; // Take the most efficient control among all available
#endif

const Int ncontrols = ncontrols_macro;
#define decomp_v_macro true
#define nsym_macro 0 // We also need the macro, otherwise issues with zero-length arrays
#include "TypeTraits.h"

#if (ndim_macro == 1)
#include "Geometry1.h"
#elif (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 4)
#include "Geometry5.h"
#endif

const Int geom_size = ncontrols*ndim; 
const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = decompdim * (controls_max_macro ? 1 : ncontrols);
#include "Constants.h"
#include "Decomp_v_.h"

bool scheme(const Scalar geom[geom_size], Scalar weights[nactx], Int offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==nfwd*nmix_macro,inconsistent_scheme_parameters)
	for(int i=0; i<ncontrols; ++i)
		decomp_v(&geom[ndim*i],&weights[decompdim*i],&offsets[decompdim*i]);
}

#include "Update.h"