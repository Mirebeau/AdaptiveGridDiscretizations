#pragma once
// Copyright 2023 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/*
This file implements a scheme for solving the PDE
<grad u(x), omega(x)> = 1 
in the viscosity sense. The corresponds to a vehicle that moves with velocity omega(x) at x,
without any ability to turn etc. The intended use is to compute multiple arrival times via dimension
lifting. 

The PDE is linear, and could be solved using a linear upwind scheme. However, that approach does not 
scale well to large problems, due to ill conditioning, and the impossibility to use a standard 
(sparse) linear solver in view of the sheer problem size. The non-linear scheme that we use here is 
causal, and can thus be solved in a single pass using the FMM on a CPU. The expectation is that it
should be behaved on the GPU.
*/

//#define nsym_macro 0 // Only forward offsets

// ndim_macro must be defined
#define decomp_v_macro true
#define nsym_macro 0 // We also need the macro, otherwise issues with zero-length arrays
#if   (ndim_macro == 1)
#include "Geometry1.h"
#elif (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 5)
#include "Geometry5.h"
#endif

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = decompdim; // Number of forward offsets
const Int geom_size = ndim; 

#include "Constants.h"
#include "Decomp_v_.h"

bool scheme(const Scalar geom[geom_size], Scalar weights[nactx], Int offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==decompdim,inconsistent_scheme_parameters)
	decomp_v(geom, weights, offsets);
}

#include "Update.h"