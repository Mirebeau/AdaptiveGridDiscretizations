#pragma once
// Copyright 2022 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=1;
#include "Geometry_.h"

// Trivial decomposition of a 1x1 symmetric positive definite matrix
const Int decompdim = symdim;
void decomp_m(const Scalar m[symdim], 
	Scalar weights[__restrict__ symdim], Int offsets[__restrict__ symdim][ndim]){
	weights[0] = m[0];
	offsets[0][0] = 1;
}
