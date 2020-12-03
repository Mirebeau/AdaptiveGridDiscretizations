#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the inf-convolution of an array with a small constant array.
The implementation could be substantially optimized, by rearranging the data in blocks, etc
*/

#include "static_assert.h"

#ifndef Int_macro
typedef int Int;
#endif

/** The following constants need to be defined by the including file
const Int ndim=2;
const Int shape_c[ndim] = {3,3};
const Int size_c = 3*3;
*/
__constant__ Int shape_tot[ndim];
__constant__ Int size_tot; // product of shape_tot

#ifndef T_macro
typedef float T;
const T T_Sup =  1./0.;
const T T_Inf = -1./0.;
#endif

// 1-> min. 0->max 
#ifndef mix_is_min_macro
#define mix_is_min_macro 1
#endif

// Work in the max-plus or min-plus algebra.
#if mix_is_min_macro
T Plus(const T a, const T b){return min(a,b);}
const T T_Neutral = T_Sup;
#else
T Plus(const T a, const T b){return max(a,b);}
const T T_Neutral = T_Inf;
#endif

// Optionally define upper_saturation_macro or lower_saturation_macro for saturated arithmetic
T Times(const T a, const T b){
	#ifdef upper_saturation_macro
	if(a>=0 && b>=T_Sup-a){return T_Sup;}
	#endif
	#ifdef lower_saturation_macro
	if(a<=0 && b<=T_Inf-a){return T_Inf;}
	#endif
	return a+b;
}

__constant__ T kernel_c[size_c];

#ifdef periodic_macro
#define PERIODIC(...) __VA_ARGS__
#else
#define PERIODIC(...) 
#endif

#include "Grid.h"

extern "C" {
__global__ void 
InfConvolution(const T * __restrict__ input, T * __restrict__ output){
	HFM_DEBUG(assert(shape2size(shape_tot,ndim)==size_tot
		&& shape2size(shape_c,ndim)==size_c);)

	// Get the position where the work is to be done.
	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}

	Int x_t[ndim];
	Grid::Position(n_t,shape_tot,x_t);
	T result = T_Neutral;
	// Access neighbor values, and perform the inf convolution
	for(Int i_c=0; i_c<size_c; ++i_c){
		Int y_t[ndim];
		Grid::Position(i_c,shape_c,y_t);
		for(Int k=0; k<ndim; ++k){
			y_t[k] += x_t[k] - shape_c[k]/2;} // Centered kernel
		if(Grid::InRange_per(y_t,shape_tot)){
			const Int ny_t = Grid::Index_per(y_t,shape_tot);
			const T prod = Times(input[ny_t],kernel_c[i_c]);
			result = Plus(result,prod);
		}
	}
	output[n_t] = result;
}

} // extern "C"

