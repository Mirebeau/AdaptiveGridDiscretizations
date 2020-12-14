#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the value pasting required when solving eikonal equations 
on manifolds described in terms of local charts.

Pasting a floating point value is an atomic operation, and the pasted values are 
always decreasing and cannot be under-estimated by construction. Therefore we do not fear 
data races when standard floating point values are used. 

However, in the case of multi-precision a float and an int must be pasted, 
and memory race issues cannot be excluded.
Thus we use distinct source and target arrays in that case.
*/

/* // Basic type traits, should be defined externally. 
typedef int Int; 
typedef float Scalar;

const Int ndim; // Number of dimensions of solution
const Int ndim_s; // Number of dimensions of broadcasted arrays
*/

#include "static_assert.h"

#ifndef periodic_macro
#define PERIODIC(...) 
#else
#define PERIODIC(...) __VA_ARGS__
//const bool periodic[ndim]={false,true}; //must be defined outside
#endif


const Scalar boundary_tol = 1e-4;
const Int ncorner_s = 1<<ndim_s;
typedef unsigned char BoolAtom;
Scalar infinity(){return 1./0.;}

// Shape of the cartesian grid on which the solution is defined
__constant__ Int shape_tot[ndim];
__constant__ Int shape_o[ndim]; 
__constant__ Int shape_i[ndim]; 
__constant__ Int size_i;
__constant__ Int size_s; // Size of broadcasted array
#define bilevel_grid_macro
#include "Grid.h"

#if multiprecision_macro
#define MULTIP(...) __VA_ARGS__
__constant__ Scalar multip_step;
#else
#define MULTIP(...) 
#endif

#if strict_iter_o_macro
#define STRICT_ITER_O(...) __VA_ARGS__
#else
#define STRICT_ITER_O(...) 
#endif

/*
Array suffix convention : 
 arr_t : bi-level array, with shape_i and shape_o sublevels.
 arr_s : single level array, possibly with independent coords.
*/

extern "C" {

__global__ void ChartPaste(
	const Scalar * __restrict__ mapping_s,

	// Solution values
	STRICT_ITER_O(const) Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) 
	STRICT_ITER_O(Scalar * __restrict__ uNext_t, MULTIP(Int * __restrict__ uqNext_t,))

	BoolAtom * __restrict__ update_o
	){
const Int ndim_b = ndim-ndim_s; // Broadcasted dimensions
HFM_DEBUG(assert( shape2size(shape_i,ndim)==size_i );)
HFM_DEBUG(assert( shape2size(shape_tot+ndim_b,ndim_s) == size_s );)

// Get the current position, array indices
const Int n_i = threadIdx.x;
const Int n_o = blockIdx.x;

Int x_o[ndim], x_i[ndim], x_t[ndim];
Grid::Position(n_i,shape_i,x_i);
Grid::Position(n_o,shape_o,x_o);
for(Int k=0; k<ndim; ++k){x_t[k] = x_o[k]*shape_i[k]+x_i[k];}

if( !Grid::InRange_per(x_t,shape_tot) ) return; // Out of domain
const Int n_t = Grid::Index_tot(x_t); // Index in arr_t
const Int n_s = Grid::Index_per(x_t,shape_tot) % size_s; // Index in arr_s

// Import the mapped point
Int q_t[ndim_s]; Scalar r_t[ndim_s]; // integer, and fractional part of mapping

for(Int i=0; i<ndim_s; ++i){
	const Scalar m = mapping_s[size_s*i+n_s];
	if(m!=m) return; // NaN mapped point
	q_t[i] = floor(m);
	r_t[i] = m-q_t[i];
	PERIODIC(if(periodic[ndim_b+i]) {continue;})

	// Move inside points which are just slightly outside the domain
	const Int q_max = shape_tot[ndim_b+i]-1;
	if(q_t[i]==q_max && r_t[i]<=boundary_tol){  q_t[i]-=1; r_t[i]=1;} //r_t[i]+=1;} // Monotony is safer..
	if(q_t[i]==-1    && r_t[i]>=1-boundary_tol){q_t[i]+=1; r_t[i]=0;} //r_t[i]-=1;}

	// Abort if point is fully outside the domain
	if(q_t[i]>=q_max || q_t[i]<0) return; 
}

const Scalar u_orig = u_t[n_t];
MULTIP(const Scalar uq_orig = uq_t[n_t];) // Used as reference multiplier

// Get the mapped value, obtained by interpolation
Scalar u_mapped = 0.;
Int y_t[ndim]; // Interpolation point
Scalar w_sum = 0.;
for(Int i=0; i<ndim_b; ++i){y_t[i] = x_t[i];}
for(Int icorner=0; icorner<ncorner_s; ++icorner){
	Scalar w = 1.; // Interpolation weight 
	for(Int i=0; i<ndim_s; ++i){
		const Int eps = ((icorner>>i) & 1); // Wether to look at next corner
		y_t[ndim_b+i] = q_t[i] + eps;
		w *= eps ? r_t[i] : (1-r_t[i]);
	}
	if(!Grid::InRange_per(y_t,shape_tot)) continue;
	const Int ny_t = Grid::Index_tot(y_t);
	u_mapped += w * (u_t[ny_t] MULTIP(+(uq_t[ny_t]-uq_orig)*multip_step) );
	w_sum+=w;
}
if(w_sum>0) {u_mapped/=w_sum;}
else {u_mapped = infinity();}

// Compare values, update if necessary
if(u_mapped < u_orig){ // Excludes NaNs, +Infs, from u_mapped. Compatible with multip.
	update_o[n_o] = 1;
	#if multiprecision_macro
	const Int uq_delta = floor(u_mapped/multip_step);
	uqNext_t[n_t] = uq_orig + uq_delta;
	uNext_t[n_t] = u_mapped - uq_delta*multip_step;
	#elif strict_iter_o_macro
	uNext_t[n_t] = u_mapped;
	#else
	u_t[n_t] = u_mapped; 
	#endif
}

} // Paste

} // extern "C"




