// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a linear update operator, in a format somewhat similar to 
the HFM eikonal update operator. It is used to solve the linear systems arising in the 
automatic differentiation of the eikonal solver. These are triangular or almost triangular
systems solved using Gauss-Siedel iteration.
*/

#include "static_assert.h"

#ifndef Scalar_macro
typedef float Scalar;
#endif
Scalar infinity(){return 1./0.;}

#ifndef Int_macro
typedef int Int;
#endif
typedef unsigned char BoolAtom;

#ifndef minchg_freeze_macro
#define minchg_freeze_macro 0
#endif
#if minchg_freeze_macro
#define MINCHG_FREEZE(...) __VA_ARGS__
#else
#define MINCHG_FREEZE(...) 
#endif

#ifndef pruning_macro
#define pruning_macro 0
#endif
#if pruning_macro
#define PRUNING(...) __VA_ARGS__
#else
#define PRUNING(...) 
#endif

#ifndef periodic_macro
#define periodic_macro 0
#endif

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
#else
#define PERIODIC(...) 
#endif

// No FIM here for now
#define FIM(...) 

#ifndef debug_print_macro
const Int debug_print = 0;
#endif

__constant__ Int shape_o[ndim]; 
__constant__ Int size_o;
__constant__ Int size_tot;

#include "Geometry_.h"
#include "Propagation.h" // Cannot include earlier

#ifndef dummy_init_macro
#define dummy_init_macro 1
#endif

#if dummy_init_macro
 /** A dummy initialization, to infinity, is passed for tracking values which have not been
visited. Actual initialization is zero.*/
Scalar init(const Scalar x){
	if(x==1./0.) {return 0.;}
	else {return x;}
} 
#else
Scalar init(const Scalar x){return x;}
#endif

__constant__ Scalar atol;
__constant__ Scalar rtol;


/* // These constants should also be defined when including the LinearUpdate.h file
const Int nrhs // Number of right hand sides
const Int nindex // Number of entries per matrix line
const Int ndim // Space dimension
const Int niter_i // Number of local iterations

const Int shape_i[ndim] = {8,8}; // Block dimension and related quantities
const Int size_i = 64;
const Int log2_size_i = 7;
*/



extern "C" {

__global__ void Update(
	Scalar * u_t, const Scalar * rhs_t, 
	const Scalar * diag_t, const Int * index_t, const Scalar * weight_t,
	MINCHG_FREEZE(const Scalar chg_t, const Scalar * minChgPrev_o, Scalar * minChgNext_o,)
	Int * updateList_o, PRUNING(BoolAtom * updatePrev_o,) BoolAtom * updateNext_o 
	){ 
	HFM_DEBUG(assert(shape2size(shape_o,ndim)==size_o && shape2size(shape_i,ndim)==size_i);)

	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if( Propagation::Abort(
		updateList_o,PRUNING(updatePrev_o,) 
		MINCHG_FREEZE(minChgPrev_o,updateNext_o)
		x_o,n_o) ){return;} // Also sets x_o, n_o

	const Int n_i = threadIdx.x;
	const Int n_t = n_o*size_i + n_i;

	__shared__ Scalar u_i_[nrhs][size_i];
	__shared__ Scalar chg_i[size_i]; // Used in the end

	Scalar rhs_[nrhs];
	Scalar u_old[nrhs];

	for(Int irhs=0; irhs<nrhs; ++irhs){
		u_old[irhs] = u_t[irhs*size_tot + n_t];
		u_i_[irhs][n_i] = init(u_old[irhs]);
		rhs_[irhs] =    rhs_t[irhs*size_tot + n_t];
	}

	const Scalar diag = diag_t[n_t];
	Int    v_i[nindex]; // Index of neighbor, if in the block
	Scalar v_o_[nrhs][nindex]; // Value of neighbor, if outside the block
	Scalar weight[nindex]; // Coefficient in the matrix

	const Int v_i_inBlock = -1; // 
	const Int v_i_invalid = -2; // Disregard entry

	for(Int k=0; k<nindex; ++k){
		weight[k] = weight_t[k*size_tot + n_t];
		if(weight[k]==0.) {v_i[k]=v_i_invalid; continue;}

		const Int index = index_t[k*size_tot + n_t];
		HFM_DEBUG(assert(0<=index && index<size_tot);)

		if(index/size_i == n_o){
			v_i[k] = index%size_i;
		} else {
			v_i[k] = v_i_inBlock;
			for(Int irhs=0; irhs<nrhs; ++irhs){
				v_o_[irhs][k] = init(u_t[irhs*size_tot + index]);}
		}

	}

	if(debug_print && n_i==0){
		printf("in Linear update, n_i=%i\n",n_i);
		printf("rhs : %f\n",rhs_[0]);
		printf("diag %f, v_i=%i,%i,v_o=%f,%f,weight=%f,%f\n",
			diag,v_i[0],v_i[1],v_o_[0][0],v_o_[0][1],weight[0],weight[1]);
		printf("n_t %i, size_tot %i\n",n_t,size_tot);
	}

	__syncthreads();

	// Gauss-Siedel iterations
	for(Int iter=0; iter<niter_i; ++iter){
		for(Int irhs=0; irhs<nrhs; ++irhs){
			Scalar * u_i = u_i_[irhs];
			const Scalar rhs = rhs_[irhs];
			const Scalar * v_o = v_o_[irhs];

			// Accumulate the weighted neighbor values
			Scalar sum=rhs;
			for(Int k=0; k<nindex; ++k){
				const Int w_i = v_i[k];
				if(w_i==v_i_invalid) {continue;}
				HFM_DEBUG(assert(w_i==v_i_inBlock || (0<=w_i && w_i<size_i));)
				const Scalar val = w_i==v_i_inBlock ? v_o[k] : u_i[w_i];
				sum += weight[k] * val;
			}

			// Normalize and store 
			u_i[n_i] = sum/diag;

		} // for irhs
		__syncthreads();
	} // for iter
	
	// Export and check for changes
	bool changed=false;
	for(Int irhs=0; irhs<nrhs; ++irhs){
		const Scalar val = u_i_[irhs][n_i];
		u_t[irhs*size_tot + n_t] = val;
		const Scalar old = u_old[irhs];
		const Scalar tol = abs(val)*rtol + atol;
		changed = changed || abs(val - old) > tol;
	}


	chg_i[n_i] = changed ? 
	#if minchg_freeze_macro
	chg_t[n_t] // Changed blocks with large values will be temporarily frozen
	#else
	0. // Dummy finite value
	#endif
	: infinity();
	__syncthreads();

	Propagation::Finalize(
		chg_i, PRUNING(updateList_o,) 
		MINCHG_FREEZE(minChgPrev_o, minChgNext_o, updatePrev_o,) updateNext_o,  
		x_o, n_o);



} // LinearUpdate


} // extern "C"