#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "Constants.h"
#include "Grid.h"
#include "HFMIter.h"
#include "REDUCE_i.h"
#include "GetBool.h"
#include "Propagation.h"
#include "FiniteDifferences.h"
/* Array suffix convention : 
 arr_t : shape_tot (Total domain shape)
 arr_o : shape_o (Grid shape)
 arr_i : shape_i (Block shape)
 arr : thread level object
*/
extern "C" {

__global__ void Update(
	// Value function (problem unknown)
	STRICT_ITER_O(const) Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) 
	STRICT_ITER_O(Scalar * __restrict__ uNext_t, MULTIP(Int * __restrict__ uqNext_t,))

	// Problem data
	const Scalar * __restrict__ geom_t, 
	const BoolPack * __restrict__ seeds_t, const Scalar * __restrict__ rhs_t, 
	WALLS(const WallT * __restrict__ wallDist_t,)

	// Export or import the finite differences scheme 
	IO_SCHEME( 	IMPORT_SCHEME(const) Scalar  * __restrict__ weights_t,
				IMPORT_SCHEME(const) OffsetT * __restrict__ offsets_t,)

	// Causality based freezing
	MINCHG_FREEZE(const Scalar * __restrict__ minChgPrev_o, Scalar * __restrict__ minChgNext_o,)

	// Exports
	FLOW_WEIGHTS(  Scalar  * __restrict__ flow_weights_t,) 
	FLOW_WEIGHTSUM(Scalar  * __restrict__ flow_weightsum_t,)
	FLOW_OFFSETS(  OffsetT * __restrict__ flow_offsets_t,) 
	FLOW_INDICES(  Int     * __restrict__ flow_indices_t,) 
	FLOW_VECTOR(   Scalar  * __restrict__ flow_vector_t,) 

	// where to update
	Int * __restrict__ updateList_o, 
	FIM(const BoolAtom * __restrict__ scorePrev_o, BoolAtom * __restrict__ scoreNext_o,) 
	PRUNING(BoolAtom * __restrict__ updatePrev_o,) BoolAtom * __restrict__ updateNext_o 
	){ 

	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if( Propagation::Abort(
		updateList_o,PRUNING(updatePrev_o,) 
		MINCHG_FREEZE(minChgPrev_o,minChgNext_o,updateNext_o,)
		x_o,n_o) ){return;} // Also sets x_o, n_o

	const Int n_i = threadIdx.x;
	Int x_i[ndim];
	Grid::Position(n_i,shape_i,x_i);

	Int x_t[ndim];
	for(Int k=0; k<ndim; ++k){x_t[k] = x_o[k]*shape_i[k]+x_i[k];}
	const Int n_t = n_o*size_i + n_i;

	// ------- Get the scheme structure -------
	#if geom_indep_macro
	const int n_geom = (n_o%size_geom_o)*size_geom_i + (n_i%size_geom_i);
	EXPORT_SCHEME(if(n_o>=size_geom_o || n_i>=size_geom_i) return;)
	#else
	const int n_geom = n_t; 
	#endif

	#if import_scheme_macro
		const Scalar * weights = weights_t+nactx*n_geom;
		const OffsetVecT offsets = (OffsetVecT) (offsets_t + ndim*nactx*n_geom);
		// Strangely, simply copying the data at this point makes the code twice slower
		DRIFT(Sorry_drift_is_not_yet_compatible_with_scheme_precomputation;)
	#else
		ADAPTIVE_WEIGHTS(Scalar weights[nactx];)
		ADAPTIVE_OFFSETS(OffsetT offsets[nactx][ndim];)
		DRIFT(Scalar drift[nmix][ndim];)

	#if geom_first_macro
		GEOM(Scalar geom[geom_size];
		for(Int k=0; k<geom_size; ++k){geom[k] = geom_t[n_geom+size_geom_tot*k];})
	#else
		const Scalar * geom = geom_t + n_geom*geom_size;
	#endif
		ADAPTIVE_MIX(const bool mix_is_min = )
		scheme(GEOM(geom,) LOCAL_SCHEME(x_t,) weights, offsets DRIFT(,drift) );
	#endif


	EXPORT_SCHEME( 
		/* This precomputation step is mostly intended for the curvature penalized
		models, which have complicated stencils, yet usually depending on 
		a single parameter : the angular coordinate.*/
		for(Int i=0; i<nactx; ++i) {
			weights_t[i+nactx*n_geom] = weights[i];
			for(Int j=0; j<ndim; ++j){
				offsets_t[j+ndim*(i+nactx*n_geom)] = offsets[i][j];}
		}
		return;
	)

	// ----------- Import the value at current position ----------
	const Scalar u_old = u_t[n_t]; 
	MULTIP(const Int uq_old = uq_t[n_t];)

	__shared__ Scalar u_i[size_i]; // Shared block values
	u_i[n_i] = u_old;
	MULTIP(__shared__ Int uq_i[size_i];
	uq_i[n_i] = uq_old;)

	// Apply boundary conditions
	const bool isSeed = GetBool(seeds_t,n_t);
	const Scalar rhs = rhs_t[n_t];
	if(isSeed){u_i[n_i]=rhs; MULTIP(uq_i[n_i]=0; Normalize(u_i[n_i],uq_i[n_i]);)}

	WALLS(
	__shared__ WallT wallDist_i[size_i];
	wallDist_i[n_i] = wallDist_t[n_t];
	__syncthreads();
	)

	// ----------- Setup the finite differences --------
	Int    v_i[ntotx]; // Index of neighbor, if inside the block
	Scalar v_o[ntotx]; // Value of neighbor, if outside the block
	MULTIP(Int vq_o[ntotx];) // Value of neighbor, complement, if outside the block
	ORDER2(
		Int v2_i[ntotx];
		Scalar v2_o[ntotx];
		MULTIP(Int vq2_o[ntotx];)
		)

	FiniteDifferences(
		u_t,MULTIP(uq_t,)
		WALLS(wallDist_t,wallDist_i,)
		offsets,DRIFT(drift,)
		v_i,v_o,MULTIP(vq_o,)
		ORDER2(v2_i,v2_o,MULTIP(vq2_o,)) 
		x_t,x_i);

	__syncthreads(); // __shared__ u_i

	FLOW(
	Scalar flow_weights[nact]; 
	NSYM(Int active_side[nsym];) // C does not tolerate zero-length arrays.
	Int kmix=0; 
	) 

	// Compute and save the values
	HFMIter(!isSeed, 
		rhs, ADAPTIVE_MIX(mix_is_min,) weights,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) 
		FLOW(, flow_weights NSYM(, active_side) MIX(, kmix) ) );

	#if strict_iter_o_macro
	uNext_t[n_t] = u_i[n_i];
	MULTIP(uqNext_t[n_t] = uq_i[n_i];)
	#else
	u_t[n_t] = u_i[n_i];
	MULTIP(uq_t[n_t] = uq_i[n_i];)
	#endif

	FLOW( // Extract and export the geodesic flow
	if(isSeed){ // HFM leaves these fields to their (unspecified) initial state
		for(Int k=0; k<nact; ++k){
			flow_weights[k]=0; 
			NSYM(active_side[k]=0;)}
		MIX(kmix=0;)
	}

	FLOW_VECTOR(Scalar flow_vector[ndim]; fill_kV(Scalar(0),flow_vector);)
	FLOW_WEIGHTSUM(Scalar flow_weightsum=0;)

	for(Int k=0; k<nact; ++k){
		FLOW_WEIGHTS(flow_weights_t[n_t+size_tot*k]=flow_weights[k];)
		FLOW_WEIGHTSUM(flow_weightsum+=flow_weights[k];)
		Int offset[ndim]; FLOW_INDICES(Int y_t[ndim];)
		const Int eps = NSYM( k<nsym ? (2*active_side[k]-1) : ) 1;
		for(Int l=0; l<ndim; ++l){
			offset[l] = eps*offsets[kmix*nact+k][l];
			FLOW_INDICES(y_t[l] = x_t[l]+offset[l];)
			FLOW_OFFSETS(flow_offsets_t[n_t+size_tot*(k+nact*l)]=offset[l];)
			FLOW_VECTOR(flow_vector[l]+=flow_weights[k]*offset[l];)
		}
		FLOW_INDICES(flow_indices_t[n_t+size_tot*k]=
			flow_weights[k]!=0 ? Grid::Index_tot(y_t) : Int_Max;) 
	}
	FLOW_WEIGHTSUM(flow_weightsum_t[n_t]=flow_weightsum;)
	FLOW_VECTOR(for(Int l=0; l<ndim; ++l){flow_vector_t[n_t+size_tot*l]=flow_vector[l];})
	) // FLOW 
	
	// Find the smallest value which was changed.
	const Scalar u_diff = 
		abs(u_old - u_i[n_i] MULTIP( + (uq_old - uq_i[n_i]) * multip_step ) );
	// Extended accuracy ditched from this point
	MULTIP(u_i[n_i] += uq_i[n_i]*multip_step;)
	const Scalar tol = atol + rtol*abs(u_i[n_i]);

	// inf-inf naturally occurs on boundary, so ignore NaNs differences
	if(isnan(u_diff) || u_diff<=tol){u_i[n_i] = infinity();}

	__syncthreads(); // Get all values before reduction

	Propagation::Finalize(
		u_i, PRUNING(updateList_o,) FIM(scorePrev_o,scoreNext_o,)
		MINCHG_FREEZE(minChgPrev_o, minChgNext_o, updatePrev_o,) updateNext_o,  
		x_o, n_o);
}

} // Extern "C"