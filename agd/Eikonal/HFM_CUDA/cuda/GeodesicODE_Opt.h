#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file implements two optional features for the geodesic ODE solver : 
- jumps according to local charts, to handle general manifolds
- recomputation of the geodesic flow vector, to save memory
*/

/* The following macros must be defined outside, see below for details
//#define chart_macro false
//#define recompute_flow_macro false
*/

// ------------- chart jumps -----------------

#if chart_macro
#define CHART(...) __VA_ARGS__
/* 
This function jumps the current geodesic point to a prescribed position. 
It is required in the case of a manifold defined by local charts, when the geodesic leaves 
a given chart. The provided mapping should map in the interior of another chart.

When three or more local charts are used, the mapping is discontinuous. 
Optionally, this can be detected, and the average jump be replaced with a 
jump from rounded point.

// The following must be defined externally
const Int ndim_s; // Number of dimensions in mapping (first are broadcasted)
const EuclT EuclT_chart; // EuclT value when a jump is to be done
*/
const Int ncorner_s = 1<<ndim_s;
__constant__ Int size_s;

#ifndef chart_jump_variance_macro
#define chart_jump_variance_macro 0
#endif

#if chart_jump_variance_macro
__constant__ Scalar chart_jump_variance;
#endif

void ChartJump(const Scalar * mapping_s, Scalar x[ndim]){
	// Replaces x with mapped point in the manifold local chart.	

	// Get integer and floating point part of x. Care about array broadcasting
	const Int ndim_b = ndim - ndim_s;
	Int    xq_s[ndim_s]; 
	Scalar xr_s[ndim_s];
	Scalar * x_s = x+ndim_b; // x_s[ndim_s]
	for(Int i=0; i<ndim_s; ++i){
		xq_s[i] = floor(x_s[i]); // Integer part
		xr_s[i] = x_s[i]-xq_s[i]; // Fractional part
		x_s[i] = 0; // Erase position, for future averaging
	}

	Int    yq[ndim]; // Interpolation point
	for(Int i=0; i<ndim_b; ++i) yq[i]=0; // Broadcasting first dimensions
	
	#if chart_jump_variance_macro
	Scalar mapping_sum_s[ndim_s]; // Coordinates sum
	Scalar mapping_sqs_s[ndim_s]; // Squared coordinates sum
	for(Int i=0; i<ndim_s; ++i){mapping_sum_s[i]=0; mapping_sqs_s[i]=0;}
	#endif

	for(Int icorner=0; icorner<ncorner_s; ++icorner){
		Scalar w=1.; // Interpolation weight
		for(Int i=0; i<ndim_s; ++i){
			const Int eps = (icorner>>i) & 1;
			yq[ndim_b+i] = xq_s[i] + eps;
			w *= eps ? xr_s[i] : (1.-xr_s[i]);
		}
		const Int ny_s = Grid::Index_per(yq,shape_tot); // % size_s (unnecessary)
		for(Int i=0; i<ndim_s; ++i){
			const Scalar mapping = mapping_s[size_s*i+ny_s];
			x_s[i] += w * mapping;
			#if chart_jump_variance_macro
			mapping_sum_s[i]+=mapping;
			mapping_sqs_s[i]+=mapping*mapping;
			#endif
		}
	} // for icorner

	#if chart_jump_variance_macro 
	// Check the variance of the mapped values. If too large, use jump at rounded point.
	Scalar mapping_var = 0.;
	for(Int i=0; i<ndim_s; ++i){
		mapping_var += mapping_sqs_s[i]/ncorner_s // Variance of i-th coordinate of mapping
		 - (mapping_sum_s[i]*mapping_sum_s[i])/(ncorner_s*ncorner_s);}

	if(mapping_var < chart_jump_variance) return;

	// Variance of mapped values is too large. Chart discontinuity detected. 
	// Using mapping from nearest point
	for(Int i=0; i<ndim_s; ++i){ yq[ndim_b+i] = xq_s[i] + (xr_s[i]>0.5 ? 1 : 0);}
	const Int ny_s = Grid::Index_per(yq,shape_tot); // % size_s (un-necessary)
	for(Int i=0; i<ndim_b; ++i){x_s[i] = mapping_s[size_s*i+ny_s];}
	#endif // chart_variance_macro
}
#else
#define CHART(...) 
#endif // chart_macro

// ----------------- flow online computation --------------------

#if online_flow_macro
/** (Optional) This function computes the geodesic flow, by calling the eikonal solver.
This allows so save a significant amount of memory in comparison with precomputing the 
geodesic flow, in particular with models whose geometric information is small 
(e.g. depending only on a subset of the coordinates, as the curvature penalized models, 
and as opposed to a point dependent Riemannian metric.) 

It overlaps significanty with Update.h, somewhat violating the DRY principle unfortunately.
*/

#define flow_args_signature_macro \
	const Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) \
	const Scalar * __restrict__ geom_t, \
	const BoolPack * __restrict__ seeds_t, const Scalar * __restrict__ rhs_t, \
	WALLS(const WallT * __restrict__ wallDist_t,) \
	IO_SCHEME( 	const Scalar  * __restrict__ weights_t, \
				const OffsetT * __restrict__ offsets_t,) 

#define flow_args_list_macro \
	u_t, MULTIP(uq_t,) geom_t, seeds_t,rhs_t, \
	WALLS(wallDist_t,) IO_SCHEME(weights_t,offsets_t,) 

void GeodesicFlow(
	flow_args_signature_macro
	const Int x_t[ndim], const Int n_t,
	Scalar flow_vector[ndim], Scalar & flow_weightsum, Scalar & dist){

	// Initializations
	const Int n_o = n_t/size_i;
	const Int n_i = threadIdx.x; // Required by HFMIter
	Int x_i[ndim]; 	
	Grid::Position(n_i,shape_i,x_i);
	
	fill_kV(Scalar(0),flow_vector);
	flow_weightsum=0;

	// ------- Get the scheme structure -------
	#if geom_indep_macro
	const Int n_geom = (n_o%size_geom_o)*size_geom_i + (n_i%size_geom_i);
	#else
	const Int n_geom = n_t;
	#endif

	#if import_scheme_macro
		const Scalar * weights = weights_t+nactx*n_geom;
		const OffsetVecT offsets = (OffsetVecT) (offsets_t + ndim*nactx*n_geom);
		DRIFT(Sorry_drift_is_not_yet_compatible_with_scheme_precomputation;)
	#else
		ADAPTIVE_WEIGHTS(Scalar weights[nactx];)
		ADAPTIVE_OFFSETS(OffsetT offsets[nactx][ndim];)
		DRIFT(Scalar drift[nmix][ndim];)

		GEOM(Scalar geom[geom_size];
		for(Int k=0; k<geom_size; ++k){geom[k] = geom_t[n_geom+size_geom_tot*k];})
		ADAPTIVE_MIX(const bool mix_is_min = )
		scheme(GEOM(geom,) CURVATURE(x_t,) weights, offsets DRIFT(,drift) );
	#endif

	// ----------- Import the value at current position ----------
	// Sharedness is not used here, since the threads do not deal with a common block.
	// It is only for compatibility with the original eikonal solver
	__shared__ Scalar u_i[size_i]; u_i[n_i] = u_t[n_t];
	MULTIP(__shared__ Int uq_i[size_i]; uq_i[n_i] = uq_t[n_t];;)
	WALLS(__shared__ WallT wallDist_i[size_i]; wallDist_i[n_i] = wallDist_t[n_t];)

	dist = u_i[n_i] MULTIP(+ uq_i[n_i]*multip_step);

	// Apply boundary conditions
	const bool isSeed = GetBool(seeds_t,n_t);
	const Scalar rhs = rhs_t[n_t];
	if(isSeed){return;} // Null flow at seed

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

	Scalar flow_weights[nact]; 
	NSYM(Int active_side[nsym];) // C does not tolerate zero-length arrays.
	Int kmix=0; 

	// Solve the eikonal equation operator
	HFMIter(!isSeed, 
		rhs, ADAPTIVE_MIX(mix_is_min,) weights,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) 
		, flow_weights NSYM(, active_side) MIX(, kmix) );

	// Compute the flow
	for(Int k=0; k<nact; ++k){
		flow_weightsum+=flow_weights[k];
		Int offset[ndim]; 
		const Int eps = NSYM( k<nsym ? (2*active_side[k]-1) : ) 1;
		for(Int l=0; l<ndim; ++l){
			offset[l] = eps*offsets[kmix*nact+k][l];
			flow_vector[l]+=flow_weights[k]*offset[l];
		}
	}
}

#else // --- Using a precomputed and imported flow vector field ---

#if multiprecision_macro
#define MULTIP(...) __VA_ARGS__
__constant__ Scalar multip_step; 
#else
#define MULTIP(...)
#endif

#define flow_args_signature_macro \
	const Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) \
	const Scalar * __restrict__ flow_vector_t, const Scalar * __restrict__ flow_weightsum_t,

#define flow_args_list_macro \
	u_t, MULTIP(uq_t,) flow_vector_t, flow_weightsum_t,

void GeodesicFlow(
	flow_args_signature_macro
	const Int x_t[ndim], const Int n_t,
	Scalar flow_vector[ndim], Scalar & flow_weightsum, Scalar & dist){

	dist = u_t[n_t] MULTIP(+uq_t[n_t]*multip_step);
	flow_weightsum = flow_weightsum_t[n_t];
	for(Int k=0; k<ndim; ++k){
		flow_vector[k] = flow_vector_t[n_t+size_tot*k];}
}
#endif // online_flow_macro