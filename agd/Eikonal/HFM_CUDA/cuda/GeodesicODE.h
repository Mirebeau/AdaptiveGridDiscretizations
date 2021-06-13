#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a basic ODE solver, devoted to backtracking the minimal geodesics
using the upwind geodesic flow computed from an Eikonal solver.
It is meant to be quite similar to the GeodesicODESolver implementation in the 
HamiltonFastMarching library.

(Note : since ODE integration is inherently a sequential process, it is admitedly a bit 
silly to solve it on the GPU. We do it here because the Python code is unacceptably slow,
and to avoid relying on compiled CPU code.)
*/

#include "static_assert.h"

/* The following, or equivalents, must be defined in including file

typedef int Int;
const Int Int_Max = 2147483647;

typedef float Scalar;

typedef unsigned int EuclT;
const EuclT EuclT_Max = 255;
const EuclT EuclT_Chart = 254;

#define ndim_macro 2;
*/

#if !online_flow_macro

const Int ndim = ndim_macro;
Scalar infinity(){return 1./0.;}

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
//const bool periodic[ndim]={false,true}; //must be defined in enclosing file
#else
#define PERIODIC(...) 
#endif

__constant__ Int shape_tot[ndim];
__constant__ Int size_tot;

#define bilevel_grid_macro 
__constant__ Int shape_o[ndim]; 
//__constant__ Int shape_i[ndim]; 
//__constant__ Int size_i;

#include "Geometry_.h"
#include "Grid.h"

#endif

const Int ncorners = 1<<ndim;
#include "GeodesicODE_Opt.h"

__constant__ Int nGeodesics;
__constant__ Int max_len = 200; // Max geodesic length
__constant__ Int pastseed_delay = 0; // Delay before checking the PastSeed stop criterion
__constant__ Scalar causalityTolerance = 4; 
__constant__ Scalar geodesicStep = 0.25;
__constant__ Scalar weight_threshold = 0.05;

/** The following constants must be defined. 
They are related with the sensitivity of the PastSeed and Stationnary stopping criteria
respectively
const Int eucl_delay 
const Int nymin_delay
*/

// History length, used for the above delayed stopping criteria
const Int hlen = 1 + (eucl_delay<nymin_delay ? nymin_delay : eucl_delay); 

namespace ODEStop {
enum Enum {
	Continue = 0, // Do not stop here
	AtSeed, // Correct termination
	InWall, // Went out of domain
	Stationnary, // Error : Stall in ODE process
	PastSeed, // Error : Moving away from target
	VanishingFlow, // Error : Vanishing flow
};
}
typedef char ODEStopT;

/** Array suffix conventions:
- t : global field [physical dims][shape_tot]
- s : data shared by all ODE solver threads [nThreads][len][physical dims]
- p : periodic buffer, for a given thread. [min_len][...]
- no suffix : basic thread dependent data.
*/

/** Computes the floor of the scalar components. Returns wether value changed.*/
bool Floor(const Scalar x[ndim], Int xq[ndim]){
	bool changed = false;
	for(Int i=0; i<ndim; ++i){
		const Int xqi = floor(x[i]);
		if(xqi!=xq[i]) changed=true;
		xq[i]=xqi;
	}
	return changed;
}

/** This function estimates the flow at position x, by a bilinear interpolation of 
the flow at neighboring corners. Some corners are excluded from the interpolation, if the
associated distance value is judged to large. The neighbor flow values are reloaded 
only if necessary. Also returns the euclidean distance (or other) from the best corner to 
the target.
Inputs : 
 - flow_args_signature_macro : data fields for flow import or recomputation
 - x : position where the flow is requested.
Outputs :
 - flow : requested flow, normalized for unit euclidean norm.
 - xq : from Floor(x). Initialize to Int_MAX before first call.
 - nymin : index of cube corner with minimal value.
 - flow_cache : flow at the cube corners.
 - dist_cache : distance at the cube corners.
 - threshold_cache : for cube corner exclusion.

 Returned value : 
 - stop : ODE stopping criterion (if any)
*/
ODEStop::Enum NormalizedFlow(
	flow_args_signature_macro
	const Scalar x[__restrict__ ndim], Scalar flow[__restrict__ ndim],
	Int xq[__restrict__ ndim], Int & nymin, Scalar & dist_threshold,
	Scalar flow_cache[__restrict__ ncorners][ndim], Scalar dist_cache[__restrict__ ncorners]){

	ODEStop::Enum result = ODEStop::Continue;
	const bool newCell = Floor(x,xq); // Get the index of the cell containing x
	Scalar weightsum_cache[ncorners];
	if(newCell){ // Load cell corners data (flow and dist)
		for(Int icorner=0; icorner< ncorners; ++icorner){
			// Get the i-th corner and its index in the total shape.
			Int yq[ndim]; 
			for(Int k=0; k<ndim; ++k){yq[k] = xq[k]+((icorner >> k) & 1);}
			if(!Grid::InRange_per(yq,shape_tot)){
				dist_cache[icorner]=infinity(); 
				continue;}
			const Int ny = Grid::Index_tot(yq);
			
			// Load or compute distance and flow 
			GeodesicFlow(
				flow_args_list_macro
				yq,ny,
				flow_cache[icorner],weightsum_cache[icorner],dist_cache[icorner]);
		} // for corner
	} // if newCell

	// Compute the bilinear weigths.
	Scalar dx[ndim]; sub_vv(x,xq,dx); 
	Scalar weights[ncorners];
	for(Int icorner=0; icorner<ncorners; ++icorner){
		weights[icorner] = 1.;
		for(Int k=0; k<ndim; ++k){
			weights[icorner]*=((icorner>>k) & 1) ? dx[k] : Scalar(1)-dx[k];}
	}
		
	if(newCell){
		// Get the point with the smallest distance, and a weight above threshold.
		Scalar dist_min=infinity();
		Int imin;
		for(Int icorner=0; icorner<ncorners; ++icorner){
			if(weights[icorner]<weight_threshold) continue;
			if(dist_cache[icorner]<dist_min) {imin=icorner; dist_min=dist_cache[icorner];}
		}

		if(dist_min==infinity()){return ODEStop::InWall;}

		// Set the distance threshold
		Int yq[ndim]; copy_vV(xq,yq); 
		for(Int k=0; k<ndim; ++k){if((imin>>k)&1) {yq[k]+=1;}}
		nymin = Grid::Index_tot(yq);
		const Scalar flow_weightsum = weightsum_cache[imin];
		if(flow_weightsum==0.){result=ODEStop::AtSeed;}
		dist_threshold=dist_min+causalityTolerance/flow_weightsum;
	} // if newCell

	// Perform the interpolation, and its normalization
	fill_kV(Scalar(0),flow);
	for(Int icorner=0; icorner<ncorners; ++icorner){
		if(dist_cache[icorner]>=dist_threshold) {continue;}
		madd_kvV(weights[icorner],flow_cache[icorner],flow);
	}
	// Not that a proper interpolation would require dividing by the weights sum
	// But this would be pointless here, due to the Euclidean normalization.
	const Scalar flow_norm = norm_v(flow);
	if(flow_norm>0){div_Vk(flow,flow_norm);}
	else if(result==ODEStop::Continue){result = ODEStop::VanishingFlow;}

	return result;
}


extern "C" {

__global__ void GeodesicODE(
	flow_args_signature_macro
	const EuclT * __restrict__ eucl_t,
	CHART(const Scalar * __restrict__ mapping_s,)
	Scalar * __restrict__ x_s, Int * __restrict__ len_s, ODEStopT * __restrict__ stop_s
	){

	const Int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=nGeodesics) return;

	// Short term periodic history introduced to avoid stalls or moving past the seed.
	EuclT eucl_p[hlen];
	Int nymin_p[hlen];
	for(Int l=0; l<hlen; ++l){
		eucl_p[l]  = EuclT_Max;
		nymin_p[l] = Int_Max;
	}

	Scalar x[ndim]; copy_vV(x_s+tid*max_len*ndim,x);
	Int xq[ndim]; fill_kV(Int_Max,xq);
	Int nymin = Int_Max;
	Scalar flow_cache[ncorners][ndim]; 
	Scalar dist_cache[ncorners];
	Scalar dist_threshold;

	Int len;
	ODEStop::Enum stop = ODEStop::Continue;
	for(len = 1; len<max_len; ++len){
		const Int l = len%hlen;
		Scalar xPrev[ndim],xMid[ndim];
		copy_vV(x,xPrev);

		// Compute the flow at the current position
		Scalar flow[ndim];
		stop = NormalizedFlow(
			flow_args_list_macro
			x,flow,
			xq,nymin,dist_threshold,
			flow_cache,dist_cache);

		if(stop!=ODEStop::Continue){break;}

		// Check PastSeed and Stationnary stopping criteria
		nymin_p[l] = nymin;
		eucl_p[l] = eucl_t[nymin];
		
		#if chart_macro 
		const bool chart = eucl_p[l]==EuclT_Chart; // Detect wether chart change needed
		if(chart) {eucl_p[l] = eucl_p[(l-1+hlen)%hlen];} // use previous value
		#endif

		if(nymin     == nymin_p[(l-nymin_delay+hlen)%hlen]){
						stop = ODEStop::Stationnary; break;}
		if(eucl_p[l] >  eucl_p[ (l-eucl_delay+hlen) %hlen && len>= pastseed_delay ]){
			stop = ODEStop::PastSeed;    break;}

		// Make a half step, to get the Euler midpoint
		madd_kvv(Scalar(0.5)*geodesicStep,flow,xPrev,xMid);

		// Compute the flow at the midpoint
		stop = NormalizedFlow(
			flow_args_list_macro
			xMid,flow,
			xq,nymin,dist_threshold,
			flow_cache,dist_cache);
		if(stop!=ODEStop::Continue){break;}

		madd_kvv(geodesicStep,flow,xPrev,x);
		copy_vV(x,x_s + (tid*max_len + len)*ndim);

		CHART(if(chart) {ChartJump(mapping_s,x);})

	}

	len_s[tid] = len;
	stop_s[tid] = ODEStopT(stop);
}

} // extern "C"