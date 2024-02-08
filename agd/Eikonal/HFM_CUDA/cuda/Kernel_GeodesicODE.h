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

/* Specific to DubinsState_.h model. How to encode states in the backtracking.
A mixed state lies in the probability simplex. The alternative is to code state
in the last coordinate, in which case it is either pure or a superposition of two 
sucessive states.
*/
#if mixed_state_macro
#define STATE(...) __VA_ARGS__
const Int nstates = nstates_macro;
const Int pdim = ndim-1; // Physical dimension
const Int xdim = pdim+nstates; // Total dimension, including state weights
const Int ncorners = (1<<pdim)*nstates;
#else
#define STATE(...) 
const Int ncorners = 1<<ndim;
const Int pdim = ndim, xdim = ndim; // only physical coordinates (no state)
#endif

__constant__ Int shape_tot[ndim];
__constant__ Int size_tot;

#define bilevel_grid_macro 
__constant__ Int shape_o[ndim]; 
//__constant__ Int shape_i[ndim]; 
//__constant__ Int size_i;

#include "Geometry_.h"
#include "Grid.h"

namespace _xdim {
	const Int ndim=xdim;
	#include "Geometry_.h"
}

#endif

#include "GeodesicODE_Opt.h"

__constant__ Int nGeodesics;
__constant__ Int max_len = 200; // Max geodesic length
__constant__ Int pastseed_delay = 0; // Delay before checking the PastSeed stop criterion
__constant__ Scalar causalityTolerance = 4; 
__constant__ Scalar geodesicStep = 0.25;
//__constant__ Scalar weight_threshold = 0.05;
const Scalar weight_threshold = 0.5/ncorners;

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
bool Floor(const Scalar x[xdim], Int xq[pdim]){
	bool changed = false;
	for(Int i=0; i<pdim; ++i){
		const Int xqi = floor(x[i]);
		if(xqi!=xq[i]) changed=true;
		xq[i]=xqi;
	}
	return changed;
}

/*Computes a neighbor of xq for the flow interpolation*/
void Neighbor(const Int xq[pdim], const Int icorner, Int yq[__restrict__ ndim]){
	for(Int k=0; k<pdim; ++k){yq[k] = xq[k]+((icorner >> k) & 1);}
	STATE(yq[pdim] = icorner>>pdim;)
}

/*The state barycentric coefficients should be non-negative and sum to one*/
void NormalizeState(Scalar x[xdim]){
	Scalar sum = 0;
	for(Int k=pdim; k<xdim; ++k){
		x[k]=max(Scalar(0),x[k]);
		sum+=x[k];
	}
	for(Int k=pdim; k<xdim; ++k){x[k]/=sum;}
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
	STATE(const Scalar * __restrict__ flow_state_t,)
	const Scalar x[__restrict__ xdim], Scalar flow[__restrict__ xdim],
	Int xq[__restrict__ pdim], Int & nymin, Scalar & dist_threshold,
	Scalar flow_cache[__restrict__ ncorners][xdim], Scalar dist_cache[__restrict__ ncorners]){

	const bool newCell = Floor(x,xq); // Get the index of the cell containing x

	// Compute the weigths using multilinear interpolation.
	Scalar dx[pdim]; 
	for(Int k=0;k<pdim;++k) dx[k]=x[k]-xq[k]; 
	Scalar weights[ncorners];
//	printf("\nweights ");
	for(Int icorner=0; icorner<ncorners; ++icorner){
		weights[icorner] = 1.;
		for(Int k=0; k<pdim; ++k){
			weights[icorner]*=((icorner>>k) & 1) ? dx[k] : Scalar(1)-dx[k];}
		STATE(weights[icorner]*=x[pdim+(icorner>>pdim)];)
//		printf("%f ",weights[icorner]);
	}


	Scalar weightsum_cache[ncorners];
	if(newCell){ // Load cell corners data (flow and dist)
		for(Int icorner=0; icorner< ncorners; ++icorner){
			// Get the i-th corner and its index in the total shape.
			Int yq[ndim]; Neighbor(xq,icorner,yq);
			if(!Grid::InRange_per(yq,shape_tot)){
				dist_cache[icorner]=infinity(); 
				continue;}
			const Int ny = Grid::Index_tot(yq);

//			printf("\nny %i",ny);
			
			// Load or compute distance and flow 
			GeodesicFlow(
				flow_args_list_macro
				yq,ny,
				flow_cache[icorner],weightsum_cache[icorner],dist_cache[icorner]);
//			printf("\n");
//			for(int k=0; k<ndim;++k) printf(" %f ",flow_cache[icorner][k]);
//			printf("\n");
			STATE(for(int k=0;k<nstates;++k){
					flow_cache[icorner][pdim+k]=flow_state_t[ny+size_tot*k];})

//			for(int k=0; k<xdim;++k) printf(" %f ",flow_cache[icorner][k]);

		} // for corner

		// Get the point with the smallest distance, and a weight above threshold.
		Scalar dist_min=infinity();
		Int imin;
		for(Int icorner=0; icorner<ncorners; ++icorner){
			if(weights[icorner]<weight_threshold) continue;
			if(dist_cache[icorner]<dist_min) {imin=icorner; dist_min=dist_cache[icorner];}
		}

		if(dist_min==infinity()){return ODEStop::InWall;}

		/* Set the distance threshold
		This computation is a bit dubious : the distance threshold is relevant only if 
		the scheme coefficients are O(1). 
		Alternative : we could look at the variance of the flows that are averaged, and 
		use the one of nymin in case of a large variance.
		*/
		Int yq[ndim]; Neighbor(xq,imin,yq);
		nymin = Grid::Index_tot(yq);
		const Scalar flow_weightsum = weightsum_cache[imin];
		if(flow_weightsum==0.){return ODEStop::AtSeed;}
		dist_threshold=dist_min+causalityTolerance/flow_weightsum;
	} // if newCell

	// Perform the interpolation, and its normalization
	_xdim::fill_kV(Scalar(0),flow); 
	for(Int icorner=0; icorner<ncorners; ++icorner){
		if(STATE(false &&) dist_cache[icorner]>=dist_threshold) {continue;}
		_xdim::madd_kvV(weights[icorner],flow_cache[icorner],flow); 
	}
	// Note that a proper interpolation would require dividing by the weights sum
	// But this would be pointless here, due to the Euclidean normalization.


#if mixed_state_macro // Normalize independently the physical and state parts of the flow.
	Scalar pnorm2=0, snorm2=0; // Squared norms of physical and state part
	for(Int k=0;    k<pdim; ++k) {pnorm2+=flow[k]*flow[k];}
	for(Int k=pdim; k<xdim; ++k) {snorm2+=flow[k]*flow[k];}
	if(pnorm2>0){const Scalar r=1/sqrt(pnorm2); for(Int k=0;   k<pdim;++k) flow[k]*=r;}
	if(snorm2>0){const Scalar r=1/sqrt(snorm2); for(Int k=pdim;k<xdim;++k) flow[k]*=r;}
	if(not (pnorm2>0 || snorm2>0)){return ODEStop::VanishingFlow;}
#else // Normalize the flow
	const Scalar flow_norm = _xdim::norm_v(flow);
	if(flow_norm>0){_xdim::div_Vk(flow,flow_norm);}
	else {return ODEStop::VanishingFlow;}
#endif
	return ODEStop::Continue;
}


extern "C" {

__global__ void GeodesicODE(
	flow_args_signature_macro
	const EuclT * __restrict__ eucl_t, // Approximate Euclidean distance to the seed(s)
	CHART(const Scalar * __restrict__ mapping_s,)
	STATE(const Scalar * __restrict__ flow_state_t,)
	Scalar * __restrict__ x_s,  // Positions of backtracked geodesic
	Int * __restrict__ len_s,   // Length of backtracked geodesic
	ODEStopT * __restrict__ stop_s // Stopping criterion which occured
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

	Scalar x[xdim]; _xdim::copy_vV(x_s+tid*max_len*xdim,x);
	Int xq[pdim]; for(Int k=0;k<pdim;++k) xq[k]=Int_Max;
	Int nymin = Int_Max;
	Scalar flow_cache[ncorners][xdim]; 
	Scalar dist_cache[ncorners];
	Scalar dist_threshold;
//	STATE(Scalar dist_prev_jump = infinity();)

	Int len;
	ODEStop::Enum stop = ODEStop::Continue;
	for(len = 1; len<max_len; ++len){
//		printf("\nIn Loop");
		const Int l = len%hlen;
		Scalar xPrev[xdim],xMid[xdim];
		_xdim::copy_vV(x,xPrev);

		// Compute the flow at the current position
		Scalar flow[xdim];
		stop = NormalizedFlow(
			flow_args_list_macro
			STATE(flow_state_t,)
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
			stop = ODEStop::PastSeed; break;}


//		STATE(flow[ndim-1]=0;) //Forbid points in between states (coded in last dimension)

//		printf("\n x %f %f %f %f ", x[0],x[1],x[2],x[3]);
//		printf("flow %f %f %f %f ", flow[0],flow[1],flow[2],flow[3]);


		// Make a half step, to get the Euler midpoint
		_xdim::madd_kvv(Scalar(0.5)*geodesicStep,flow,xPrev,xMid);
		STATE(NormalizeState(xMid);)

		// Compute the flow at the midpoint
		stop = NormalizedFlow(
			flow_args_list_macro
			STATE(flow_state_t,)
			xMid,flow,
			xq,nymin,dist_threshold,
			flow_cache,dist_cache);
		if(stop!=ODEStop::Continue){break;}
		
/*		
		STATE( // Jump to adequate new state if needed
		if(flow[ndim-1]!=0){ // There may be a jump
			flow[ndim-1]=0; //Forbid points in between states (coded in last dimension)
			// Find if there is a jump associated to a strictly smaller value than previously
			// (Strict decreasing required to avoid oscillation between states)
			//Scalar dist_min_jump=dist_prev_jump;
			Int i_jump=-1;
			for(Int icorner=0; icorner<ncorners; ++icorner){
				if(flow_cache[icorner][ndim-1]!=0 && dist_cache[icorner]<dist_prev_jump){
					dist_prev_jump = dist_cache[icorner];
					i_jump = icorner;
				}
			}
			printf("\nJump attempt ");
			printf("d %f, i_jump %i ",dist_prev_jump,i_jump);
			printf("xq %i, %i, %i, %i ", xq[0],xq[1],xq[2],xq[3]); 

			if(i_jump!=-1){
				// Get the coordinates yq of the jump, the grid index ny.
				Int yq[ndim]; 
				for(Int k=0; k<ndim; ++k){yq[k] = xq[k]+((i_jump >> k) & 1);}
				const Int ny = Grid::Index_tot(yq);
				const Int flow_weightpos = flow_weightpos_t[ny];
				// Caution : these constants nact, nmix, are specific to the DubinsState_.h model
				const int nact = (ndim*(ndim-1))/2 + 2*(ndim==5), // decompdim for ndim-1 
				nmix = ncontrols_macro+1; 
				Int newState = __ffs(flow_weightpos)-1; // Equivalent to __ctz, which is not implemented 
				newState += (xPrev[ndim-1]<=newState); // See DubinsState_.h for jump coding
				xPrev[ndim-1] = newState;
				zero_V(flow); // We jumped, which invalidates the rest of the flow.
			}*/

			/*
			const Scalar dist_nymin = u_t[nymin] MULTIP(+uq_t[nymin]*multip_step);
			const Int flow_weightpos = flow_weightpos_t[nymin];
			// Caution : these constants nact, nmix, are specific to the DubinsState_.h model
			const int nact = (ndim*(ndim-1))/2 + 2*(ndim==5), // decompdim for ndim-1 
			nmix = ncontrols_macro+1; 
			printf("nymin, %i,  imax %i", nymin, flow_weightpos>>nact);
			if( 
				(flow_weightpos>>nact) == nmix-1 // best neighbor nymin requires a jump
//				&& dist_nymin < dist_prev_jump // and value has improved since last jump
			){//Effectively jumping to another state
				printf("Jump done");
				dist_prev_jump = dist_nymin;
				Int newState = __ffs(flow_weightpos)-1; // Equivalent to __ctz, which is not implemented 
				newState += (xPrev[ndim-1]<=newState); // See DubinsState_.h for jump coding
				xPrev[ndim-1] = newState;
				zero_V(flow); // We jumped, which invalidates the rest of the flow.
			} 
			flow[ndim-1]=0;//Forbid points in between states (coded in last dimension)
			*/
//		}) // STATE

		_xdim::madd_kvv(geodesicStep,flow,xPrev,x);
		STATE(NormalizeState(x);)
		_xdim::copy_vV(x,x_s + (tid*max_len + len)*xdim);

		CHART(if(chart) {ChartJump(mapping_s,x);})

	}

	len_s[tid] = len;
	stop_s[tid] = ODEStopT(stop);
}

} // extern "C"