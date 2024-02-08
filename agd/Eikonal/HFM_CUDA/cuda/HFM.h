#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0
/**
This file implements common rountines for HFM-type fast marching methods 
running on the GPU based on CUDA.
*/

#include "NetworkSort.h"

/// Normalizes a multi-precision variable so that u is as small as possible
MULTIP( 
void Normalize(Scalar & u, Int & uq){
	if( u<multip_max ){
		const Int n = Int(u / multip_step);
		u -= n*multip_step;
		uq += n;
	} 
} )

/// Compares u and v, possibly in multi-precision
bool Greater(const Scalar u MULTIP(, const Int uq), const Scalar v MULTIP(, const Int vq) ){
	NOMULTIP(return u>v;)
	MULTIP(return u-v > (vq-uq)*multip_step; )
}

// --- Gets all the neighbor values ---
void HFMNeighbors(
	const Scalar v_o[__restrict__ ntot],   MULTIP(const Int vq_o[__restrict__ ntot],) 
	const Int v_i[__restrict__ ntot], 
	ORDER2(const Scalar v2_o[__restrict__ ntot],   MULTIP(const Int vq2_o[__restrict__ ntot],) 
		const Int v2_i[__restrict__ ntot], const Scalar weights[__restrict__ nact],)
	const Scalar u_i[__restrict__ size_i], MULTIP(const Int uq_i[__restrict__ size_i],)
	Scalar v[__restrict__ nact], MULTIP(Int & vqmin,) // The neighbor values
	ORDER2(bool order2[__restrict__ nact],) // Wether the second order scheme is used
	Int order[__restrict__ nact] // The order in which the neighbor values are sorted
	FLOW(NSYM(, Int side[__restrict__ nsym]) )){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	MULTIP(Int vq[nact];)
	ORDER2(NOFLOW(NSYM(Int side[nsym];)))
	for(Int k=0; k<nsym; ++k){
		for(Int s=0; s<=1; ++s){
			const Int ks = 2*k+s;
			const Int w_i = v_i[ks];
			HFM_DEBUG(assert(-1<=w_i && w_i<size_i);)
			Scalar v_ MULTIP(,vq_);
			if(w_i>=0){
				v_ = u_i[w_i] SHIFT(+v_o[ks]);
				MULTIP(vq_ = uq_i[w_i];)
			} else {
				v_ = v_o[ks];
				MULTIP(vq_ = vq_o[ks];)
			}

			if(s==0) { 
				v[k] = v_; MULTIP(vq[k] = vq_;) ORDER2_OR_FLOW(NSYM(side[k] = 0;))
			} else if( Greater(v[k] MULTIP(, vq[k]), v_ MULTIP(, vq_)) ){
				v[k] = v_; MULTIP(vq[k] = vq_;) ORDER2_OR_FLOW(NSYM(side[k] = 1;))
			}
		}
	}

	// Get the value for the forward offsets
	for(Int k=0; k<nfwd; ++k){
		const Int nk = nsym+k, n2k = 2*nsym+k;
		const Int w_i = v_i[n2k];
		HFM_DEBUG(assert(-1<=w_i && w_i<size_i);)
		if(w_i>=0){
			v[nk] = u_i[w_i] SHIFT(+v_o[n2k]);
			MULTIP(vq[nk] = uq_i[w_i];)
		} else {
			v[nk] = v_o[n2k];
			MULTIP(vq[nk] = vq_o[n2k];)
		}
	}

	// Find the minimum value for the multi-precision int, and account for it
	MULTIP(
	vqmin = Int_Max;
	for(Int k=0; k<nact; ++k){
		if(v[k]<infinity()){
			vqmin = min(vqmin,vq[k]);}
	}

	for(Int k=0; k<nact; ++k){
		v[k] += (vq[k]-vqmin)*multip_step;}
	)

	ORDER2(
	// Set the threshold for second order accuracy
	const Int n_i = threadIdx.x;
	const Scalar u0 = u_i[n_i] MULTIP(+ (uq_i[n_i] - vqmin)*multip_step);

	#if order2_threshold_weighted_macro
	Scalar diff1_sum = 0.,diff1_sum2=0.;
	for(Int k=0; k<nact; ++k){
		const Scalar diff1 = max(0.,u0 - v[k]);
		const Scalar w = weights[k];
		diff1_sum  += w*diff1;
		diff1_sum2 += (w*diff1)*diff1;}
	const Scalar diff1_bound = diff1_sum>0. ? diff1_sum2/diff1_sum : 0.;
	#else 
	Scalar diff1_bound = 0.;
	for(Int k=0; k<nact; ++k){diff1_bound=max(diff1_bound,u0-v[k]);}
	#endif

	for(Int k=0; k<nact; ++k){
		// Get the further neighbor value
		const Int ks = NSYM( k<nsym ? (2*k+side[k]) : ) (k+nsym);
		const Int w_i=v2_i[ks];
		Scalar v2;
		if(w_i>=0){
			// Drift alone only affects first order
			v2 = u_i[w_i] MULTIP(+ (uq_i[w_i]-vqmin)*multip_step) FACTOR(+v2_o[ks]); 
		} else {
			v2 = v2_o[ks] MULTIP(+ (vq2_o[ks]-vqmin)*multip_step);
		}

		// Compute the second order difference, and compare
		const Scalar diff2 = abs(u0-2*v[k]+v2);
		if(diff2 < order2_threshold*diff1_bound
		#if order2_causal_macro
		&& u0>v[k]
		#endif
		){
			order2[k]=true;
			v[k] += (v[k]-v2)/3.;
		} else {
			order2[k]=false;
		}
	}
	)

	fixed_length_sort<nact>(v,order);

} // HFMNeighbors


/// --------- Eulerian fast marching update operator -----------
void HFMUpdate(const Scalar rhs, const Scalar weights[__restrict__ nact],
	const Scalar v_o[__restrict__ ntot], MULTIP(const Int vq_o[__restrict__ ntot],) 
	const Int v_i[__restrict__ ntot],
	ORDER2(const Scalar v2_o[__restrict__ ntot],MULTIP(const Int vq2_o[__restrict__ ntot],) 
		const Int v2_i[__restrict__ ntot],)
	const Scalar u_i[__restrict__ size_i], MULTIP(const Int uq_i[__restrict__ size_i],)
	Scalar & u_out MULTIP(,Int & uq_out) 
	FLOW(, Scalar flow_weights[__restrict__ nact] NSYM(, Int active_side[__restrict__ nsym])) 
	){
	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	Scalar v[nact]; 
	MULTIP(Int vqmin;) // shared value vqmin
	ORDER2(bool order2[nact];) // Wether second order is active for this neighbor
	Int order[nact];

	HFMNeighbors(
		v_o MULTIP(,vq_o), v_i,
		ORDER2(v2_o MULTIP(,vq2_o), v2_i, weights,)
		u_i MULTIP(,uq_i), 
		v MULTIP(,vqmin) ORDER2(,order2),
		order FLOW(NSYM(, active_side)) );

	// Compute the update
	const Int k=order[0];
	const Scalar vmin = v[k]; 
	if(vmin==infinity()){
		u_out = vmin; MULTIP(uq_out=0;) 
		FLOW(for(Int k=0; k<nact; ++k){flow_weights[k]=0.;})
		return;
	}
	
	Scalar w = weights[k]; 
	ORDER2(if(order2[k]) w*=9./4.;)
	Scalar a=w, b=0., c=-rhs*rhs;

	for(Int k_=1; k_<nact; ++k_){
		const Int k = order[k_];
		const Scalar t = v[k] - vmin;
		if( (a*t-2*b)*t+c > 0 ) break; // The solution is in this interval
		Scalar w = weights[k]; 
		ORDER2(if(order2[k]) w*=9./4.;)
		a+=w;
		b+=w*t;
		c+=w*t*t;
	}
	// Delta is expected to be non-negative by Cauchy-Schwartz inequality
	// but roundoff errors happen
	const Scalar delta = b*b-a*c; 
	const Scalar sdelta = sqrt(max(Scalar(0),delta));
	const Scalar value = a==Scalar(0) ? infinity() : (b+sdelta)/a;

	const Scalar val = vmin+value;
	u_out = val; MULTIP(uq_out = vqmin; Normalize(u_out,uq_out); )

	FLOW(
	if(u_out==infinity()){
		for(Int k=0; k<nact; ++k){flow_weights[k]=0;}
		return;}
	
	for(Int k=0; k<nact; ++k){
		const Scalar w = weights[k];
		const Scalar diff = max(Scalar(0),val - v[k]);
		flow_weights[k] = w*diff;
		ORDER2(if(order2[k]) {flow_weights[k]*=3./2.;})
	})
}
