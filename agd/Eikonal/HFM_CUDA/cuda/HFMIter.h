#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0
/**
This file implements the of a block of values, in the HFM algorithm.
*/

#include "HFM.h"

void HFMIter(const bool active, 
	const Scalar rhs, ADAPTIVE_MIX(const bool mix_is_min,) const Scalar weights[__restrict__ nactx],
	const Scalar v_o[__restrict__ ntotx], MULTIP(const Int vq_o[__restrict__ ntotx],) 
	const Int v_i[__restrict__ ntotx], 
	ORDER2(const Scalar v2_o[__restrict__ ntotx], MULTIP(const Int vq2_o[__restrict__ ntotx],) 
		const Int v2_i[__restrict__ ntotx],)
	Scalar u_i[__restrict__ size_i] MULTIP(, Int uq_i[__restrict__ size_i]) 
	FLOW(, Scalar flow_weights[__restrict__ nact] NSYM(, Int active_side[__restrict__ nsym]) 
		MIX(, Int & kmix_) ) ){
	const Int n_i = threadIdx.x;

/*After update to cupy 8.6, cuda toolkit 11.2, the code computes incorrect values 
without pragma unroll, for the ReedsShepp2 model, with precomputed stencils.
Could not find a sensible reason.*/
	#pragma unroll 
	for(int iter_i=0; iter_i<niter_i; ++iter_i){

	#if strict_iter_i_macro // Always on if MULTIP or NMIX are on. 

	Scalar u_new MIX(=mix_neutral(mix_is_min ALTERNATING_MIX([0])) ); 
	MULTIP(Int uq_new MIX(=0);)

	NOMIX(Scalar & u_mix = u_new; MULTIP(Int & uq_mix = uq_new;)
		FLOW(Scalar * const flow_weights_mix = flow_weights; 
			 NSYM(Int * const active_side_mix = active_side;)) )
	MIX(Scalar u_mix; MULTIP(Int uq_mix;) 
		FLOW(Scalar flow_weights_mix[nact]; NSYM(Int active_side_mix[nsym];)) )

	for(Int kmix=0; kmix<nmix; ++kmix){
		const Int s = kmix*ntot;
		HFMUpdate(
			rhs, weights+kmix*nact,
			v_o+s MULTIP(,vq_o+s), v_i+s,
			ORDER2(v2_o+s MULTIP(,vq2_o+s), v2_i+s,)
			u_i MULTIP(,uq_i),
			u_mix MULTIP(,uq_mix) 
			FLOW(, flow_weights_mix NSYM(, active_side_mix))
			);
		MIX(if(mix_is_min ALTERNATING_MIX([kmix])==
			Greater(u_new MULTIP(,uq_new), u_mix MULTIP(,uq_mix) ) ){
			u_new=u_mix; MULTIP(uq_new=uq_mix;)
			FLOW(kmix_=kmix; 
				for(Int k=0; k<nact; ++k){flow_weights[k]=flow_weights_mix[k];}
				NSYM(for(Int k=0; k<nsym; ++k){active_side[k]=active_side_mix[k];}))
		}) // Mix and better update value
	}
	__syncthreads();
	if(active DECREASING(&& Greater(u_i[n_i] MULTIP(,uq_i[n_i]),
									u_new    MULTIP(,uq_new)))) {
		u_i[n_i]=u_new; MULTIP(uq_i[n_i] = uq_new;)}
	__syncthreads();

	#else // Without strict_iter_i
	MIX(strict_iter_i_is_needed_with_mix)
	MULTIP(strict_iter_i_is_needed_with_multip)

	if(active) {
		Scalar u_new; MULTIP(Int uq_new;)
		HFMUpdate(
			rhs, weights,
			v_o MULTIP(,vq_o), v_i,
			ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
			u_i MULTIP(,uq_i),
			u_new MULTIP(,uq_new) 
			FLOW(, flow_weights NSYM(, active_side))
			);
		if(true DECREASING(&& Greater(u_i[n_i] MULTIP(,uq_i[n_i]),
									  u_new    MULTIP(,uq_new))) ) {
			u_i[n_i]=u_new; MULTIP(uq_i[n_i] = uq_new;)}
	}
	__syncthreads();

	#endif // strict_iter_i

	} // for 0<=iter_i<niter_i

}
