/** This file implements a 'vector decomposition' based on the decomposition of a the 
positive definite matrix obtained by relaxing the self outer product of that vector.
In particular, decomp_m must be defined.
*/

// Based on Selling decomposition, with some relaxation, reorienting of offsets, and pruning of weights
void decomp_v(const Scalar v[ndim], Scalar weights[decompdim], OffsetT offsets[decompdim][ndim]){

	// Build and decompose the relaxed self outer product of v
	Scalar m[symdim];
	self_outer_relax_v(v,decomp_v_relax,m);	
	decomp_m(m,weights,offsets);
	const Scalar vv = scal_vv(v,v);

	for(Int k=0; k<decompdim; ++k){
		OffsetT * e = offsets[k]; // e[ndim]
		const Scalar ve = scal_vv(v,e), ee = scal_vv(e,e);
		// Eliminate offsets which deviate too much from the direction of v
		DECOMP_V_ALIGN(if(ve*ve < vv*ee* decomp_v_cosmin2){weights[k]=0; continue;})
		// Redirect offsets in the direction of v
		if(ve>0){neg_V(e);} // Note : we want ve<0.
	}
}