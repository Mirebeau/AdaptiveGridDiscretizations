#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file setups the finite difference scheme used in the Eulerian fast marching method.
The scheme parameters (weights, offsets,drift,mix) are called. 
The finite differences which fall in the shape_i block are identified. 
The values associated to other finite diffferences are imported.
*/

#include "Walls.h"

typedef const OffsetT (*OffsetVecT)[ndim]; // OffsetVecT[][ndim]
typedef const Scalar (*DriftVecT)[ndim]; // DriftVectT[][ndim]


void FiniteDifferences(
	// Value function (problem unknown)
	const Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) 
	WALLS(const WallT * __restrict__ wallDist_t, const WallT * __restrict__ wallDist_i,)

	// Structure of the finite differences (input, output)
	const OffsetVecT __restrict__ offsets, DRIFT(const DriftVecT __restrict__ drift,)
	Int * __restrict__ v_i, Scalar * __restrict__ v_o, MULTIP(Int * __restrict__ vq_o,)
	ORDER2(Int * __restrict__ v2_i, Scalar * __restrict__ v2_o, MULTIP(Int * __restrict__ vq2_o,))
	// Position of current point
	const Int * __restrict__ x_t, const Int * __restrict__ x_i
	){

	FACTOR(
	Scalar x_rel[ndim]; // Relative position wrt the seed.
	const bool factors = factor_rel(x_t,x_rel);
	)

	// Get the neighbor values, or their indices if interior to the block
	Int koff=0,kv=0; 
	for(Int kmix=0; kmix<nmix; ++kmix){
	for(Int kact=0; kact<nact; ++kact){
		const OffsetT * e = offsets[koff]; // e[ndim]
		++koff;
		SHIFT(
			Scalar fact[2]={0.,0.}; ORDER2(Scalar fact2[2]={0.,0.};)
			FACTOR( if(factors){factor_sym(x_rel,e,fact ORDER2(,fact2));} )
			DRIFT( const Scalar s = scal_vv(drift[kmix],e); fact[0] +=s; fact[1]-=s; )
			)

		for(Int s=0; s<2; ++s){
			if(s==0 && kact>=nsym) continue;
			OffsetT offset[ndim];
			const Int eps=2*s-1; // direction of offset
			mul_kv(eps,e,offset);

			WALLS(
			const bool visible = Visible(offset, x_t,wallDist_t, x_i,wallDist_i);
			if(!visible){
				v_i[kv]=-1; ORDER2(v2_i[kv]=-1;)
				v_o[kv]=infinity(); ORDER2(v2_o[kv]=infinity();)
				MULTIP(vq_o[kv]=0;  ORDER2(vq2_o[kv]=0;) )
				{++kv; continue;}
			})

			Int y_t[ndim], y_i[ndim]; // Position of neighbor. 
			add_vv(offset,x_t,y_t);
			add_vv(offset,x_i,y_i);

			if(local_i_macro && Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)))  {
				v_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v_o[kv] = fact[s];)
			} else {
				v_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot)) {
					const Int ny_t = Grid::Index_tot(y_t);
					v_o[kv] = u_t[ny_t] SHIFT(+fact[s]);
					MULTIP(vq_o[kv] = uq_t[ny_t];)
				} else {
					v_o[kv] = infinity();
					MULTIP(vq_o[kv] = 0;)
				}
			}

			ORDER2(
			add_vV(offset,y_t);
			add_vV(offset,y_i);

			if(local_i_macro && Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)) ) {
				v2_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v2_o[kv] = fact2[s];)
			} else {
				v2_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot) ) {
					const Int ny_t = Grid::Index_tot(y_t);
					v2_o[kv] = u_t[ny_t] SHIFT(+fact2[s]);
					MULTIP(vq2_o[kv] = uq_t[ny_t];)
				} else {
					v2_o[kv] = infinity();
					MULTIP(vq2_o[kv] = 0;)
				}
			}
			) // ORDER2

			++kv;
		} // for s 
	} // for kact
	} // for kmix
	HFM_DEBUG(assert(kv==ntotx && koff==nactx);)

}