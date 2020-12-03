#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements "dimension generic" (a.k.a dimension 4 and 5) tools 
for Voronoi's first reduction of quadratic forms.*/

#ifndef Voronoi_maxiter_macro
const Int Voronoi_maxiter=100;
#endif

namespace Voronoi {


void KKT(const SimplexStateT & state, Scalar weights[decompdim], OffsetT offsets[decompdim][ndim]);

void FirstGuess(SimplexStateT & state){
	state.objective = 1./0.; 
	for(int ivertex=0; ivertex<nvertex; ++ivertex){
		const Scalar obj = scal_mm(state.m,vertex_[ivertex]);
		if(obj>=state.objective) continue;
		state.vertex=ivertex;
		state.objective=obj;
	}
}

#if ndim_macro<6
void SetNeighbor(SimplexStateT & state,const Int neigh){
	// Record the new change of coordinates
//	const small * neigh_chg_flat = state.vertex==0 ? neigh_chg0[neigh] : neigh_chg1[neigh];
//	typedef const small (*smallMatrixT)[ndim];
//	const small (* neigh_chg)[ndim] = (smallMatrixT) neigh_chg_flat;
	Scalar a[ndim][ndim];  copy_aA(neigh_chg_[state.vertex][neigh],a);
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa);
	dot_aa(a,sa,state.a);
	
	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm);
	tgram_am(a,sm,state.m);

	state.vertex = neigh_vertex_[state.vertex][neigh];
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateT & state){
	const uchar * iw   = iw_[state.vertex];
	const uchar * iwend = iw+iwlen_[state.vertex];
	const uchar * stop = stop_[state.vertex];
	Scalar obj  = state.objective;
	Scalar bestObj=obj;
	int k=0, bestK = -1;
	const uchar * stopIt=stop; Int stop8=0;
	for(const uchar * iwIt=iw; iwIt!=iwend; ++iwIt, ++stop8){
		if(stop8==8){stop8=0; ++stopIt;}
		uchar s = *iwIt;
		const int ind = int(s >> 4);
		s = s & 15;
		const Scalar wei = Scalar(s) - Scalar(s>=2 ? 1: 2);
		obj += wei*state.m[ind];

		if(!(((*stopIt)>>stop8)&1)) continue;
		if(obj<bestObj) {
			bestObj=obj;
			bestK=k;}
		++k;
	}
	if(bestK==-1) return false;
	state.objective=bestObj; // Note : roundoff error could be an issue ?
	SetNeighbor(state,bestK); // neighs[bestK]
	return true;
}
#endif
} // namespace Voronoi

void decomp_m(const Scalar m[symdim],
	Scalar weights[decompdim], OffsetT offsets[decompdim][ndim]){
	using namespace Voronoi;
	SimplexStateT state;
	copy_mM(m,state.m);
	identity_A(state.a);
	FirstGuess(state); 
	for(Int i=0; i<Voronoi_maxiter; ++i){if(!BetterNeighbor(state)){break;}} 
	KKT(state,weights,offsets);
}
