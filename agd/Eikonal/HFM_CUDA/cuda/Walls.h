#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#ifndef WallT_macro
typedef unsigned char WallT;
const WallT WallT_Max = 255; 
#endif

#include "Grid.h"

//https://stackoverflow.com/a/18067292/12508258
Int divRoundClosest(const Int n, const Int d){
  return ((n < 0) ^ (d < 0)) ? ((n - d/2)/d) : ((n + d/2)/d);}


// Checks wether the line [x, x+v] is exempt from walls
bool Visible(const Int v[ndim], 
	const Int x_t[ndim], const WallT * __restrict__ wallDist_t,
	const Int x_i[ndim], const WallT   wallDist_i[size_i]){
	const Int n_i = threadIdx.x;
	HFM_DEBUG(assert(n_i<size_i);)
	if(wallDist_i[n_i]==WallT_Max) return true;		

	// L1 norm of the vector will be compared with L1 distance to the walls.
	Int vl1 = 0; for(Int i=0; i<ndim; ++i) {vl1 += abs(v[i]);}
	if(vl1 < wallDist_i[n_i]) return true; // Walls are far enough

	// Start walking from the source point to the tip
	Int vlinf = 0; for(Int i=0; i<ndim; ++i) {vlinf = max(vlinf,abs(v[i]));}

	for(Int k=1; k<=vlinf; ++k){
		WallT value=WallT_Max;
		Int w[ndim]; Int diffl1=0;
		for(Int i=0; i<ndim; ++i){
			w[i] = divRoundClosest(v[i]*k,vlinf);
			diffl1 += abs(v[i]-w[i]);} // L1 norm of v-w

		Int y_i[ndim]; add_vv(x_i,w,y_i);
		if(Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_i,shape_tot)) && local_i_macro){
			value = wallDist_i[Grid::Index(y_i,shape_i)];
		} else {
			Int y_t[ndim]; add_vv(x_t,w,y_t);
			if(Grid::InRange_per(y_t,shape_tot)){value = wallDist_t[Grid::Index_tot(y_t)];}
		}
		if(diffl1 < value)  {return true;}
		else if(value==0) {return false;}
	}
	return true;
}