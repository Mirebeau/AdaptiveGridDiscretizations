#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements common facilities for bounds checking and array access.*/

namespace Grid {

Int mod_pos(Int x,const Int n){
	// Positive residue of x modulo n
	x=x%n;
	return x>=0 ? x : (x+n);
}

Int Index_tot(const Int x[ndim], 
	const Int shape_tot[ndim], const Int shape_o[ndim], 
	const Int shape_i[ndim], const Int size_i){
	// Get the index of a point in the full array.
	// No bounds check 
	Int n_o=0,n_i=0;
	for(Int k=0; k<ndim; ++k){
		Int xk=x[k];
		PERIODIC(if(periodic_axes[k]){xk = mod_pos(xk,shape_tot[k]);})
		HFM_DEBUG(assert(0<=xk && xk<shape_tot[k]);)
		const Int 
		s_i = shape_i[k],
		x_o = xk/s_i,
		x_i = xk%s_i;
		if(k>0) {n_o*=shape_o[k]; n_i*=s_i;}
		n_o+=x_o; n_i+=x_i; 
	}
	HFM_DEBUG(assert(0<=n_o && 0<=n_i && n_i<size_i);)
	return n_o*size_i+n_i;
}

#ifdef bilevel_grid_macro // Only if shape_tot, shape_o and shape_i are defined
Int Index_tot(const Int x[ndim]){return Index_tot(x,shape_tot,shape_o,shape_i,size_i);}
#endif

bool InRange(const Int x[ndim], const Int shape_[ndim]){
	for(int k=0; k<ndim; ++k){
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index(const Int x[ndim], const Int shape_[ndim]){
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		n+=x[k];
		HFM_DEBUG(assert(0<=x[k] && x[k]<shape_[k]);)
	}
	return n;
}

bool InRange_per(const Int x[ndim], const Int shape_[ndim]){
for(int k=0; k<ndim; ++k){
		PERIODIC(if(periodic_axes[k]) continue;)
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index_per(const Int x[ndim], const Int shape_[ndim]){
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		Int xk=x[k];
		PERIODIC(if(periodic_axes[k]){xk=mod_pos(xk,shape_[k]);})
		n+=xk;
		HFM_DEBUG(assert(0<=xk && xk<shape_[k]);)
	}
	return n;
}


void Position(Int n, const Int shape_[ndim], Int x[ndim]){
	for(Int k=ndim-1; k>=1; --k){
		x[k] = n % shape_[k];
		n /= shape_[k];
		HFM_DEBUG(assert(0<=x[k] && x[k]<shape_[k]);)
	}
	x[0] = n;
	HFM_DEBUG(assert(0<=x[0] && x[0]<shape_[0]);)
}

}

Int shape2size(const Int * shape_, Int ndim_){ // Intended for debug asserts
	Int size=1;
	for(int i=0; i<ndim_; ++i){size*=shape_[i];}
	return size;
}
