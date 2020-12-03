#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* The asymmetric quadratic scheme reads max(min(a,b),c), 
where a is a Rander scheme, and b,c are Riemann schemes*/
#define nmix_macro 3 
#define alternating_mix_macro 1
const bool mix_is_min[3] = {true,true,false}; // First element doesn't really count
#define drift_macro 1

// ndim_macro must be defined
#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 5)
#include "Geometry5.h"
#endif

const Int nsym = decompdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int geom_size = symdim + ndim;
const Int factor_size = geom_size;
// nactx = nmix*nsym

#include "Constants.h"

void scheme(const Scalar geom[geom_size], 
	Scalar weights[nactx], OffsetT offsets[nactx][ndim], Scalar drift[3][ndim] ){
	STATIC_ASSERT(nact==decompdim && nactx==3*decompdim, inconsistent_scheme_structure);

	const Scalar * m = geom; // m[symdim]
	const Scalar * eta = geom+symdim; // eta[ndim]
	Scalar w[ndim]; dot_mv(m,eta,w);
	Scalar wwT[symdim]; self_outer_v(w,wwT);

	// Produce the two ellipses to be glued
	decomp_m(m,weights+nact,offsets+nact); zero_V(drift[1]);

	Scalar mwwT[symdim]; add_mm(m,wwT,mwwT);
	decomp_m(mwwT,weights+2*nact,offsets+2*nact); zero_V(drift[2]);

	// Computes an ellipse in between the two halves
	const Scalar n2 = scal_vv(w,eta); // | w |_{M^{-1}}
	const Scalar n = sqrt(n2);
	const Scalar in2 = sqrt(1.+n2);
	const Scalar iin2 = 1.+in2;
	const Scalar iin2_2 = iin2*iin2;
	const Scalar iin2_3 = iin2*iin2_2;
	const Scalar lambda = n/(2.*in2*iin2);
	const Scalar mu = 4.*in2/iin2_3;
	const Scalar gamma = 4.*in2/iin2_2; //4.*(1.+n2)/iin2_2 - n2*mu;

	for(Int i=0; i<symdim; ++i){mwwT[i] = gamma*m[i]+mu*wwT[i];}
	decomp_m(mwwT,weights,offsets);
	mul_kv(-lambda,eta,drift[0]);
}

#include "EuclideanFactor.h"

FACTOR(
Scalar asym_quad_norm(const Scalar m[symdim], const Scalar w[ndim], const Scalar v[ndim]){
	const Scalar s = max(Scalar(0.),scal_vv(w,v));
	return sqrt(scal_vmv(v,m,v)+s*s);
}

Scalar asym_quad_norm(const Scalar m[symdim], const Scalar w[ndim], const Scalar v[ndim],
	Scalar grad[__restrict__ ndim]){ // Similar but returning also the gradient
	const Scalar s = max(Scalar(0.),scal_vv(w,v));
	dot_mv(m,v,grad);
	const Scalar norm = sqrt(scal_vv(grad,v)+s*s);
	madd_kvV(s,w,grad);
	div_Vk(grad,norm);
	return norm;
}

/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products and norms
	const Scalar * m = factor_metric; // m[symdim]
	const Scalar * w = factor_metric+symdim; // w[ndim]

	Scalar grad[ndim]; const Scalar Nx = asym_quad_norm(m,w, x,grad);
	Scalar xpe[ndim],xme[ndim]; add_vv(x,e,xpe); sub_vv(x,e,xme);
	const Scalar Nxpe = asym_quad_norm(m,w,xpe); 
	const Scalar Nxme = asym_quad_norm(m,w,xme); 

	ORDER2(
	Scalar xpe2[ndim],xme2[ndim]; add_vv(xpe,e,xpe2); sub_vv(xme,e,xme2);
	const Scalar Nxpe2 = asym_quad_norm(m,w,xpe2); 
	const Scalar Nxme2 = asym_quad_norm(m,w,xme2); 
	)

	generic_factor_sym(scal_vv(grad,e),Nx,Nxpe,Nxme,fact ORDER2(,Nxpe2,Nxme2,fact2));
}
) 

#include "Update.h"