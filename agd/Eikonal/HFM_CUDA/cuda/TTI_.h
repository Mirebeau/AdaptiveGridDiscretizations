#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements numerical scheme for a class of Finslerian eikonal,
known as tilted transversally isotropic, and arising in seismology.

 The dual unit ball is defined by
 < linear,p > + (1/2)< p,quadratic,p > = 1
 where p is the vector containing the squares of transform*p0.
*/

/** The following constant must be defined in including file.
// Number of schemes of which to take the minimum or maximum.
#define nmix_macro 7
*/

// Min or max of schemes depending on the data
#define adaptive_mix_macro 1

#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#endif

namespace dim2 { // Some two dimensional linear algebra is needed, even in dimension 3.
	const Int ndim = 2;
	#include "Geometry_.h"
}

const Int nsym = symdim;
const Int nfwd = 0;
const Int geom_size = dim2::ndim + dim2::symdim + ndim*ndim;

// For factorization
const Int factor_size = geom_size;

#include "Constants.h"

/*Returns the two roots of a quadratic equation, a + 2 b t + c t^2.
The discriminant must be non-negative, but aside from that the equation may be degenerate*/
void solve2(const Scalar a, const Scalar b, const Scalar c, Scalar r[2]){
	const Scalar sdelta = sqrt(b*b-a*c);
	const Scalar u = - b + sdelta, v = -sdelta-b;
	if(abs(c)>abs(a)) {r[0] = u/c; r[1] = v/c;        return;}
	else if(a!=0)     {r[0] = a/u; r[1] = a/v;        return;}
	else              {r[0] = 0;   r[1] = infinity(); return;}
}

/*Returns the smallest root of the considered quadratic equation above the given threshold.
Such a root is assumed to exist.*/
Scalar solve2_above(const Scalar a, const Scalar b, const Scalar c, const Scalar above){
	Scalar r[2]; solve2(a,b,c,r);
	const bool ordered = r[0]<r[1];
	const Scalar rmin = ordered ? r[0] : r[1], rmax = ordered ? r[1] : r[0];
	return rmin>=above ? rmin : rmax;
}

namespace dim2 {

/*Scalar det_m(const Scalar m[symdim]){
	return coef_m(m,0,0)*coef_m(m,1,1)-coef_m(m,0,1)*coef_m(m,1,0);}*/
Scalar det_vv(const Scalar x[ndim], const Scalar y[ndim]){
	return x[0]*y[1]-x[1]*y[0];}

/** Returns df(x)/<x,df(x)> where f(x):= C + 2 <l,x> + <qx,x> */
void grad_ratio(const Scalar l[2], const Scalar q[3], const Scalar x[2], Scalar g[2]){
		Scalar hgrad[2]; dot_mv(q,x,hgrad); add_vV(l,hgrad); // df(x)/2
		const Scalar xhg = scal_vv(x,hgrad);
		g[0]=hgrad[0]/xhg; g[1]=hgrad[1]/xhg;
}

struct tti_data_t {Scalar L[2]; Scalar Q[3]; Scalar a,b;};
void tti_data_init(const Scalar l[2], const Scalar q[3], tti_data_t & data){
	// Equation is <l,x> + 0.5 <x,q,x> = 1
	data.a = solve2_above(-2,l[0],q[0],0.); // (a,0) is on the curve
	data.b = solve2_above(-2,l[1],q[2],0.); // (0,b) is on the curve

	// Change of basis 
	const Scalar e0[2] = {1/2.,1/2.}, e1[2] = {1/2.,-1/2.};
	data.L[0] = scal_vv(l,e0); data.L[1]=scal_vv(l,e1);
	data.Q[0]=scal_vmv(e0,q,e0); data.Q[1]=scal_vmv(e0,q,e1); data.Q[2]=scal_vmv(e1,q,e1);
}

/** Samples the curve defined by f(x)=0, x>=0, 
where f(x):= -2 + 2 <l,x> + <qx,x>,
and returns diag(i) := grad f(x)/<x,grad f(x)>.*/
bool diags(const Scalar l[2], const Scalar q[3], Scalar diag_s[nmix][2]){
	tti_data_t data; tti_data_init(l,q,data);

	Scalar x_s[nmix][2]; // Curve sampling
	Scalar * xbeg = x_s[0], * xend = x_s[nmix-1];
	xbeg[0]=data.a; xbeg[1]=0; xend[0]=0; xend[1]=data.b;
	for(Int i=1;i<nmix-1; ++i){
		const Scalar t = i/Scalar(nmix-1);
		const Scalar v = (1-t)*data.a - t*data.b;
		// Solving f(u e0+ v e_1) = 0 w.r.t u
		const Scalar u = solve2_above(-2+2*data.L[1]*v+data.Q[2]*v*v, 
			data.L[0]+data.Q[1]*v, data.Q[0], abs(v));
		// Inverse change of basis
		Scalar * x = x_s[i];
		x[0] = (u+v)/2; x[1] = (u-v)/2;
	} 
	for(Int i=0; i<nmix; ++i){grad_ratio(l,q,x_s[i],diag_s[i]);}
	return det_vv(diag_s[0],diag_s[nmix-1])>0;
}

} // namespace dim2


bool scheme(const Scalar geom[geom_size], 
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==symdim*nmix,inconsistent_scheme_parameters)
	
	const Scalar * linear = geom; // linear[2]
	const Scalar * quadratic = geom + 2; // quadratic[dim2::symdim]
	const Scalar * transform = geom + (2+dim2::symdim); // transform[ndim * ndim]

	Scalar diag_s[nmix][2];
	const bool mix_is_min = dim2::diags(linear,quadratic,diag_s);
	Scalar D0[symdim]; self_outer_v(transform,D0);
	Scalar D1[symdim]; self_outer_v(transform+(ndim-1)*ndim,D1);
	if(ndim==3){Scalar D2[symdim]; self_outer_v(transform+ndim,D2);
		for(Int i=0; i<symdim; ++i) D0[i]+=D2[i];}

	for(Int kmix=0; kmix<nmix; ++kmix){
		const Scalar * diag = diag_s[kmix]; // diag[2];
		Scalar D[symdim];
		for(Int i=0; i<symdim; ++i) {D[i] = diag[0]*D0[i] + diag[1]*D1[i];} 
		decomp_m(D, weights+kmix*symdim, offsets+kmix*symdim);
	}
	return mix_is_min;
}

#include "GoldenSearch.h"
#include "EuclideanFactor.h"

FACTOR(
#ifndef niter_golden_search_macro
const Int niter_golden_search = 8;
#endif

namespace dim2 {

/// Compute the norm associated with the tangent ellipsoid of parameter v
Scalar _tti_norm(const Scalar l[2], const Scalar q[3], const tti_data_t & data, 
	const Scalar s[2], const Scalar v, Scalar diag[2]){
	const Scalar u = solve2_above(-2+2*data.L[1]*v+data.Q[2]*v*v, 
		data.L[0]+data.Q[1]*v, data.Q[0], abs(v));
	const Scalar x[2] = {(u+v)/2,(u-v)/2};
	grad_ratio(l,q,x,diag);
	diag[0]=1./diag[0]; diag[1]=1./diag[1];
	return sqrt(scal_vv(s,diag));
}

/// Compute the norm after transformation by A, by optimizing over tangent ellipsoids
Scalar _tti_norm(const Scalar l[2], const Scalar q[3], const tti_data_t & data,
	const Scalar s[2], Scalar diag[2]){

	const Scalar bounds[2] = {data.a,-data.b};
	Scalar mid[2]; golden_search::init(bounds,mid);
	Scalar diag_[2][2];
	Scalar values[2] = {
		_tti_norm(l,q,data,s,mid[0],diag_[0]), 
		_tti_norm(l,q,data,s,mid[1],diag_[1])}; 
	const bool mix_is_min = det_vv(diag_[0],diag_[1])<=0; 
	Int next;
	for(Int i=0; i<niter_golden_search; ++i){
		next = golden_search::step(mid,values,mix_is_min);
		values[next] = _tti_norm(l,q,data,s,mid[next],diag);}
	return values[next];
}
} //namespace dim2

Scalar tti_norm(const Scalar l[2], const Scalar q[3], const dim2::tti_data_t & data,
	const Scalar A[ndim][ndim], const Scalar x[ndim], Scalar * gradient=NULL){
	Scalar Ax[ndim]; dot_av(A,x,Ax);
	Scalar s[2] = {Ax[0]*Ax[0],Ax[ndim-1]*Ax[ndim-1]};
	if(ndim==3) {s[0]+=Ax[1]*Ax[1];}

	Scalar diag[2];
	const Scalar norm = _tti_norm(l,q,data,s,diag);
	if(gradient!=NULL) {
		for(Int i=0; i<ndim; ++i) {Ax[i]*=diag[ndim==2 ? i : max(i-1,0)]/norm;}
		tdot_av(A,Ax,gradient);}
	return norm;
}


void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	const Scalar * l = factor_metric; // linear[2]
	const Scalar * q = factor_metric + 2; // quadratic[dim2::symdim]
	const Scalar * A_ = factor_metric + (2+dim2::symdim); // transform[ndim * ndim]
	const Scalar (* A)[ndim] = (const Scalar (*)[ndim]) A_;
	dim2::tti_data_t data; tti_data_init(l,q,data);

	Scalar grad[ndim]; const Scalar Nx = tti_norm(l,q,data,A, x, grad);
	Scalar xpe[ndim],xme[ndim]; add_vv(x,e,xpe); sub_vv(x,e,xme);
	const Scalar Nxpe = tti_norm(l,q,data,A,xpe); 
	const Scalar Nxme = tti_norm(l,q,data,A,xme); 

	ORDER2(
	Scalar xpe2[ndim],xme2[ndim]; add_vv(xpe,e,xpe2); sub_vv(xme,e,xme2);
	const Scalar Nxpe2 = tti_norm(l,q,data,A,xpe2); 
	const Scalar Nxme2 = tti_norm(l,q,data,A,xme2); 
	)

	generic_factor_sym(scal_vv(grad,e),Nx,Nxpe,Nxme,fact ORDER2(,Nxpe2,Nxme2,fact2));
}

/*
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	const Scalar * l = factor_metric; // linear[2]
	const Scalar * q = factor_metric + 2; // quadratic[dim2::symdim]
	const Scalar * A_ = factor_metric + (2+dim2::symdim); // transform[ndim * ndim]
	const Scalar (* A)[ndim] = (const Scalar (*)[ndim]) A_;
	dim2::tti_data_t data; tti_data_init(l,q,data);

	Scalar grad[ndim];
	const Scalar Nx = tti_norm(l,q,data,A, x, grad);
	const Scalar grad_e = scal_vv(grad,e);
	Scalar xpe[ndim],xme[ndim]; add_vv(x,e,xpe); sub_vv(x,e,xme);
	const Scalar Nxpe = tti_norm(l,q,data,A,xpe); 
	const Scalar Nxme = tti_norm(l,q,data,A,xme); 

	fact[0] = -grad_e + Nx - Nxme; 
	fact[1] =  grad_e + Nx - Nxpe; 

	ORDER2(
	Scalar xpe2[ndim],xme2[ndim]; add_vv(xpe,e,xpe2); sub_vv(xme,e,xme2);
	const Scalar Nxpe2 = tti_norm(l,q,data,A,xpe2); 
	const Scalar Nxme2 = tti_norm(l,q,data,A,xme2); 
	fact2[0] = 2*fact[0]-(Nx - 2*Nxme + Nxme2); 
	fact2[1] = 2*fact[1]-(Nx - 2*Nxpe + Nxpe2); 
	)
}
*/

) // FACTOR

#include "Update.h"