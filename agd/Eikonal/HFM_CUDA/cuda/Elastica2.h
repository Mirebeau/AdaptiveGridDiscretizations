#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define nsym_macro 0 // Only uses forward offsets
#define curvature_macro true
#include "Geometry3.h"

/**Fejer quadrature rule for integration
#define nFejer_macro 5 // Must be defined in enclosing file
*/

const Int nFejer = nFejer_macro;

/*Fejer weights are for one dimensional quadrature on [-pi/2,pi/2] with cosine weight
 Array suffix _s indicates stencil data.
// Generate the cosine and sine tables
np.set_printoptions(edgeitems=30, linewidth=100000,formatter=dict(float=lambda x: "%5.9g" % x))
def cos_sin_table(n): 
    angles = (np.arange(n)+0.5)*np.pi/n
    return np.cos(angles),np.sin(angles)*/
#if nFejer_macro==2
const Scalar wFejer_s[nFejer]={1.,1.};
const Scalar cosPhi_s[nFejer]={0.707106781, -0.707106781};
const Scalar sinPhi_s[nFejer]={0.707106781, 0.707106781};

#elif nFejer_macro==3
const Scalar wFejer_s[nFejer]={0.444444, 1.11111, 0.444444};
const Scalar cosPhi_s[nFejer]={0.866025404, 0., -0.866025404};
const Scalar sinPhi_s[nFejer]={0.5,     1,   0.5};

#elif nFejer_macro==4
const Scalar wFejer_s[nFejer]={0.264298, 0.735702, 0.735702, 0.264298};
const Scalar cosPhi_s[nFejer]={0.923879533, 0.382683432, -0.382683432, -0.923879533};
const Scalar sinPhi_s[nFejer]={0.382683432, 0.923879533, 0.923879533, 0.382683432};

#elif nFejer_macro==5
const Scalar wFejer_s[nFejer]={0.167781, 0.525552, 0.613333, 0.525552, 0.167781};
const Scalar cosPhi_s[nFejer]={0.951056516, 0.587785252, 0., -0.587785252, -0.951056516};
const Scalar sinPhi_s[nFejer]={0.309016994, 0.809016994, 1., 0.809016994, 0.309016994};

#elif nFejer_macro==9
const Scalar wFejer_s[nFejer]={0.0527366, 0.179189, 0.264037, 0.330845, 
const Scalar cosPhi_s[nFejer]={0.984807753, 0.866025404, 0.64278761, 0.342020143, 0.,
	-0.342020143, -0.64278761, -0.866025404, -0.984807753};
const Scalar sinPhi_s[nFejer]={0.173648178,   0.5, 0.766044443, 0.939692621, 1., 
	0.939692621, 0.766044443,   0.5, 0.173648178};
#endif

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = nFejer*decompdim; // Number of forward offsets

#include "Constants.h"
#include "Decomp_v_.h"

#if !precomputed_scheme_macro
void scheme(GEOM(const Scalar geom[geom_size],) const Int x[ndim],
	Scalar weights[nactx], OffsetT offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==nfwd,inconsistent_scheme_parameters)

	XI_VAR(Scalar ixi;) KAPPA_VAR(Scalar kappa;) 
	Scalar cT, sT; // cos(theta), sin(theta)
	get_ixi_kappa_theta(GEOM(geom,) x, XI_VAR(ixi,) KAPPA_VAR(kappa,) cT, sT);

	for(Int l=0; l<nFejer; ++l){
//		const Scalar phi = pi*(l+0.5)/nFejer; // Now using tables for speed
//		const Scalar cP = cos(phi), sP = sin(phi);
		const Scalar cP = cosPhi_s[l], sP = sinPhi_s[l];
		const Scalar v[ndim]={sP*cT,sP*sT,(sP*kappa+cP*ixi)};

		decomp_v(v, &weights[l*decompdim], &offsets[l*decompdim]);

		const Scalar s = wFejer_s[l] 
		#if convex_curvature_macro // Model variant where the vehicle always turns left
		* (2*l<nFejer-1 ? 1. : 2*l==nFejer-1 ? 0.5 : 0.)
		#endif
		;
		
		for(Int i=0; i<decompdim; ++i) weights[l*decompdim+i] *= s;
	}
} 
#endif

#include "Update.h"