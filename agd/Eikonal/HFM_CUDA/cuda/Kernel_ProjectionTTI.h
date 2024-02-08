#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file implements the projection of a Hooke tensor onto the TTI model, 
which can be regarded as a polynomial minimization problem over a three dimensional sphere.
It uses a combination of exhaustive search and of a Newton method.

The problem is embarassingly parallel, and no attempt is made to make threads collaborate.
Note : one could easily alleviate register pressure by having each thread of a warp work 
on a different seed point for the Newton method.
*/

typedef int Int;
#include "static_assert.h"
/* // Must be defined externally
typedef float Scalar;
const int xdim=3;
const int n_samples_bound = ??; // upper bound on n_samples
*/

const int xdim = xdim_macro;
STATIC_ASSERT(1<=xdim && xdim<=3,Unsupported_dimension);
const int ndim = xdim==1 ? 2 : 3;
const int symdim = (ndim*(ndim+1))/2;

#include "Dense2.h"
typedef Dense2<Scalar,xdim> AD; // Automatic differentiation type (second order, dense)
typedef GeometryT<xdim>   GX;   // Geometry in dimension xdim
typedef GeometryT<symdim> GM;   // Geometry in dimension symdim

const int hdim = GM::symdim; // Number of independent coefficients of Hooke tensor
template<typename T> T co(const T hooke[hdim], int i, int j){
	return GM::coef_m(hooke,i,j);}
template<typename T> T sq(const T & x){return x*x;}

/// Generate a matrix which can be used to apply a rotation to a Hooke tensor
template<typename T>
void make_hooke_rotation(const T x[xdim], T R[symdim][symdim]){
	// We first build a standard rotation matrix
#if xdim_macro==1 // Build a rotation from an angle
	const T c = cos(x[0]), s = sin(x[0]);
	const T r[ndim][ndim] = {{c,-s},{s,c}};
#elif xdim_macro==2 // Build a rotation from two angles (take advantage of vti z-invariance)
	const T c0 = cos(x[0]), s0 = sin(x[0]), c1 = cos(x[1]), s1 = sin(x[1]);
	const T r[ndim][ndim] = {
		{c1,Scalar(0.),s1},
		{s0*s1,c0,-s0*c1},
		{-c0*s1,s0,c0*c1}};
#elif xdim_macro==3 // Build a rotation from a point in the unit ball
	// First build a unit quaternion. See agd/Sphere.py
	const T xn2 = GX::norm2(x), den=1./(1.+xn2), den2=2.*den;
	T qr=(1.-xn2)*den,qi=x[0]*den2,qj=x[1]*den2,qk=x[2]*den2;

	// Build the corresponding 3x3 rotation matrix. See agd/Sphere.py
	T r[ndim][ndim] = {
        {0.5-(sq(qj)+sq(qk)), qi*qj-qk*qr, qi*qk+qj*qr},
        {qi*qj+qk*qr, 0.5-(sq(qi)+sq(qk)), qj*qk-qi*qr},
        {qi*qk-qj*qr, qj*qk+qi*qr, 0.5-(sq(qi)+sq(qj))}};
    GX::mul_kA(Scalar(2.),r); 
#endif

    // Then we expand it to be applicable to Hooke tensors
    // Define the Voigt coefficient indexation
#if xdim_macro==1 
    const int Voigti[symdim][2]={{0,0},{1,1},{1,0}};
#else
    const int Voigti[symdim][2]={{0,0},{1,1},{2,2},{2,1},{2,0},{1,0}};
#endif

    // Build the 6x6 rotation matrix. See agd/Metrics/Seismic/Hooke.py
	for(int I=0; I<symdim; ++I){
		const int i = Voigti[I][0], j = Voigti[I][1];
		for(int K=0; K<symdim; ++K){
			const int k = Voigti[K][0], l=Voigti[K][1];
			R[I][K] = r[i][k]*r[j][l];
			if(k!=l) {R[I][K] += r[j][k]*r[i][l];}
		}
	}

}

/** Returns the square L2 error of projection of the rotated matrix onto hexagonal
 *  Hooke tensors.
 *  See notebook SeismicNorm for an explanation and a (slow but readable) Python implem */
template<typename T>
T projection_error(const Scalar hooke[hdim], const T x[xdim]){
	// Rotate Hooke tensor
	T R[symdim][symdim];
	make_hooke_rotation(x,R);
	T h[hdim];
	GM::tgram_am(R,hooke,h);

	// Project Hooke onto hexagonal VTI structure. See notebook Notebooks_Algo/SeismicNorm
#if xdim_macro==1 
	const T res = sq(co(h,2,0))+sq(co(h,2,1)); // Other coefs are reproduced exactly
#else
	const T 
    alpha=(3*(co(h,0,0)+co(h,1,1))+2.*co(h,0,1)+4.*co(h,5,5))/8.,
    beta =(co(h,0,0)+co(h,1,1)+6*co(h,0,1)-4.*co(h,5,5))/8.,
    gamma=(co(h,0,2)+co(h,1,2))/2.,
    delta=(co(h,3,3)+co(h,4,4))/2.;

//    printf("%f, %f, %f, %f\n",alpha,beta,gamma,delta);
    // Calculate residual. See Notebooks_Algo/SeismicNorm
    // Hooke.from_orthorombic(α,β,γ,α,γ,c33,δ,δ,(α-β)/2).hooke
    // c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2],c[3,3],c[4,4],c[5,5]
    T res = 0.5*sq(co(h,0,0)-alpha)+sq(co(h,0,1)-beta)+sq(co(h,0,2)-gamma)
    +0.5*sq(co(h,1,1)-alpha)+sq(co(h,1,2)-gamma) // No co(h,2,2).
    +4.*sq(co(h,3,3)-delta) // Same +sq(co(h,4,4)-delta) 
    +2.*sq(co(h,5,5)-(alpha-beta)/2.); // Here and below we take into account Mandel's coeffs
    for(int i=0; i<3; ++i) {for(int j=0; j<3; ++j) res+=2.*sq(co(h,3+i,  j));}
    for(int i=0; i<3; ++i) {for(int j=0; j<i; ++j) res+=4.*sq(co(h,3+i,3+j));}
#endif

	return res;
}

/** Return the coefficients of the orthogonal projextion onto hexagonal Hooke tensors
// Equal to c11,c12,c13,c33,c44 if already hexagonal */
void projection_coefficients(const Scalar hooke[hdim], const Scalar x[xdim], Scalar hexa[5]){
	Scalar R[symdim][symdim];
	make_hooke_rotation(x,R);
	Scalar h[hdim];
	GM::tgram_am(R,hooke,h);

#if xdim_macro==1 
	hexa[0] = co(h,0,0);
	hexa[1] = 0.;
	hexa[2] = co(h,1,0);
	hexa[3] = co(h,1,1);
	hexa[4] = co(h,2,2);
#else
    hexa[0] = (3*(co(h,0,0)+co(h,1,1))+2.*co(h,0,1)+4.*co(h,5,5))/8.;
    hexa[1] = (co(h,0,0)+co(h,1,1)+6*co(h,0,1)-4.*co(h,5,5))/8.;
    hexa[2] = (co(h,0,2)+co(h,1,2))/2.;
    hexa[3] = co(h,2,2);
    hexa[4] = (co(h,3,3)+co(h,4,4))/2.;
#endif
}

// Number of hooke tensors, of sphere seeds, and of newton steps
__constant__ int n_hooke, n_samples, n_newton; 

void TestAD(){
	typedef Dense2<Scalar,1> AD1;
	AD1 x,y; 
	x.a=2; x.v[0]=3; x.m[0]=4;
	y.a=5; y.v[0]=6; y.m[0]=7;

	printf("\nsum\n");  show(x+y-x-y);
	printf("\nprod\n"); show(x*y/x/y - 1.);
	printf("\nneg\n");  show(-x+x);
	printf("\nksum\n"); show((x+1.)-x-1.);
	printf("\nkprod\n");show(2.*x/2./x -1.);
}

extern "C" {

__global__ void ProjectionTTI(
	const Scalar * __restrict__ hooke_in, 
	Scalar * __restrict__ v_out, 
	Scalar * __restrict__ x_out,
	Scalar * __restrict__ x_in,
	Scalar * __restrict__ hexa){

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=n_hooke) return;

	Scalar hooke[hdim];
	for(int i=0; i<hdim; ++i) {hooke[i] = hooke_in[tid*hdim+i];}

	Scalar v_opt = 1./0.; // Best value found for projection error
	Scalar x_opt[xdim];

	AD x_ad[xdim];
	AD::Identity(x_ad);

	for(int i_samples=0; i_samples<n_samples; ++i_samples){
		
		for(int i=0; i<xdim; ++i) {x_ad[i].a = x_in[i_samples*xdim+i];}

		for(int i=0; i<n_newton; ++i){
			// Evaluate objective function
			const AD obj = projection_error(hooke,x_ad);

			// Register if better than before
			if(obj.a<v_opt){
				v_opt=obj.a; 
				for(int i=0; i<xdim; ++i) {x_opt[i]=x_ad[i].a;}
			}

			// Update position using Newton method
			if(i==n_newton-1) break; // No need in last iteration
			Scalar a[xdim][xdim], ai[xdim][xdim], dx[xdim];
			GX::copy_mA(obj.m,a);
			GX::inv_a(a,ai);
			GX::dot_av(ai,obj.v,dx);
			GX::sub(dx,x_ad);
		}
	}

	v_out[tid] = v_opt;
	for(int i=0; i<xdim; ++i){x_out[tid*xdim+i] = x_opt[i];}
	projection_coefficients(hooke,x_opt,hexa+tid*5);
}

}