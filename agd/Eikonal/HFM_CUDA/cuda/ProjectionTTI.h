#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file implements the projection of a Hooke tensor onto the TTI model, 
which can be regarded as a polynomial minimization problem over a three dimensional sphere.
It uses a combination of exhaustive search and of a Newton method.

The problem is embarassingly parallel, and no attempt is made to make threads collaborate.
*/

typedef int Int;
#include "static_assert.h"
/* // Must be defined externally
typedef float Scalar;
const int ndim=3;
const int n_samples_bound = ??; // upper bound on n_samples
*/
STATIC_ASSERT(ndim==2 || ndim==3,Unsupported_dimension);

const int xdim = ndim==2 ? 1 : 3;
const int symdim = (ndim*(ndim+1))/2;

#include "Dense2.h"
typedef Dense2<Scalar,xdim> AD; // Automatic differentiation type (second order, dense)
typedef GeometryT<xdim>   GX;   // Geometry in dimension xdim
typedef GeometryT<symdim> GM;   // Geometry in dimension symdim

const int hdim = GM::symdim; // Number of independent coefficients of Hooke tensor
template<typename T> T co(const T hooke[hdim], int i, int j){
	return GM::coef_m(hooke,i,j);}
template<typename T> T sq(const T & x){return x*x;}

template<typename T>
T projection_error(const Scalar hooke[hdim], const T x[xdim]){
	// See notebook SeismicNorm for a explanation and (slow but readable) Python implem
//	printf("\nx\n"); GX::show_v(x);

	// Build the unit quaternion. See agd/Sphere.py
	const T xn2 = GX::norm2(x), den=1./(1.+xn2), den2=2.*den;
	T qr=(1.-xn2)*den,qi=x[0]*den2,qj=x[1]*den2,qk=x[2]*den2;

//	printf("\nq\n"); show(qr); show(qi); show(qj); show(qk); 

	// Build the 3x3 rotation matrix. See agd/Sphere.py
	T r[ndim][ndim] = {
        {0.5-(sq(qj)+sq(qk)), qi*qj-qk*qr, qi*qk+qj*qr},
        {qi*qj+qk*qr, 0.5-(sq(qi)+sq(qk)), qj*qk-qi*qr},
        {qi*qk-qj*qr, qj*qk+qi*qr, 0.5-(sq(qi)+sq(qj))}};
    GX::mul_kA(Scalar(2.),r); 

//   printf("\nr\n"); GX::show_a(r);

    // Build the 6x6 rotation matrix. See agd/Metrics/Seismic/Hooke.py
	T R[symdim][symdim];
	const int Voigti[symdim][2]={{0,0},{1,1},{2,2},{2,1},{2,0},{1,0}};
	for(int I=0; I<symdim; ++I){
		const int i = Voigti[I][0], j = Voigti[I][1];
		for(int K=0; K<symdim; ++K){
			const int k = Voigti[K][0], l=Voigti[K][1];
			R[I][K] = r[i][k]*r[j][l];
			if(k!=l) {R[I][K] += r[j][k]*r[i][l];}
		}
	}

	// Rotate Hooke tensor
	T h[hdim];
	GM::tgram_am(R,hooke,h);
//	printf("\nhooke\n"); GM::show_m(hooke);
//	printf("\nh\n"); GM::show_m(h);

	// Project Hooke onto hexagonal VTI structure. See notebook Notebooks_Algo/SeismicNorm
    const T 
    alpha=(3*(co(h,0,0)+co(h,1,1))+2.*co(h,0,1)+4.*co(h,5,5))/8.,
    beta =(co(h,0,0)+co(h,1,1)+6*co(h,0,1)-4.*co(h,5,5))/8.,
    gamma=(co(h,0,2)+co(h,1,2))/2.,
    delta=(co(h,3,3)+co(h,4,4))/2.;

//    printf("%f, %f, %f, %f\n",alpha,beta,gamma,delta);
    // Calculate residual. See Notebooks_Algo/SeismicNorm
    // c[0,0],c[0,1],c[0,2],c[1,1],c[1,2],c[2,2],c[3,3],c[4,4],c[5,5]
    // Hooke.from_orthorombic(α,β,γ,α,γ,c33,δ,δ,(α-β)/2).hooke
    T res = 0.5*sq(co(h,0,0)-alpha)+sq(co(h,0,1)-beta)+sq(co(h,0,2)-gamma)
    +0.5*sq(co(h,1,1)-alpha)+sq(co(h,1,2)-gamma) // No co(h,2,2).
    +sq(co(h,3,3)-delta) // Same +sq(co(h,4,4)-delta) 
    +0.5*sq(co(h,5,5)-(alpha-beta)/2.);
    for(int i=0; i<3; ++i) {for(int j=0; j<3; ++j) res+=sq(co(h,3+i,  j));}
    for(int i=0; i<3; ++i) {for(int j=0; j<i; ++j) res+=sq(co(h,3+i,3+j));}

//    printf("\nres\n"); show(res);
	// Return residual
	return res;
}

// Number of hooke tensors, of sphere seeds, and of newton steps
__constant__ int n_hooke, n_samples, n_newton; 

__constant__ float x_in[n_samples_bound*xdim];

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
	Scalar * __restrict__ x_out){

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid>=n_hooke) return;

	Scalar hooke[hdim];
	for(int i=0; i<hdim; ++i) {hooke[i] = hooke_in[tid*hdim+i];}

	Scalar v_opt = 1./0.; // Best value found for projection error
	Scalar x_opt[xdim];

	AD x_ad[xdim];
	AD::Identity(x_ad);

	// DEBUG
//	TestAD();
//	for(int i=0; i<xdim; ++i) {x_opt[i]=x_in[i];}
//	projection_error(hooke,x_opt); 
//	return;

//	printf("Hi %i, %i\n",n_samples,n_newton);

	for(int i_samples=0; i_samples<n_samples; ++i_samples){
		
		for(int i=0; i<xdim; ++i) {x_ad[i].a = x_in[i_samples*xdim+i];}

		for(int i=0; i<n_newton; ++i){
			// Evaluate objective function
			const AD obj = projection_error(hooke,x_ad);

//			printf("%f\n",obj.a);
			// Register if better than before
			if(obj.a<v_opt){
				v_opt=obj.a; 
				for(int i=0; i<xdim; ++i) {x_opt[i]=x_ad[i].a;}
			}

			// Update position using Newton method
			Scalar a[xdim][xdim], ai[xdim][xdim], dx[xdim];
			GX::copy_mA(obj.m,a);
			GX::inv_a(a,ai);
			GX::dot_av(ai,obj.v,dx);
			GX::sub(dx,x_ad);
		}
		// TODO : one iteration for almost free by evaluating without AD on last result
	}

	v_out[tid] = v_opt;
	for(int i=0; i<xdim; ++i){x_out[tid*xdim+i] = x_opt[i];}
}

}