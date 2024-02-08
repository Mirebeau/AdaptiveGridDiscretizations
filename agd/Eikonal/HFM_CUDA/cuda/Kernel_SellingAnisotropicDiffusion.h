#pragma once
// Copyright 2022 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a non-negative anisotropic diffusion scheme, gpu accelerated, based on :
Fehrenbach, Mirebeau, Sparse non-negative stencils for anisotropic diffusion, JMIV, 2013
*/

/** The following need to be defined in including file (example)
typedef int Int;
typedef float Scalar;
#define ndim_macro 3
*/


#if   (ndim_macro == 1)
#include "Geometry1.h"
#elif (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 5)
#include "Geometry5.h"
#elif (ndim_macro == 6)
#include "Geometry6.h"
#endif 

#undef bilevel_grid_macro
#include "Grid.h"

#if ced_macro
#include "CoherenceEnhancingDiffusion.h"
#endif

__constant__ Int shape_tot[ndim];
__constant__ Int size_tot; // product of shape_tot
__constant__ Scalar dx[ndim],dt; // grid scale, time step

extern "C" {
__global__ void 
anisotropic_diffusion_scheme(const Scalar * __restrict__ D_t, 
	Scalar * __restrict__ wdiag_t, Scalar * __restrict__ wneigh_t, Int * __restrict__ ineigh_t
	#if retD_macro
	,Scalar * __restrict__ retD_t
	#endif
	){

	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}

	// Get the position where the work is to be done.
	Int x_t[ndim];
	Grid::Position(n_t,shape_tot,x_t);

	// Optional : structure tensor transformation into diffusion tensor
	Scalar D[symdim];
	for(int i=0; i<symdim; ++i) D[i]=D_t[n_t*symdim+i];
	#if ced_macro
	Scalar lambda[ndim], mu[ndim];
	eigvalsh(D,lambda);
	ced(lambda,mu);
	map_eigvalsh(D,lambda,mu);
	#endif
	#if retD_macro
	for(int i=0; i<symdim; ++i) retD_t[n_t*symdim+i]=D[i];
	#endif
	for(int i=0,k=0;i<ndim;++i) for(int j=0;j<=i;++j,++k) D[k]/=dx[i]*dx[j];

	// Selling decomposition
	Scalar weights[decompdim]; 
	Int offsets[decompdim][ndim];
	decomp_m(D,weights,offsets);

	// Conversion to linear indices, and storage
	Scalar wsum=0;
	for(int i=0,k=0; i<decompdim; ++i){
		const Scalar weight = weights[i]/2;
		const Int * offset = offsets[i];
		wneigh_t[n_t*decompdim+i] = weight; // Export the decomposition weight
		Int y_t[ndim];

#if fourth_order_macro
/*{{18,-12,-12, 3, 3}, // Each full ter creates the following diagonal block
   {-12,20, -4,-4, 0},
   {-12,-4, 20, 0,-4},
   {3,  -4,  0, 1, 0},
   {3,   0, -4, 0, 1}} */

		const Int indstart = n_t*(decompdim*4)+4*i; //k+=4;
		const int badIndex = -(1<<30);
		Int indices[4]={badIndex,badIndex,badIndex,badIndex};
		bool fourth_active = true;
		sub_vv(x_t,offset,y_t); 
		if(Grid::InRange_per(y_t,shape_tot)) indices[0]=Grid::Index_per(y_t,shape_tot); 
		else fourth_active=false;
		sub_vV(offset,y_t);
		if(Grid::InRange_per(y_t,shape_tot)) indices[2]=Grid::Index_per(y_t,shape_tot);
		else fourth_active=false;
		add_vv(x_t,offset,y_t); 
		if(Grid::InRange_per(y_t,shape_tot)) indices[1]=Grid::Index_per(y_t,shape_tot);
		else fourth_active=false;
		add_vV(offset,y_t);
		if(Grid::InRange_per(y_t,shape_tot)) indices[3]=Grid::Index_per(y_t,shape_tot);
		else fourth_active=false;
		
		if(fourth_active){ // Use the fourth order scheme
			const Scalar w = weight/12;
			wsum+=18*w; 
			atomicAdd(wdiag_t+indices[0],20*w); 
			atomicAdd(wdiag_t+indices[1],20*w); 
			atomicAdd(wdiag_t+indices[2],w);
			atomicAdd(wdiag_t+indices[3],w);
			for(int j=0; j<4; ++j) {ineigh_t[indstart+j]=indices[j];}
		} else { // Fall back to the second order scheme
			for(int j=0; j<2; ++j){ // Neuman boundary conditions
				if(indices[j]!=badIndex){
					ineigh_t[indstart+j]=indices[j];
					atomicAdd(wdiag_t+indices[j],weight);
					wsum+=weight;
				}
			}
		}
#else // second order scheme
		for(int s=0; s<=1; ++s,++k){
			if(s) add_vv(x_t,offset,y_t); 
			else sub_vv(x_t,offset,y_t);

			if(Grid::InRange_per(y_t,shape_tot)) {
				const Int ny_t = Grid::Index_per(y_t,shape_tot);
				ineigh_t[n_t*(decompdim*2)+k] = ny_t;
				atomicAdd(wdiag_t+ny_t,weight);
				wsum+=weight;
			} // if in range
		} // for s
#endif // second or fourth order scheme
	} // for i
	atomicAdd(wdiag_t+n_t,wsum);
}

__global__ void 
anisotropic_diffusion_step(const Scalar * __restrict__ uold_t, Scalar * __restrict__ unew_t,
	const Scalar * __restrict__ wdiag_t, 
	const Scalar * __restrict__ wneigh_t, const Int * __restrict__ ineigh_t){

	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}

	const Scalar uold = uold_t[n_t];
	Scalar uinc = uold*(Scalar(prev_coef)-dt*wdiag_t[n_t]); 
	for(int i=0,k=0; i<decompdim; ++i){
		const Scalar weight = dt*wneigh_t[n_t*decompdim+i];

		#if fourth_order_macro
		Int indices[4]; 
		const Int indstart = n_t*(decompdim*4)+4*i;
		for(int j=0; j<4; ++j) {indices[j] = ineigh_t[indstart+j];}
		if(indices[2]>=0){ // Effectively using the fourth order scheme
			Scalar u[5], du[5];
			u[0] = uold;
			for(int j=0; j<4; ++j) {u[j+1]=uold_t[indices[j]];}
			// Diagonal coefficients have already been handled
			du[0] =          -12*u[1] -12*u[2] +3*u[3] +3*u[4];
			du[1] = -12*u[0]          -4*u[2]  -4*u[3];
			du[2] = -12*u[0] -4*u[1]                   -4*u[4];
			du[3] = 3*u[0]   -4*u[1];
			du[4] = 3*u[0]            -4*u[2];

			const Scalar w = -weight/12; // Note the minus sign, because minus signs above
			uinc += w*du[0];
			for(int j=0; j<4; ++j){atomicAdd(unew_t+indices[j],w*du[j+1]);}

		} else { // Fall back to second order scheme
			for(int j=0; j<2; ++j){
				const Int ineigh = indices[j];
				if(ineigh>=0){
					uinc+=weight*uold_t[ineigh];
					atomicAdd(unew_t+ineigh,weight*uold);
				} // if within domain
			} // for j neighbor
		}

		#else // second order scheme
		
		for(int j=0; j<2; ++j,++k){
			const Int ineigh = ineigh_t[n_t*(decompdim*2)+k];
			if(ineigh>=0) {
				uinc+=weight*uold_t[ineigh];
				atomicAdd(unew_t+ineigh,weight*uold);
			}
		}
		
		#endif // second or fourth order scheme
	}
	atomicAdd(unew_t+n_t,uinc);
}

__global__


} // extern "C"