// Copyright 2024 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the linear acoustic wave equation operators, for a general metric. 
The tensor must be decomposed for finite differences using, e.g., Selling's decomposition.
*/

/* -- Wave equation vs diffusion --
The contents are quite similar to the Kernel_SellingAnisotropicDiffusion.h file, 
but eventually a new file had to be created due to diverging requirements.
(Here : no CED transformation, but one may use a decomposition other than Selling, and 
forward and reverse automatic differentiation are implemented.)
*/

/* -- Array indexing --
Contrary to the Kernel_ElasticWave.h file, we do not use bilevel indexing, and the 
c-contiguity of the AD arrays is straigthforward. 

Those indexing transformation improve performance, a bit, but they are annoying to setup,
prone to bugs, and cause inconvenience with the code afterwards. (Reshaping operations.)

The acoustic wave equation is here regarded as a toy model for the elastic wave, hence
we favor simplicity over performance.  
*/

#include "static_assert.h"

/* // The following must be defined externally
typedef float Scalar
typedef int Int;
typedef int8_t OffsetT
#define fourth_order_macro false
#define ndim_macro 2
#define decompdim 3
#define size_ad_macro 0
#define fwd_macro True
#define periodic_macro false
*/

// Boundary conditions support
#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
// const bool periodic_axes[ndim] = {false,false,true}; // must be defined externally
#else
#define PERIODIC(...) 
#endif

#if size_ad_macro>0
const int size_ad = size_ad_macro;
#define AD(...) __VA_ARGS__ 
#if fwd_macro
#define FWD(...) __VA_ARGS__ // Forward autodiff
#define REV(...) 
#else
#define FWD(...)
#define REV(...) __VA_ARGS__ // Reverse autodiff
#endif // if fwd_macro
#else
#define AD(...)
#define FWD(...)
#endif // if size_ad_macro>0

const int ndim = ndim_macro;
#include "Geometry_.h"
__constant__ Int shape_tot[ndim];
__constant__ Int size_tot; // product of shape_tot
#undef bilevel_grid_macro
#include "Grid.h"
const int badIndex = -(1<<30); // Used for out of domain points

const int order  = fourth_order_macro ? 4 : 2; // Consistency order of the numerical scheme
const int nneigh = 1+order; // Number of neighbors involved in the finite differences

__constant__ Scalar DqH_mult; //dt/(c*dx**2) where c = 2 if order==2, and c=24 if order==4
__constant__ Scalar DpH_mult; //dt

/** The finite differences scheme associated with a single term of Selling's decomposition.
Either second or  fourth order, and involved in the evaluation of DqH.*/
void DqH_scheme(const Scalar q[nneigh], Scalar dq[__restrict__ nneigh]){
#if fourth_order_macro
/*	{{18,-12,-12,3,3},
	{-12,20,-4,-4,0},
	{-12,-4,20,0,-4},
	{3,-4,0,1,0},
	{3,0,-4,0,1}} */
	dq[0] =  18*q[0] -12*q[1] -12*q[2] +3*q[3] +3*q[4];
	dq[1] = -12*q[0] +20*q[1] - 4*q[2] -4*q[3];
	dq[2] = -12*q[0] - 4*q[1] +20*q[2]         -4*q[4];
	dq[3] =   3*q[0] - 4*q[1]          +  q[3];
	dq[4] =   3*q[0]          - 4*q[2]         +  q[4];
#else
//	{{2, -1, -1}, {-1, 1, 0}, {-1, 0, 1}}
	dq[0] =2*q[0] -q[1] -q[2];
	dq[1] = -q[0] +q[1];
	dq[2] = -q[0]       +q[2];
#endif
	for(int i=0; i<nneigh; ++i) dq[i] *= DqH_mult;
}

/** The computation of DpH does not involve any finite differences, but 
for symmetry with DqH_scheme we define a similarly named function.*/
Scalar DpH_scheme(const Scalar p){return p*DpH_mult;}

extern "C" {

__global__ void 
get_indices(
	const OffsetT * offsets_t, // [d,decompdim,n1,...,nd] ! Non standard in my kernels
	      Int     * ineigh_t // [n1,...,nd,decompdim,order]
){
	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}

	// Get the position where the work is to be done.
	Int x_t[ndim];
	Grid::Position(n_t,shape_tot,x_t);

	for(int decomp=0; decomp<decompdim; ++decomp){
		Int offset[ndim];
		for(int i=0; i<ndim; ++i){offset[i]=offsets_t[(i*decompdim+decomp)*size_tot+n_t];}
		Int y_t[ndim];

/*		if(n_t==0){
			printf("Position : %i, offset : %i\n",x_t[0],offset[0]);
		}
*/
#if fourth_order_macro
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
		
		// Fall back to the second order scheme
		if(!fourth_active){indices[2]=badIndex; indices[3]=badIndex;}
		const Int nstart_ineigh = n_t*(decompdim*4)+decomp*4;
		for(int i=0; i<4; ++i) {ineigh_t[nstart_ineigh+i]=indices[i];}

#else // second order scheme
		Int indices[2]={badIndex,badIndex};
		sub_vv(x_t,offset,y_t); 
		if(Grid::InRange_per(y_t,shape_tot)) indices[0]=Grid::Index_per(y_t,shape_tot); 
		add_vv(x_t,offset,y_t); 
		if(Grid::InRange_per(y_t,shape_tot)) indices[1]=Grid::Index_per(y_t,shape_tot);
/*		if(n_t==1){
			printf("Position : %i, offset : %i, total %i\n",x_t[0],offset[0],y_t[0]);
		}
*/
		const Int nstart_ineigh = n_t*(decompdim*2)+2*decomp;
		for(int i=0; i<2; ++i) {ineigh_t[nstart_ineigh+i]=indices[i];}
#endif // second or fourth order scheme
	} // for i
} // get_indices

/** Differentiate the Acoustic potential energy.
 * weights, offsets : Voronoi decomposition of the dual metric tensor.
 * q : position variable
 * wq : desired derivative
 * q_ad,w_ad,wq_ad : first order autodiff, forward or reverse.
 */
__global__ void DqH(
	const Scalar * __restrict__ weights_t,   // [sizetot,decompdim]
	const Int    * __restrict__ ineigh_t,    // [sizetot,decompdim,order]
	const Scalar * __restrict__ q_t,         // [sizetot]
AD(
FWD(const)Scalar * __restrict__ weights_ad_t,// [sizetot,decompdim,size_ad]
	const Scalar * __restrict__ q_ad_t,      // [sizetot,size_ad]
	      Scalar * __restrict__ wq_ad_t,)    // [sizetot,size_ad]
	      Scalar * __restrict__ wq_t         // [sizetot]
	){
	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}
	const Scalar q = q_t[n_t];
	Scalar dwq = 0;

AD( Scalar q_ad_all[size_ad];
	for(int ad=0; ad<size_ad; ++ad) {q_ad_all[ad] = q_ad_t[n_t*size_ad+ad];}
	Scalar dwq_ad_all[size_ad];
	for(int ad=0; ad<size_ad; ++ad) {dwq_ad_all[ad] = 0;}
	) // AD
	
	for(int decomp=0; decomp<decompdim; ++decomp){
		// Load the indices, neighbor values	
		Int i_neigh[order];
		for(int i=0; i<order; ++i){i_neigh[i] = ineigh_t[(n_t*decompdim+decomp)*order+i];}
		
		Scalar q_neigh[nneigh];
		q_neigh[0] = q;
		for(int i=0; i<order; ++i){ // Dirichlet boundary conditions
			q_neigh[1+i] = (i_neigh[i]!=badIndex) ? q_t[i_neigh[i]] : Scalar(0);}

		// Evaluate the scheme
		Scalar dq_neigh[nneigh];
		DqH_scheme(q_neigh,dq_neigh);

		// Update the neighbor values
		const Scalar w = weights_t[n_t*decompdim+decomp];
		dwq += w*dq_neigh[0];
		for(int i=0; i<order; ++i){
			if(i_neigh[i]!=badIndex){atomicAdd(wq_t+i_neigh[i], w*dq_neigh[1+i]);}}

	AD( for(int ad=0; ad<size_ad; ++ad){
		// Lod the neighbor values
		Scalar q_ad_neigh[nneigh];
		q_ad_neigh[0] = q_ad_all[ad];
		for(int i=0; i<order; ++i){
			q_ad_neigh[1+i] = i_neigh[i]!=badIndex ? q_ad_t[i_neigh[i]*size_ad+ad] : Scalar(0);}

		// Evaluate the scheme
		Scalar dq_ad_neigh[nneigh];
		DqH_scheme(q_ad_neigh,dq_ad_neigh);

	FWD(const Scalar w_ad = weights_ad_t[(n_t*decompdim+decomp)*size_ad+ad];)

		// Update the neighbor values
		dwq_ad_all[ad] += w*dq_ad_neigh[0] FWD(+ w_ad*dq_neigh[0]);
		for(int i=0; i<order; ++i){
			if(i_neigh[i]!=badIndex){atomicAdd(wq_ad_t+i_neigh[i]*size_ad+ad, 
				w*dq_ad_neigh[1+i] FWD(+ w_ad*dq_neigh[1+i]));}} // for i

	REV(Scalar w_ad = 0;
		for(int i=0; i<nneigh; ++i) w_ad += q_neigh[i]*dq_ad_neigh[i];
			// w_ad += dq_neigh[i]*q_ad_neigh[i]; // Equivalent by symmetry
		weights_ad_t[(n_t*decompdim+decomp)*size_ad+ad] += w_ad;) // REV
		}) // for ad // AD
	}
	atomicAdd(wq_t+n_t,dwq);
AD( for(int ad=0; ad<size_ad; ++ad){atomicAdd(wq_ad_t+n_t*size_ad+ad,dwq_ad_all[ad]);})
} // DqH

/* Differentiate the acoustic kinetic energy.
- m : inverse density, also denoted irho in the python code
- p : momentum variable
- mp : desired derivative (yes, this is just m*p)
- p_ad,m_ad,mp_ad : first order autodiff, forward or reverse.

 * Since this is a purely local and quite trivial operation, 
 * the need for a kernel is not obvious. We do it for consistency.
*/
__global__ void DpH(
	const Scalar * __restrict__ m_t,      // [n1,...,nd]
	const Scalar * __restrict__ p_t,      // [n1,...,nd]
AD( // Autodiff variables, forward or reverse
FWD(const)Scalar * __restrict__ m_ad_t,   // [n1,...,nd,size_ad]
	const Scalar * __restrict__ p_ad_t,   // [n1,...,nd,size_ad]
	      Scalar * __restrict__ mp_ad_t,) // [n1,...,nd,size_ad]
	      Scalar * __restrict__ mp_t      // [n1,...,nd]
	){
	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) {return;}

	const Scalar m = m_t[n_t];
	const Scalar p = p_t[n_t];
	const Scalar dp = DpH_scheme(p);

	const Scalar dmp = m*dp;
	mp_t[n_t] += dmp;

AD( for(int ad=0; ad<size_ad; ++ad){ // Autodiff over all channels
	const Int nstart_ad = n_t*size_ad;
	const Scalar p_ad = p_ad_t[nstart_ad+ad];
	const Scalar dp_ad = DpH_scheme(p_ad);

FWD(const Scalar m_ad = m_ad_t[nstart_ad+ad];) // FWD
	mp_ad_t[nstart_ad+ad] += m*dp_ad FWD(+ m_ad*dp);

REV(m_ad_t[nstart_ad+ad]  += p*dp_ad;) // REV
	}) // for ad // AD
} // DpH

} // extern "C"