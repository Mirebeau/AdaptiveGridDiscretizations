#pragma once
// Copyright 2022 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements low-level GPU routines for the a relaxed Bellman formulation of the 
 optimal transport problem.
*/


/** The following need to be defined in including file (example)
typedef int Int;
typedef float Scalar;
#define ndim_macro 3
const int shape_i[ndim] = {4,4,4}; 
const int size_i = 64;
const int shape_e[ndim] = {5,5,5};
const int size_e = 125;
const int size_bd_e = size_e-size_i;
x_top_e[size_bd_e][ndim] = {...}; // Top and bottom points of shape_e in memory friendly order
x_bot_e[size_bd_e][ndim] = {...};
const int nchannels = 3;
*/

#if   ndim_macro == 1
#include "Geometry1.h"
#elif ndim_macro == 2
#include "Geometry2.h"
#elif ndim_macro == 3
#include "Geometry3.h"
#else
STATIC_ASSERT(false,Unsupported_dimension);
#endif 

#include "REDUCE_i.h"

// We use a two-level grid. shape_tot_s = shape_tot_v + 1
__constant__ Int shape_o[ndim]; 
__constant__ Int size_io;
__constant__ Int shape_tot_v[ndim]; // Shape of domain for vector fields
__constant__ Int shape_tot_s[ndim]; // Shape of domain for scalar fcts

__constant__ Scalar tau_primal, tau_dual; // Proximal time steps
__constant__ Scalar idx[ndim]; // Inverse grid scale
__constant__ Scalar lambda,ilambda; // relaxation parameter and its inverse
__constant__ Scalar irelax_norm_constraint; // Characteristic function of Lipschitz maps
__constant__ Scalar rho_overrelax; // Over relaxation parameter for the iterations
__constant__ Scalar atol,rtol; // tolerance for the termination criterion

#undef bilevel_grid_macro
#include "Grid.h"
STATIC_ASSERT(size_bd_e==size_e-size_i && size_bd_e<size_i, Incorrect_size_bd_e);

extern "C" {

/** Generates a new primal point. */
__global__ void primal_step(
	Scalar * __restrict__ phi_t, // primal variable (input and output)
	Scalar * __restrict__ phi_ext_t, // extrapolated primal variable (output)
	const Scalar * __restrict__ eta_t, // dual variable 
	const Scalar * __restrict__ xi_t // target measure
#if checkstop_macro
 ,Scalar * __restrict__ primal_value_o,  // primal objective value of the block
	Scalar * __restrict__ dual_value_o, // dual objective value of the block
	char * __restrict__ stabilized_o // Wether the block was effectively updated
#endif
	){

// Compute the position the local and global grids
const int n_i = threadIdx.x;
const int n_o = blockIdx.x;
const int n_t = n_o*size_i + n_i;
int x_i[ndim];
Grid::Position(n_i,shape_i,x_i);
__shared__ int x_o[ndim];
if(n_i==0){Grid::Position(n_o,shape_o,x_o);}
__syncthreads();
int x_t[ndim];
for(int k=0; k<ndim; ++k){x_t[k]=x_o[k]*shape_i[k]+x_i[k];}
const bool inRange = Grid::InRange(x_t,shape_tot_s);

// Initially included a weight to better match the reflected boundary conditions.
// Eventually removed since it leads to incorrect normgrad2
const Scalar weight = 1.; 
//for(int k=0; k<ndim; ++k){if(x_t[k]==0 || x_t[k]==shape_tot_s[k]-1) weight/=2.;}

__shared__ Scalar eta_e[size_e][ndim];
#if checkstop_macro
__shared__ Scalar primal_value_i[size_i], dual_value_i[size_i];
__shared__ Scalar phi_delta_max_i[size_i], phi_max_i[size_i];
primal_value_i[n_i]=0; dual_value_i[n_i]=0;
phi_delta_max_i[n_i]=0; phi_max_i[n_i]=0;
#endif
// Each channel can be dealt with independently
#pragma unroll 
for(int ichannel=0; ichannel<nchannels; ++ichannel
	,phi_t += size_io, phi_ext_t += size_io, // Offset the arrays
	eta_t += ndim*size_io, xi_t += size_io
	){

// ---- Load the vector field ----
{	// At the current point
	int x_e[ndim]; // position in shared array
	for(int k=0; k<ndim; ++k) {x_e[k] = x_i[k]+1;}
	const int n_e = Grid::Index(x_e,shape_e);
	if(Grid::InRange(x_t,shape_tot_v)){
		for(int k=0; k<ndim; ++k) {eta_e[n_e][k] = eta_t[n_t+size_io*k];}
	} else {for(int k=0; k<ndim; ++k) {eta_e[n_e][k]=0;}}
}
if(n_i<size_bd_e){ // At the chosen boundary point
	const int n_e = Grid::Index(x_bot_e[n_i],shape_e);
	int q_t[ndim];
	for(int k=0; k<ndim; ++k){q_t[k] = x_o[k]*shape_i[k]+x_bot_e[n_i][k]-1;}	
	if(Grid::InRange(q_t,shape_tot_v)) {
		const int n_t = Grid::Index_tot(q_t,shape_tot_v,shape_o,shape_i,size_i);
		for(int k=0; k<ndim; ++k) {eta_e[n_e][k] = eta_t[n_t+size_io*k];}
	} else {for(int k=0; k<ndim; ++k) {eta_e[n_e][k]=0;}}
}
__syncthreads();

// ------ Compute the divergence of the vector field ------
// Does not take into account the boundary weights
Scalar div_eta;
const int n_e = Grid::Index(x_i,shape_e);

#if ndim_macro==1
const int n0 = n_e, n1 = n_e+1;
div_eta = (eta_e[n1][0]-eta_e[n0][0])*idx[0];

#elif ndim_macro==2
const int n00 = n_e, n01 = n_e+1, n10 = n_e+shape_e[1], n11 = n10+1;
div_eta =
  (eta_e[n11][0] - eta_e[n01][0])*idx[0]
+ (eta_e[n11][1] - eta_e[n10][1])*idx[1];
#elif ndim_macro==3
const int 
n000 = n_e, n001 = n_e+1, n010 = n_e+shape_e[2], n011 = n010+1,
n100 = n_e+shape_e[1]*shape_e[2], n101=n100+1, n110 = n100+shape_e[2], n111 = n110+1;

div_eta = 
  (eta_e[n111][0] - eta_e[n011][0])*idx[0]
+ (eta_e[n111][1] - eta_e[n101][1])*idx[1] 
+ (eta_e[n111][2] - eta_e[n110][2])*idx[2];
#endif


const Scalar phi_old = phi_t[n_t];
const Scalar xi = xi_t[n_t];

const Scalar phi_in = phi_old + tau_primal*div_eta/weight;
const Scalar phi_new = (phi_in+tau_primal*xi)/(Scalar(1)+tau_primal*ilambda); //prox_f 
const Scalar phi_ext = 2*phi_new - phi_old; // Extrapolation step
const Scalar phi_delta = phi_new-phi_old;

phi_t[n_t] = phi_old+rho_overrelax*phi_delta;
phi_ext_t[n_t] = phi_ext;

// Compute primal and dual energy values, and check stopping criteria
#if checkstop_macro
primal_value_i[n_i] += Scalar(0.5)*ilambda*phi_ext*phi_ext - xi*phi_ext; // function f
const Scalar s = xi+div_eta/weight;
dual_value_i[n_i] += Scalar(0.5)*lambda*s*s; // function f^*

phi_delta_max_i[n_i] = max(phi_delta_max_i[n_i], abs(phi_delta));
phi_max_i[n_i] = max(phi_max_i[n_i],abs(phi_new));
#endif
__syncthreads(); // IMPORTANT, since shared values are modified in next iteration
} // for ichannel

// Export primal and dual energy values, stopping criteria
#if checkstop_macro
primal_value_i[n_i]*=weight; dual_value_i[n_i]*=weight;
if(!inRange) {primal_value_i[n_i]=0; dual_value_i[n_i]=0;}
__syncthreads();
REDUCE_i(  
	primal_value_i[n_i] += primal_value_i[m_i]; 
	dual_value_i[n_i]   += dual_value_i[m_i];
	phi_delta_max_i[n_i] = max(phi_delta_max_i[n_i],phi_delta_max_i[m_i]);
	phi_max_i[n_i]       = max(phi_max_i[n_i],phi_max_i[m_i]);
	)
if(n_i==0){
	primal_value_o[n_o] =primal_value_i[n_i]; 
	dual_value_o[n_o] =dual_value_i[n_i];
	if(phi_delta_max_i[n_i] < atol+rtol*phi_max_i[n_i]) stabilized_o[n_o] = 1;
}
#endif
} // primal step


/** Generates a new dual point. */
__global__ void dual_step( 
	      Scalar * __restrict__ eta_t, // Dual variable (input and output)
	const Scalar * __restrict__ phi_ext_t // Primal variable, extrapolated
#if checkstop_macro
 ,Scalar * __restrict__ primal_value_o, // primal objective value of the block
	Scalar * __restrict__ dual_value_o, // dual objective value of the block
	char * __restrict__ stabilized_o
#endif
	){

// Get current position in grid
const int n_i = threadIdx.x;
const int n_o = blockIdx.x;
const int n_t = n_o*size_i + n_i;
int x_i[ndim];
Grid::Position(n_i,shape_i,x_i);
__shared__ int x_o[ndim];
if(n_i==0){Grid::Position(n_o,shape_o,x_o);}
const int n_e = Grid::Index(x_i,shape_e);
__syncthreads();
int x_t[ndim];
for(int k=0; k<ndim; ++k){x_t[k]=x_o[k]*shape_i[k]+x_i[k];}
const bool inRange = Grid::InRange(x_t,shape_tot_v);

// --- Load data ---
__shared__ Scalar phi_e[size_e];
Scalar multi_grad_phi[nchannels*ndim];
Scalar * grad_phi = multi_grad_phi;

#pragma unroll 
for(int ichannel=0; ichannel<nchannels; ++ichannel
   ,phi_ext_t+=size_io, grad_phi+=ndim
	){

phi_e[n_e]  = phi_ext_t[n_t];  // Load the interior value
if(n_i<size_bd_e){ // Load the boundary value
	const int n_e = Grid::Index(x_top_e[n_i],shape_e);
	int q_t[ndim], k;
	for(k=0; k<ndim; ++k){
		q_t[k] = x_o[k]*shape_i[k]+x_top_e[n_i][k];
		if(q_t[k]>=shape_tot_s[k]) break;
	}
	if(k==ndim){
		const int n_t = Grid::Index_tot(q_t,shape_tot_s,shape_o,shape_i,size_i);
		phi_e[n_e] = phi_ext_t[n_t];
	} else {phi_e[n_e]=0;} // Not really needed
}
__syncthreads();

// ------ Gradient computation ------
#if ndim_macro==1
const int n0 = n_e, n1 = n_e+1;
grad_phi[0] = phi_e[n1]-phi_e[n0];

#elif ndim_macro==2
const int n00 = n_e, n01 = n_e+1, n10 = n_e+shape_e[1], n11 = n10+1;

const Scalar p00 = phi_e[n00], p01 = phi_e[n01], p10=phi_e[n10];
grad_phi[0] = p10 - p00;
grad_phi[1] = p01 - p00;

#elif ndim_macro==3
const int 
n000 = n_e, n001 = n_e+1, n010 = n_e+shape_e[2], n011 = n010+1,
n100 = n_e+shape_e[1]*shape_e[2], n101=n100+1, n110 = n100+shape_e[2], n111 = n110+1;

const Scalar p000 = phi_e[n000], p001 = phi_e[n001], p010=phi_e[n010], p100=phi_e[n100];
grad_phi[0] = p100 - p000;
grad_phi[1] = p010 - p000;
grad_phi[2] = p001 - p000;
#endif

for(int k=0; k<ndim; ++k) {grad_phi[k]*=idx[k];}
__syncthreads(); // IMPORTANT, since shared values are modified in next iteration

} // for ichannel

// --- Computation of the new dual point ----
Scalar eta_old[nchannels*ndim], proxin[nchannels*ndim];
Scalar proxin_norm2=0;
#if checkstop_macro
Scalar eta_norm2=0, grad_phi_norm2=0;
#endif

for(int k=0; k<nchannels*ndim; ++k){
	eta_old[k] = eta_t[k*size_io+n_t];
	proxin[k] = eta_old[k] + tau_dual*multi_grad_phi[k];
	proxin_norm2 += proxin[k]*proxin[k];
	// eta_norm2 and grad_phi_norm2 are only used for evaluating primal and dual energies
#if checkstop_macro
	eta_norm2 += eta_old[k]*eta_old[k];
	grad_phi_norm2 += multi_grad_phi[k]*multi_grad_phi[k];
#endif
}

#if checkstop_macro
__shared__ Scalar eta_delta_max_i[size_i], eta_max_i[size_i];
eta_delta_max_i[n_i]=0; eta_max_i[n_i]=0;
#endif

// Proximal operator of x -> tau_dual |x|
const Scalar prox_mult = Scalar(1)-Scalar(1)/max(Scalar(1), sqrt(proxin_norm2)/tau_dual);
Scalar eta_new[nchannels*ndim];
for(int k=0; k<nchannels*ndim; ++k){
	eta_new[k] = prox_mult*proxin[k];
	const Scalar eta_delta = eta_new[k]-eta_old[k];
	eta_t[k*size_io+n_t] = eta_old[k]+rho_overrelax*eta_delta;
#if checkstop_macro
	eta_delta_max_i[n_i]=max(eta_delta_max_i[n_i],abs(eta_delta));
	eta_max_i[n_i]=max(eta_max_i[n_i],abs(eta_new[k]));
#endif
}
// Export primal and dual energy values, and check stopping criteria
#if checkstop_macro
__shared__ Scalar primal_value_i[size_i], dual_value_i[size_i];
// Contribution to the primal energy
const Scalar excess=max(Scalar(0),sqrt(grad_phi_norm2)-Scalar(1))*irelax_norm_constraint;
primal_value_i[n_i] = excess*excess;// function g

// Contribution to the dual energy
dual_value_i[n_i] = sqrt(eta_norm2); // function g^*

if(!inRange){primal_value_i[n_i]=0; dual_value_i[n_i]=0;}
__syncthreads();
REDUCE_i(  
	primal_value_i[n_i]+=primal_value_i[m_i]; 
	dual_value_i[n_i]+=dual_value_i[m_i];
	eta_delta_max_i[n_i] = max(eta_delta_max_i[n_i],eta_delta_max_i[m_i]);
	eta_max_i[n_i]       = max(eta_max_i[n_i],eta_max_i[m_i]);
	)
if(n_i==0){
	primal_value_o[n_o]+=primal_value_i[n_i]; 
	dual_value_o[n_o]+=dual_value_i[n_i];
	if(eta_delta_max_i[n_i]<atol+rtol*eta_max_i[n_i]) stabilized_o[n_o]=1;
}
#endif
} // dual_step

} // extern "C"