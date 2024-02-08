// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file computes on the GPU and exports a Voronoi decomposition of a quadratic form.*/

/* // The following typedefs, or equivalent, must be defined in enclosing file
typedef int Int;
typedef float Scalar;
typedef char OffsetT;
*/  


#if ndim_macro==2
#include "Geometry2.h"
#elif ndim_macro==3
#include "Geometry3.h"
#elif ndim_macro==4
#include "Geometry4.h"
#elif ndim_macro==5
#include "Geometry5.h"
#elif ndim_macro==6
#define CUDA_DEVICE // Do not include <math.h>, and do not use exit(1) in linprog
#define SIMPLEX_TOL SIMPLEX_TOL_macro
#include "Geometry6.h"
#endif

__constant__ int size_tot;
const Int nsym = decompdim;

void scheme(const Scalar dual_metric[symdim], Scalar weights[nsym], OffsetT offsets[nsym][ndim]){
	// For some mysterious reason, decomp_m needs to be called from a __device__ function
	// otherwise cudaIllegalAddressError ?!? (Problem related with embedded lp solver ?!?)
	decomp_m(dual_metric,weights,offsets);
}

#if ndim_macro>=4
void KKT_(const Voronoi::SimplexStateT & state, Scalar weights[decompdim], 
	OffsetT offsets[decompdim][ndim]){
	KKT(state,weights,offsets);
}
#endif

extern "C" {

__global__ void VoronoiDecomposition(const Scalar * __restrict__ m_t,
	Scalar * __restrict__ weights_t, OffsetT * __restrict__ offsets_t){
	
	const int n_i = threadIdx.x;
	const int n_o = blockIdx.x;
	const int n_t = n_o*blockDim.x + n_i;
	if(n_t>=size_tot) return;

	// Load the data
	Scalar m[symdim];
	Scalar weights[decompdim];
	OffsetT offsets[decompdim][ndim]; // geometry last
	for(Int i=0; i<symdim; ++i){m[i] = m_t[n_t+size_tot*i];}

	// Voronoi decomposition
	scheme(m,weights,offsets); // Cannot call decomp_m directly
	
	// Export
	for(Int i=0; i<decompdim; ++i){
		weights_t[n_t+i*size_tot]=weights[i];} // geometry first
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<decompdim; ++j){
			offsets_t[n_t+size_tot*(j+decompdim*i)]=offsets[j][i];}} // geometry first
			
}

#if ndim_macro>=4

__global__ void VoronoiMinimization(Scalar * __restrict__ m_t,Scalar * __restrict__ a_t, 
	int * __restrict__ vertex_t, Scalar * __restrict__ objective_t){

	const int n_i = threadIdx.x;
	const int n_o = blockIdx.x;
	const int n_t = n_o*blockDim.x + n_i;
	if(n_t>=size_tot) return;

	// Load the data
	Voronoi::SimplexStateT state;
	for(Int i=0; i<symdim; ++i){state.m[i] = m_t[n_t+size_tot*i];}

	// Do the minimization
	identity_A(state.a);
	Voronoi::FirstGuess(state); 
	for(int i=0; i<Voronoi_maxiter; ++i){if(!Voronoi::BetterNeighbor(state)){break;}} 

	// Export the results
	for(int i=0; i<symdim; ++i){m_t[n_t+size_tot*i] = state.m[i];}
	for(int i=0; i<ndim; ++i){
		for(int j=0; j<ndim; ++j){
			a_t[n_t + size_tot*(j+ndim*i)] = state.a[i][j];}
	}
	vertex_t[n_t] = state.vertex;
	objective_t[n_t] = state.objective;
}

__global__ void VoronoiKKT(const Scalar * __restrict__ m_t, const Scalar * __restrict__ a_t, 
	const int * __restrict__ vertex_t, const Scalar * __restrict__ objective_t,
	Scalar * __restrict__ weights_t, OffsetT * __restrict__ offsets_t){

	const int n_i = threadIdx.x;
	const int n_o = blockIdx.x;
	const int n_t = n_o*blockDim.x + n_i;
	if(n_t>=size_tot) return;

	// Load the data
	Voronoi::SimplexStateT state;
	for(int i=0; i<symdim; ++i){state.m[i] = m_t[n_t+size_tot*i];}
	for(int i=0; i<ndim; ++i){
		for(int j=0; j<ndim; ++j){
			state.a[i][j] = a_t[n_t + size_tot*(j+ndim*i)];}
	}
	state.vertex = vertex_t[n_t];
	state.objective = objective_t[n_t];

	// Solve the linear program
	Scalar weights[decompdim];
	OffsetT offsets[decompdim][ndim]; // geometry last
	KKT_(state,weights,offsets);

	// Export the results
	for(int i=0; i<decompdim; ++i){
		weights_t[n_t+i*size_tot]=weights[i];} // geometry first
	for(int i=0; i<ndim; ++i){
		for(Int j=0; j<decompdim; ++j){
			offsets_t[n_t+size_tot*(j+decompdim*i)]=offsets[j][i];}} // geometry first
}

#endif

} // Extern "C"