// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the linear elastic wave equation operators, in the case of a fully 
generic hooke tensor. The tensor must be decomposed for finite differences using, e.g.,
Voronoi's first reduction.
*/

/* // The following must be defined externally
typedef float Scalar; 
#define fourth_order_macro false
#define ndim_macro 2
#define periodic_macro false
*/
typedef char OffsetT;
typedef int OffsetPack;

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
#else
#define PERIODIC(...) 
#endif

const int ndim = ndim_macro;
#include "Geometry_.h"
__constant__ int shape_o[ndim];
__constant__ int shape_tot[ndim];
#include "Grid.h"
// TODO : It could make sense to introduce one additional level for better memory coherency

namespace geom_symdim {
	const int ndim = symdim;
	#include "Geometry_.h"
}

const int decompdim = geom_symdim::symdim; // Voronoi decomposition of hooke tensor
const int firstdim = ndim*symdim;

__constant__ Scalar dtQ = NAN, dtP = NAN;
__constant__ Scalar idx[ndim]; // Inverse grid scale along each axis

// Unpack the offsets of Voronoi's first reduction
void offset_expand(OffsetPack pack, OffsetT exp[symdim]){
	const int nbit = ndim==2 ? 10 : 5; // Number of bits for each offset
	const int mask = (1<<nbit)-1;
	const int zero = 1<<(nbit-1);

	for(int i=0; i<symdim; ++i){exp[i] = ((pack >> (i*nbit)) & mask) - zero;}
}

extern "C" {

void UpdateQ(
	const Scalar * __restrict__ weights_t,     // [size_o,decompdim,size_i]
	const OffsetPack * __restrict__ offsets_t, // [size_o,decompdim,size_i]
	const Scalar * __restrict__ firstorder_t,  // [size_o,firstdim,size_i]
	const Scalar * __restrict__ damping_t,     // [size_o,size_i]
	const Scalar * __restrict__ qold_t,        // [size_o,ndim,size_i]
	const Scalar * __restrict__ p_t,           // [size_o,ndim,size_i]
	Scalar * __restrict__ qnew_t               // [size_o,ndim,size_i]
	){

	// Compute position
	Int x_o[ndim], x_i[ndim];
	x_o[0] = blockIdx.x; x_i[0] = threadIdx.x; 
	x_o[1] = blockIdx.y; x_i[1] = threadIdx.y; 
	#if ndim_macro==3
	x_o[2] = blockIdx.z; x_i[2] = threadIdx.z; 
	#endif

	Int x_t[ndim]; 
	for(int i=0; i<ndim; ++i){x_t[i] = x_i[i]+x_o[i]*shape_i[i];}

	const int n_o = Grid::Index(x_o,shape_o);
	const int n_i = Grid::Index(x_i,shape_i); 
	const int n_oi = n_o*size_i; 
	int nstart;// Mutable, used for array data start
	
	// Weights and offsets are needed one at a time in the loop, 
	// but we load them all here since they are contiguous in memory.
	Scalar weights[decompdim];
	OffsetPack offsets[decompdim];
	nstart = n_oi*decompdim + n_i;
	for(int i=0; i<decompdim; ++i){weights[i] = weights_t[nstart + size_i * i];}
	for(int i=0; i<decompdim; ++i){offsets[i] = offsets_t[nstart + size_i * i];}

	Scalar firstorder[firstdim];
	nstart = n_oi*firstdim + n_i;
	for(int i=0; i<firstdim; ++i){firstorder[i] = firstorder_t[nstart + size_i * i];}

	const Scalar damping = damping_t[noi+n_i];

	Scalar qold[ndim];
	Scalar p[ndim];
	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){q_old[i] = qold_t[nstart + size_i * i];}
	for(int i=0; i<ndim; ++i){p[i]     = p_t[   nstart + size_i * i];}

	// Contribution of zero-th order term
	Scalar qnew[ndim];
	mul_kv(1.-dtQ*damping,qold,qnew); 

	Scalar stress[symdim];
	geom_symdim::fill_kV(0,stress);

	for(int decomp_i=0; decomp_i<decompdim; ++decomp_i){

		// Load one weight and offset
		Scalar weight = weights[decomp_i];
		OffsetT offset[symdim]; 
		offset_expand(offsets[decomp_i],offset);

		// First and second order finite differences of the i-th component of qold
		// along the i-th line of offset
		Scalar diff1[ndim];
		Scalar diff2[ndim];

		for(int i=0; i<ndim; ++i){
			// Values of the i-th component of qold along the i-th line of offset
			Scalar qneigh1[2];
			#if fourth_order_macro
			Scalar qneigh2[2];
			#endif

			OffsetT offset_i[ndim]; // i-th line of offset 
			for(int j=0; j<ndim; ++j){offset_i[j] = coef_m(offset,i,j);}

			for(int side=0; side<=1; ++side){

			#if fourth_order_macro 
			for(int dist=1; dist<=2; ++dist){ // Fetch at distance two along offset
				const int eps = dist*(2*side-1);
			#else
				const int eps = 2*side-1:
			#endif

				Scalar value; // value to be fetched. i-th component along i-th offset
				Int y_t[ndim]; 
				madd_kvv(eps,offset_i,x_t,y_t);

				if(Grid::InRange_per(y_t,shape_tot)){
					Int y_o[ndim],y_i[ndim];
					for(int k=0; k<ndim; ++k){
						const int yk = PERIODIC(periodic[k] ? mod_pos(y_t[k]) :) y_t[k];
						y_o[k] = yk / shape_i[k]; y_i[k] = yk%shape_i[k];}
					const int 
					ny_o = Grid::Index(y_o,shape_o),
					ny_i = Grid::Index(y_i,shape_i);
					const int ny_oi = ny_o*size_i;
					value = qold_t[ny_oi + ny_i + i*size_i];
				} else { // y_t is out of range
					value = 0; // Null dirichlet boundary conditions
				}

			#if fourth_order_macro
				if(dist==1) {qneigh1[side] = value;}
				else        {qneigh2[side] = value;}

			} // for dist
			#else
			qneigh1[i] = value;
			#endif
			}

			// Finite differences
			#if fourth_order_macro
			diff1[i] =  (4./3.)*(qneigh1[1] - qneigh1[0])
					   -(1./3.)*(qneigh2[1] - qneigh2[0]);
			diff2[i] =  (4./3.)*(qneigh1[1] + qneigh1[0])
					   -(1./3.)*(qneigh2[1] + qneigh2[0])
					   - 2. * qold[i];
			#else
			diff1[i] = qneigh1[1] - qneigh1[0];
			diff2[i] = qneigh1[1] + qneigh1[0] - 2. * qold[i];
			#endif

			// Take into account the grid scales
			diff1[i] *= idx[i]/2.;
			diff2[i] *= idx[i]*idx[i];
		}

		// Contribution of the second order term
		madd_kvV(dtQ*weight,diff2,qnew);

		// Build the stress tensor
		Scalar diff1sum = 0; 
		for(int i=0; i<ndim; ++i) {diff1sum+=diff1[i];}
		geom_symdim::madd_kvV(diff1sum*weight,offset,stress);
	}
	
	// Contribution of the first order term
	for(int i=0; i<ndim; ++i){qnew[i] += dtQ*scal_mm(firstorder + i*symdim,stress);}

	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){qnew_t[nstart + size_i * i] = qnew[i];}
}

void UpdateP(
	const Scalar * __restrict__ metric_t, // [size_o,symdim,size_i]
	const Scalar * __restrict__ damping_t,// [size_o,size_i]
	const Scalar * __restrict__ q_t,      // [size_o,ndim,size_i]
	const Scalar * __restrict__ pold_t,   // [size_o,ndim,size_i]
	Scalar * __restrict__ pnew_t          // [size_o,ndim,size_i]
	){

	// Since UpdateP is a purely local operation, the need for a kernel is not obvious...

	// Compute position
	const Int n_o = blockIdx.x;
	const Int n_i = threadIdx.x;
	const Int n_oi = n_o*size_i; 
	Int nstart;// Mutable, used for array data start
	// Int x_t[ndim]; // Useless, this update does not involve finite differences

	// Load data
	Scalar metric[symdim];
	nstart = n_oi*symdim + n_i;
	for(int i=0; i<symdim; ++i){metric[i] = metric_t[nstart + size_i * i];}

	const Scalar damping = damping_t[noi + n_i];
	
	Scalar q[ndim];
	Scalar pold[ndim];
	nstart = n_oi*ndim+n_i,
	for(int i=0; i<ndim; ++i){q[i]    = q_t[   nstart + size_i * i];}
	for(int i=0; i<ndim; ++i){pold[i] = pold_t[nstart + size_i * i];}

	// Update
	Scalar pnew[ndim];
	mul_kv(1-damping*dtP,pold);

	Scalar mq[ndim];
	mul_mv(m,q,mq);

	madd_kv(dtQ,mq,pnew);

	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){pnew_t[nstart + size_i * i] = pnew[i];}
}

}
