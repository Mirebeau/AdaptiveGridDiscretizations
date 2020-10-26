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
*/
typedef int OffsetPack;
typedef int Int;

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
// const bool periodic_axes[ndim] = {false,false,true}; // must be defined externally
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

__constant__ Scalar dt; // time step
__constant__ Scalar idx; // Inverse grid scale 

/** Bypass computations involving null weights or offsets. Should speed up computations 
involving isotropic hooke tensors in particular.*/
#if bypass_zeros_macro
#define BYPASS_ZEROS(...) __VA_ARGS__
#else 
#define BYPASS_ZEROS(...) 
#endif

/** Replace cross differences involving the same vectors, or opposite vectors, with 
second order finite differences. Not much effect expected.*/
#if compact_scheme_macro
#define COMPACT_SCHEME(...) __VA_ARGS__
#else
#define COMPACT_SCHEME(...) 
#endif

// ------------- Offset manipulation -------------

/// Unpack the offsets of Voronoi's first reduction
void offset_expand(OffsetPack pack, Int exp[symdim]){
	const int nbit = ndim==2 ? 10 : 5; // Number of bits for each offset
	const int mask = (1<<nbit)-1;
	const int zero = 1<<(nbit-1);

	for(int i=0; i<symdim; ++i){exp[i] = ((pack >> (i*nbit)) & mask) - zero;}
}


bool is_zero(const Int e[ndim]){
	for(int i=0; i<ndim; ++i){if(e[i]!=0) return false;}
	return true;
}

bool is_same(const Int e[__restrict__ ndim], const Int f[__restrict__ ndim]){
	for(int i=0; i<ndim; ++i){if(e[i]!=f[i]) return false;}
	return true;
}

bool is_opp(const Int e[__restrict__ ndim], const Int f[__restrict__ ndim]){
	for(int i=0; i<ndim; ++i){if(e[i]!=-f[i]) return false;}
	return true;
}

// -------------------- Vector field component access -------------------

/// Return a given component, at a given position, of a vector field
Scalar component(const Int comp, const Int x_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t){

	if(Grid::InRange_per(x_t,shape_tot)){
		Int x_o[ndim],x_i[ndim];
		for(int k=0; k<ndim; ++k){
			const int xk = 
			PERIODIC(periodic_axes[k] ? Grid::mod_pos(x_t[k],shape_tot[k]) :) 
			x_t[k];
			x_o[k] = xk / shape_i[k]; x_i[k] = xk % shape_i[k];}
		const int 
		n_o = Grid::Index(x_o,shape_o),
		n_i = Grid::Index(x_i,shape_i);
		const int n_oi = n_o*size_i;
		const int nstart = n_oi*ndim + n_i;
		return q_t[nstart + size_i * comp];
	} else {
		return 0.; // Null dirichlet boundary conditions
	}
}

/// Like component, but includes a correction for the fourth order scheme
Scalar component_corrected(const Int comp, const Int offset[__restrict__ ndim],
	const Int x0_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t){

	Int x1_t[ndim];
	add_vv(offset,x0_t,x1_t);

	#if fourth_order_macro
	Int x2_t[ndim];
	add_vv(offset,x1_t,x2_t);
	return (4./3.)*component(comp,x1_t,q_t) - (1./3.)*component(comp,x2_t,q_t);
	#else
	return component(comp,x1_t,q_t);
	#endif
}

extern "C" {

__global__ void AdvanceP(
	const Scalar * __restrict__ weights_t,     // [size_o,decompdim,size_i]
	const OffsetPack * __restrict__ offsets_t, // [size_o,decompdim,size_i]
	const Scalar * __restrict__ firstorder_t,  // [size_o,firstdim,size_i]
	const Scalar * __restrict__ damping_t,     // [size_o,size_i]
	const Scalar * __restrict__ q_t,           // [size_o,ndim,size_i]
	const Scalar * __restrict__ pold_t,        // [size_o,ndim,size_i]
	Scalar       * __restrict__ pnew_t         // [size_o,ndim,size_i]
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

	if(n_i==0 && n_o==0){
		printf("x_i %i,%i, x_o %i,%i \n",x_i[0],x_i[1],x_o[0],x_o[1]);
	} 
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

	const Scalar damping = damping_t[n_oi+n_i];

	Scalar q[ndim];
	Scalar pold[ndim];
	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){q[i]    = q_t[   nstart + size_i * i];}
	for(int i=0; i<ndim; ++i){pold[i] = pold_t[nstart + size_i * i];}

	// Contribution of zero-th order term
	Scalar pnew[ndim];
	mul_kv(1.-dt*damping,pold,pnew); 

	Scalar stress[symdim];
	geom_symdim::fill_kV(Scalar(0),stress);

	for(int decomp=0; decomp<decompdim; ++decomp){

		// Load one weight and offset. Expand offset.
		Scalar weight = weights[decomp];
		BYPASS_ZEROS(if(weight==0) continue;)
		Int offset[symdim]; 
		offset_expand(offsets[decomp],offset);

		Int moffset[ndim][ndim];
		for(int i=0; i<ndim; ++i){
			for(int j=0; j<ndim; ++j){
				moffset[i][j] = coef_m(offset,i,j);
			}
		}

		// Contribution from the second order operator dvi = m_ij m_kl D_jk v_l
		const Scalar w2 = dt*weight*idx*idx; // Rescaled weight for second order diff
		for(int i=0; i<ndim; ++i){
			const Int * e = moffset[i]; // e[ndim]
			BYPASS_ZEROS(if(is_zero(e)) continue;)
			for(int l=0; l<ndim; ++l){
				const Int * f = moffset[l];
				BYPASS_ZEROS(if(is_zero(f)) continue;)
				// Evaluate the cross derivative of v_l w.r.t moffsets[i] and moffsets[l]
				Scalar cross;

				if(i==l COMPACT_SCHEME(||is_same(e,f)||is_opp(e,f))){
					Int ne[ndim]; neg_v(e,ne);
					cross = 
					  component_corrected(l,e,x_t,q_t) 
					 -2*q[l]
					 +component_corrected(l,ne,x_t,q_t);
					 COMPACT_SCHEME(if(is_opp(e,f)) cross*=-1;)
				} else {
					// Note : if e=f or e=-f, the previous more compact scheme could be used.
					Int pp[ndim],pm[ndim],mp[ndim],mm[ndim];
					for(int k=0; k<ndim; ++k){
						pp[k] =  e[k] +f[k];
						pm[k] =  e[k] -f[k];
						mp[k] = -e[k] +f[k];
						mm[k] = -e[k] -f[k];
					}
					cross = (
					  component_corrected(l,pp,x_t,q_t) 
					 -component_corrected(l,pm,x_t,q_t)
					 -component_corrected(l,mp,x_t,q_t)
					 +component_corrected(l,mm,x_t,q_t) )/4.;
				}

				pnew[i]+=w2*cross;
			}
		}

		// Reconstruction of the stress tensor m_ij m_kl D_k v_l
		Scalar diff1sum = 0;
		for(int l=0; l<ndim; ++l){
			const Int * e = moffset[l]; // e[ndim]
			BYPASS_ZEROS(if(is_zero(e)) continue;)
			Int ne[ndim]; neg_v(e,ne);

			const Scalar diff1 = 
			  component_corrected(l,e,x_t,q_t) 
			 -component_corrected(l,ne,x_t,q_t);

			diff1sum+=diff1;
		}
		const Scalar w1 = weight*idx/2;
		geom_symdim::madd_kvV(diff1sum*w1,offset,stress);
	}
	if(n_i==0 && n_o==0){
		printf("pnew %f,%f\n", pnew[0],pnew[1]);
	}
	// Contribution of the first order term
	for(int i=0; i<ndim; ++i){pnew[i] -= dt*scal_mm(firstorder + i*symdim,stress);}

	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){pnew_t[nstart + size_i * i] = pnew[i];}
}

__global__ void AdvanceQ(
	const Scalar * __restrict__ metric_t, // [size_o,symdim,size_i]
	const Scalar * __restrict__ damping_t,// [size_o,size_i]
	const Scalar * __restrict__ qold_t,        // [size_o,ndim,size_i]
	const Scalar * __restrict__ p_t,           // [size_o,ndim,size_i]
	Scalar       * __restrict__ qnew_t               // [size_o,ndim,size_i]
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

	const Scalar damping = damping_t[n_oi + n_i];
	
	Scalar qold[ndim];
	Scalar p[ndim];
	nstart = n_oi*ndim+n_i;
	for(int i=0; i<ndim; ++i){qold[i] = qold_t[nstart + size_i * i];}
	for(int i=0; i<ndim; ++i){p[i]    = p_t[   nstart + size_i * i];}

	// Update
	Scalar qnew[ndim];
	mul_kv(1-damping*dt,qold,qnew);

	Scalar mp[ndim];
	dot_mv(metric,p,mp);

	madd_kvV(dt,mp,qnew);

	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){qnew_t[nstart + size_i * i] = qnew[i];}
}

}
