// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the linear elastic wave equation operators, in the case of a fully 
generic hooke tensor. The tensor must be decomposed for finite differences using, e.g.,
Voronoi's first reduction.
*/

#include "static_assert.h"

/* // The following must be defined externally
typedef float Scalar; 
#define fourth_order_macro false
#define ndim_macro 2
#define isotropic_metric_macro false // true for a metric proportional to the identity
#define vertical_macro 0 // 1 : Hexagonal, 2 : Tetragonal, 3 : Orthorombic
*/
typedef int OffsetPack;
typedef int Int;

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
// const bool periodic_axes[ndim] = {false,false,true}; // must be defined externally
#else
#define PERIODIC(...) 
#endif

#if vertical_macro
#define VERTICAL(...) __VA_ARGS__
#else
#define VERTICAL(...) 
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
const int metricdim = isotropic_metric_macro ? 1 : symdim;

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

#if fourth_order_macro
#define FOURTH_ORDER(...) __VA_ARGS__
#else
#define FOURTH_ORDER(...) 
#endif

// ------------- Offset manipulation -------------

namespace Offset {

const int nbit = ndim==2 ? 10 : 5; // Number of bits for each offset
const int mask = (1<<nbit)-1;
const int zero = 1<<(nbit-1);

/// Unpack the offsets of Voronoi's first reduction
void offset_expand(OffsetPack pack, Int exp[symdim]){
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

} // Namespace offset

// -------------------- Vector field component access -------------------

/// Return a given component, at a given position, of a vector field
// EXPECTS : Grid::InRange_per(x_t,shape_tot)
Scalar component(const Int comp, const Int x_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t){
	HFM_DEBUG(assert(0<=comp && comp<ndim);)

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
}

// ------- Finite difference operators, second order and fourth order accurate. -------

#if fourth_order_macro
bool 
#else
void
#endif
components(const Int comp,
	const Int offset[__restrict__ ndim], const Int x_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t,
	Scalar values[__restrict__ 2] FOURTH_ORDER(, Scalar values2[__restrict__ 2])
	){
	FOURTH_ORDER(bool fourth_active=true;)
	Int y_t[ndim];
	for(int side=0; side<=1; ++side){
		const int eps = 2*side-1;
		madd_kvv(eps,offset,x_t,y_t);
		if(Grid::InRange_per(y_t,shape_tot)){values[side] = component(comp,y_t,q_t);}

		#if fourth_order_macro
		madd_kvV(eps,offset,y_t);
		if(Grid::InRange_per(y_t,shape_tot)){values2[side]=component(comp,y_t,q_t);}
		else{fourth_active=false;}
		#endif
	}
	FOURTH_ORDER(return fourth_active;)
}

Scalar diff_second(const Int comp, 
	const Int offset[__restrict__ ndim], const Int x_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t, const Scalar q[__restrict__ ndim]){
	Scalar values[2]={0.,0.}; // Null Dirichlet boundary conditions
	FOURTH_ORDER(Scalar values2[2];)
	FOURTH_ORDER(const bool fourth_active =)
	components(comp,offset,x_t,q_t,values FOURTH_ORDER(,values2));
	const Scalar result = FOURTH_ORDER(fourth_active ? 
		-(values2[1]+values2[0])/12. +(values[1]+values[0])*(4/3.) -q[comp]*(15/6.) : )
		values[1]+values[0]-2.*q[comp];
	return result*(idx*idx);
}

Scalar diff_cross(const Int comp, 
	const Int offseta[__restrict__ ndim], const Int offsetb[__restrict__ ndim], 
	const Int x_t[__restrict__ ndim], const Scalar * __restrict__ q_t){
	Int offsetp[2],offsetm[2]; 
	add_vv(offseta,offsetb,offsetp); 
	sub_vv(offseta,offsetb,offsetm);

	Scalar valuesp[2]={0.,0.}, valuesm[2]={0.,0.}; // Null Dirichlet boundary conditions
	FOURTH_ORDER(Scalar valuesp2[2],valuesm2[2];)
	FOURTH_ORDER(bool fourth_active = )
	components(comp,offsetp,x_t,q_t,valuesp FOURTH_ORDER(,valuesp2));
	FOURTH_ORDER(fourth_active = fourth_active && )
	components(comp,offsetm,x_t,q_t,valuesm FOURTH_ORDER(,valuesm2));

	const Scalar result = FOURTH_ORDER(fourth_active ? 
		-(valuesp2[1]+valuesp2[0])/12. +(valuesp[1]+valuesp[0])*(4/3.)
		+(valuesm2[1]+valuesm2[0])/12. -(valuesm[1]+valuesm[0])*(4/3.) :)	
		(valuesp[1]+valuesp[0])-(valuesm[1]+valuesm[0]);
	return result * ((idx*idx)/4.);

}

Scalar diff_centered(const Int comp, 
	const Int offset[__restrict__ ndim], const Int x_t[__restrict__ ndim], 
	const Scalar * __restrict__ q_t){
	Scalar values[2]={0.,0.}; // Null Dirichlet boundary conditions
	FOURTH_ORDER(Scalar values2[2];)
	FOURTH_ORDER(const bool fourth_active =)
	components(comp,offset,x_t,q_t,values FOURTH_ORDER(,values2));
	const Scalar result = FOURTH_ORDER(fourth_active ? 
		-(values2[1]-values2[0])/6. +(values[1]-values[0])*(4/3.) : )
		values[1]-values[0];
	return result*(idx/2);
}

// ----------- Vertical geometry ------------

#if vertical_macro
const int vertdim = 
	ndim_macro==2     ? 4 : // (2,1) block diag
	vertical_macro==1 ? 5 : // hexagonal
	vertical_macro==2 ? 6 : // tetragonal
	vertical_macro==3 ? 9;  // Orthorombic 

/// Produces the block diagonal hooke tensor from the raw coefficients
void vertical_dispatch(const Scalar c[__restrict__ vertdim], 
	Scalar b[__restrict__ symdim], Scalar d[__restrict__ symdim-ndim]){
	#if ndim_macro==2 || vertical_macro==3 // Orthorombic
	for(int i=0; i<symdim; ++i) {b[i]=c[i];}
	for(int i=0; i<symdim-ndim; ++i){d[i]=c[symdim+i];}
	#else
	b[0]=c[0]; b[1]=c[1]; b[2]=c[0]; b[3]=c[2]; b[4]=c[2]; b[5]=c[3];
	d[0]=c[4]; d[1]=c[5];
	d[2] = vertical_macro==1 ? (c[0]-c[1])/2. : c[vertdim-1];
	#endif
}

const int vertdecompdim = symdim + (symdim-ndim);
STATIC_ASSERT(vertdecompdim<decompdim, inconsistent_scheme_structure)

void vertical_decomp(
	const Scalar b[__restrict__ symdim], // ndim x ndim block
	const Scalar d[__restrict__ symdim-ndim], // remaining diagonal
	Scalar       w[__restrict__ decompdim], // weights
	OffsetPack   o[__restrict__ decompdim]){// offsets
	// Note : it is a bit silly to compress the offsets only to uncompress them afterwards.
	// The vertical structure means that fewer weights and offsets are needed
	for(int i=vertdecompdim; i<decompdim; ++i){w[i]=0;} // Will be bypassed

	// Decomposition associated to the block
	Int offsets[symdim][ndim]; // Weights and offsets for the block
	decomp_m(b,w,offsets); // Fills the symdim first weights
	for(int i=0; i<symdim; ++i){
		#if ndim_macro==2
		o[i] = 
		  ((offsets[i][0]+Offset::zero) << (0*Offset::nbits)) 
		+ ((offsets[i][1]+Offset::zero) << (2*Offset::nbits));
		#else
		o[i] = 
		  ((offsets[i][0]+Offset::zero) << (0*Offset::nbits)) 
		+ ((offsets[i][1]+Offset::zero) << (2*Offset::nbits))
		+ ((offsets[i][2]+Offset::zero) << (5*Offset::nbits));
		#endif
	}

	// Decomposition associated to the remaining diagonal coefficients
	for(int i=0; i<symdim-ndim; ++i){w[symdim+i] = d[i];}
	#if ndim_macro==2
	o[symdim] = (1+Offset::zero) << (1*Offset::nbits);
	#else
	o[symdim] = (1+Offset::zero) << (4*Offset::nbits);
	o[symdim] = (1+Offset::zero) << (3*Offset::nbits);
	o[symdim] = (1+Offset::zero) << (1*Offset::nbits);
	#endif
}
#endif // vertical_macro

// -------- Main functions -----------

extern "C" {

__global__ void AdvanceP(
	VERTICAL(// Simplified vertical geometry
	const Scalar * __restrict__ vertical_t,    // [size_o,vertdim,size_i]
	const Int * __restrict__ geomindex_t,)     // [size_o,size_i]
	// Full Hooke tensor geometry
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
	const int n_oi = n_o*size_i; 
	int nstart;// Mutable, used for array data start

	// Weights and offsets are needed one at a time in the loop, 
	// but we load them all here since they are close in memory.
	Scalar weights[decompdim];
	OffsetPack offsets[decompdim];
	Scalar firstorder[firstdim];

	#if vertical_macro // Switch to vertical geometry when possible (Memory optimization)
	Scalar vert_coefs[vertdim];
	nstart = n_oi*vertdim;
	for(int i=0; i<decompdim; ++i){vert_coefs[i] = vertical_t[nstart + size_i * i];
	Scalar vert_block[symdim]; // Dense block at beginning of Hooke tensor
	Scalar vert_diag[symdim-ndim]; // Diagonal terms

	const Int geomindex = geomindex_t[n_oi+n_i];
	if(geomindex>=0){ // Full anisotropic geometry. Precomputed decomposition
		for(int i=0; i<decompdim; ++i){weights[i] = weights_t[geomindex*decompdim+i];}
		for(int i=0; i<decompdim; ++i){offsets[i] = offsets_t[geomindex*decompdim+i];}
		for(int i=0; i<firstdim; ++i){firstorder[i] = firstorder_t[geomindex*firstdim+i];}
	} else { // Use vertical geometry
		vertical_dispatch(vert_coefs,vert_block,vert_diag);
		vertical_decomp(vert_block,vert_diag,weights,offsets);
		__TODO__ // First order (Compare with neighbors to get divergence, etc.
	}
	#else // Load full geometry everywhere
	nstart = n_oi*decompdim + n_i;
	for(int i=0; i<decompdim; ++i){weights[i] = weights_t[nstart + size_i * i];}
	for(int i=0; i<decompdim; ++i){offsets[i] = offsets_t[nstart + size_i * i];}
	nstart = n_oi*firstdim + n_i;
	for(int i=0; i<firstdim; ++i){firstorder[i] = firstorder_t[nstart + size_i * i];}
	#endif

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
		Offset::expand(offsets[decomp],offset);

		Int moffset[ndim][ndim];
		for(int i=0; i<ndim; ++i){
			for(int j=0; j<ndim; ++j){
				moffset[i][j] = coef_m(offset,i,j);
			}
		}

		// Contribution from the second order operator dvi = m_ij m_kl D_jk v_l
		for(int i=0; i<ndim; ++i){
			const Int * e = moffset[i]; // e[ndim]
			BYPASS_ZEROS(if(Offset::is_zero(e)) continue;)
			for(int l=0; l<ndim; ++l){
				const Int * f = moffset[l];
				BYPASS_ZEROS(if(Offset::is_zero(f)) continue;)
				// Evaluate the cross derivative of v_l w.r.t moffsets[i] and moffsets[l]
				Scalar cross;

				if(i==l COMPACT_SCHEME(||Offset::is_same(e,f)||Offset::is_opp(e,f))){
					cross = diff_second(l,e,x_t,q_t,q);
					COMPACT_SCHEME(if(Offset::is_opp(e,f)) cross*=-1;)
				} else {
					cross = diff_cross(l,e,f,x_t,q_t);
				}
				pnew[i]+=(dt*weight)*cross;
			}
		}

		// Reconstruction of the stress tensor m_ij m_kl D_k v_l
		Scalar diff1sum = 0;
		for(int l=0; l<ndim; ++l){
			const Int * e = moffset[l]; // e[ndim]
			BYPASS_ZEROS(if(is_zero(e)) continue;)
			diff1sum += diff_centered(l,e,x_t,q_t);
		}
		geom_symdim::madd_kvV(diff1sum*weight,offset,stress);
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
	Scalar metric[metricdim];
	nstart = n_oi*metricdim + n_i;
	for(int i=0; i<metricdim; ++i){metric[i] = metric_t[nstart + size_i * i];}

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
	#if isotropic_metric_macro
	mul_kv(metric[0],p,mp);
	#else
	dot_mv(metric,p,mp);
	#endif

	madd_kvV(dt,mp,qnew);

	nstart = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){qnew_t[nstart + size_i * i] = qnew[i];}
}

}
