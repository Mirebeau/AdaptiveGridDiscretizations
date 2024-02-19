// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the linear elastic wave equation operators, in the case of a fully 
generic hooke tensor. The tensor must be decomposed for finite differences using, e.g.,
Voronoi's first reduction.
*/

#include "static_assert.h"

/* // The following must be defined externally (example)
typedef float Scalar; 
#define fourth_order_macro false
#define ndim_macro 2
#define isotropic_metric_macro false // true for a metric proportional to the identity
*/
typedef int OffsetPack;
typedef int Int;

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
__constant__ int shape_o[ndim];
__constant__ int size_o;
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
const int badIndex = -(1<<30);

/** Bypass computations involving null weights or offsets. Should speed up computations 
involving isotropic hooke tensors in particular.*/
#if bypass_zeros_macro
#define BYPASS_ZEROS(...) __VA_ARGS__
#else 
#define BYPASS_ZEROS(...) 
#endif

#if fourth_order_macro
#define FOURTH_ORDER(...) __VA_ARGS__
#else
#define FOURTH_ORDER(...) 
#endif

#if isotropic_metric_macro
#define ISOTROPIC_METRIC(a,b) a
#else
#define ISOTROPIC_METRIC(a,b) b
#endif

//	const Scalar normalization = (idx*idx)/(4 FOURTH_ORDER(*12));
__constant__ Scalar DqH_mult; //dt/(c*dx**2) where c = 4 if order==2, and c=48 if order==4
__constant__ Scalar DpH_mult; //dt


/// Multiplication by the metric (scalar of symmetric matrix) 
void mul_metricv(const Scalar m[metricdim], const Scalar p[ndim], 
	Scalar mp[__restrict__ ndim]){
	ISOTROPIC_METRIC( mul_kv(m[0],p,mp) , dot_mv(m,p,mp) );
}

// ------------- Offset manipulation -------------

namespace Offset {

const int nbit = ndim<=2 ? 10 : 5; // Number of bits for each offset
const int mask = (1<<nbit)-1;
const int zero = 1<<(nbit-1);

/// Unpack the offsets of Voronoi's first reduction
void expand(OffsetPack pack, Int exp[symdim]){
	for(int i=0; i<symdim; ++i){exp[i] = ((pack >> (i*nbit)) & mask) - zero;}
}


bool is_zero(const Int e[ndim]){
	for(int i=0; i<ndim; ++i){if(e[i]!=0) return false;}
	return true;
}

} // Namespace offset

// -------------------- Vector field component access -------------------

/** Return the index a given component, at a given position, of a vector field
Vector field array shape : (*shape_o,ndim,*shape_i)
EXPECTS : Grid::InRange_per(x_t,shape_tot)
*/
Int component_index(const Int comp, const Int x_t[__restrict__ ndim]){
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
	return nstart + size_i * comp;
}

/// Number of neighbors in the divergence form finite difference scheme stencil (single term)
const Int nneigh_comp = 3 FOURTH_ORDER(+2); // Per component (dimension)
const Int nneigh = ndim*nneigh_comp;    // Total

/** The finite differences scheme associated with a single term of Voronoi's decomposition
 * of the Hooke tensor. 
 * Either second or  fourth order, and involved in the evaluation of DqH.*/
void DqH_scheme(const Scalar q[__restrict__ nneigh], Scalar dq[__restrict__ nneigh]){

// __TODO__ case where gcd of a column is not one.
// __TODO__ First order term

// See notebook QuadraticFormTerm.nb
#if fourth_order_macro
#if ndim_macro==1
	dq[0]=36*q[0]-24*q[1]-24*q[2]+6*q[3]+6*q[4];
	dq[1]=-24*q[0]+40*q[1]-8*q[2]-8*q[3];
	dq[2]=-24*q[0]-8*q[1]+40*q[2]-8*q[4];
	dq[3]=6*q[0]-8*q[1]+2*q[3];
	dq[4]=6*q[0]-8*q[2]+2*q[4];
#elif ndim_macro==2
	dq[0]=36*q[0]-24*q[1]-24*q[2]+6*q[3]+6*q[4];
	dq[1]=-24*q[0]+40*q[1]-8*q[2]-8*q[3]+24*q[6]-24*q[7]-4*q[8]+4*q[9];
	dq[2]=-24*q[0]-8*q[1]+40*q[2]-8*q[4]-24*q[6]+24*q[7]+4*q[8]-4*q[9];
	dq[3]=6*q[0]-8*q[1]+2*q[3]-4*q[6]+4*q[7]+q[8]-q[9];
	dq[4]=6*q[0]-8*q[2]+2*q[4]+4*q[6]-4*q[7]-q[8]+q[9];
	dq[5]=36*q[5]-24*q[6]-24*q[7]+6*q[8]+6*q[9];
	dq[6]=24*q[1]-24*q[2]-4*q[3]+4*q[4]-24*q[5]+40*q[6]-8*q[7]-8*q[8];
	dq[7]=-24*q[1]+24*q[2]+4*q[3]-4*q[4]-24*q[5]-8*q[6]+40*q[7]-8*q[9];
	dq[8]=-4*q[1]+4*q[2]+q[3]-q[4]+6*q[5]-8*q[6]+2*q[8];
	dq[9]=4*q[1]-4*q[2]-q[3]+q[4]+6*q[5]-8*q[7]+2*q[9];
#else // dimension 3
	dq[0]=36*q[0]-24*q[1]-24*q[2]+6*q[3]+6*q[4];
	dq[1]=-24*q[0]+40*q[1]-8*q[2]-8*q[3]+24*q[6]-24*q[7]-4*q[8]+4*q[9]+24*q[11]-24*q[12]-4*q[13]+4*q[14];
	dq[2]=-24*q[0]-8*q[1]+40*q[2]-8*q[4]-24*q[6]+24*q[7]+4*q[8]-4*q[9]-24*q[11]+24*q[12]+4*q[13]-4*q[14];
	dq[3]=6*q[0]-8*q[1]+2*q[3]-4*q[6]+4*q[7]+q[8]-q[9]-4*q[11]+4*q[12]+q[13]-q[14];
	dq[4]=6*q[0]-8*q[2]+2*q[4]+4*q[6]-4*q[7]-q[8]+q[9]+4*q[11]-4*q[12]-q[13]+q[14];
	dq[5]=36*q[5]-24*q[6]-24*q[7]+6*q[8]+6*q[9];
	dq[6]=24*q[1]-24*q[2]-4*q[3]+4*q[4]-24*q[5]+40*q[6]-8*q[7]-8*q[8]+24*q[11]-24*q[12]-4*q[13]+4*q[14];
	dq[7]=-24*q[1]+24*q[2]+4*q[3]-4*q[4]-24*q[5]-8*q[6]+40*q[7]-8*q[9]-24*q[11]+24*q[12]+4*q[13]-4*q[14];
	dq[8]=-4*q[1]+4*q[2]+q[3]-q[4]+6*q[5]-8*q[6]+2*q[8]-4*q[11]+4*q[12]+q[13]-q[14];
	dq[9]=4*q[1]-4*q[2]-q[3]+q[4]+6*q[5]-8*q[7]+2*q[9]+4*q[11]-4*q[12]-q[13]+q[14];
	dq[10]=36*q[10]-24*q[11]-24*q[12]+6*q[13]+6*q[14];
	dq[11]=24*q[1]-24*q[2]-4*q[3]+4*q[4]+24*q[6]-24*q[7]-4*q[8]+4*q[9]-24*q[10]+40*q[11]-8*q[12]-8*q[13];
	dq[12]=-24*q[1]+24*q[2]+4*q[3]-4*q[4]-24*q[6]+24*q[7]+4*q[8]-4*q[9]-24*q[10]-8*q[11]+40*q[12]-8*q[14];
	dq[13]=-4*q[1]+4*q[2]+q[3]-q[4]-4*q[6]+4*q[7]+q[8]-q[9]+6*q[10]-8*q[11]+2*q[13];
	dq[14]=4*q[1]-4*q[2]-q[3]+q[4]+4*q[6]-4*q[7]-q[8]+q[9]+6*q[10]-8*q[12]+2*q[14];
#endif // by dimension
#else // second order scheme

/* Matrices corresponding to the quadratic term.
{{4,-2,-2},
{-2,2,0},
{-2,0,2}}

{{4,-2,-2,0,0,0},
{-2,2,0,0,1,-1},
{-2,0,2,0,-1,1},
{0,0,0,4,-2,-2},
{0,1,-1,-2,2,0},
{0,-1,1,-2,0,2}}

{{4,-2,-2,0,0,0,0,0,0},
{-2,2,0,0,1,-1,0,1,-1},
{-2,0,2,0,-1,1,0,-1,1},
{0,0,0,4,-2,-2,0,0,0},
{0,1,-1,-2,2,0,0,1,-1},
{0,-1,1,-2,0,2,0,-1,1},
{0,0,0,0,0,0,4,-2,-2},
{0,1,-1,0,1,-1,-2,2,0},
{0,-1,1,0,-1,1,-2,0,2}}
*/

#if ndim_macro==1
	dq[0]=4*q[0]-2*q[1]-2*q[2];
	dq[1]=-2*q[0]+2*q[1];
	dq[2]=-2*q[0]+2*q[2];
#elif ndim_macro==2
	dq[0]=4*q[0]-2*q[1]-2*q[2];
	dq[1]=-2*q[0]+2*q[1]+q[4]-q[5];
	dq[2]=-2*q[0]+2*q[2]-q[4]+q[5];
	dq[3]=4*q[3]-2*q[4]-2*q[5];
	dq[4]=q[1]-q[2]-2*q[3]+2*q[4];
	dq[5]=-q[1]+q[2]-2*q[3]+2*q[5];
#else // dimension 3
	dq[0]=4*q[0]-2*q[1]-2*q[2];
	dq[1]=-2*q[0]+2*q[1]+q[4]-q[5]+q[7]-q[8];
	dq[2]=-2*q[0]+2*q[2]-q[4]+q[5]-q[7]+q[8];
	dq[3]=4*q[3]-2*q[4]-2*q[5];
	dq[4]=q[1]-q[2]-2*q[3]+2*q[4]+q[7]-q[8];
	dq[5]=-q[1]+q[2]-2*q[3]+2*q[5]-q[7]+q[8];
	dq[6]=4*q[6]-2*q[7]-2*q[8];
	dq[7]=q[1]-q[2]+q[4]-q[5]-2*q[6]+2*q[7];
	dq[8]=-q[1]+q[2]-q[4]+q[5]-2*q[6]+2*q[8];
#endif // by dimension
#endif // fourth/second order scheme

	for(int i=0; i<nneigh; ++i) {dq[i]*=DqH_mult;}
} // div scheme

/** The computation of DpH does not involve any finite differences, but 
for symmetry with DqH_scheme we define a similarly named function.*/
void DpH_scheme(const Scalar p[__restrict__ ndim], Scalar dp[__restrict__ ndim]){
	for(int i=0; i<ndim; ++i) dp[i] = p[i]*DpH_mult;}

// -------- Main functions -----------

extern "C" {

/** Differentiate the Elastic potential energy.
 * weights, offsets : Voronoi decomposition of the Hooke tensor.
 * q : position variable
 * wq : desired derivative
 * q_ad,w_ad,wq_ad : first order autodiff, forward or reverse.
 */
__global__ void DqH(
	const Scalar * __restrict__ weights_t,     // [size_o,decompdim,size_i]
	const OffsetPack * __restrict__ offsets_t, // [size_o,decompdim,size_i]
	const Scalar * __restrict__ q_t,           // [size_o,ndim,size_i]
AD( // Autodiff variables
FWD(const)Scalar * __restrict__ weights_ad_t,  // [size_o,size_ad,metricdim,size_i]
	const Scalar * __restrict__ q_ad_t,        // [size_o,size_ad,ndim,size_i]
	      Scalar * __restrict__ wq_ad_t,)      // [size_o,size_ad,ndim,size_i]
	      Scalar * __restrict__ wq_t           // [size_o,ndim,size_i]
	){
	// Compute position
	Int x_o[ndim], x_i[ndim];
	x_o[0] = blockIdx.x; x_i[0] = threadIdx.x; 
	#if ndim_macro>=2
	x_o[1] = blockIdx.y; x_i[1] = threadIdx.y;
	#endif 
	#if ndim_macro==3
	x_o[2] = blockIdx.z; x_i[2] = threadIdx.z; 
	#endif

	Int x_t[ndim]; 
	for(int i=0; i<ndim; ++i){x_t[i] = x_i[i]+x_o[i]*shape_i[i];}
	if(!Grid::InRange(x_t,shape_tot)) return; // Do not use InRange_per here !

	const int n_o = Grid::Index(x_o,shape_o);
	const int n_i = Grid::Index(x_i,shape_i);
	const int n_oi = n_o*size_i; 

	Scalar q[ndim]; // Constant after set
	const Int nstart_q = n_oi*ndim + n_i;
	for(int i=0; i<ndim; ++i){q[i] = q_t[nstart_q + size_i * i];}

	Scalar wq[ndim]; // The local part of the computed update
	zero_V(wq);
	
AD( Scalar q_ad_all[size_ad][ndim];
	const Int nstart_q_ad = n_oi*(ndim*size_ad) + n_i;
	for(int ad=0; ad<size_ad; ++ad){
		for(int i=0; i<ndim; ++i){q_ad_all[ad][i] 
			= q_ad_t[nstart_q_ad+(size_i*ndim)*ad+size_i*i];}} // for i // for ad

	Scalar wq_ad_all[size_ad][ndim];
	for(int ad=0; ad<size_ad; ++ad) {zero_V(wq_ad_all[ad]);}
	) // AD

	const Int nstart_w = n_oi*decompdim + n_i;

	for(int decomp=0; decomp<decompdim; ++decomp){
		// Load one weight and offset. 
		const Scalar weight = weights_t[nstart_w + size_i*decomp];
		BYPASS_ZEROS(if(weight==0) continue;)
		Int offset[symdim]; 
		Offset::expand(offsets_t[nstart_w + size_i*decomp], offset);

		// Expand the offset as a symmetric matrix
		Int moffset[ndim][ndim];
		for(int i=0; i<ndim; ++i){
			for(int j=0; j<ndim; ++j){
				moffset[i][j] = coef_m(offset,i,j);
			}
		}
		bool zoffset[ndim]; // Check if some columns are zero
		for(int i=0; i<ndim; ++i) {zoffset[i]=Offset::is_zero(moffset[i]);}

	// Load the values of the displacement q involved in the finite difference scheme
	Int ind_neigh[nneigh]; // Indices of neighbors of q
	Scalar q_neigh[nneigh];// Corresponding values of q

	// Load the values used in the scheme, and the corresponding indices
	for(int comp=0, r=0; comp<ndim; ++comp){ // Iterate over omponents of the vector field
		if(zoffset[comp]){ // Bypass this offset, which is zero, hence does not contribute
			for(int i=0; i<nneigh_comp; ++i, ++r){ind_neigh[r]=badIndex; q_neigh[r]=0;}
			continue;
		}
		const Int * offset = moffset[comp]; 
		ind_neigh[r] = -comp-1; // No index in global array, we use the local cache
		q_neigh[r] = q[comp]; 
		++r;
		for(int side=0; side<=1; ++side, ++r){
			const int eps = 2*side-1;
			Int y_t[ndim];
			madd_kvv(eps,offset,x_t,y_t);
			if(Grid::InRange_per(y_t,shape_tot)){
				// We do not use a __shared__ cache for q_t since likely too few hits.
				const Int index = component_index(comp,y_t);
				ind_neigh[r] = index;
				q_neigh[r] = q_t[index];
			} else {
				ind_neigh[r] = badIndex;
				q_neigh[r] = 0.; // Dirichlet boundary conditions
			}

		#if fourth_order_macro 
			madd_kvV(eps,offset,y_t);
			if(Grid::InRange_per(y_t,shape_tot)){
				const Int index = component_index(comp,y_t);
				ind_neigh[r+2] = index;
				q_neigh[r+2] = q_t[index];
			} else {
				ind_neigh[r+2] = badIndex;
				q_neigh[r+2] = 0; // TODO : deactivate fourth order in these cases ? 
			}
		#endif
		}
		FOURTH_ORDER(r+=2;)
	} // for comp

	HFM_DEBUG(for(int i=0; i<nneigh; ++i) {assert(q_neigh[i]==q_neigh[i]); 
	assert( (ind_neigh[i]<0) || (0<=ind_neigh[i] && ind_neigh[i]<size_i*size_o*ndim));})

	Scalar dq_neigh[nneigh]; // Incremental update of wq associated to this term
	DqH_scheme(q_neigh,dq_neigh);

	// Add the increments using atomic operations
	for(int i=0,r=0; i<ndim; ++i){
		if(zoffset[i]) {r+=nneigh_comp; continue;}
		wq[i] += weight*dq_neigh[r]; ++r;
		for(int j=1; j<nneigh_comp; ++j, ++r) {
			if(ind_neigh[r]!=badIndex) atomicAdd(wq_t+ind_neigh[r],weight*dq_neigh[r]);}
	} // For i, export values

AD(	Int ind_ad_neigh[nneigh];  // Neighbor indices need to be adjusted if fwd_ad>0
	for(int i=0; i<nneigh; ++i){
		const Int ind = ind_neigh[i];
		ind_ad_neigh[i] = (size_ad==1 || ind < 0) ? ind : 
		(ind / (size_i*ndim))*(size_i*ndim*size_ad) + (ind % (size_i*ndim));}
	const Int nstart_w_ad = n_oi*(decompdim*size_ad) + size_i*decomp + n_i;

	for(int ad=0; ad<size_ad; ++ad){ // Do each ad component
	const Scalar * q_ad  = q_ad_all[ad]; // q_ad[ndim]

	Scalar q_ad_neigh[nneigh]; // Neighbor values of q_ad
	for(int i=0; i<nneigh; ++i) {
		const int ind = ind_ad_neigh[i];
		q_ad_neigh[i] = ind==badIndex ? 0 : ind<0 ? q_ad[-ind-1] : 
		q_ad_t[ind+(size_i*ndim)*ad];
	} // for i ineigh

	Scalar dq_ad_neigh[nneigh];
	DqH_scheme(q_ad_neigh,dq_ad_neigh);

FWD(const Scalar weight_ad = weights_ad_t[nstart_w_ad+(size_i*decompdim)*ad];) // FWD 
REV(Scalar weight_ad = 0;) // REV
	Scalar * wq_ad = wq_ad_all[ad]; //wq_ad[ndim] 
	for(int i=0,r=0; i<ndim; ++i){
		if(zoffset[i]) {r+=nneigh_comp; continue;}
		wq_ad[i] += weight*dq_ad_neigh[r] FWD(+ weight_ad*dq_neigh[r]); 
		REV(weight_ad += dq_ad_neigh[r]*q_neigh[r];)
//		REV(weight_ad += dq_neigh[r]*q_ad_neigh[r];) // Equivalent, by symmetry
		++r;
		for(int j=1; j<nneigh_comp; ++j, ++r) {
			if(ind_neigh[r]!=badIndex) {
				atomicAdd(wq_ad_t+ind_ad_neigh[r] + (size_i*ndim)*ad,
				weight*dq_ad_neigh[r] FWD(+ weight_ad*dq_neigh[r]));
				REV(weight_ad += dq_ad_neigh[r]*q_neigh[r];)
//				REV(weight_ad += dq_neigh[r]*q_ad_neigh[r];) // Equivalent, by scheme symmetry
			}
		} // for j
	} // For i, export values
REV(weights_ad_t[nstart_w_ad+(size_i*decompdim)*ad] += weight_ad;) //REV
}) // for ad // AD

	} // For decomp

	for(int i=0; i<ndim; ++i){atomicAdd(wq_t + (nstart_q+size_i*i), wq[i]);}
AD(	for(int ad=0; ad<size_ad; ++ad){ for(int i=0; i<ndim; ++i){
		atomicAdd(wq_ad_t + (nstart_q_ad+(size_i*ndim)*ad+size_i*i), wq_ad_all[ad][i]);}})
}

/** Differentiate the Elastic kinetic energy.
 * metric : defines the kinetic energy.
 * p : momentum variable
 * mp : desired derivative
 * p_ad,metric_ad,mp_ad : first order autodiff, forward or reverse.
 * 
 * Since this is a purely local and quite trivial operation, 
 * the need for a kernel is not obvious.
 */
__global__ void DpH(
	const Scalar * __restrict__ metric_t,    // [size_o,metricdim,size_i]
	const Scalar * __restrict__ p_t,         // [size_o,ndim,size_i]
AD( // Autodiff variables, forward or reverse
FWD(const)Scalar * __restrict__ metric_ad_t, // [size_o,size_ad,metricdim,size_i]
	const Scalar * __restrict__ p_ad_t,      // [size_o,size_ad,ndim,size_i]
	      Scalar * __restrict__ mp_ad_t,)    // [size_o,size_ad,ndim,size_i]
	      Scalar * __restrict__ mp_t         // [size_o,ndim,size_i] 
	){

	// Compute position
	const Int n_o = blockIdx.x;
	const Int n_i = threadIdx.x;
	const Int n_oi = n_o*size_i; 
	
	// Load data
	Scalar m[metricdim];
	const Int nstart_m = n_oi*metricdim + n_i;
	for(int i=0; i<metricdim; ++i){m[i] = metric_t[nstart_m + size_i * i];}
	
	Scalar p[ndim];
	const Int nstart_p = n_oi*ndim+n_i;
	for(int i=0; i<ndim; ++i){p[i] = p_t[nstart_p + size_i * i];}
	Scalar dp[ndim];
	DpH_scheme(p,dp);

	// Make product and export
	Scalar dmp[ndim];
	mul_metricv(m,dp,dmp);
	for(int i=0; i<ndim; ++i){mp_t[nstart_p + size_i * i] += dmp[i];}

AD( for(int ad=0; ad<size_ad; ++ad){ // Autodiff over all channels
	// Load data
	Scalar p_ad[ndim];
	const Int nstart_p_ad = n_oi*(ndim*size_ad) + (size_i*ndim)*ad + n_i;
	for(int i=0; i<ndim; ++i){p_ad[i] = p_ad_t[nstart_p_ad + size_i * i];}

	Scalar dp_ad[ndim];
	DpH_scheme(p_ad,dp_ad);

	mul_metricv(m,dp_ad,dmp); // Metric times derivative of momentum
	
	const Int nstart_m_ad = n_oi*(metricdim*size_ad) + (size_i*metricdim)*ad + n_i;
FWD(Scalar m_ad[metricdim];
	for(int i=0; i<metricdim; ++i){m_ad[i] = metric_ad_t[nstart_m_ad + size_i * i];}
	Scalar dmp1[ndim];
	mul_metricv(m_ad,dp,dmp1); // Derivative of metric times momentum
	) // FWD
	
	for(int i=0; i<ndim; ++i){mp_ad_t[nstart_p_ad + size_i * i] += dmp[i] FWD(+ dmp1[i]);}

REV(ISOTROPIC_METRIC(
	Scalar dm_ad=0.;
	for(int i=0; i<ndim; ++i) dm_ad+=dp_ad[i]*p[i];
	metric_ad_t[nstart_m_ad] += dm_ad;
	, // ANISOTROPIC_METRIC
	for(int i=0,k=0; i<ndim; ++i)
		for(int j=0; j<=i; ++j,++k){
			// Caution : the 0.5 is takes into account the Frobenius scalar product
			metric_ad_t[nstart_m_ad + size_i*k] 
			+= (i==j) ? dp_ad[i]*p[i] : 0.5*(dp_ad[i]*p[j] + dp_ad[j]*p[i]);} // for j
	) )// ISOTROPIC_METRIC //  REV
	}) // for ad // AD

} // DpH

} // Extern "C"