#pragma once
// Copyright 2022 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements low-level GPU routines for the mincut problem.
*/

/** The following need to be defined in including file (example)
typedef int Int;
typedef float Scalar;
#define ndim_macro 3
#define rander_macro true
#define grad_macro 1 // gradient discretization
const int newton_maxiter = 7;
// Subgrid used for gradient computation
const int shape_i[ndim] = {4,4,4}; 
const int size_i = 64;
const int shape_e[ndim] = {5,5,5};
const int size_e = 125;
const int size_bd_e = size_e-size_i;
x_top_e[size_bd_e][ndim] = {...}; // Top and bottom points of shape_e in memory friendly order
x_bot_e[size_bd_e][ndim] = {...};
#define preproc_randers_macro false
*/

#if   ndim_macro == 1
#include "Geometry1.h"
#elif ndim_macro == 2
#include "Geometry2.h"
#elif ndim_macro == 3
#include "Geometry3.h"
#endif 
STATIC_ASSERT(1<=ndim_macro && ndim_macro<=3, Unsupported_dimension);
STATIC_ASSERT(size_bd_e==size_e-size_i && size_bd_e<size_i, Incorrect_size_bd_e);

#include "REDUCE_i.h"

#define gradb_macro 1 // gradient computed from bottom corner
#define gradc_macro 2 // centered gradient (accurate but unstable)
#define grad2_macro 3 // uses two gradients, computed from the bottom and the top corners
STATIC_ASSERT(grad_macro==gradb_macro || grad_macro==gradc_macro 
	|| grad_macro==grad2_macro, Unsupported_grad_macro);
STATIC_ASSERT(ndim_macro>1 || grad_macro!=grad2_macro, Unsupported_grad_macro);
#if grad_macro==grad2_macro 
#define GRAD2(...) __VA_ARGS__
const int graddim = 2*ndim;
#else
#define GRAD2(...) 
const int graddim = ndim;
#endif

// NEVER USE DEFAULT VALUES for __constants__ (ignored hen soure is "recompiled")
// We use a two-level grid. shape_tot_s = shape_tot_v + 1
__constant__ Int shape_o[ndim]; 
__constant__ Int size_io;
__constant__ Int shape_tot_v[ndim]; // Shape of domain for vector fields
__constant__ Int shape_tot_s[ndim]; // Shape of domain for scalar fcts

__constant__ Scalar tau_primal, tau_dual; // proximal time steps
__constant__ Scalar rho_overrelax;
__constant__ Int niter; // (Debug only) Iteration count
__constant__ Int preproc_randers; // (Randers only) Updates ground cost based on drift

#undef bilevel_grid_macro
#include "Grid.h"

// --------------- Projections on anisotropic (dual) unit balls and related ----------------

#define metric_type_iso 1
#define metric_type_iso_asym 2
#define metric_type_riemann 3
#define metric_type_riemann_asym 4
STATIC_ASSERT(1<=metric_type_macro && metric_type_macro<=4, Unsupported_metric_type);


// Dimension for the diagonal prox
const int pdim = (metric_type_macro == metric_type_iso_asym) ? 2 : ndim;

// Dimension for the quaternion data (d=3), or counterpart if d<3
const int qdim = (ndim_macro==1) ? 1: (ndim_macro==2) ? 2 : 4;
void copy_eigenvectors(const Scalar v0[qdim], Scalar v[__restrict__ ndim][ndim]){
#if ndim_macro==1 // 1D : there is only one possible eigenvector
	v[0][0]=1.; 
#elif ndim_macro==2 // 2D : importing only the first eigenvector, second one is perpendicular
	for(int i=0; i<ndim; ++i){v[0][i] = v0[i];} 
	perp_v(v[0],v[1]);
	trans_A(v);
#elif ndim_macro==3  // 3D : importing the eigenvector matrix as a unit quaternion
	Scalar q[qdim];
	for(int i=0; i<qdim; ++i){q[i] = v0[i];}
	rotation3_from_sphere3(q,v);
#endif
}

const int geomsize = 
(metric_type_macro==metric_type_iso) ? 1 : 
(metric_type_macro==metric_type_iso_asym) ? (1+ndim) :
(metric_type_macro==metric_type_riemann) ? (ndim+qdim) : 
(metric_type_macro==metric_type_riemann_asym) ? (ndim+qdim+ndim+qdim+ndim) : -1;

/** Projection onto the dual unit ball of an isotropic norm*/
void proj_iso(
	const Scalar eta[ndim], // Projected variable
	const Scalar a, // norm multiplier
	Scalar sol[__restrict__ ndim] // minimizer
	){
	const Scalar norm = norm_v(eta);
	const Scalar k = Scalar(1)/max(Scalar(1),norm/a); // Todo : precompute 1/a ? 
	mul_kv(k,eta,sol);
}
Scalar norm_iso(const Scalar eta[ndim],const Scalar a){return a*norm_v(eta);}

/** Projection onto the dual unit ball of a diagonal norm.*/
void proj_diagonal(
	const Scalar eta[pdim], // projected variable
	const Scalar lambda[pdim], // eigenvalues
	Scalar sol[__restrict__ pdim] // minimizer
	){ 
	// Since pdim!=ndim in general, we cannot use the Geometry_.h header here, e.g. norm2_v
	Scalar eta2[pdim];
	for(int i=0; i<pdim; ++i){eta2[i] = eta[i]*eta[i];}

	Scalar norm2=0.;
	for(int i=0; i<pdim; ++i){norm2 += eta2[i]/lambda[i];}

	// Case where eta already lies in the unit ball
	if(norm2<=1){for(int i=0; i<pdim; ++i) sol[i]=eta[i]; return;}

	// Compute beta via a Newton method, solving the equation f(β)=0 where
	//f(β) = sum(λi*coef**2/(λi+β)**2,axis=0)-1.
	// The system should behave well enough that no damping or 
	// fancy stopping criterion is needed

	// Initial guess is exact in isotropic case
	Scalar lambda_min = lambda[0];
	for(int i=1; i<pdim; ++i) lambda_min = min(lambda_min,lambda[i]);
	Scalar beta = lambda_min * (sqrt(norm2)-1);
	Scalar lc2[pdim];
	for(int i=0; i<pdim; ++i){lc2[i]=lambda[i]*eta2[i];}

	for(int newton_iter=0; newton_iter<newton_maxiter; ++newton_iter){
		Scalar val = -Scalar(1); // f(β)
		Scalar der = Scalar(0);  // f'(β)
		for(int i=0; i<pdim; ++i){
			const Scalar ilb = Scalar(1)/(lambda[i]+beta);
			const Scalar a = lc2[i]*ilb*ilb;
			val += a;
			der += a*ilb;
		}
		der *= -Scalar(2);
		beta -= val/der;
	}
	for(int i=0; i<pdim; ++i){sol[i] = eta[i] * lambda[i]/(beta+lambda[i]);}
}
Scalar norm2_diagonal(const Scalar eta[pdim],const Scalar lambda[pdim]){
	Scalar norm2=0;
	for(int i=0; i<pdim; ++i) norm2+=eta[i]*eta[i]*lambda[i];
	return norm2; 
}
Scalar norm_diagonal(const Scalar eta[pdim],const Scalar lambda[pdim]){
	return sqrt(norm2_diagonal(eta,lambda));}

/**
Projection onto the dual unit ball of an "asymmetric isotropic" norm.
sqrt(a^2 |x|^2+<w,x>_+^2)
*/
#if metric_type_macro==metric_type_iso_asym
void proj_iso_asym(
	const Scalar eta[ndim], // Projected variable
	const Scalar a, // norm multiplier
	const Scalar w[ndim], // anisotropy vector
	Scalar sol[__restrict__ ndim] // minimizer
	){
	const Scalar s = scal_vv(eta,w);
	const Scalar sgn_a = a>=0. ? Scalar(1) : -Scalar(1);

	// Case where eta is in the isotropic half space, or the norm is isotropic
	const Scalar norm2_w = norm2_v(w);
	if(s<=0 || norm2_w==0){proj_iso(eta,abs(a),sol); return;}

#if ndim_macro==1
	const Scalar b = sqrt(a*a+sgn_a*w[0]*w[0]);
	proj_iso(eta,b,sol);
#else
	STATIC_ASSERT(pdim==2,inconsistent_scheme_data);
	// We generate an orthonormal basis of the space containing w and eta,
	// and solve the projection in the transformed coordinates.
	const Scalar lambda[2] = {a*a+sgn_a*norm2_w, a*a}; // Eigenvalues in these coords

	const Scalar inorm_w = Scalar(1)/sqrt(norm2_w);
	Scalar w_unit[ndim]; mul_kv(inorm_w,w,w_unit); // First basis vector : w normalized
	const Scalar eta0 = s*inorm_w; // = <eta,w_unit>
	Scalar e_unit[ndim];  // Second basis vector
#if ndim_macro==2
	perp_v(w_unit,e_unit);
	const Scalar eta1 = scal_vv(eta,e_unit); 
#elif ndim_macro==3
	madd_kvv(-eta0,w_unit,eta,e_unit);//Gram-Schmidt. Alternatively, could use cross products
	const Scalar norm_e = norm_v(e_unit);
	if(norm_e>0){mul_kV(Scalar(1)/norm_e,e_unit);} 
	const Scalar eta1 = norm_e; 
#endif
	// Solve the proj in the transformed coordinates
	const Scalar eta_coef[2] = {eta0, eta1};
	Scalar sol_coef[2]; 
	proj_diagonal(eta_coef,lambda,sol_coef);
	mul_kv(  sol_coef[0],w_unit,sol);
	madd_kvV(sol_coef[1],e_unit,sol);
#endif
}
Scalar norm_iso_asym(const Scalar eta[ndim], const Scalar a, const Scalar w[ndim]){
	const Scalar norm2 = norm2_v(eta), scal = max(Scalar(0),scal_vv(eta,w)), 
	sgn_a = a>=0. ? Scalar(1) : -Scalar(1);
	return sqrt(a*a*norm2 + sgn_a*scal*scal);
}
#endif

/** 
Projection onto the dual unit ball.
*/
void proj_riemann(
	const Scalar eta[ndim], // Projected variable
	const Scalar lambda[ndim], // Eigenvalues
	const Scalar v[ndim][ndim], // Eigenvectors
	Scalar sol[__restrict__ ndim] // Minimizer
	){ 
	Scalar eta_coef[ndim], sol_coef[ndim];
	tdot_av(v,eta,eta_coef);
	proj_diagonal(eta_coef,lambda,sol_coef);
	dot_av(v,sol_coef,sol);
}
Scalar norm2_riemann(const Scalar eta[ndim],const Scalar lambda[ndim],const Scalar v[ndim][ndim]){
	Scalar eta_coef[ndim];
	tdot_av(v,eta,eta_coef);
	return norm2_diagonal(eta_coef,lambda);
}
Scalar norm_riemann(const Scalar eta[ndim],const Scalar lambda[ndim],const Scalar v[ndim][ndim]){
	return sqrt(norm2_riemann(eta,lambda,v));}

void proj_riemann_asym(
	const Scalar eta[ndim], // Projected variable
	const Scalar lambda0[ndim], // Eigen decomposition of first matrix 
	const Scalar v0[ndim][ndim], 
	const Scalar lambda1[ndim], // Eigen decomposition of second matrix 
	const Scalar v1[ndim][ndim], 
	const Scalar w[ndim], // Separating plane between the two ellipsoids
	Scalar sol[__restrict__ ndim] // Minimizer
){
	Scalar sol1[ndim];
	proj_riemann(eta,lambda0,v0,sol);
	proj_riemann(eta,lambda1,v1,sol1);
	if(scal_vv(sol,w)>0){copy_vV(sol1,sol);}
}
Scalar norm_riemann_asym(const Scalar eta[ndim], 
	const Scalar lambda0[ndim], const Scalar v0[ndim][ndim],const Scalar w[ndim]){
	const Scalar scal = max(Scalar(0),scal_vv(eta,w));
	return sqrt(norm2_riemann(eta,lambda0,v0)+scal*scal);
}

// -------------------------------------------------------------------------

extern "C" {

/** Generates a new primal point. */
__global__ void primal_step(
	Scalar * __restrict__ phi_t, // primal variable (input and output)
	Scalar * __restrict__ phi_ext_t, // extrapolated primal variable (output)
	const Scalar * __restrict__ eta_t, // dual variable 
	const Scalar * __restrict__ g_t, // ground cost
	Scalar * __restrict__ primal_value_o,  // primal objective value of the block
	Scalar * __restrict__ dual_value_o // dual objective value of the block
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

// Weight initially introduced to better match reflected boundary conditions
// However, this causes the norm of gradb and gradt to be doubled, and thus reduces 
// admissible time step, for little benefit
const Scalar weight = 1.; 
//for(int k=0; k<ndim; ++k){if(x_t[k]==0 || x_t[k]==shape_tot_s[k]-1) weight/=2.;}

// ---- Load the vector field ----
__shared__ Scalar eta_e[size_e][graddim];

{	// At the current point
	int x_e[ndim]; // position in shared array
	for(int k=0; k<ndim; ++k) {x_e[k] = x_i[k]+1;}
	const int n_e = Grid::Index(x_e,shape_e);
	if(Grid::InRange(x_t,shape_tot_v)){
		for(int k=0; k<graddim; ++k) {eta_e[n_e][k] = eta_t[n_t+size_io*k];}
	} else {for(int k=0; k<graddim; ++k) {eta_e[n_e][k]=0;}}
}
if(n_i<size_bd_e){ // At the chosen boundary point
	const int n_e = Grid::Index(x_bot_e[n_i],shape_e);
	int q_t[ndim];
	for(int k=0; k<ndim; ++k){q_t[k] = x_o[k]*shape_i[k]+x_bot_e[n_i][k]-1;}	
	if(Grid::InRange(q_t,shape_tot_v)) {
		const int n_t = Grid::Index_tot(q_t,shape_tot_v,shape_o,shape_i,size_i);
		for(int k=0; k<graddim; ++k) {eta_e[n_e][k] = eta_t[n_t+size_io*k];}
	} else {for(int k=0; k<graddim; ++k) {eta_e[n_e][k]=0;}}
}
__syncthreads();

// ------ Compute the divergence of the vector field ------
// Does not take into account the boundary weights
const int n_e = Grid::Index(x_i,shape_e);

#if ndim_macro==1
const int n0 = n_e, n1 = n_e+1;
const Scalar div_eta = eta_e[n1][0]-eta_e[n0][0];

#elif ndim_macro==2
const int n00 = n_e, n01 = n_e+1, n10 = n_e+shape_e[1], n11 = n10+1;

#if   grad_macro == gradb_macro
const Scalar div_eta = (
  eta_e[n11][0] - eta_e[n01][0] 
+ eta_e[n11][1] - eta_e[n10][1]);
#elif grad_macro == gradc_macro
const Scalar div_eta = Scalar(0.5)*(
  eta_e[n11][0] + eta_e[n10][0] - eta_e[n01][0] - eta_e[n00][0]
+ eta_e[n11][1] - eta_e[n10][1] + eta_e[n01][1] - eta_e[n00][1]);
#elif grad_macro==grad2_macro
const Scalar div_eta = Scalar(0.5)*(
  eta_e[n11][0] - eta_e[n01][0] + eta_e[n10][2] - eta_e[n00][2]
+ eta_e[n11][1] - eta_e[n10][1] + eta_e[n01][3] - eta_e[n00][3]);
#endif

#elif ndim_macro==3
const int 
n000 = n_e, n001 = n_e+1, n010 = n_e+shape_e[2], n011 = n010+1,
n100 = n_e+shape_e[1]*shape_e[2], n101=n100+1, n110 = n100+shape_e[2], n111 = n110+1;

#if   grad_macro == gradb_macro
const Scalar div_eta = (
  eta_e[n111][0] - eta_e[n011][0] 
+ eta_e[n111][1] - eta_e[n101][1] 
+ eta_e[n111][2] - eta_e[n110][2]);
#elif grad_macro == gradc_macro
const Scalar div_eta = Scalar(0.25)*(
  eta_e[n111][0] + eta_e[n110][0] + eta_e[n101][0] + eta_e[n100][0] 
  	- eta_e[n011][0] - eta_e[n010][0] - eta_e[n001][0] - eta_e[n000][0]
+ eta_e[n111][1] + eta_e[n110][1] - eta_e[n101][1] - eta_e[n100][1] 
	+ eta_e[n011][1] + eta_e[n010][1] - eta_e[n001][1] - eta_e[n000][1]
+ eta_e[n111][2] - eta_e[n110][2] + eta_e[n101][2] - eta_e[n100][2] 
	+ eta_e[n011][2] - eta_e[n010][2] + eta_e[n001][2] - eta_e[n000][2]);
#elif grad_macro==grad2_macro
const Scalar div_eta = Scalar(0.5)*(
  eta_e[n111][0] - eta_e[n011][0] + eta_e[n100][3] - eta_e[n000][3]
+ eta_e[n111][1] - eta_e[n101][1] + eta_e[n010][4] - eta_e[n000][4] 
+ eta_e[n111][2] - eta_e[n110][2] + eta_e[n001][5] - eta_e[n000][5]);
#endif

#endif

#if preproc_randers_macro
if(preproc_randers){
	phi_t[n_t]-=div_eta/weight; 
	return;
}
#endif

// ----- Compute the prox operator of chi_[-1,1](phi) + g*phi -----
// Load
const Scalar phi_old = phi_t[n_t];
const Scalar g = g_t[n_t];

const Scalar phi_in = phi_old + tau_primal*div_eta/weight;
const Scalar phi_new = max(Scalar(-1),min(1.,phi_in-tau_primal*g)); // Prox operator
const Scalar phi_ext = 2*phi_new - phi_old;

// Export
if(inRange){
	const Scalar phi_delta = phi_new-phi_old;
	phi_t[n_t] = phi_old + rho_overrelax*phi_delta;
	phi_ext_t[n_t] = phi_ext;
}

//return;
// ----- Evaluate the dual energy -----
__shared__ Scalar value_i[size_i];
value_i[n_i] = inRange ? (abs(div_eta-g*weight)) : Scalar(0); 
__syncthreads();
REDUCE_i(value_i[n_i]+=value_i[m_i];)
if(n_i==0){dual_value_o[n_o] = value_i[n_i];}

// ----- Evaluate the primal energy (zero-th order part) -----
value_i[n_i] = inRange ? (phi_ext*g*weight) : Scalar(0); 
__syncthreads();
REDUCE_i(value_i[n_i]+=value_i[m_i];)
if(n_i==0){primal_value_o[n_o] = value_i[n_i];}


} // proxg

// ------------------------------------------------------------------------------------

/** Generates a new dual point. */
__global__ void dual_step( 
	      Scalar * __restrict__ eta_t, // Dual variable (input and output)
	const Scalar * __restrict__ phi_ext_t, // Primal variable, extrapolated
	const Scalar * __restrict__ geom_t, // The metric data
	Scalar * __restrict__ primal_value_o // primal objective value of the block
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
// Metric
Scalar geom[geomsize];
for(int k=0; k<geomsize; ++k) {geom[k] = geom_t[k*size_io+n_t];}
Scalar eta_old[graddim];
for(int k=0; k<graddim; ++k) {eta_old[k] = eta_t[n_t+size_io*k];}

__shared__ Scalar phi_e[size_e];
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
	}
}
__syncthreads();

// ------ Gradient computation ------
Scalar grad_phi[graddim];
#if ndim_macro==1
const int n0 = n_e, n1 = n_e+1;
grad_phi[0] = phi_e[n1]-phi_e[n0];

#elif ndim_macro==2
const int n00 = n_e, n01 = n_e+1, n10 = n_e+shape_e[1], n11 = n10+1;

#if   grad_macro == gradb_macro // Upwind scheme, from the bottom of the cell
const Scalar p00 = phi_e[n00], p01 = phi_e[n01], p10=phi_e[n10];
grad_phi[0] = p10 - p00;
grad_phi[1] = p01 - p00;
#elif grad_macro == gradc_macro // Centered scheme, accurate but unstable
const Scalar p00 = phi_e[n00], p01 = phi_e[n01], p10=phi_e[n10], p11 = phi_e[n11];
grad_phi[0] = Scalar(0.5)*(p11 + p10 - p01 - p00);
grad_phi[1] = Scalar(0.5)*(p11 - p10 + p01 - p00);
#elif grad_macro==grad2_macro // Use both upwind and downwind schemes 
const Scalar p00 = phi_e[n00], p01 = phi_e[n01], p10=phi_e[n10], p11 = phi_e[n11];
grad_phi[0] = Scalar(0.5)*(p10 - p00);
grad_phi[1] = Scalar(0.5)*(p01 - p00);

grad_phi[2] = Scalar(0.5)*(p11 - p01);
grad_phi[3] = Scalar(0.5)*(p11 - p10);
#endif

#elif ndim_macro==3
const int 
n000 = n_e, n001 = n_e+1, n010 = n_e+shape_e[2], n011 = n010+1,
n100 = n_e+shape_e[1]*shape_e[2], n101=n100+1, n110 = n100+shape_e[2], n111 = n110+1;

#if   grad_macro == gradb_macro // Upwind scheme, from the bottom of the cell
const Scalar p000 = phi_e[n000], p001 = phi_e[n001], p010=phi_e[n010], p100=phi_e[n100];
grad_phi[0] = p100 - p000;
grad_phi[1] = p010 - p000;
grad_phi[2] = p001 - p000;
#elif grad_macro == gradc_macro
const Scalar 
p000 = phi_e[n000], p001 = phi_e[n001], p010 = phi_e[n010], p011 = phi_e[n011],
p100 = phi_e[n100], p101 = phi_e[n101], p110 = phi_e[n110], p111 = phi_e[n111];
grad_phi[0] = Scalar(0.25)*(p111 + p110 + p101 + p100 - p011 - p010 - p001 - p000);
grad_phi[1] = Scalar(0.25)*(p111 + p110 - p101 - p100 + p011 + p010 - p001 - p000);
grad_phi[2] = Scalar(0.25)*(p111 - p110 + p101 - p100 + p011 - p010 + p001 - p000);
#elif grad_macro==grad2_macro
const Scalar 
p000 = phi_e[n000], p001 = phi_e[n001], p010 = phi_e[n010], p011 = phi_e[n011],
p100 = phi_e[n100], p101 = phi_e[n101], p110 = phi_e[n110], p111 = phi_e[n111];
grad_phi[0] = Scalar(0.5)*(p100 - p000);
grad_phi[1] = Scalar(0.5)*(p010 - p000);
grad_phi[2] = Scalar(0.5)*(p001 - p000);

grad_phi[3] = Scalar(0.5)*(p111 - p011);
grad_phi[4] = Scalar(0.5)*(p111 - p101);
grad_phi[5] = Scalar(0.5)*(p111 - p110);
#endif

#endif

// --- Computation of the new dual point ----
// Update eta <- eta + tau_dual * grad_phi
Scalar eta[graddim];
madd_kvv(tau_dual,grad_phi,eta_old,eta); 
GRAD2(madd_kvv(tau_dual,grad_phi+ndim,eta_old+ndim,eta+ndim);)
// Project onto the unit ball of the dual metric
Scalar proj_eta[graddim]; 
#if metric_type_macro == metric_type_iso
proj_iso(eta,geom[0],proj_eta); 
GRAD2(proj_iso(eta+ndim,geom[0],proj_eta+ndim);)
#elif metric_type_macro == metric_type_iso_asym
proj_iso_asym(eta,geom[0],geom+1,proj_eta);
GRAD2(proj_iso_asym(eta+ndim,geom[0],geom+1,proj_eta+ndim);)
#elif metric_type_macro == metric_type_riemann
Scalar v[ndim][ndim];
copy_eigenvectors(geom+ndim,v);
proj_riemann(eta,geom,v,proj_eta);
GRAD2(proj_riemann(eta+ndim,geom,v,proj_eta+ndim);)
#elif metric_type_macro == metric_type_riemann_asym
Scalar v0[ndim][ndim], v1[ndim][ndim];
copy_eigenvectors(geom+ndim,v0);
copy_eigenvectors(geom+(ndim+qdim+ndim),v1);
proj_riemann_asym(eta,geom,v0,geom+(ndim+qdim),v1,geom+(ndim+qdim+ndim+qdim),proj_eta);
GRAD2(proj_riemann_asym(eta+ndim,geom,v0,geom+(ndim+qdim),v1,geom+(ndim+qdim+ndim+qdim),
	proj_eta+ndim);)
#endif


// Export the new dual point 
if(inRange){for(int k=0; k<graddim; ++k){
	const Scalar eta_new = proj_eta[k], old=eta_old[k];
	const Scalar eta_delta = eta_new-old;
	eta_t[n_t+size_io*k] = rho_overrelax*eta_delta + old;
}}

//return;
// ----- Evaluate the primal energy (gradient part) -----
// Note that grad_phi is the gradient of phi_ext.
// We could use the gradient of phi instead, but this would require duplicating 
// significant portions of the code above, and likely it does not make much difference.

const Scalar grad_phi_norm = 
#if metric_type_macro == metric_type_iso
norm_iso(grad_phi,geom[0]) GRAD2(+norm_iso(grad_phi+ndim,geom[0]));
#elif metric_type_macro == metric_type_iso_asym
norm_iso_asym(grad_phi,geom[0],geom+1) GRAD2(+norm_iso_asym(grad_phi+ndim,geom[0],geom+1));
#elif metric_type_macro == metric_type_riemann
norm_riemann(grad_phi,geom,v) GRAD2(+norm_riemann(grad_phi+ndim,geom,v));
#elif metric_type_macro == metric_type_riemann_asym
norm_riemann_asym(grad_phi,geom,v0,geom+(ndim+qdim+ndim+qdim))
GRAD2(+norm_riemann_asym(grad_phi+ndim,geom,v0,geom+(ndim+qdim+ndim+qdim)));
#endif


__shared__ Scalar value_i[size_i];
value_i[n_i] = inRange ? grad_phi_norm : Scalar(0); 
__syncthreads();

// Reduce the sum over kernels and export
REDUCE_i(value_i[n_i]+=value_i[m_i];)
if(n_i==0){primal_value_o[n_o] += value_i[n_i];}


} // proxf

} // extern "C"
