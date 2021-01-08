#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

// This file implements a solver of the shape from shading equation

/* // Must be defined in external file
const int niter_i = 8;
const int side_i = 8; // at least 4
*/

const int ndim=2;

const int shape_i[ndim] = {side_i,side_i};

const int side_e = side_i+2;
const int shape_e[ndim] = {side_e,side_e}; // Important : shape_e = shape_i+2 (boundary layer)
const int size_e = side_e * side_e; // Important : size_e = product(shape_e)
typedef unsigned char boolatom;

__constant__ float params[4]; // alpha (signed),beta (signed), gamma, h;
__constant__ int shape_o[ndim]; 
__constant__ int shape_tot[ndim];

#include "Grid.h"
#include "Geometry_.h"

void sq(float x){return x*x;}

void update(float * u_e,const float cp,const int n_e){
	// Adapted from the Python notebook ShapeFromShading
	// Getting neighbor values
	const float u0 = u_e[n_e]; // Old value
	const float // Finite differences in four directions
	umx = u_e[n_e-side_e] -u0, 
	upx = u_e[n_e+side_e] -u0,
	umy = u_e[n_e-1]      -u0,
	upy = u_e[n_e+1]      -u0;

	const float // Monotone scheme
	wx = params[0]>0 ? umx : upx,
	wy = params[1]>0 ? umy : upy,
	vy = min(umx,upx);
	vy = min(umy,upy);

	const float 
	alpha = abs(params[0]),
	beta  = abs(params[1]),
	gamma = params[2],
	h = params[3];

    // ---- Trying with two active positive parts ----

    // Quadratic equation coefficients.
    // a lambda^2 - 2 b lambda + c =0
    const float cp2 = sq(cp);
    float a = cp2 - sq(alpha+beta);
    float b = cp2 * (vx+vy) - (alpha+beta)*(alpha*wx+beta*wy+h*gamma);
    float c = cp2*(sq(h)+sq(vx)+sq(vy))-sq(gamma*h+alpha*wx+beta*wy);

    float delta = sq(b) - a*c;
    if(delta>=0 && a!=0){
		const float u = (b+np.sqrt(delta))/a;
		const float vmax = max(vx,vy)
		if(u>=vmax) return u;
    }

    // ---- Trying with one active positive part ----
    // TODO : restrict computations to not good points to save cpu time ?

    const float vmin = min(vx,vy);
	// a is unchanged
    b = cp2 * vmin - (alpha+beta)*(alpha*wx+beta*wy+h*gamma);
    c = cp2*(sq(h)+sq(vmin))-sq(gamma*h+alpha*wx+beta*wy);
    delta = sq(b) - a*c;

    if(delta>=0 && a!=0){
    	const float u = (b+sqrt(delta))/a;
    	if(u>=vmin) return u;
    }

    // No active positive part
    // equation becomes linear, a lambda - b = 0
    a = alpha+beta;
    b = alpha*wx+beta*wy +gamma*h - cp*h;
    const float u=b/a;
    return u

}

export "C" {

__global__ void sfs(
	      float    * __restrict__ u_t, 
	const float    * __restrict__ cp_t, 
	const boolatom * __restrict__ mask_t, 
	const int      * __restrict__ update_o){

	// Set position and indices
	int x_i[ndim]; 
	x_i[0] = threadIdx.x;
	x_i[1] = threadIdx.y;

	const int n_o = update_o[blockIdx.x];
	int x_o[ndim];
	Grid::Position(n_o,shape_o,x_o);

	int x_e[ndim]; 
	add_kvV(1,x_i,x_e);

	int x_t[ndim];
	madd_kvv(side_i,x_o,x_i,x_t);
	const int n_t = Grid::Index_tot(x_t);
	const int n_e = Grid::Index(x_e,shape_e);

	// Load the data
	const float cp = cp_t[n_t];
	const float mask = mask_t[n_t];
	__shared__ u_e[size_e];
	u_e[n_e] = u_t[n_t];
	
	if(n_i<4*side_i){ // Load value from boundary layer
		const int r = 1+(n_i%side_i);
		int y_e[ndim];
		if     (m_i<side_i)   {y_e[0]=0;      y_e[1]=r;}
		else if(m_i<2*side_i) {y_e[0]=side_i; y_e[1]=r;}
		else if(m_i<2*side_i) {y_e[0]=r;      y_e[1]=0;}
		else if(m_i<2*side_i) {y_e[0]=r;      y_e[1]=side_i;}
		
		const int m_e = Grid::Index(y_e,shape_e);
		int y_t[ndim];
		madd_kvv(side_i,x_o,y_e,y_t);
		add_kV(-1,y_t);

		u_e[m_e] = Grid::InRange(y_t,shape_tot) ?
		u_t[Grid::Index_t(y_t)] : 0.;
	}
	__syncthreads();

	// Iterate the scheme
	for(int i=0; i<niter; ++i){
		if(mask) update(u_e,cp,n_e);
		__syncthreads()

	}

	// Export the data
	u_t[n_t] = u_e[n_e];
}

}