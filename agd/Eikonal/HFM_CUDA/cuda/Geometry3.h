#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=3;
#include "Geometry_.h"

#ifndef Selling_maxiter_macro
const Int Selling_maxiter = 100;
#endif

/// Cross product, in dimension three. 
template<typename Tx, typename Ty, typename Tout=Tx>
void cross_vv(const Tx x[3], const Ty y[3], Tout out[__restrict__ 3]){
	for(Int i=0; i<3; ++i){
		const Int j=(i+1)%3, k=(i+2)%3;
		out[i]=x[j]*y[k]-x[k]*y[j];
	}
}

// the first two elements of these permutations range among all possible pairs
const Int iterReducedMax = 6;
const Int Selling_permutations[iterReducedMax][ndim+1] = { 
	{0,1,2,3},{0,2,1,3},{0,3,1,2},{1,2,0,3},{1,3,0,2},{2,3,0,1}};

// Computation of an obtuse superbase of a positive definite matrix, by Selling's algorithm
void obtusesuperbase_m(const Scalar m[symdim], OffsetT sb[__restrict__ ndim+1][ndim]){
	canonicalsuperbase(sb);
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int it = iter%6; 
		const Int * perm = Selling_permutations[it];
		const Int i = perm[0], j=perm[1];
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			const Int k=perm[2], l=perm[3];
			add_vV(sb[i],sb[k]);
			add_vV(sb[i],sb[l]);
			neg_V(sb[i]);
			iterReduced=0;
		}
	}
}

// Selling's decomposition of a positive definite matrix
const Int decompdim = symdim;
void decomp_m(const Scalar m[symdim], Scalar weights[__restrict__ symdim], 
	OffsetT offsets[__restrict__ symdim][ndim]){
	OffsetT sb[ndim+1][ndim];
	obtusesuperbase_m(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int * perm = Selling_permutations[r];
		const Int i=perm[0],j=perm[1],k=perm[2],l=perm[3];
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]) );
		cross_vv(sb[k],sb[l],offsets[r]);
	}
}

Scalar det_m(const Scalar m[symdim]){ 
	return -m[2]*m[3]*m[3]+2*m[1]*m[3]*m[4]-m[0]*m[4]*m[4]-m[1]*m[1]*m[5]+m[0]*m[2]*m[5];}

Scalar trace_m(const Scalar m[symdim]){return m[0]+m[2]+m[5];}

/**  
Returns : the middle root of a degree three polynomial. 
Assumption : a+bx+cx^2+x^3 has three real roots.
*/
Scalar _eigvalsh_midroot(Scalar a, Scalar b, Scalar c){
	Scalar xo,x,px,p1x,p2x; // px = p(x); p1x = p'(x); p2x = p''(x)/2
	// px=a+x*(b+x*(c+x)); p1x=b+x*(2*c+3*x); p2x=c+3*x

	// Test wether the value is positive at inflexion point. If not consider -p(-x)
	x=-c/3; px=a+x*(b+x*(c+x)); 
	const bool reversed = px<0;
	if(reversed){x=-x; px=-px; a=-a; c=-c;}
	p1x=b+x*(2*c+3*x); 
	if(!(p1x<0)) return reversed ? (-x) : x; // p1x<0, except in the case of a triple root where p1x=0
	xo=x; x-=px/p1x; 	// Make a Newton step. 
	// xmax is a local minimizer of P, by construction larger than the midroot
	const Scalar xmax=(-c+sqrt(max(Scalar(0),c*c-3*b)))/3;
	for(int i=0; i<10; ++i){
		// By construction, x increases, and still underestimates the root.
		px=a+x*(b+x*(c+x)); 
		if(!(xo<x && px>0)) return reversed ? (-x) : x;
		// Find the tangent quadratic polynomial, and take the first root.
		p1x=b+x*(2*c+3*x); p2x=c+3*x; 
		const Scalar delta = p1x*p1x-2*px*p2x;
		if(!(delta>=0 && p2x>0)) return reversed ? (-x) : x;
		xo=x; x+=(-p1x-sqrt(delta))/p2x;
		if(x>xmax) return reversed ? (-xo) : xo;
	}
	return reversed ? (-x) : x;
}

/** 
Returns the eigenvalues of a symmetric matrix, sorted from smallest to largest.
There are official cupy functions for this, which are likely more accurate.
However, they but they are very slow, and have high memory usage.
*/
void eigvalsh(const Scalar m[__restrict__ symdim], Scalar lambda[__restrict__ ndim]){
	// Compute the characteristic polynomial det(x*I-m) = a+bx+cx^2+x^3
	const Scalar a=-det_m(m), c=-trace_m(m),
	b=-m[1]*m[1] + m[0]*m[2] - m[3]*m[3] - m[4]*m[4] + m[0]*m[5] + m[2]*m[5];
	const Scalar r = _eigvalsh_midroot(a,b,c);
	// Divide : a+bx+cx^2+x^3 = (x-r)(d+ex+x^2)
	const Scalar d=b+r*(c+r),e=c+r;
	const Scalar delta = sqrt(max(0.,e*e-4*d));
	lambda[0] = (-e-delta)/2;
	lambda[1] = r;
	lambda[2] = (-e+delta)/2;
}

/// Evaluates (m-lambda)*(m-lambda1), and returns the largest column
void _eigh_helper(const Scalar m[symdim], 
	const Scalar m2[symdim], // m squared
	const Scalar lambda0, const Scalar lambda1, // Expected to be eigenvalues
	Scalar v[__restrict__ ndim] // Contents of the largest column
	){ 
	const Scalar s = lambda0+lambda1, p = lambda0*lambda1;
	Scalar mp[symdim];
	Scalar mp_max = -1.; int i_max=0;
	for(int i=0,k=0; i<ndim; ++i){
		for(int j=0; j<=i; ++j,++k){
			mp[k]=m2[k]-s*m[k]+p*(i==j); // (m-lambda)*(m-lambda1)
			const Scalar a = abs(mp[k]);
			if(mp_max<=a){i_max=i; mp_max=a;}
		}
	}
	for(int j=0; j<ndim; ++j) v[j]=coef_m(mp,i_max,j);
} 

/** Returns the eigenvalues and eigenvectors of a symmetric matrix.
(The cuda routines are likely more accurate, but they are slow and memory intensive)
*/
void eigh(const Scalar m[symdim], // In : symmetric matrix
	Scalar lambda[__restrict__ ndim], // Out : eigenvalues
	Scalar v[__restrict__ ndim][ndim]){ // Out : eigenvectors
	/// Compute the eigenvalues, and find the one farthest from the other two
	eigvalsh(m,lambda);
	const int i0 = (lambda[0]+lambda[2]<2*lambda[1]) ? 0 : 2, i1=1, i2=2-i0;
	//compute m^2
	Scalar m2[symdim]; 
	zero_M(m2);
	for(int i=0,k=0; i<ndim; ++i) {
		for(int j=0; j<=i; ++j,++k) {
			for(int r=0; r<ndim; ++r) m2[k] += coef_m(m,i,r)*coef_m(m,r,j);
		}
	}
	// Evaluate on m a polynomial vanishing at other eigenvalues, to get the eigenvectors
	_eigh_helper(m,m2,lambda[i1],lambda[i2],v[i0]);
	_eigh_helper(m,m2,lambda[i0],lambda[i1],v[i2]);

	 // Normalize v0
	Scalar norm2 = norm2_v(v[i0]);
	if(norm2>0) mul_kV(1/sqrt(norm2),v[i0]); else v[i0][0]=1;

	// Orthonormalize v2 (Needed for edge cases where the matrix has identical eigenvals)
	const Scalar s02 = scal_vv(v[i0],v[i2]);
	madd_kvV(-s02,v[i0],v[i2]);
	norm2 = norm2_v(v[i2]);
	if(norm2>0) mul_kV(1/sqrt(norm2),v[i2]);
	if(norm2==0 || abs(scal_vv(v[i0],v[i2]))>1e-2){
	// Orthonormalization failed. The matrix must be almost proportional to identity
	// We use some arbitrary nonzero vector orthogonal to v[i0]
		int i=0; // Becomes the index of the largest coordinate of v[i0]
		Scalar v_max = abs(v[i0][0]), v_abs = abs(v[i0][1]);
		if(v_abs > v_max){i=1; v_max=v_abs;}
		v_abs = abs(v[i0][2]);
		if(v_abs > v_max) i=2;
		norm2 = norm2_v(v[i2]);
		const int j = (i+1)%ndim, k = (i+2)%ndim;
		v[i2][i] = -v[i0][j];
		v[i2][j] =  v[i0][i];
		v[i2][k] = 0;
		mul_kV(1/sqrt(norm2_v(v[i2])),v[i2]);
	}
	
	// Generate v1
	cross_vv(v[2],v[0],v[1]);
	// Transpose (convention for compatibilty with numpy)
	trans_A(v);
}

/// Produces the rotation associated with a unit quaternion. See agd/Sphere.py
void rotation3_from_sphere3(const Scalar q[4], Scalar r[__restrict__ ndim][ndim]){
	const Scalar qr=q[0], qi=q[1], qj=q[2], qk=q[3];
	r[0][0] = Scalar(0.5)-(qj*qj+qk*qk); r[0][1] = qi*qj-qk*qr; r[0][2] = qi*qk+qj*qr;
	r[1][0] = qi*qj+qk*qr; r[1][1] = Scalar(0.5)-(qi*qi+qk*qk); r[1][2] = qj*qk-qi*qr;
	r[2][0] = qi*qk-qj*qr; r[2][1] = qj*qk+qi*qr; r[2][2] = Scalar(0.5)-(qi*qi+qj*qj);
	for(int i=0; i<ndim; ++i){for(int j=0; j<ndim; ++j) r[i][j] *= Scalar(2);}
}

/// Produces the unit quaternion, with positive real part, associated with a rotation.
void sphere3_from_rotation3(const Scalar r[ndim][ndim], Scalar q[__restrict__ 4]){
    q[0] = sqrt(Scalar(1)+trace_a(r))/Scalar(2);
    const Scalar s = Scalar(0.25)/q[0];
    q[1] = (r[2][1]-r[1][2])*s;
    q[2] = (r[0][2]-r[2][0])*s;
    q[3] = (r[1][0]-r[0][1])*s;
}

