#pragma once 
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.
xx,xe,ee : scalar products of the vectors x and e*/
void euclidean_factor_sym(const Scalar xx, const Scalar xe, const Scalar ee,
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	const Scalar Nx = sqrt(xx), // |x|_m
	Nxme = sqrt(xx-2*xe+ee), // |x-e|_m
	Nxpe = sqrt(xx+2*xe+ee); // |x+e|_m
	const Scalar 
	Nx_Nxme = ( 2*xe - ee)/(Nx + Nxme), // |x|_m-|x-e|_m computed in a stable way
	Nx_Nxpe = (-2*xe - ee)/(Nx + Nxpe), // |x|_m-|x+e|_m computed in a stable way
	grad_e = xe/Nx; // <e, m x/|x|_m>
	fact[0] = -grad_e + Nx_Nxme; // <-e,m x/|x|_m> + |x|_m-|x-e|_m
	fact[1] =  grad_e + Nx_Nxpe; // < e,m x/|x|_m> + |x|_m-|x+e|_m

	ORDER2(
	const Scalar 
	Nxme2 = sqrt(xx-4*xe+4*ee), // |x-2e|_m
	Nxpe2 = sqrt(xx+4*xe+4*ee); // |x+2e|_m
	const Scalar 
	Nxme2_Nxme = (-2*xe + 3*ee)/(Nxme+Nxme2), // |x-2e|_m-|x-e|_m computed in a stable way
	Nxpe2_Nxpe = ( 2*xe + 3*ee)/(Nxpe+Nxpe2); // |x+2e|_m-|x+e|_m computed in a stable way
	fact2[0] = 2*fact[0]-(Nx_Nxme + Nxme2_Nxme); // parenth : |x|_m-2|x-e|_m+|x-2e|_m
	fact2[1] = 2*fact[1]-(Nx_Nxpe + Nxpe2_Nxpe); // parenth : |x|_m-2|x+e|_m+|x+2e|_m
	)
}

/** Returns the perturbation involved in the factored fast marching method,
for a generic norm.*/
void generic_factor_sym(const Scalar grad_e, // Gradient at x in direction e
	const Scalar Nx, const Scalar Nxpe, const Scalar Nxme, // Norm at x and neighbors
	Scalar fact[2] // Scheme first order perturbaton 
	ORDER2(,const Scalar Nxpe2, const Scalar Nxme2, // Norm at farther neighbors
	Scalar fact2[2]) ){ // Scheme second order perturbation

	fact[0] = -grad_e + Nx - Nxme; 
	fact[1] =  grad_e + Nx - Nxpe; 

	ORDER2(
	fact2[0] = 2*fact[0]-(Nx - 2*Nxme + Nxme2); 
	fact2[1] = 2*fact[1]-(Nx - 2*Nxpe + Nxpe2); 
	)
}