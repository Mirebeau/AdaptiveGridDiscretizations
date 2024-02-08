#pragma once 
// Copyright 2023 Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/*
This file implements a Dubins-like vehicle model with several states.
The state space is Omega x A, where Omega is a domain of R^d (or R^2xS1 for actual Dubins) 
and A is an abstract set of states.

In a given state a in A, the vehicle can choose between various controls omega_i(a,x).
A transition cost between the states is given. For simplicity, we assume that it is independent of 
the current point x. 
*/

// nstates must be defined
// ncontrols must be defined 

#if controls_max_macro // Hamiltonian is max_i <p,omega_i>_+^2
// Choose between the provided controls and a transition between states.
#define nmix_macro (ncontrols_macro+(nstates_macro>1)) 
#if nmix_macro>=2
const bool mix_is_min = true; // Take the most efficient control among all available
#endif
#endif

const Int ncontrols = ncontrols_macro, nstates = nstates_macro;
__constant__ Scalar state_transition_costs_m2[nstates][nstates]; 
#define decomp_v_macro true
#define local_scheme_macro true // The coordinates of the points are used to get the state
#define nsym_macro 0 // We also need the macro, otherwise issues with zero-length arrays0
const int ndim = ndim_macro;
#define ndim_nostate (ndim_macro - 1)
const Int geom_size = ncontrols*ndim_nostate; 
#include "TypeTraits.h"

namespace nostate { // Drop last dimension
	#if (ndim_nostate == 1)
	#include "Geometry1.h"
	#elif (ndim_nostate == 2)
	#include "Geometry2.h"
	#elif (ndim_nostate == 3)
	#include "Geometry3.h"
	#endif
}

const Int nsym = 0; // Number of symmetric offsets
const Int ddim_nostate = nostate::decompdim; // (Selling/Voronoi decomposition)
#if controls_max_macro
const Int nfwd = ddim_nostate; 
#else 
const Int nfwd =  (ncontrols * ddim_nostate) + (nstates-1);
#endif
#include "Constants.h"

namespace nostate {
	#include "Decomp_v_.h"
}


STATIC_ASSERT(ndim>=2, inconsistent_dimension)
STATIC_ASSERT(nstates-1<=ddim_nostate || !controls_max_macro, unsupported_scheme_parameters)

bool scheme(const Scalar geom[geom_size], const Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	STATIC_ASSERT(nactx==nfwd*nmix_macro,inconsistent_scheme_parameters)

	for(int i=0; i<nactx; ++i){for(int j=0; j<ndim; ++j){offsets[i][j]=0;}}

	for(int i=0; i<ncontrols; ++i){
		Int offsets_nostate[ddim_nostate][ndim_nostate];
		nostate::decomp_v(&geom[ndim_nostate*i],&weights[ddim_nostate*i],offsets_nostate);
		for(int j=0; j<ddim_nostate; ++j){ // Last offset component is zero
			for(int k=0; k<ndim_nostate; ++k){
				offsets[ddim_nostate*i+j][k]=offsets_nostate[j][k];} 
		}
	}

/*	if(blockIdx.x==0 && threadIdx.x ==0){
		printf("In scheme %i %i %i %i %i %i\n", ncontrols, ndim_nostate, nfwd, geom_size, nactx, nmix);
		printf("... strict %i nmix_macro %i geom_first %i\n",strict_iter_i_macro,nmix_macro,geom_first_macro);
		printf("... %f, %f, %f, %f, %f \n",geom[0],geom[1],weights[0],weights[1],weights[2]);
		printf("... %i, %i, %i, %i\n",offsets[0][0],offsets[0][1], offsets[1][0], offsets[1][1]);
	}*/

	// Weights and offsets corresponding to transistion costs between states
	if(nstates==1) return;
	const Int state = x[ndim_nostate]; // State of the current point
	for(int i=0,j=0; i<nstates; ++i){
		if(i==state) continue; // Skip entry which amounts to stay in same state
		weights[ddim_nostate*ncontrols+j] = state_transition_costs_m2[state][i];
		offsets[ddim_nostate*ncontrols+j][ndim_nostate] = i-state;//First offset components = 0
		++j;
	}
	#if controls_max_macro //Zero fill unused part of the scheme. (Corresponding offset=0)
	for(int i=nstates-1;i<ddim_nostate;++i){weights[ddim_nostate*ncontrols+i]=0;}
	#endif
}

#include "Geometry_.h"
#include "Update.h"