#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements Voronoi's first reduction 
of six dimensional positive definite matrices.
*/


#include "TypeTraits.h"

#define ndim_macro 6
const int ndim=ndim_macro;
#include "Geometry_.h"
#include "Inverse_.h"
#include "NetworkSort.h"

#define SIMPLEX_VERBOSE 0 

// linear programming 
#ifdef SIMPLEX_VERBOSE // Use simplex algorithm
#define SIMPLEX_MAX_M 21 // Number of constraints
#define SIMPLEX_MAX_N 16 // Number of variables. Initialization increases dimension by one.
#include "SimplexAlgorithm.h"

#else /* Use Siedel Homeyer algorithm. 
This is left here for comparison in case the simplex fails, testing, but should not really
be used since it is extremely slow in dimension 15.*/
//#define CHECK
#ifndef LINPROG_DIMENSION_MAX 
#define LINPROG_DIMENSION_MAX 15 // Use a non-recursive linprog
#endif
#endif


/* Select a Voronoi decomposition with Lipschitz dependency w.r.t parameters
Not yet implemented. */
#ifndef GEOMETRY6_NORMALIZE_SOLUTION
#define GEOMETRY6_NORMALIZE_SOLUTION 0
#endif
 
namespace Voronoi {

namespace dim_symdim {
	const Int ndim=symdim;
	#include "Geometry_.h"
}

typedef char small; // Small type to avoid overusing memory
typedef unsigned char uchar;
#define nullptr NULL
typedef unsigned int uint;

#include "Geometry6/Geometry6_data.h"
#include "Geometry6/Geometry6_datag.h"
#include "Geometry6/Geometry6_datakkt.h"

#ifndef GEOMETRY6_DATA2//This file is a bit huge, so it is not embedded in the agd library
#include "../../../../Miscellaneous/Geometry6_data2.h"
#endif


struct SimplexStateT {
	Scalar m[symdim];
	Scalar a[ndim][ndim];
	Int vertex;
	Scalar objective;
};

// The seven six dimensional perfect forms, vertices of Ryskov's polyhedron
const Int nvertex = 7;

const Scalar vertex_[nvertex][symdim] = {
 {2. ,1. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,0.5,0.5,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0.5,1. ,2. ,1. ,1. ,1. ,2. ,1. ,0.5,1. ,1. ,2. ,0.5,1. ,1. ,0.5,0. ,2. }
};

// ------ For GroupElem ------

// Number of neighbors of the perfect forms.
const Int nneigh_[nvertex] = {21, 6336, 38124, 21, 621, 46, 21};
// Number of classes of neighbors of each perfect form
const int nneigh_base_[7] = {1, 8, 11, 3, 3, 5, 1} ;
// The number of active constraints, at each perfect form
const int nsupport_[7] = {21, 30, 36, 21, 27, 22, 21} ;
typedef const small (*vertex_supportT)[6]; // small[][6]
const int ndiff_[nvertex] = {ndiff0,ndiff1,ndiff2,ndiff3,ndiff4,ndiff5,ndiff6};
typedef const small (*keyT)[symdim]; // small[][symdim]
typedef const small (*kkt_2weightsT)[symdim]; // small[symdim][symdim]


// ----- Group all those things togeter ----
struct vertex_dataT {
	const Scalar * vertex;
	
	// ------ For GroupElem ------
	
	// Number of neighbors
	const int nneigh;
	// The class of each neighbor vertex
	const uchar * neigh_vertex;
	// The next two encode the change of variable from neighbor toward reference form
	const uint * neigh_choice; // Note : 2^31 < 36^6 < 2^32
	const uchar * neigh_signs;
	
	// Number of classes of neighbors of each perfect form
	const int nneigh_base;
	// The vertex type of each neighbor class
	const int * neigh_base_v;
	// The change of variables from the neighbor, to the reference perfect form
	const chgi_jT * neigh_base_c;
	
	// The number and the list of the active constraints, at this vertex
	const int nsupport;
	const vertex_supportT vertex_support;
	
	// ----- For Better neighbor ------

	// The number of elementwise differences between the successive neighbors
	const int ndiff;
	// One key neighbor is placed every 1024, to avoid roundoff error accumulation
	const keyT key;
	// The place where successive neighbors differ
	const uchar * diff_i;
	// By how much the successive neighbors differ, at the given place
	const small * diff_v;
	
	// ----- For KKT -----

	const kkt_2weightsT kkt_2weights;
	const kkt_constraintsT kkt_constraints;

	vertex_dataT(const Scalar * _vertex, 
		const Int _nneigh, const uchar * _neigh_vertex,
		const uint * _neigh_choice, const uchar * _neigh_signs, 
		const int _nneigh_base, const int * _neigh_base_v, const chgi_jT * _neigh_base_c,
		const int _nsupport, const vertex_supportT _vertex_support,
		const int _ndiff, const keyT _key, const uchar * _diff_i, const small * _diff_v,
		const kkt_2weightsT _kkt_2weights, const kkt_constraintsT _kkt_constraints
		):
	vertex(_vertex),
	nneigh(_nneigh),neigh_vertex(_neigh_vertex),
	neigh_choice(_neigh_choice),neigh_signs(_neigh_signs),
	nneigh_base(_nneigh_base),neigh_base_v(_neigh_base_v),neigh_base_c(_neigh_base_c),
	nsupport(_nsupport),vertex_support(_vertex_support),
	ndiff(_ndiff),key(_key),diff_i(_diff_i),diff_v(_diff_v),
	kkt_2weights(_kkt_2weights),kkt_constraints(_kkt_constraints){};

};

#ifdef COMPILE_TIME_CSTRUCT
const vertex_dataT vertex_data_[nvertex] = {
	{vertex_[0], nneigh_[0],neigh_vertex0,neigh_choice0,neigh_signs0, nneigh_base_[0],neigh0_base_v,neigh0_base_c, nsupport_[0],vertex_support0, ndiff0,key0,diff0_i,diff0_v, kkt_2weights0,kkt_constraints0},
	{vertex_[1], nneigh_[1],neigh_vertex1,neigh_choice1,neigh_signs1, nneigh_base_[1],neigh1_base_v,neigh1_base_c, nsupport_[1],vertex_support1, ndiff1,key1,diff1_i,diff1_v, kkt_2weights1,kkt_constraints1},
	{vertex_[2], nneigh_[2],neigh_vertex2,neigh_choice2,neigh_signs2, nneigh_base_[2],neigh2_base_v,neigh2_base_c, nsupport_[2],vertex_support2, ndiff2,key2,diff2_i,diff2_v, kkt_2weights2,kkt_constraints2},
	{vertex_[3], nneigh_[3],neigh_vertex3,neigh_choice3,neigh_signs3, nneigh_base_[3],neigh3_base_v,neigh3_base_c, nsupport_[3],vertex_support3, ndiff3,key3,diff3_i,diff3_v, kkt_2weights3,kkt_constraints3},
	{vertex_[4], nneigh_[4],neigh_vertex4,neigh_choice4,neigh_signs4, nneigh_base_[4],neigh4_base_v,neigh4_base_c, nsupport_[4],vertex_support4, ndiff4,key4,diff4_i,diff4_v, kkt_2weights4,kkt_constraints4},
	{vertex_[5], nneigh_[5],neigh_vertex5,neigh_choice5,neigh_signs5, nneigh_base_[5],neigh5_base_v,neigh5_base_c, nsupport_[5],vertex_support5, ndiff5,key5,diff5_i,diff5_v, kkt_2weights5,kkt_constraints5},
	{vertex_[6], nneigh_[6],neigh_vertex6,neigh_choice6,neigh_signs6, nneigh_base_[6],neigh6_base_v,neigh6_base_c, nsupport_[6],vertex_support6, ndiff6,key6,diff6_i,diff6_v, kkt_2weights6,kkt_constraints6},
};
const vertex_dataT vertex_data(int i){return vertex_data_[i];}
#else
const vertex_dataT vertex_data(int i){
	switch(i){
		case 0: return vertex_dataT(vertex_[0], nneigh_[0],neigh_vertex0,neigh_choice0,neigh_signs0, nneigh_base_[0],neigh0_base_v,neigh0_base_c, nsupport_[0],vertex_support0, ndiff0,key0,diff0_i,diff0_v, kkt_2weights0,kkt_constraints0);
		case 1: return vertex_dataT(vertex_[1], nneigh_[1],neigh_vertex1,neigh_choice1,neigh_signs1, nneigh_base_[1],neigh1_base_v,neigh1_base_c, nsupport_[1],vertex_support1, ndiff1,key1,diff1_i,diff1_v, kkt_2weights1,kkt_constraints1);
		case 2: return vertex_dataT(vertex_[2], nneigh_[2],neigh_vertex2,neigh_choice2,neigh_signs2, nneigh_base_[2],neigh2_base_v,neigh2_base_c, nsupport_[2],vertex_support2, ndiff2,key2,diff2_i,diff2_v, kkt_2weights2,kkt_constraints2);
		case 3: return vertex_dataT(vertex_[3], nneigh_[3],neigh_vertex3,neigh_choice3,neigh_signs3, nneigh_base_[3],neigh3_base_v,neigh3_base_c, nsupport_[3],vertex_support3, ndiff3,key3,diff3_i,diff3_v, kkt_2weights3,kkt_constraints3);
		case 4: return vertex_dataT(vertex_[4], nneigh_[4],neigh_vertex4,neigh_choice4,neigh_signs4, nneigh_base_[4],neigh4_base_v,neigh4_base_c, nsupport_[4],vertex_support4, ndiff4,key4,diff4_i,diff4_v, kkt_2weights4,kkt_constraints4);
		case 5: return vertex_dataT(vertex_[5], nneigh_[5],neigh_vertex5,neigh_choice5,neigh_signs5, nneigh_base_[5],neigh5_base_v,neigh5_base_c, nsupport_[5],vertex_support5, ndiff5,key5,diff5_i,diff5_v, kkt_2weights5,kkt_constraints5);
		default: // Should not happen
		case 6: return vertex_dataT(vertex_[6], nneigh_[6],neigh_vertex6,neigh_choice6,neigh_signs6, nneigh_base_[6],neigh6_base_v,neigh6_base_c, nsupport_[6],vertex_support6, ndiff6,key6,diff6_i,diff6_v, kkt_2weights6,kkt_constraints6);
	}
}

#endif


/** Generates an isometry for the given vertex, 
which puts the corresponding neighbor in reference position.
Returns the index of the reference form.
*/
int GroupElem(const int ivertex, const int neighbor,
	small g[__restrict__ ndim][ndim]){
	const vertex_dataT & data = vertex_data(ivertex);
	const int nsupport = data.nsupport;
	const uchar edge = data.neigh_vertex[neighbor]; //unsigned to silence warning
	uint choice = data.neigh_choice[neighbor];
	char sign = data.neigh_signs[neighbor];

	/*
	std::cout << "choice " << choice << " and sign" << int(sign) << std::endl;
	std::cout << "ivertex " <<ivertex << std::endl;
*/
	// Decompose the choice and signs
	uint choices[ndim]; 	small signs[ndim];
	for(int i=0; i<ndim; ++i){
		choices[i] = choice % nsupport;
		choice /= nsupport;
		signs[i] = 1-2*(sign % 2);
		sign /= 2;
	}

	// Build the change of variables from the support vectors
	small g0[ndim][ndim];
	for(int j=0; j<ndim; ++j){
		const uint k = choices[j];
		const small * v = data.vertex_support[k];
//		show_v(std::cout,v);
		const small s = signs[j];
//		std::cout << k << " " << int(s) << std::endl;
		for(int i=0; i<ndim; ++i){
			g0[i][j] = s*v[i];}
	}
/*
	std::cout << "choices and signs" << std::endl;
	show_v(std::cout, choices);
	show_v(std::cout, signs);
	std::cout << "g0 " << std::endl;
	show_a(std::cout, g0); std::cout << std::endl;*/
	// If necessary, compose with the base change of variables
	chgi_jT chg = data.neigh_base_c[edge];
	if(chg==nullptr){copy_aA(g0,g);}
	else {dot_aa(chg,g0,g);}

	return data.neigh_base_v[edge];
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateT & state){
	const int ivertex = state.vertex;
	const vertex_dataT & data = vertex_data(ivertex);

	Scalar obj = dim_symdim::scal_vv(state.m,data.key[0]);
	int best_neigh = 0;
	Scalar best_obj = obj;
	for(int idiff=0,ineigh=1; idiff<data.ndiff; ++idiff){
		const uchar index = data.diff_i[idiff];
		obj += data.diff_v[idiff] * state.m[index & 31];
		if(index & 32){ // Completed neighbor
			if((ineigh & 1023)==0){
				// Use the key points to avoid roundoff error accumulation
//				const Scalar obj_old = obj;
				obj = dim_symdim::scal_vv(state.m, data.key[ineigh>>10]);
//				std::cout << obj << "," << (obj-obj_old) << std::endl;
			}
			if(obj<best_obj){
				best_obj = obj;
				best_neigh = ineigh;
			}
			++ineigh;
		}
	}

	// Now set that neighbor in the state, if necessary
	if(best_obj>=state.objective) return false;
	state.objective = best_obj;

	// Record the new vertex
	small a_[ndim][ndim];
	state.vertex = GroupElem(ivertex,best_neigh,a_);
	
	// Record the new change of coordinates
	Scalar a[ndim][ndim]; copy_aA(a_,a); // cast to scalar to avoid small overflow
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa); 
	dot_aa(a,sa,state.a);

	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm); 
	tgram_am(a,sm,state.m);

	return true;
}

// Sign of b-a, zero in the equality case.
template<typename T> int sign_diff(const T& a, const T& b){return (a<b) - (b<a);}

void KKT(const SimplexStateT & state, Scalar weights[symdim], 
	OffsetT offsets[symdim][ndim]){
	const vertex_dataT & data = vertex_data(state.vertex);
	
	// Compute a decomposition, possibly with negative entries
	dim_symdim::dot_av(data.kkt_2weights,state.m,weights);
	dim_symdim::div_Vk(weights, 2);
	
	// Change of variables toward original coordinates.
	Scalar aInv_[ndim][ndim]; inv_a(state.a,aInv_);
	Int aInv[ndim][ndim]; round_a(aInv_,aInv);

#ifndef nsupport_max_macro
	/* Upper bounds for the number of minimal vectors, 
	and the dimension of the linear program */
	const int nsupport_max = 36; // Upper bound
#endif

	// Number of minimal vectors for the perfect form
	const int nsupport = data.nsupport;
	OffsetT offsets_[nsupport_max][ndim]; // Using [nsupport][ndim]
	for(int i=0; i<nsupport; ++i){dot_av(aInv,data.vertex_support[i],offsets_[i]);}
	
	if(nsupport==symdim){
		// Case where the vertex is non-degenerate.
		// There is only one possible decomposition, and it must be non-negative.
		for(int i=0; i<symdim; ++i){copy_vV(offsets_[i],offsets[i]);}
		return;
	} else {
		// Case where the vertex is degenerate.
		// Solve a linear program to find a non-negative decomposition

		// Dimension of the linear program
		const int d = nsupport - symdim;
		const int d_max = nsupport_max - symdim;

#if GEOMETRY6_NORMALIZE_SOLUTION
		// We want the 'best' solution, w.r.t suitable ordering of the support vectors
		int offset_norm2_[nsupport_max]; 
		for(int k=0; k<nsupport; ++k){
			// First, normalize those, so that their first coordinate is a positive number
			OffsetT * offset = offsets_[k];
			OffsetT sign=0;
			for(int i=0; i<ndim; ++i){if(offset[i]!=0) {sign=2*(offset[i]>0)-1;break;}}
			for(int i=0; i<ndim; ++i){offset[i]*=sign;}
			// Also, we want to promote small vectors, so let us compute the norms
			int & offset_norm2 = offset_norm2_[k];
			offset_norm2=0;
			for(int i=0; i<ndim; ++i){offset_norm2+=int(offset[i])*int(offset[i]);}
		}
		
//		for(int k=0; k<nsupport; ++k){
//			show_v(std::cout, offsets_[k]); std::cout << std::endl;}

		// We use a basic bubble sort to sort the keys
		// We do not use the sorting functions, as it would require 
		// to define a struct with comparison operators... 
		int support_order[nsupport_max];
		for(int k=0; k<nsupport_max; ++k) support_order[k]=k;
		for(int k=0; k<nsupport; ++k){
			for(int l=0; l<nsupport-k-1; ++l){
				const int u = support_order[l], v = support_order[l+1];
				// Norm equality, then lexicographic ordering
				int ordered = sign_diff(offset_norm2_[u], offset_norm2_[v]);
				for(int i=0; i<ndim; ++i){
					if(ordered==0) ordered = sign_diff(offsets_[u][i],offsets_[v][i]);}
				// Put the smallest norm first
				if(ordered==-1){support_order[l]=v; support_order[l+1]=u;}
			}
		}
		Int support_priority[nsupport_max]; Int support_tmp[nsupport_max];
		variable_length_sort(support_order,support_priority,support_tmp,nsupport);
/*		std::cout<< "offset_norm2_,priority "; 
		for(int i=0; i<nsupport; ++i) std::cout << "(" << offset_norm2_[i]<<
		","<< support_priority[i]<<")"; std::cout<<std::endl;
		*/

		
		// Now, we must reflect this ordering on the linear programming variables
		// Priority : lower number goes first
		int priority[d_max]; 
		int direction[d_max];
		for(int k=0; k<d; ++k) {
			priority[k]  = support_priority[symdim+k];
			direction[k] = 1;
			for(int l=0; l<symdim; ++l){
				if(data.kkt_constraints[k][l]!=0 && support_priority[l]<priority[k]){
					priority[k] = support_priority[l];
					direction[k] = data.kkt_constraints[k][l]>0;
				}
			}
			direction[k] = 2*direction[k]-1; // Turn sign bit into +/- value
		}
		// Set the objective function to favor unknowns with low priority number
		Int unknowns_order[d_max]; Int unknowns_tmp[d_max];
		variable_length_sort(priority,unknowns_order,unknowns_tmp,d);
		Scalar objective[d_max]; Scalar w=1.;
		for(int i=0; i<d; ++i) {
			const int j=unknowns_order[i];
			objective[j] = -direction[j] * w;
			w/=16.;
		}
		
		
//		std::cout<< "offset_norm2_ "; for(int i=0; i<nsupport; ++i) std::cout << offset_norm2_[i]<<","; std::cout<<std::endl;
//		std::cout<< "support_priority ";for(int i=0; i<nsupport; ++i) std::cout << support_priority[i]<<","; std::cout<<std::endl;
#endif // Normalize solution

#ifdef SIMPLEX_VERBOSE // Simplex algorithm
		
		SimplexData sdata;
		sdata.n = d; // number of variables (all positive)
		sdata.m = symdim; // Number of constraints (all positivity constraits)
		Scalar opt[nsupport_max]; // optimal solution
		Scalar wfeas = 0.; //SIMPLEX_TOL; // Added to ensure feasibility

//		for(int i=0; i<symdim;++i) printf(" %f",weights[i]); printf("\n");

		for(int i=0; i<symdim; ++i){ // Specify the constraints
			for(int j=0; j<d; ++j){
				sdata.A[i][j] = data.kkt_constraints[j][i];}
			sdata.b[i] = weights[i] + wfeas; // No normalization here
		}

		for(int i=0; i<sdata.n; ++i){
			sdata.c[i] = // Linear form to be maximized
			#if GEOMETRY6_NORMALIZE_SOLUTION
			objective[i]; // Select Lipschitz continuous representative
			#else
			1; // Arbitrary
			#endif
		}

		const Scalar value = simplex(sdata,opt);   
//		assert(!isinf(value));
		/* +/-Infinity values mean failure (unbounded or infeasible problem).
		 However, we choose not to crash the program, but to detect the
		 invalid decompositions a posteriori, and recompute them differently.
		 (Typical : failure using floats, success using double.) */

/*		std::cout << "Value of the linear program " << value << std::endl;
		if(isinf(value)) {std::cout << opt[0] << std::endl;}
		std::cout << "state.vertex" << state.vertex << std::endl;*/
		Scalar sol[nsupport_max]; // solution
		for(int i=0; i<d; ++i){sol[symdim+i] = opt[i];}
		for(int i=0; i<symdim; ++i){sol[i] = opt[d+i];}

#else // Siedel Homeyer linprog

		// ---- Define the half spaces intersections. (linear constraints) ----
		Scalar halves[(nsupport_max+1)*(d_max+1)]; // used as Scalar[nsupport+1][d+1];
		
		Scalar maxWeight = 0;
		for(Int i=0; i<symdim; ++i) maxWeight = max(maxWeight,abs(weights[i]));
		if(maxWeight==0) maxWeight=1;
		
		// The old components must remain positive
		for(int i=0; i<symdim; ++i){
			for(int j=0; j<d; ++j){
				halves[i*(d+1)+j] = data.kkt_constraints[j][i];}
			halves[i*(d+1)+d] = weights[i]/maxWeight;
		}
		
		// The new components must be positive
		for(int i=symdim; i<nsupport; ++i){
			for(int j=0; j<d; ++j){
				halves[i*(d+1)+j] = (i-symdim)==j;}
			halves[i*(d+1)+d] = 0;
		}
		
		// Projective component positive
		for(int j=0; j<d; ++j){halves[nsupport*(d+1)+j] = 0;}
		halves[nsupport*(d+1)+d] = 1;
						
		Scalar n_vec[d_max+1]; // used as Scalar[d+1]
		Scalar d_vec[d_max+1]; // used as Scalar[d+1]
#if GEOMETRY6_NORMALIZE_SOLUTION
		for(int i=0; i<d; ++i) {n_vec[i]=objective[i];}
#else
		// Minimize some arbitrary linear form (we only need a feasible solution)
		for(int i=0; i<d; ++i) {n_vec[i]=1;}
#endif
		for(int i=0; i<d; ++i) {d_vec[i]=0;}
		n_vec[d]=0; d_vec[d]=1;

			
		Scalar opt[d_max+1];
		const int size = nsupport+1;
		const int size_max = nsupport_max+1;

		Scalar work[((size_max+3)*(d_max+2)*(d_max-1))/2]; // Scalar[4760]
		
		int next[size_max];
		int prev[size_max];
		for(int i=0; i<size_max; ++i){
			next[i] = i+1;
			prev[i] = i-1;
		}
		linprog(halves, 0, size, n_vec, d_vec, d, opt, work, next, prev, size);
		
		//switch(d){
		//	case 15: linprog_templated<15>::go(halves, 0, size, n_vec, d_vec, /*d,*/ opt, work, next, prev, size); break;
		//}
	
		// TODO : check that status is correct
		// The solution is "projective". Let's normalize it, dividing by the last coord.
		for(int i=0; i<d; ++i){opt[i]/=opt[d];}


		// Get the solution, and find the non-zero weights, which should be positive.
		Scalar sol[nsupport_max]; // Using sol[nsupport]
		for(int i=0; i<symdim; ++i){
			Scalar s=0;
			for(int j=0; j<d; ++j) {s+=opt[j]*halves[i*(d+1)+j];}
			s*=maxWeight;
			sol[i]=s+weights[i];
		}
		for(int i=0; i<d; ++i){sol[symdim+i] = maxWeight*opt[i];}
#endif // Siedel Homeyer linprog

		// We only need to exclude the d smallest elements. For simplicity, we sort all.
		Int ord[nsupport_max], tmp[nsupport_max]; // using Int[nsupport]
		variable_length_sort(sol, ord, tmp, nsupport);

		for(int i=0; i<symdim; ++i){
			const int j=ord[i+d];
			weights[i] = sol[j];
			copy_vV(offsets_[j],offsets[i]);
		} // for i
		
		
	}
}


} // Namespace Voronoi

const Int decompdim=symdim;
#include "Voronoi_.h"
