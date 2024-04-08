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


/* Select a Voronoi decomposition with Lipschitz dependency w.r.t parameters. */
#ifndef GEOMETRY6_NORMALIZE_SOLUTION
#define GEOMETRY6_NORMALIZE_SOLUTION 1
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

#ifndef GEOMETRY6_DATA2 // This file is a bit huge, sorry
#include "Geometry6/Geometry6_data2.h"
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

// Chosen by fair dice rool, guaranteed to be random (xkcd 221)
const int hash_base = 13, hash_mod=256;
const Scalar hash_rnd[hash_mod] = {0.690697337782161, 0.483722331116672, 0.743753753429354, 0.721002138241528, 0.889101224551914, 0.700742469711384, 0.115054049449868, 0.482771316762674, 0.967218217812245, 0.191506875825576, 0.73755020982563 , 0.307231789670071, 0.150910615586556, 0.948304757861257, 0.461940751717273, 0.045885478525466, 0.319677532667326, 0.862256548592645, 0.195060126713638, 0.049453630566158, 0.169899660203638, 0.644513634809654, 0.990324311046253, 0.138508154706999, 0.117350538337899, 0.264385619222806, 0.137010952394882, 0.855897532653139, 0.772557659555397, 0.817922362469687, 0.448468222323861, 0.965797868913469, 0.789511339778968, 0.948396161132307, 0.163279182883155, 0.757266668795819, 0.861751358261795, 0.278942661588243, 0.087666585622725, 0.441987761951863, 0.804788798580161, 0.499044783763613, 0.865147587165297, 0.90631506634091 , 0.531028309425607, 0.009067113221537, 0.760767603278517, 0.675151447628338, 0.181166194746267, 0.839149506359144, 0.275833809396651, 0.800011811333069, 0.449447334673229, 0.633919620316714, 0.035842161856303, 0.126884179544837, 0.24197674458787 , 0.737620672712132, 0.432434750271531, 0.536786909511855, 0.322564326204125, 0.427764047684999, 0.633346735357506, 0.184132265987251, 0.419530069293309, 0.677312347753029, 0.848564991204658, 0.416721752433328, 0.739174170698364, 0.073902108569917, 0.543739094996271, 0.339099542453608, 0.447096506785153, 0.157967379648128, 0.153618472458268, 0.488213807928168, 0.639204733387567, 0.853497390377125, 0.923751397099138, 0.725328064366064, 0.338742925499795, 0.411127592284021, 0.614523420991109, 0.746495239112596, 0.486074712458846, 0.673266205343198, 0.305129440204979, 0.152273160713366, 0.414683294845447, 0.048226246479783, 0.359688061543675, 0.372884661794097, 0.941698247158122, 0.319133621569881, 0.854110174457866, 0.407944326379071, 0.881508028708691, 0.918672739735375, 0.030968598377933, 0.567394455317863, 0.885031377294322, 0.795017002347453, 0.398280563315029, 0.622495752319105, 0.305857937634608, 0.594543998702211, 0.454503908099653, 0.43800338083708 , 0.840089897438398, 0.771623297633423, 0.938080644307669, 0.030076405448699, 0.366199123494718, 0.575838134091073, 0.269993529162751, 0.304847765866286, 0.296151692492209, 0.233120797908678, 0.728068104255968, 0.205611937046943, 0.138203545229527, 0.189881421649331, 0.841213292701945, 0.198196203780208, 0.826948459157772, 0.676250124347325, 0.848155013485828, 0.978497904088677, 0.872051613641737, 0.639694262066783, 0.510553460583134, 0.715777284759136, 0.32169308931643 , 0.384125788762886, 0.519969599671708, 0.115623542800208, 0.851783095144246, 0.035133755582779, 0.446925409147525, 0.707531737188635, 0.509306095498739, 0.708009698579086, 0.909215448079746, 0.818132011774798, 0.311858526091468, 0.251244005139463, 0.538327831666644, 0.017845110915726, 0.515593759159651, 0.016808379113491, 0.612632900497085, 0.663654291497305, 0.325150983071195, 0.487228745016513, 0.827683997959527, 0.855897038738031, 0.499675548290629, 0.894090783226805, 0.48012205386361 , 0.173994329061333, 0.138695264158256, 0.91656800653439 , 0.181687754133206, 0.567850567390956, 0.20106107674288 , 0.346576746926752, 0.10501264034626 , 0.006939690800771, 0.416598293234668, 0.644943120905739, 0.648065723032423, 0.073259771969248, 0.585074759212088, 0.036253566938045, 0.769073168086859, 0.091488936654992, 0.658841701234514, 0.880976006039527, 0.48024319554174 , 0.927177795813044, 0.038448547787501, 0.515933752433032, 0.178307138179843, 0.238030747304187, 0.830152166595131, 0.677672430824361, 0.806627867202981, 0.754463899293375, 0.040151358312631, 0.581928277391582, 0.155818062465667, 0.902278682719631, 0.548872070785091, 0.832303271885468, 0.485158468356476, 0.333846478537863, 0.371795205820647, 0.045251313787544, 0.553739603107236, 0.270311836103049, 0.802448207167068, 0.381825672505056, 0.5801987865483  , 0.372864283888511, 0.855628774420025, 0.211748107572083, 0.663183471119694, 0.13759764841393 , 0.604562770548906, 0.215348633244505, 0.299268829250946, 0.393330121325129, 0.892995491689337, 0.721896701431277, 0.912448869989175, 0.363584144617152, 0.60714792698288 , 0.665136855720383, 0.390044043096082, 0.443087863196345, 0.222758426827245, 0.354968105124743, 0.373955617679775, 0.811868311065631, 0.787121511357079, 0.069684579114568, 0.696041589565845, 0.300063883334412, 0.114176527676988, 0.757833040996555, 0.536603088025427, 0.672418413178956, 0.3721166534874  , 0.380915925291777, 0.748153052512397, 0.30587193831285 , 0.491472571982712, 0.752156004886554, 0.219245471826598, 0.929804837662696, 0.817675871515081, 0.24508402947411 , 0.380523526477658, 0.597445391493939, 0.875893164176974, 0.282397344081379, 0.34195599691193 , 0.850432704943723, 0.988273800387934, 0.145013881622962, 0.779097052415496, 0.571665949345111, 0.528688058059305, 0.394989412916223, 0.401464199237052, 0.730248551620968};

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
        /*
         We want to select a decomposition continuously, among the many possible ones, in a family
         of linear programs whose constraint bounds vary linearly. Options :
         - Choosing the minimal decomposition w.r.t. lexicographic ordering selects in a Lipschitz
         manner, but it is a pain to implement, and may have significantly higher cost.
         - Other abstract options include the Steiner point of a convex body, also Lipschitz and with
         likely a better constant, but it is uncomputable realistically.
         - Minimizing a linear form whose kernel contains no edge of the polytopes involved in the
         linear programs. The Lipschitz constant depends on the angle between the kernel and the
         edges in question. The set of edges is finite would be a pain to compute. Choosing a random
         linear form works with probability one (as well as choosing a linear form with coefficients
         (pi^N(e))_(e in Z^d) where pi can be replaced with any transcendental number, and
         N(e) is a unique index). The Lipschtitz constants may be poor, but this is the simplest and
         
         Notes :
         - Objectives of the form a+l(ee^t), where a is a constant and l is a linear form over
         symmetric matrices, are constant over the set of solutions.
         - More generally, objectives which are too simple, and with too many symmetries, led to
         discontinuities in the decomposition, which means that an edge was contained of the kernel.
         - Objective is maximized.
         */
        Scalar offset_weight_[nsupport_max]; // Some function of the offset whose weighted sum is minimized
        for(int k=0; k<nsupport; ++k){
            // Normalize offset to have a positive first coefficient. (Opposite offsets are equivalent)
            OffsetT * offset = offsets_[k];
            OffsetT sign=0;
            for(int i=0; i<ndim; ++i){if(offset[i]!=0) {sign=2*(offset[i]>0)-1;break;}}
            for(int i=0; i<ndim; ++i){offset[i]*=sign;}
            
            // Build some weight associated to the offset
            int hash = 0;
            for(int i=0; i<ndim; ++i){hash = hash_base*hash + offset[i];}
            hash %= hash_mod;
            if(hash<0) {hash+=hash_mod;} // Positive residue
            offset_weight_[k] = hash_rnd[hash]-0.5; // -0.5 changes nothing
        }
        Scalar objective[d_max];
        for(int k=0; k<d; ++k) {
            objective[k] = offset_weight_[symdim+k];
            for(int l=0; l<symdim; ++l){
                objective[k] += data.kkt_constraints[k][l]*offset_weight_[l];
            }
        }
#endif // Normalize solution

#ifdef SIMPLEX_VERBOSE // Simplex algorithm
		SimplexData sdata;
		sdata.n = d; // number of variables (all positive)
		sdata.m = symdim; // Number of constraints (all positivity constraits)
        sdata.v = 0; // Arbitrary value, added to the objective
		Scalar opt[nsupport_max]; // optimal solution
        
        Scalar weight_max=0;
        for(int i=0; i<symdim; ++i) weight_max = max(weight_max,abs(weights[i]));
        // Relax a bit the positivity constraint to ensure positivity
        Scalar wfeas = 0.2*weight_max*SIMPLEX_TOL; // 0;
        /* With wfeas = 0, we do see simplex failures with doubles in some edge cases,
        typically Hooke TTI tensors rotated almost along some axis. For a matrix with approx unit
        entries, reconstruction error is approx 4e-13 with wfeas=0,
        and 8e-12 with wfeas = 0.2*weight_max*SIMPLEX_TOL. Sounds reasonable.*/
        
//		for(int i=0; i<symdim;++i) printf(" %f",weights[i]); printf("\n");

		for(int i=0; i<symdim; ++i){ // Specify the constraints
			for(int j=0; j<d; ++j){
				sdata.A[i][j] = data.kkt_constraints[j][i];}
			sdata.b[i] = weights[i] + wfeas;
		}

		for(int i=0; i<sdata.n; ++i){
			sdata.c[i] = // Linear form to be maximized
			#if GEOMETRY6_NORMALIZE_SOLUTION
			objective[i]; // Select Lipschitz continuous representative
			#else
			1; // Arbitrary
			#endif
		}
        
        
		Scalar value = simplex(sdata,opt);
#ifdef DOUBLE
        /* +/-Infinity values mean failure (unbounded or infeasible problem).
         With doubles, we relax a bit the positivity constraints, and hope for the best.
         With floats we detect the invalid decompositions a posteriori, and recompute using doubles.
         */
        if(isinf(value)){
            wfeas*=5;
            for(int i=0; i<symdim; ++i){sdata.b[i] = weights[i] + wfeas;}
            value = simplex(sdata,opt);
        }
#endif
//		assert(!isinf(value));

/*		std::cout << "Value of the linear program " << value << std::endl;
		if(isinf(value)) {std::cout << opt[0] << std::endl;}
		std::cout << "state.vertex : " << state.vertex << std::endl;*/
        
		Scalar sol[nsupport_max]; // solution
		for(int i=0; i<d; ++i){sol[symdim+i] = opt[i];}
		for(int i=0; i<symdim; ++i){sol[i] = opt[d+i];}
        /*
        std::cout ExportArrayArrow(sol) << std::endl;
        std::cout << "max min sol : "
        << *std::max_element(std::begin(sol),std::end(sol)) << " "
        << *std::min_element(std::begin(sol),std::end(sol)) << " "
        ExportVarArrow(sdata.v)
        ExportArrayArrow(objective)
        <<" Bye" << std::endl;*/

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
