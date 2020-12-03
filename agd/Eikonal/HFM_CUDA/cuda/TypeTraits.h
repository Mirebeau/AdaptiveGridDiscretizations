#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "static_assert.h"

#ifndef Scalar_macro
typedef float Scalar;
#endif

#ifndef Int_macro
typedef int Int;
const Int Int_Max = 2147483647;
#endif

#ifndef OffsetT_macro
typedef int OffsetT;
#endif

typedef unsigned char BoolPack;
typedef unsigned char BoolAtom;

// ----------- Flags -------------

/// A positive value may cause debug messages to be printed
#ifndef debug_print_macro
const int debug_print = 0;
#endif

/*The solution is provided on a two level block grid, defined by shape_o and shape_i.*/
#define bilevel_grid_macro
//#define local_i_macro true // shape_i shared variables are a local block in space

/** In multi-precision, we address float roundoff errors 
by representing a real in the form u+uq*multip_step, where
u is a float, uq is an integer, and multip_step is a constant.*/
#ifndef multiprecision_macro
#define multiprecision_macro false
#endif

#if multiprecision_macro
#define MULTIP(...) __VA_ARGS__
#define NOMULTIP(...) 
#else
#define MULTIP(...) 
#define NOMULTIP(...) __VA_ARGS__
#endif

/** Wether the finite differences scheme uses any symmetric offsets.
(Possibly in addition to forward offsets.) */
#ifndef nsym_macro
#define nsym_macro true
#endif

#if nsym_macro
#define NSYM(...) __VA_ARGS__
#else
#define NSYM(...) 
#endif

/** Number of schemes of which we take the max of min*/
#ifndef nmix_macro
#define nmix_macro 1
#endif

#if nmix_macro >= 2
#define MIX(...) __VA_ARGS__
#define NOMIX(...) 
#else
#define MIX(...) 
#define NOMIX(...) __VA_ARGS__
#endif

// Wether the mix (min/max) depends on the geometry
#ifndef adaptive_mix_macro
#define adaptive_mix_macro false
#endif

#if adaptive_mix_macro
#define ADAPTIVE_MIX(...) __VA_ARGS__
#else
#define ADAPTIVE_MIX(...) 
#endif

// Wether the mix depends on the step (e.g. successively min-max-min)
#ifndef alternating_mix_macro
#define alternating_mix_macro false
#endif

#if alternating_mix_macro
#define ALTERNATING_MIX(...) __VA_ARGS__
#else
#define ALTERNATING_MIX(...) 
#endif

/** strict_iter_i_macro = 1 causes the input and output values 
within a block to be stored separately and synced at the end of 
each iteration*/
#ifndef strict_iter_i_macro
#define strict_iter_i_macro (multiprecision_macro || (nmix_macro >= 2) )
#endif

/** strict_iter_o_macro causes a similar behavior, but for the global iterations */ 
#ifndef strict_iter_o_macro
#define strict_iter_o_macro multiprecision_macro
#endif

#if strict_iter_o_macro
#define STRICT_ITER_O(...) __VA_ARGS__
#else 
#define STRICT_ITER_O(...) 
#endif


/** Source factorization allows to improve the solution accuracy by subtracting, before 
the finite differences computation, a expansion of the solution near the source.*/
#ifndef factor_macro
#define factor_macro false
#endif

#if factor_macro
#define FACTOR(...) __VA_ARGS__
#define NOFACTOR(...) 
#else
#define FACTOR(...) 
#define NOFACTOR(...) __VA_ARGS__
#endif

/** A drift can be introduced in some schemes */
#ifndef drift_macro
#define drift_macro false
#endif

#if drift_macro
#define DRIFT(...) __VA_ARGS__
#else
#define DRIFT(...) 
#endif

/** factorization and drift act similarly, by introducing a shift in the finite differences*/
#define shift_macro (factor_macro+drift_macro)

#if shift_macro
#define SHIFT(...) __VA_ARGS__
#else
#define SHIFT(...) 
#endif

/** The second order scheme allows to improve accuracy*/
#ifndef order2_macro
#define order2_macro false
#endif

#if order2_macro
#define ORDER2(...) __VA_ARGS__
#else
#define ORDER2(...) 
#endif

// An inconclusive experiment on setting the switching threshold adaptively
#ifndef order2_threshold_weighted_macro
#define order2_threshold_weighted_macro false
#endif

#ifndef order2_causal_macro
#define order2_causal_macro true
#endif
/** Curvature penalized models have share a few specific features : 
relaxation parameter, periodic boundary condition, xi and kappa constants, 
position dependent metric. */
#ifndef curvature_macro
#define curvature_macro false
#endif

#if curvature_macro
#define CURVATURE(...) __VA_ARGS__
#define periodic_macro true 
#else
#define CURVATURE(...) 
#endif

#if curvature_macro

#define local_scheme_macro true
#define decomp_v_macro true

#define geom_macro (xi_var_macro + kappa_var_macro + 2*theta_var_macro)
const Int geom_size = geom_macro;

#if xi_var_macro
#define XI_VAR(...) __VA_ARGS__
#else
#define XI_VAR(...) 
#endif

#if kappa_var_macro
#define KAPPA_VAR(...) __VA_ARGS__
#else
#define KAPPA_VAR(...) 
#endif

#endif // curvature_macro

/** Some discretization schemes may be exported to speed up iterations*/
#ifndef export_scheme_macro
#define export_scheme_macro false
#endif

#if export_scheme_macro
#define EXPORT_SCHEME(...) __VA_ARGS__
#else
#define EXPORT_SCHEME(...) 
#endif

#ifndef import_scheme_macro
#define import_scheme_macro false
#endif

#if import_scheme_macro
#define IMPORT_SCHEME(...) __VA_ARGS__
#else
#define IMPORT_SCHEME(...) 
#endif

#define io_scheme_macro (export_scheme_macro || import_scheme_macro)

#if io_scheme_macro 
#define IO_SCHEME(...) __VA_ARGS__
#else
#define IO_SCHEME(...)
#endif

/** Wether the model depends on local geometrical data, aside from the cost function.*/
#ifndef geom_macro
#define geom_macro true
#endif

#if geom_macro
#define GEOM(...) __VA_ARGS__
#else 
#define GEOM(...)
#endif

/** Wether the scheme depends on the position in the domain*/
#ifndef local_scheme_macro
#define local_scheme_macro false
#endif

#if local_scheme_macro
#define LOCAL_SCHEME(...) __VA_ARGS__
#else
#define LOCAL_SCHEME(...) 
#endif

/** Apply periodic boundary conditions on some of the axes.*/
#ifndef periodic_macro
#define periodic_macro false
#endif

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
#define APERIODIC(...) 
#else
#define PERIODIC(...) 
#define APERIODIC(...) __VA_ARGS__
#endif

/** Since the schemes are monotone (except with the second-order enhancement), and we start 
from a super-solution, the solution values should be decreasing as the iterations proceed. 
We can take advantage of this property to achieve better robustness of the solver. 
(Otherwise, floating point roundoff errors often cause multiple useless additional iterations)
*/
#ifndef decreasing_macro
#define decreasing_macro true
#endif

#if decreasing_macro
#define DECREASING(...) __VA_ARGS__
#else 
#define DECREASING(...) 
#endif

/** The implemented schemes are causal except in the following cases:
- second order enhancement (mild non-causality)
- source factorization (mild non-causality)
- drift (strong non-causality, but possibly not such an issue due to the large block size)
We can take advantage of this property to improve computation time, by freezing computations
in the far future until the past has suitably converged. For that purpose, a target number of 
active blocks is specified. Blocks are then frozen, or not, depending on their minChg
(minimal change) value.
*/
#ifndef minChg_freeze_macro
#define minChg_freeze_macro false
#endif

#if minChg_freeze_macro
#define MINCHG_FREEZE(...) __VA_ARGS__
#else
#define MINCHG_FREEZE(...) 
#endif

/** The pruning macro maintains a list of the active nodes at any time.
It is slightly more flexible than the default method, and needed for minChg_freeze_macro
to take effect.
*/
#ifndef pruning_macro
#define pruning_macro minChg_freeze_macro
#endif

#if pruning_macro
#define PRUNING(...) __VA_ARGS__
#else
#define PRUNING(...) 
#endif

/* The fast iterative method is an alternative to the AGSI, which limits the front width
(to a single pixel in the original implementation). 
*/
#ifndef fim_macro
#define fim_macro false
#endif

#if fim_macro
#define FIM(...) __VA_ARGS__
#else
#define FIM(...) 
#endif

/** The following macros are for the extraction of the upwind geodesic flow. */
// weights
#ifndef flow_weights_macro 
#define flow_weights_macro false
#endif

#if flow_weights_macro
#define FLOW_WEIGHTS(...) __VA_ARGS__
#else
#define FLOW_WEIGHTS(...) 
#endif

// weightsum
#ifndef flow_weightsum_macro
#define flow_weightsum_macro false
#endif

#if flow_weightsum_macro
#define FLOW_WEIGHTSUM(...) __VA_ARGS__
#else
#define FLOW_WEIGHTSUM(...)
#endif

// offets
#ifndef flow_offsets_macro
#define flow_offsets_macro false
#endif

#if flow_offsets_macro
#define FLOW_OFFSETS(...) __VA_ARGS__
#else
#define FLOW_OFFSETS(...) 
#endif

// indices
#ifndef flow_indices_macro
#define flow_indices_macro false
#endif

#if flow_indices_macro
#define FLOW_INDICES(...) __VA_ARGS__
#else
#define FLOW_INDICES(...) 
#endif

// vector
#ifndef flow_vector_macro
#define flow_vector_macro false
#endif

#if flow_vector_macro
#define FLOW_VECTOR(...) __VA_ARGS__
#else
#define FLOW_VECTOR(...) 
#endif

// Any of these
#ifndef flow_macro // Compute the upwind geodesic flow, in one form or another
#define flow_macro (flow_weights_macro || flow_weightsum_macro \
	|| flow_offsets_macro || flow_indices_macro || flow_vector_macro) 
#endif 

#if flow_macro
#define FLOW(...) __VA_ARGS__
#define NOFLOW(...) 
#else 
#define FLOW(...)
#define NOFLOW(...) __VA_ARGS__
#endif
#if order2_macro || flow_macro
#define ORDER2_OR_FLOW(...) __VA_ARGS__
#else
#define ORDER2_OR_FLOW(...) 
#endif

/** Isotropic and diagonal metrics have special treatment, since they and offsets,
and weights in the anisotropic case.*/
#ifndef adaptive_weights_macro
#define adaptive_weights_macro true
#endif

#if adaptive_weights_macro
#define ADAPTIVE_WEIGHTS(...) __VA_ARGS__
#else 
#define ADAPTIVE_WEIGHTS(...)
#endif

#ifndef adaptive_offsets_macro
#define adaptive_offsets_macro true
#endif

#if adaptive_offsets_macro
#define ADAPTIVE_OFFSETS(...) __VA_ARGS__
#else
#define ADAPTIVE_OFFSETS(...)
#endif

/** Dealing with walls in the domain */
#ifndef walls_macro
#define walls_macro false
#endif

#if walls_macro
#define WALLS(...) __VA_ARGS__
#else
#define WALLS(...) 
#endif

/** Method used for sorting the values before the update*/
#ifndef merge_sort_macro
#define merge_sort_macro false
#endif

#ifndef network_sort_macro
#define network_sort_macro false
#endif

/** Apply Voronoi's decomposition to a relaxation of the rank one matrix v v^T*/
#ifndef decomp_v_macro
#define decomp_v_macro false
#endif

// Eliminate offsets which deviate too much from the direction of v
#ifndef decomp_v_align_macro
#define decomp_v_align_macro true
#endif

#if decomp_v_align_macro
#define DECOMP_V_ALIGN(...) __VA_ARGS__
#else
#define DECOMP_V_ALIGN(...) 
#endif
