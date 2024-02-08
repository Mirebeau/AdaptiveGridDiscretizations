//
//  Siedel_Hohmeyer_LinProg.h
//  This header collects the files for
//  Michael Hohmeyer's implementation of R. Siedel's linear programming solver

// A few, very minor, adjustments had to be made to the source.

// Original source :
// https://github.com/cgrushko/seidel-lp-solver

// Academic paper :
// Seidel, R. (1991), "Small-dimensional linear programming and convex hulls made easy", Discrete & Computational Geometry 6 (1): 423–434, doi:10.1007/BF02574699

//  Voronoi45
//
//  Created by Jean-Marie Mirebeau on 15/02/2018.
//  Copyright © 2018 Jean-Marie Mirebeau. All rights reserved.
//

#ifndef Siedel_Hohmeyer_LinProg_h
#define Siedel_Hohmeyer_LinProg_h

// #define DOUBLE // Define this macro for double

/* A non-recursive version is used if this variable is set to a positive value
Can be better on gpus, probably not on cpus */
#ifndef LINPROG_DIMENSION_MAX
#define LINPROG_DIMENSION 0
#endif

#ifdef CUDA_DEVICE // math.h and exit(1) are not available on cuda
#define NOINCLUDE_MATH_H
#define NOEXIT1
#endif


// Define this macro to NOT include include math.h 
#ifndef NOINCLUDE_MATH_H
#include <math.h>
#endif

#include "lp.h"

// Define this macro to NOT use exit(1) 
#ifdef NOEXIT1
#define EXIT1 return ERRORED;
#else
#define EXIT1 exit(1);
#endif

#include "localmath.h"
#include "tol.h"

#include "unit2.c"
#include "linprog.c"
#include "lp_base_case.c"
#include "vector_up.c"
//#include "linprog_templated.h"


#if LINPROG_DIMENSION_MAX==0
#define linprog(v,istart, n,num,den,dim,opt,work,next,prev,max_size)  \
linprog_recursive(v,istart, n,num,den,dim,opt,work,next,prev,max_size)
#else
#include "linprog_flattened.h"
#define linprog(v,istart, n,num,den,dim,opt,work,next,prev,max_size)  \
linprog_flattened(v,istart, n,num,den,dim,opt,work,next,prev,max_size)
#endif

#endif /* Siedel_Hohmeyer_LinProg_h */
