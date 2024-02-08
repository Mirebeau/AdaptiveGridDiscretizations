#pragma once
/*
 Petar 'PetarV' Velickovic
 Algorithm: Simplex Algorithm
 
 
 Minor modifications by J.M. Mirebeau, 2020
 (avoid use of global variables, C compilation, remove useless headers, tol auxiliary pb,
 silent call)
*/

#ifdef CUDA_DEVICE
#define INFINITY (1./0.)
#else
// Those are actually needed, but explicit inclusion bugs the nvcc cuda compiler
#include <math.h>
#include <iostream>
#endif

#ifndef SIMPLEX_VERBOSE
#define SIMPLEX_VERBOSE 1
#define SIMPLEX_MAX_N 1001 // Max number of variables
#define SIMPLEX_MAX_M 1001 // Max number of constraints
typedef double Scalar;
#endif

//#ifndef SIMPLEX_TOL
//#define SIMPLEX_TOL 0
// Typical values : 1e-5 for float, 2e-14 for double
//#endif

void swap_int(int & a, int & b){const int c = a; a=b; b=c;}

/*
 The Simplex algorithm aims to solve a linear program - optimising a linear function subject
 to linear constraints. As such it is useful for a very wide range of applications.
 
 N.B. The linear program has to be given in *slack form*, which is as follows:
 maximise
     c_1 * x_1 + c_2 * x_2 + ... + c_n * x_n + v
 subj. to
     a_11 * x_1 + a_12 * x_2 + ... + a_1n * x_n + b_1 = s_1
     a_21 * x_1 + a_22 * x_2 + ... + a_2n * x_n + b_2 = s_2
     ...
     a_m1 * x_1 + a_m2 * x_2 + ... + a_mn * x_n + b_m = s_m
 and
     x_1, x_2, ..., x_n, s_1, s_2, ..., s_m >= 0
 
 Every linear program can be translated into slack form; the parameters to specify are:
     - the number of variables, n, and the number of constraints, m;
     - the matrix A = [[A_11, A_12, ..., A_1n], ..., [A_m1, A_m2, ..., A_mn]];
     - the vector b = [b_1, b_2, ..., b_m];
     - the vector c = [c_1, c_2, ..., c_n] and the constant v.
 
 Complexity:    O(m^(n/2)) worst case
                O(n + m) average case (common)
*/

struct SimplexData {
	int n, m;
	Scalar A[SIMPLEX_MAX_M][SIMPLEX_MAX_N], b[SIMPLEX_MAX_M], c[SIMPLEX_MAX_N], v;
	int N[SIMPLEX_MAX_N], B[SIMPLEX_MAX_M]; // nonbasic & basic
};

// pivot yth variable around xth constraint
inline void pivot(int x, int y, SimplexData & d)
{
#if SIMPLEX_VERBOSE
    printf("Pivoting variable %d around constraint %d.\n", y, x);
#endif
    // first rearrange the x-th row
    for (int j=0;j<d.n;j++)
    {
        if (j != y)
        {
			d.A[x][j] /= -d.A[x][y];
        }
    }
    d.b[x] /= -d.A[x][y];
    d.A[x][y] = 1.0 / d.A[x][y];
    
    // now rearrange the other rows
    for (int i=0;i<d.m;i++)
    {
        if (i != x)
        {
            for (int j=0;j<d.n;j++)
            {
                if (j != y)
                {
                    d.A[i][j] += d.A[i][y] * d.A[x][j];
                }
            }
            d.b[i] += d.A[i][y] * d.b[x];
            d.A[i][y] *= d.A[x][y];
        }
    }
    
    // now rearrange the objective function
    for (int j=0;j<d.n;j++)
    {
        if (j != y)
        {
            d.c[j] += d.c[y] * d.A[x][j];
        }
    }
    d.v += d.c[y] * d.b[x];
    d.c[y] *= d.A[x][y];
    
    // finally, swap the basic & nonbasic variable
    swap_int(d.B[x], d.N[y]);
}

// Run a single iteration of the simplex algorithm.
// Returns: 0 if OK, 1 if STOP, -1 if UNBOUNDED
inline int iterate_simplex(SimplexData & d)
{
#if SIMPLEX_VERBOSE
    printf("--------------------\n");
    printf("State:\n");
    printf("Maximise: ");
    for (int j=0;j<d.n;j++) printf("%lfx_%d + ", d.c[j], d.N[j]);
    printf("%lf\n", d.v);
    printf("Subject to:\n");
    for (int i=0;i<d.m;i++)
    {
        for (int j=0;j<d.n;j++) printf("%lfx_%d + ", d.A[i][j], d.N[j]);
        printf("%lf = x_%d\n", d.b[i], d.B[i]);
    }

    // getchar(); // uncomment this for debugging purposes!
#endif
	
    int ind = -1, best_var = -1;
    for (int j=0;j<d.n;j++)
    {
        if (d.c[j] > SIMPLEX_TOL)
        {
            if (best_var == -1 || d.N[j] < ind)
            {
                ind = d.N[j];
                best_var = j;
            }
        }
    }

/*    for(int j=0; j<d.n; ++j) printf("%f ",1000*d.c[j]); printf("\n");
    for(int j=0; j<d.n; ++j) printf("%i ",d.c[j]>SIMPLEX_TOL); printf("\n");
    printf("best_var %i", best_var);*/

    if (ind == -1) return 1;
    
    Scalar max_constr = INFINITY;
    int best_constr = -1;
    for (int i=0;i<d.m;i++)
    {
        if (d.A[i][best_var] < -SIMPLEX_TOL)
        {
            Scalar curr_constr = -d.b[i] / d.A[i][best_var];
            if (curr_constr < max_constr)
            {
                max_constr = curr_constr;
                best_constr = i;
            }
        }
    }
    if (isinf(max_constr)) return -1;
    else pivot(best_constr, best_var,d);
    
    return 0;
}

// (Possibly) converts the LP into a slack form with a feasible basic solution.
// Returns 0 if OK, -1 if INFEASIBLE
inline Scalar initialise_simplex(SimplexData & d)
{
    int k = -1;
    Scalar min_b = -1;
    for (int i=0; i<d.m; i++)
    {
        if (k == -1 || d.b[i] < min_b)
        {
            k = i;
            min_b = d.b[i];
        }
    }
    
    if (d.b[k] >= 0) // basic solution feasible!
    {
        for (int j=0;j<d.n;j++) d.N[j] = j;
        for (int i=0;i<d.m;i++) d.B[i] = d.n + i;
        return 0;
    }
    
    // generate auxiliary LP
    d.n++;
    for (int j=0;j<d.n;j++) d.N[j] = j;
    for (int i=0;i<d.m;i++) d.B[i] = d.n + i;
    
    // store the objective function
    Scalar c_old[SIMPLEX_MAX_N];
    for (int j=0;j<d.n-1;j++) c_old[j] = d.c[j];
    Scalar v_old = d.v;
    
    // aux. objective function
    d.c[d.n-1] = -1;
    for (int j=0;j<d.n-1;j++) d.c[j] = 0;
    d.v = 0;
    // aux. coefficients
    for (int i=0;i<d.m;i++) d.A[i][d.n-1] = 1;
    
    // perform initial pivot
    pivot(k, d.n - 1,d);
    
    // now solve aux. LP
    int code;
    while (!(code = iterate_simplex(d))); 
    
    assert(code == 1); // aux. LP cannot be unbounded!!!
    
	/* The value of the auxiliary problem is non-positive.
	 It is zero if feasible, and negative if infeasible. (Returned value in that case.)*/
//	std::cout << "Value of initial problem " << d.v << std::endl;
	if(d.v < -SIMPLEX_TOL){--d.n; return -d.v;}

    int z_basic = -1;
    for (int i=0;i<d.m;i++)
    {
        if (d.B[i] == d.n - 1)
        {
            z_basic = i;
            break;
        }
    }
    
    // if x_n basic, perform one degenerate pivot to make it nonbasic
    if (z_basic != -1) pivot(z_basic, d.n - 1, d);

    int z_nonbasic = -1;
    for (int j=0;j<d.n;j++)
    {
        if (d.N[j] == d.n - 1)
        {
            z_nonbasic = j;
            break;
        }
    }
    assert(z_nonbasic != -1);
    
    for (int i=0;i<d.m;i++)
    {
        d.A[i][z_nonbasic] = d.A[i][d.n-1];
    }
    swap_int(d.N[z_nonbasic], d.N[d.n - 1]);
    
    d.n--;
    for (int j=0;j<d.n;j++) if (d.N[j] > d.n) d.N[j]--;
    for (int i=0;i<d.m;i++) if (d.B[i] > d.n) d.B[i]--;
    
    for (int j=0;j<d.n;j++) d.c[j] = 0;
    d.v = v_old;
    
    for (int j=0;j<d.n;j++)
    {
        bool ok = false;
        for (int jj=0;jj<d.n;jj++)
        {
            if (j == d.N[jj])
            {
                d.c[jj] += c_old[j];
                ok = true;
                break;
            }
        }
        if (ok) continue;
        for (int i=0;i<d.m;i++)
        {
            if (j == d.B[i])
            {
                for (int jj=0;jj<d.n;jj++)
                {
                    d.c[jj] += c_old[j] * d.A[i][jj];
                }
                d.v += c_old[j] * d.b[i];
                break;
            }
        }
    }
    
    return 0;
}

// Runs the simplex algorithm to optimise the LP.
// If infeasible, returns -Infinity.
// If unbounded, returns +Infinity.
Scalar simplex(SimplexData & d, Scalar ret[SIMPLEX_MAX_M+SIMPLEX_MAX_N])
{
	const double infeasible = initialise_simplex(d);
    if (infeasible) { // infeasible
//        printf("infeasible ? %f, %i\n",1000000*infeasible, infeasible!=0);
		ret[0] = infeasible; // How much to offset the constraint values to be feasible
		return -INFINITY;
    }
    
    int code;
    while (!(code = iterate_simplex(d)));
    
	if (code == -1) { return INFINITY;} // unbounded
    
    for (int j=0;j<d.n;j++){ret[d.N[j]] = 0;}
    for (int i=0;i<d.m;i++){ret[d.B[i]] = d.b[i];}
    
	return d.v;
}

