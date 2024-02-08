// templates allow the compiler to determine stack depth statically.

template<int d> struct linprog_templated {
	static int go(FLOAT halves[], 
	int istart, 
	int m,  	
	FLOAT n_vec[], 	
	FLOAT d_vec[], 	
//	int d, 		
	FLOAT opt[],	
	FLOAT work[], 	
	int next[], 	
	int prev[], 	
	int max_size);
};

template<> struct linprog_templated<0> {
	static int go(FLOAT halves[], 
	int istart, 
	int m,  	
	FLOAT n_vec[], 	
	FLOAT d_vec[], 	
//	int d, 		
	FLOAT opt[],	
	FLOAT work[], 	
	int next[], 	
	int prev[], 	
	int max_size){return ERRORED;} // Actually stops at dimension 1
};

template<int d>
int linprog_templated<d>::go
(FLOAT halves[], /* halves  --- half spaces */
	int istart,     /* istart  --- should be zero
				 unless doing incremental algorithm */
	int m,  	/* m       --- terminal marker */
	FLOAT n_vec[], 	/* n_vec   --- numerator vector */
	FLOAT d_vec[], 	/* d_vec   --- denominator vector */
//	int d, 		/* d       --- projective dimension */
	FLOAT opt[],	/* opt     --- optimum */
	FLOAT work[], 	/* work    --- work space (see below) */
	int next[], 	/* next    --- array of indices into halves */
	int prev[], 	/* prev    --- array of indices into halves */
	int max_size) 	/* max_size --- size of halves array */
/*
**
** half-spaces are in the form
** halves[i][0]*x[0] + halves[i][1]*x[1] + 
** ... + halves[i][d-1]*x[d-1] + halves[i][d]*x[d] >= 0
**
** coefficients should be normalized
** half-spaces should be in random order
** the order of the half spaces is 0, next[0] next[next[0]] ...
** and prev[next[i]] = i
**
** halves[max_size][d+1]
**
** the optimum has been computed for the half spaces
** 0 , next[0], next[next[0]] , ... , prev[istart]
** the next plane that needs to be tested is istart
**
** m is the index of the first plane that is NOT on the list
** i.e. m is the terminal marker for the linked list.
**
** the objective function is dot(x,nvec)/dot(x,dvec)
** if you want the program to solve standard d dimensional linear programming
** problems then n_vec = ( x0, x1, x2, ..., xd-1, 0)
** and           d_vec = (  0,  0,  0, ...,    0, 1)
** and halves[0] = (0, 0, ... , 1)
**
** work points to (max_size+3)*(d+2)*(d-1)/2 FLOAT space
*/
{
	if(debug_print){
		printf("Entering linprog %d\n", d);
	}

	int status;
	int i, j, imax;
	#ifdef CHECK
	int k;
	#endif
	FLOAT *new_opt, *new_n_vec, *new_d_vec,  *new_halves, *new_work;
	FLOAT *plane_i;
	FLOAT val;

	if(d==1 && m!=0) {
		return(lp_base_case((FLOAT (*)[2])halves,m,n_vec,d_vec,opt,
			next,prev,max_size));
	} else {
		int d_vec_zero;
		val = 0.0;
		for(j=0; j<=d; j++) val += d_vec[j]*d_vec[j];
		d_vec_zero = (val < (d+1)*EPS*EPS);

/* find the unconstrained minimum */
		if(!istart) {
			status = lp_no_constraints(d,n_vec,d_vec,opt); 
		} else {
			status = MINIMUM;
		}
		if(m==0) return(status);
/* allocate memory for next level of recursion */
		new_opt = work;
		new_n_vec = new_opt + d;
		new_d_vec = new_n_vec + d;
		new_halves = new_d_vec + d;
		new_work = new_halves + max_size*d;
		for(i = istart; i!=m; i=next[i]) {
#ifdef CHECK
			if(i<0 || i>=max_size) {
				printf("index error\n");
				EXIT1;
			}
#endif
/* if the optimum is not in half space i then project the problem
** onto that plane */
			plane_i = halves + i*(d+1);
/* determine if the optimum is on the correct side of plane_i */
			val = 0.0;
			for(j=0; j<=d; j++) val += opt[j]*plane_i[j];
			if(val<-(d+1)*EPS) {
/* find the largest of the coefficients to eliminate */
			    findimax(plane_i,d,&imax);
/* eliminate that variable */
			    if(i!=0) {
				FLOAT fac;
				fac = 1.0/plane_i[imax];
				for(j=0; j!=i; j=next[j]) {
					FLOAT *old_plane, *new_plane;
					int k;
					FLOAT crit;

					old_plane = halves + j*(d+1);
					new_plane = new_halves + j*d;
					crit = old_plane[imax]*fac;
					for(k=0; k<imax; k++)  {
						new_plane[k] = old_plane[k] - plane_i[k]*crit;
					}
					for(k=imax+1; k<=d; k++)  {
						new_plane[k-1] = old_plane[k] - plane_i[k]*crit;
					}
				}
			    }
/* project the objective function to lower dimension */
			    if(d_vec_zero) {
				vector_down(plane_i,imax,d,n_vec,new_n_vec);
				for(j=0; j<d; j++) new_d_vec[j] = 0.0;
			    } else {
			        plane_down(plane_i,imax,d,n_vec,new_n_vec);
			        plane_down(plane_i,imax,d,d_vec,new_d_vec);
			    }
/* solve sub problem */
			    status = linprog_templated<d-1>::go(new_halves,0,i,new_n_vec,
			    new_d_vec,/*d-1,*/new_opt,new_work,next,prev,max_size);
/* back substitution */
			    if(status!=INFEASIBLE) {
				    vector_up(plane_i,imax,d,new_opt,opt);
				{
/* in line code for unit */
				FLOAT size;
				size = 0.0;
				for(j=0; j<=d; j++) 
				    size += opt[j]*opt[j];
				size = 1.0/sqrt(size);
				for(j=0; j<=d; j++)
				    opt[j] *= size;
				}
			    } else {
				    return(status);
			    }
/* place this offensive plane in second place */
			    i = move_to_front(i,next,prev,max_size);
#ifdef CHECK
			    j=0;
			    while(1) {
/* check the validity of the result */
				val = 0.0;
				for(k=0; k<=d; k++) 
					val += opt[k]*halves[j*(d+1)+k];
				if(val <-(d+1)*EPS) {
				    printf("error\n");
					EXIT1;
				}
				if(j==i)break;
				j=next[j];
			    }
#endif
			} 
		}
 		return(status);
	}
}