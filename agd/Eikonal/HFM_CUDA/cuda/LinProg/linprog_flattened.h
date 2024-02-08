/* In this file, we write a recursion free version of linprog.
The objective is to 
- avoid stack depth issues, while avoiding templates (which produce very long compile times)
- extract some more parallelism on the GPU.

The idea is to split lin_prog in a coroutine manner.
*/

struct linprog_args {
	FLOAT * halves;
	int istart;
	int m;
	FLOAT * n_vec;
	FLOAT * d_vec;
	int d;
	FLOAT * opt;
	FLOAT * work;
	int * next;
	int * prev;
	int max_size;
	linprog_args(FLOAT * _halves,int _istart,int _m,FLOAT * _n_vec,FLOAT * _d_vec,
				 int _d,FLOAT * _opt,FLOAT * _work,int * _next,int * _prev,int _max_size):
	halves(_halves),istart(_istart),m(_m),n_vec(_n_vec),d_vec(_d_vec),d(_d),opt(_opt),
	work(_work),next(_next),prev(_prev),max_size(_max_size){}
	linprog_args(){}
};

struct linprog_state {
	int status;
	int i, j, imax;
	#ifdef CHECK
	int k;
	#endif
	FLOAT *new_opt, *new_n_vec, *new_d_vec,  *new_halves, *new_work;
	FLOAT *plane_i;
	FLOAT val;
	int d_vec_zero;
};

//const int linprog_d_max; // This compile time constant must be provided
int linprog_counter = 0; const int linprog_counter_max = 1000;

int linprog_init(const linprog_args & args, linprog_state & state);
int linprog_loop(const linprog_args & args, linprog_state & state); // false -> loop interrupted, true -> loop terminated
void linprog_sub_begin(const linprog_args & args, linprog_state & state, linprog_args & rec_args);
int linprog_sub_end(const linprog_args & args, linprog_state & state);

int linprog_flattened(FLOAT halves[], /* halves  --- half spaces */
	int istart,     /* istart  --- should be zero
				 unless doing incremental algorithm */
	int m,  	/* m       --- terminal marker */
	FLOAT n_vec[], 	/* n_vec   --- numerator vector */
	FLOAT d_vec[], 	/* d_vec   --- denominator vector */
	int d_max, 		/* d       --- projective dimension */
	FLOAT opt[],	/* opt     --- optimum */
	FLOAT work[], 	/* work    --- work space (see below) */
	int next[], 	/* next    --- array of indices into halves */
	int prev[], 	/* prev    --- array of indices into halves */
	int max_size) 	/* max_size --- size of halves array */
{

linprog_args args[(LINPROG_DIMENSION_MAX)+1];
linprog_state state[(LINPROG_DIMENSION_MAX)+1];
int d=d_max;
args[d] = linprog_args(halves,istart,m,n_vec,d_vec,d_max,opt,work,next,prev,max_size);
int status;
while(true){
	
	// Init state
	status = linprog_init(args[d],state[d]);
	if(status!=INIT_DONE){ // No loop, go up one level
		if(d==d_max) {return status;}
		while(true){ // Rewind recursion
			++d;
			state[d].status = status;
			status = linprog_sub_end(args[d],state[d]);
			if(status==SUBPROBLEM_TERMINATED){break;}
			else if(d==d_max){return status;}
		}
	}
	
	//find the next subproblem to work on
	while(true){
		status = linprog_loop(args[d],state[d]);
		if(status==LOOP_INTERRUPTED){break;}
		else if(d==d_max) {return status;}
		while(true){ // Rewind recursion
			++d;
			state[d].status = status;
			status = linprog_sub_end(args[d],state[d]);
			if(status==SUBPROBLEM_TERMINATED){break;}
			else if(d==d_max){return status;}
		}
	}
	// Recurse in lower dimension
	linprog_sub_begin(args[d],state[d],args[d-1]); // Most of real work here
	--d;
}
}

int linprog_init(const linprog_args & a, linprog_state & s){
	if(debug_print){
		printf("Entering linprog_init %d\n", a.d);
//		++linprog_counter;
//		assert(linprog_counter<linprog_counter_max);
	}
/* // linprog_state
	int status;
	int i, j, imax;
	#ifdef CHECK
	int k;
	#endif
	FLOAT *new_opt, *new_n_vec, *new_d_vec,  *new_halves, *new_work;
	FLOAT *plane_i;
	FLOAT val;
*/
	if(a.d==1 && a.m!=0) {
		return(lp_base_case((FLOAT (*)[2])a.halves,a.m,a.n_vec,a.d_vec,a.opt,
			a.next,a.prev,a.max_size));
	} else {
		s.val = 0.0;
		for(s.j=0; s.j<=a.d; s.j++) s.val += a.d_vec[s.j]*a.d_vec[s.j];
		s.d_vec_zero = (s.val < (a.d+1)*EPS*EPS);

/* find the unconstrained minimum */
		if(!a.istart) {
			s.status = lp_no_constraints(a.d,a.n_vec,a.d_vec,a.opt); 
		} else {
			s.status = MINIMUM;
		}
		if(a.m==0) return(s.status);
/* allocate memory for next level of recursion */
		s.new_opt = a.work;
		s.new_n_vec = s.new_opt + a.d;
		s.new_d_vec = s.new_n_vec + a.d;
		s.new_halves = s.new_d_vec + a.d;
		s.new_work = s.new_halves + a.max_size*a.d;
	}
	return INIT_DONE;
}


int linprog_loop(const linprog_args & a, linprog_state & s){
	if(debug_print){
		printf("linprog_loop %d\n", a.d);
//		++linprog_counter;
//		assert(linprog_counter<linprog_counter_max);
	}
	for(s.i = a.istart; s.i!=a.m; s.i=a.next[s.i]) {
#ifdef CHECK
	if(s.i<0 || s.i>=a.max_size) {
		printf("index error\n");
		EXIT1;
	}
#endif
/* if the optimum is not in half space i then project the problem
** onto that plane */
	s.plane_i = a.halves + s.i*(a.d+1);
/* determine if the optimum is on the correct side of plane_i */
	s.val = 0.0;
	for(s.j=0; s.j<=a.d; s.j++) s.val += a.opt[s.j]*s.plane_i[s.j];
	if(s.val<-(a.d+1)*EPS) {return LOOP_INTERRUPTED;} // Need to work on that subproblem
	}
	return s.status; // This problem terminated
}

void linprog_sub_begin(const linprog_args & a, linprog_state & s, linprog_args & b){
	if(debug_print){
		printf("linprog_sub_begin %d\n", a.d);
//		++linprog_counter;
//		assert(linprog_counter<linprog_counter_max);
	}
/* find the largest of the coefficients to eliminate */
    findimax(s.plane_i,a.d,&s.imax);
/* eliminate that variable */
    if(s.i!=0) {
	FLOAT fac;
	fac = 1.0/s.plane_i[s.imax];
	for(s.j=0; s.j!=s.i; s.j=a.next[s.j]) {
		FLOAT *old_plane, *new_plane;
		int k;
		FLOAT crit;

		old_plane = a.halves + s.j*(a.d+1);
		new_plane = s.new_halves + s.j*a.d;
		crit = old_plane[s.imax]*fac;
		for(k=0; k<s.imax; k++)  {
			new_plane[k] = old_plane[k] - s.plane_i[k]*crit;
		}
		for(k=s.imax+1; k<=a.d; k++)  {
			new_plane[k-1] = old_plane[k] - s.plane_i[k]*crit;
		}
	}
    }
/* project the objective function to lower dimension */
    if(s.d_vec_zero) {
		vector_down(s.plane_i,s.imax,a.d,a.n_vec,s.new_n_vec);
		for(s.j=0; s.j<a.d; s.j++) s.new_d_vec[s.j] = 0.0;
    } else {
        plane_down(s.plane_i,s.imax,a.d,a.n_vec,s.new_n_vec);
        plane_down(s.plane_i,s.imax,a.d,a.d_vec,s.new_d_vec);
    }
/* solve sub problem */
	b = linprog_args(s.new_halves,0,s.i,s.new_n_vec,
		s.new_d_vec,a.d-1,s.new_opt,s.new_work,a.next,a.prev,a.max_size);
}

int linprog_sub_end(const linprog_args & a, linprog_state & s){
	if(debug_print){
		printf("linprog_sub_end %d\n", a.d);
//		++linprog_counter;
//		assert(linprog_counter<linprog_counter_max);
	}
/* back substitution */
    if(s.status!=INFEASIBLE) {
	    vector_up(s.plane_i,s.imax,a.d,s.new_opt,a.opt);
	{
/* in line code for unit */
	FLOAT size;
	size = 0.0;
	for(s.j=0; s.j<=a.d; s.j++) 
	    size += a.opt[s.j]*a.opt[s.j];
	size = 1.0/sqrt(size);
	for(s.j=0; s.j<=a.d; s.j++)
	    a.opt[s.j] *= size;
	}
    } else {
	    return(s.status);
    }
/* place this offensive plane in second place */
    s.i = move_to_front(s.i,a.next,a.prev,a.max_size);
#ifdef CHECK
    s.j=0;
    while(1) {
/* check the validity of the result */
	s.val = 0.0;
	for(s.k=0; s.k<=a.d; s.k++)
		s.val += a.opt[s.k]*a.halves[s.j*(a.d+1)+s.k];
	if(s.val <-(a.d+1)*EPS) {
	    printf("error\n");
		EXIT1;
	}
	if(s.j==s.i) break;
	s.j=a.next[s.j];
    }
#endif
//	} // if subproblem 
//} // for subproblem
	return SUBPROBLEM_TERMINATED;
}
