#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0


/*This file defines a class implementing basic geometrical operations. 
It is templated over the dimension. (Previously used namespaces, which to compiler errors.)
Indeed, the compiler seemed to be confused by the use of namespaces 
and multiple inclusions of the file Geometry_.h.*/

/**
Naming conventions : 
- k : scalar 
- v,V : vector 
- m,M : symmetric matrix 
- a,A : square matrix 
- lower case : input
- upper case : output (terminal output may be omitted from function name)

IMPORTANT : NO-ALIAS ASSUMPTION ! Outputs are assumed to be __restrict__.
*/

void show(float x){printf("%f\n",x);}
void show(int x)  {printf("%i\n",x);}
void show(char x) {show(int(x));}

template<int ndim> struct GeometryT {

static const int symdim = (ndim*(ndim+1))/2;

// ------------------ Vector elementary operations ------------------
template<typename T>
static void zero(T out[ndim]){for(int i=0; i<ndim; ++i){out[i]=T(0);}}
template<typename S, typename T> 
static void cast(const S x[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]=x[i];}
template<typename T>
static void fill_kV(const T k, T v[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){v[i]=k;}}


// In place operations
template<typename T>
static void neg(const T x[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]=-x[i];}
template<typename Tx, typename T>
static void add(const Tx x[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]+=x[i];}
template<typename Tx, typename T>
static void sub(const Tx x[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]-=x[i];}
template<typename Tk, typename T>
static void mul(const Tk k,       T out[ndim]){for(int i=0; i<ndim; ++i) out[i]*=k;}
template<typename T>
static void madd(const T k, const T x[ndim], T out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]+=k*x[i];}}

// Out of place operations
template<typename T>
static void add(const T x[ndim], const T y[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]=x[i]+y[i];}
template<typename T>
static void sub(const T x[ndim], const T y[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]=x[i]-y[i];}
template<typename T>
static void mul(const T k,       const T y[ndim], T out[ndim]){for(int i=0; i<ndim; ++i) out[i]=k*y[i];}
template<typename T>
static void madd(const T k, const T x[ndim], const T y[ndim], T out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=k*x[i]+y[i];}}

// Euclidean geometry
template<typename Tx, typename Ty, typename T=Tx>
static T scal(const Tx x[ndim], const Ty y[ndim]){
	T r=0.; for(Int i=0; i<ndim; ++i){r+=x[i]*y[i];} return r;}
template<typename T>
static T norm2(const T x[ndim]){return scal(x,x);}

// -------------------- Matrix initialization ------------------
template<typename T>
static T coef_m(const T m[symdim], const Int i, const Int j){
	const Int i_ = max(i,j), j_=min(i,j);
	const Int k = (i_*(i_+1))/2+j_;
	HFM_DEBUG(assert(0<=i && i<ndim && 0<=j && j<ndim && 0<=k && k<symdim);)
	return m[k];
}

template<typename T>
static void copy_mA(const T m[symdim], T a[__restrict__ ndim][ndim]){
	for(int i=0,k=0; i<ndim; ++i){
		for(int j=0; j<=i; ++j,++k){
			a[i][j]=m[k];
			if(i!=j) a[j][i]=m[k];
		}
	}
}

template<typename T>
static void copylower_aM(const T a[ndim][ndim], T m[__restrict__ symdim]){
	for(int i=0,k=0; i<ndim; ++i){for(int j=0; j<=i; ++j,++k) m[k]=a[i][j];} }

template<typename T, typename Tout>
static void copy_aA(const T a[ndim][ndim], Tout out[__restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) out[i][j] = a[i][j];}}

template<typename T>
static void identity_A(T a[ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) a[i][j]=(i==j);} }

static void identity_M(Scalar m[symdim]){
	for(Int i=0,k=0; i<ndim; ++i){for(Int j=0; j<=i; ++j,++k) m[k]=(i==j);} }

template<typename T>
static void self_outer(const T x[ndim],T m[symdim]){
	for(int i=0,k=0; i<ndim; ++i){for(int j=0; j<=i; ++j,++k) m[k]=x[i]*x[j];}}

// ------------- dot products -----------

template<typename Tm, typename Tv, typename T>
static void dot_mv(const Tm m[symdim], const Tv v[ndim], T out[__restrict__ ndim]){
	zero(out);
	for(Int i=0,k=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j,++k){
			out[i]+=m[k]*v[j];
			if(i!=j) {out[j]+=m[k]*v[i];}
		}
	}
}

/// Matrix vector product
template<typename Ta,typename Tx,typename Tout>
static void dot_av(const Ta a[ndim][ndim], const Tx x[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i) out[i] = scal<Ta,Tx,Tout>(a[i],x);}
/// Transposed matrix vector product
static void tdot_av(const Scalar a[ndim][ndim], const Scalar x[ndim], Scalar out[__restrict__ ndim]){
	fill_kV(Scalar(0),out);
	for(Int i=0; i<ndim; ++i) {for(Int j=0; j<ndim; ++j) out[i] += a[j][i]*x[j];}}


template<typename Ta, typename Tm, typename T>
static void tgram_am(const Ta a[ndim][ndim], const Tm m[symdim], T out[__restrict__ symdim]){
	for(Int i=0,k=0; i<ndim; ++i){
		T mai[ndim]; 
		dot_mv(m,a[i],mai);
		for(Int j=0; j<=i; ++j,++k){
			out[k] = scal(mai,a[j]); // Accumulates with type Tout
		}
	}
}


// ------------------- Matrix inversion and related ----------------------

template<typename T> // Matrix determinant ! In dimension <=3 !
static T det_a(const T a[ndim][ndim]){
	if(ndim==1) {return a[0][0];}
	else if (ndim==2) {return a[0][0]*a[1][1] - a[1][0]*a[0][1];}
	else if (ndim==3) {
		T det=0;
		for (Int i = 0; i < 3; ++i) 
			det += a[i][0]*a[(i+1)%3][1]*a[(i+2)%3][2]-a[i][2]*a[(i+1)%3][1]*a[(i+2)%3][0];
		return det;
	} else {return 0./0.;}

}

template<typename Tk, typename T>
static void mul_kA(const Tk k, T out[ __restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j){out[i][j]*=k;}}}
template<typename Tk, typename T>
static void div_kA(const Tk k, T out[ __restrict__ ndim][ndim]){mul_kA<Tk,T>(1./k,out);}

/** 
Matrix inversion using : 
 - Explicit formulas in dimension d<=3.
 - Gauss pivot otherwise.
*/
template<typename T>
static void inv_a(const T a[ndim][ndim], T out[__restrict__ ndim][ndim]){
	if(ndim==1){
		out[0][0]=T(1);
	} else if(ndim==2) { // transposed comatrix
		out[0][0]= a[1][1];
		out[1][0]=-a[1][0];
		out[0][1]=-a[0][1];
		out[1][1]= a[0][0];
	} else if (ndim==3){ // transposed comatrix
		for(int i=0; i<3; ++i)
			for(int j=0; j<3; ++j)
				out[j][i]=
				a[(i+1)%3][(j+1)%3]*a[(i+2)%3][(j+2)%3]-
				a[(i+1)%3][(j+2)%3]*a[(i+2)%3][(j+1)%3];
	} else {
		// Gauss pivot
		Scalar m[ndim][ndim], b[ndim][ndim];
		copy_aA(a,m);
		identity_A(b);
		Int i2j[ndim], j2i[ndim]; 
		fill_kV(-1,i2j); fill_kV(-1,j2i);
		for(int j=0; j<ndim; ++j){
			// Get largest coefficient in column
			T cMax = 0;
			int iMax=0;
			for(int i=0; i<ndim; ++i){
				if(i2j[i]>=0) continue;
				const T c = m[i][j];
				if(abs(c)>abs(cMax)){
					cMax=c; iMax=i;}
			}
			i2j[iMax]=j;
			j2i[j]=iMax;
//			assert(cMax!=0); // Otherwise, matrix is not invertible
			
			const Scalar invcMax = 1./cMax;
			// Remove line from other lines, while performing likewise on b
			for(int i=0; i<ndim; ++i){
				if(i2j[i]>=0) continue;
				const Scalar r = m[i][j]*invcMax;
				for(int k=j+1; k<ndim; ++k){m[i][k]-=m[iMax][k]*r;}
				for(int l=0;   l<ndim; ++l){b[i][l]-=b[iMax][l]*r;}
			}
		}
		// Solve the remaining triangular system
		for(int j=ndim-1; j>=0; --j){
			const int i=j2i[j];
			for(int l=0; l<ndim; ++l){
				out[j][l]=b[i][l];
				for(int k=j+1; k<ndim; ++k) {out[j][l]-=out[k][l]*m[i][k];}
				out[j][l]/=m[i][j];
			}
		}
		return;
	}
	// In dimension <=3, transposed comatrix must be divided by determinant
	div_kA<T,T>(det_a(a),out);
}

/**
Solve the linear system ax = b using the Gauss pivot method. 
(This version overwrites the inputs a and b)
*/
template<typename T> 
static void solve_av_overwrite(
	T a[__restrict__ ndim][ndim], 
	T b[__restrict__ ndim], 
	T out[__restrict__ ndim][ndim]){
	// A basic Gauss pivot
	Int i2j[ndim], j2i[ndim]; 
	fill_kV(-1,i2j); fill_kV(-1,j2i);
    for(int j=0; j<n; ++j){
		// Get largest coefficient in column
		T cMax = 0;
		int iMax=0;
		for(int i=0; i<ndim; ++i){
			if(i2j[i]>=0) continue;
			const T c = a[i][j];
			if(abs(c)>abs(cMax)){
				cMax=c; iMax=i;}
		}
		i2j[iMax]=j;
		j2i[j]=iMax;

		const Scalar invcMax = 1./cMax;
		// Remove line from other lines, while performing likewise on b
		for(int i=0; i<ndim; ++i){
			if(i2j[i]>=0) continue;
			const Scalar r = a[i][j]*invcMax;
			for(int k=j+1; k<ndim; ++k){a[i][k]-=a[iMax][k]*r;}
			b[i]-=b[iMax]*r;
        }
    }
    // Solve the remaining triangular system
    for(int j=n-1; j>=0; --j){
        const int i=j2i[j];
        T & r = out[j];
        r=b[i];
        for(int k=j+1; k<n; ++k){r-=out[k]*a(i,k);}
        r/=a(i,j);
    }
}

/// Solve the linear system ax = b using the Gauss pivot method. 
template<typename T>
static void solve_av(const T a[ndim][ndim],const T b[ndim],T out[__restrict__ ndim]){
	T a_[ndim][ndim]; copy_aA(a,a_);
	T b_[ndim]; cast(b,b_);
	solve_av_overwrite(a_,b_,out);
}

// ---------------------- Display --------------------
template<typename T>
static void show_v(const T v[ndim]){
	for(int i=0; i<ndim; ++i){
		if(i==0) printf("{"); else printf(",");
		show(v[i]);
		if(i==ndim-1) printf("}");
	}
}

template<typename T>
static void show_m(const T m[symdim]){
	for(int i=0; i<ndim; ++i){
		if(i==0) printf("{"); else printf(",");
		for(int j=0; j<ndim; ++j){
			if(j==0) printf("{"); else printf(",");
			show(coef_m(m,i,j));
			if(j==ndim-1) printf("}");
		}
		if(i==ndim-1) printf("}");
	}
}

template<typename T>
static void show_a(const T a[ndim][ndim]){
	for(int i=0; i<ndim; ++i){
		if(i==0) printf("{"); else printf(",");
		for(int j=0; j<ndim; ++j){
			if(j==0) printf("{"); else printf(",");
			show(a[i][j]);
			if(j==ndim-1) printf("}");
		}
		if(i==ndim-1) printf("}");
	}
}


};
