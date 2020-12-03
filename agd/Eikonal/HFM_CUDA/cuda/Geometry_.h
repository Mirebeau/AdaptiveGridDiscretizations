/* This file is purposedly missing a #pragma once, because it is sometimes useful to 
include it within several namespaces, where the constant ndim takes different values.*/ 

// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

const Int symdim = (ndim*(ndim+1))/2; // Dimension of the space of symmetric matrices.

/**
Naming conventions : 
- k : scalar 
- v,V : vector 
- m,M : symmetric matrix 
- a,A : square matrix 
- lower case : input
- upper case : output (terminal output may be omitted)

IMPORTANT : NO-ALIAS ASSUMPTION ! Outputs are assumed to be __restrict__.
*/

// ------ Copies and casts -----
template<typename T>
void copy_vV(const T x[ndim], T out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=x[i];}} 

template<typename T, typename Tout>
void copy_mM(const T m[symdim], Tout out[__restrict__ symdim]){
	for(Int i=0; i<symdim; ++i){out[i]=m[i];}} 

template<typename T, typename Tout>
void copy_aA(const T a[ndim][ndim], Tout out[__restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) out[i][j] = a[i][j];}}

template<typename T, typename Tout>
void round_a(const T a[ndim][ndim], Tout out[__restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) out[i][j] = round(a[i][j]);}}

template<typename T> 
void zero_V(T out[ndim]){for(Int i=0;   i<ndim; ++i){out[i]=0;}}

template<typename T> 
void zero_M(T out[symdim]){for(Int i=0; i<symdim; ++i){out[i]=0;}}

template<typename T>
void zero_A(T out[ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0;j<ndim; ++j) out[i][j]=0;}}

// ------ vector algebra -----

/// Sum 
template<typename Tx,typename Ty,typename Tout>
void add_vv(const Tx x[ndim], const Ty y[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=x[i]+y[i];}}

template<typename T>
void add_vV(const T x[ndim], T y[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){y[i]+=x[i];}}

template<typename Tx,typename Ty,typename Tout>
void add_mm(const Tx x[symdim], const Ty y[symdim], Tout out[__restrict__ symdim]){
	for(Int i=0; i<symdim; ++i){out[i]=x[i]+y[i];}}

/// Difference
template<typename Tx, typename Ty, typename Tout>
void sub_vv(const Tx x[ndim], const Ty y[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=x[i]-y[i];}}

/// Opposite vector
template<typename T>
void neg_v(const T x[ndim], T out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=-x[i];}}

template<typename T>
void neg_V(T x[ndim]){
	for(Int i=0; i<ndim; ++i){x[i]=-x[i];}}

template<typename T>
void fill_kV(const T k, T v[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){v[i]=k;}}

template<typename T>
void mul_kV(const T k, T v[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){v[i]*=k;}}

template<typename Tk,typename Tv,typename Tout>
void mul_kv(const Tk k, const Tv v[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i] = k * v[i];}}

void div_Vk(Scalar v[__restrict__ ndim], const Scalar k){
	const Scalar l=1./k; mul_kV(l,v);}

template<typename Tk, typename Tx, typename Ty, typename Tout>
void madd_kvv(const Tk k, const Tx x[ndim], const Ty y[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=k*x[i]+y[i];}}

template<typename Tk, typename Tx, typename Ty>
void madd_kvV(const Tk k, const Tx x[ndim], Ty y[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i){y[i]+=k*x[i];} }


// ----------- Scalar products -----------

/// Euclidean scalar product
template<typename Tx, typename Ty, typename Tout=Tx>
Tout scal_vv(const Tx x[ndim], const Ty y[ndim]){
	Tout result=0.;
	for(Int i=0; i<ndim; ++i){
		result+=x[i]*y[i];}
	return result;
}
// Squared Euclidean norm
template<typename T>
T norm2_v(const T x[ndim]){return scal_vv(x,x);}
Scalar norm_v(const Scalar x[ndim]){return sqrt(norm2_v<Scalar>(x));}

/// Scalar product associated with a symmetric matrix
template<typename Tx,typename Ty>
Scalar scal_vmv(const Tx x[ndim], const Scalar m[symdim], const Ty y[ndim]){
	Scalar result=0.;
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			result += (i==j ? x[i]*y[i] : (x[i]*y[j]+x[j]*y[i]))*m[k];
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
	return result;
}
/// Squared norm associated with a symmetric matrix
template<typename T>
T norm2_vm(const T x[ndim], const Scalar m[symdim]){return scal_vmv(x,m,x);}


// Scalar product associated with a diagonal matrix
template<typename Tx, typename Ty>
Scalar scal_vdv(const Tx x[ndim], const Scalar diag[ndim], const Ty y[ndim]){
	Scalar result=0; 
	for(Int i=0; i<ndim; ++i){result+=x[i]*y[i]*diag[i];}
	return result;
}

// Frobenius scalar product of two matrices
template<typename Tx, typename Ty>
Scalar scal_mm(const Tx mx[symdim],const Ty my[symdim]){
	Scalar result=0; 
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			result+=mx[k]*my[k]*(i==j ? 1 : 2);
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
	return result;
}

// -------- Outer products -------

template<typename T, typename Tout>
void self_outer_v(const T x[ndim], Tout m[__restrict__ ndim]){
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			m[k] = x[i]*x[j]; 
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
}

void self_outer_relax_v(const Scalar x[ndim], const Scalar relax, 
	Scalar m[__restrict__ ndim]){
	const Scalar eps = scal_vv(x,x)*relax;
	Int k=0;
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			m[k] = x[i]*x[j]*(1-eps) + (i==j)*eps; 
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
}

template<typename T>
void madd_kmM(const T k, const T x[symdim], T y[__restrict__ symdim]){
	for(Int i=0; i<symdim; ++i){y[i]+=k*x[i];} }

// ------ Special matrices ------

void identity_M(Scalar m[symdim]){
	Int k=0;
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			m[k]=(i==j);
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
}

template<typename T>
void identity_A(T a[ndim][ndim]){
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<ndim; ++j){
			a[i][j]=(i==j);}}
}

template<typename T>
void canonicalsuperbase(T sb[ndim+1][ndim]){
	identity_A(sb);
	for(Int j=0; j<ndim; ++j){
		sb[ndim][j] = -1;
	}
}
// ------- Matrix products ------

template<typename T>
T coef_m(const T m[symdim], const Int i, const Int j){
	const Int i_ = max(i,j), j_=min(i,j);
	const Int k = (i_*(i_+1))/2+j_;
	HFM_DEBUG(assert(0<=i && i<ndim && 0<=j && j<ndim && 0<=k && k<symdim);)
	return m[k];
}

/// Dot product of symmetric matrix times vector
template<typename Tm,typename Tv, typename Tout>
void dot_mv(const Tm m[symdim], const Tv v[ndim], Tout out[__restrict__ ndim]){
	zero_V(out);
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			out[i]+=m[k]*v[j];
			if(i!=j) {out[j]+=m[k]*v[i];}
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
}

/// Matrix vector product
template<typename Ta,typename Tx,typename Tout>
void dot_av(const Ta a[ndim][ndim], const Tx x[ndim], Tout out[__restrict__ ndim]){
	for(Int i=0; i<ndim; ++i) out[i] = scal_vv<Ta,Tx,Tout>(a[i],x);}
/// Transposed matrix vector product
void tdot_av(const Scalar a[ndim][ndim], const Scalar x[ndim], Scalar out[__restrict__ ndim]){
	fill_kV(Scalar(0),out);
	for(Int i=0; i<ndim; ++i) {for(Int j=0; j<ndim; ++j) out[i] += a[j][i]*x[j];}}

/// Matrix transposition
template<typename Ta, typename Tout>
void trans_a(const Ta a[ndim][ndim], Tout out[__restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j) out[i][j]=a[j][i];}}

/// Matrix-Matrix product
template<typename Ta,typename Tb,typename Tout>
void dot_aa(const Ta a[ndim][ndim], const Tb b[ndim][ndim],
	Tout out[__restrict__ ndim][ndim]){
	zero_A(out);
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j){for(Int k=0; k<ndim; ++k){
		out[i][k]+=a[i][j]*b[j][k];}}}
}

template<typename Ta, typename Tm, typename Tout>
void tgram_am(const Ta a[ndim][ndim], const Tm m[symdim],
	Tout out[__restrict__ symdim]){
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		Tout mai[ndim]; dot_mv(m,a[i],mai);
		for(Int j=0; j<=i; ++j){
			out[k] = scal_vv(mai,a[j]); // Accumulates with type Tout
			++k;
		}
	}
	HFM_DEBUG(assert(k==symdim);)
}

// ----- Display ------

#ifdef IOSTREAM
template<typename T>
void show_v(std::ostream & os, const T v[ndim]){
	for(int i=0; i<ndim; ++i){
		os << (i==0 ? "{" : ",");
		os << Scalar(v[i]);
		if(i==ndim-1) os << "}";
	}
}

template<typename T>
void show_m(std::ostream & os, const T m[symdim]){
	for(int i=0; i<ndim; ++i){
		os << (i==0 ? "{" : ",");
		for(int j=0; j<ndim; ++j){
			os << (j==0 ? "{" : ",");
			os << Scalar(coef_m(m,i,j));
			if(j==ndim-1) os << "}";
		}
		if(i==ndim-1) os << "}";
	}
}

template<typename T>
void show_a(std::ostream & os, const T a[ndim][ndim]){
	for(int i=0; i<ndim; ++i){
		os << (i==0 ? "{" : ",");
		for(int j=0; j<ndim; ++j){
			os << (j==0 ? "{" : ",");
			os << Scalar(a[i][j]);
			if(j==ndim-1) os << "}";
		}
		if(i==ndim-1) os << "}";
	}
}
#endif
