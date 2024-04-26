#pragma once

#include "GeometryT.h"
// Classes for first and second order dense automatic differentiation

// Until now I avoided C++ with cuda, but it seems to work after all.
// Pasted from HamiltonFastMarching/ LinearAlgebra/FriendOperators.h
// Overload operators
template<typename TDifferentiation, typename TScalar> struct DifferentiationTypeOperators {
	typedef TDifferentiation AD;
	typedef TScalar K;
	friend AD operator + (AD a, AD const & b) { a += b; return a; }
	friend AD operator + (AD a, K const & b)  { a += b; return a; }
	friend AD operator + (K const & a, AD b)  { b += a; return b; }

	friend AD operator - (AD a, AD const & b) { a -= b; return a; }
	friend AD operator - (AD a, K const & b)  { a -= b; return a; }
	friend AD operator - (K const & a, AD b)  { b -= a; return -b; }

	friend AD operator * (AD a, AD const & b) { a *= b; return a; }
	friend AD operator * (AD a, K const & b)  { a *= b; return a; }
	friend AD operator * (K const & a, AD b)  { b *= a; return b; }
	
	friend AD operator / (const AD & a, const AD & b){return a*b.Inverse();}
	friend AD operator / (const K & a,  const AD & b){return a*b.Inverse();}
	friend AD operator / (AD a, const K & b) {a/=b; return a;}
	friend void operator /= (AD & a, const AD & b){a*=b.Inverse();}

	// Totally ordered
	friend bool operator >  (AD const & a, AD const & b) { return b < a; }
	friend bool operator >= (AD const & a, AD const & b) { return !(a < b); }
	friend bool operator <= (AD const & a, AD const & b) { return !(a > b); }

	// Totally ordered 2
	friend bool operator <= (const AD & a, const K & b) { return !(a > b); }
	friend bool operator >= (const AD & a, const K & b) { return !(a < b); }

	friend bool operator <  (const K & b, const AD & a) { return a > b; }
	friend bool operator >  (const K & b, const AD & a) { return a < b; }
	friend bool operator <= (const K & b, const AD & a) { return a >= b; }
	friend bool operator >= (const K & b, const AD & a) { return a <= b; }

	friend bool operator != (const AD &a, const AD &b) { return !(a == b); }
};

template<typename Scalar> struct _Taylor1 { // first order Taylor expansions of elem functions
	static void _log(const Scalar x, Scalar y[2]){y[0]=log(x); y[1]=Scalar(1)/x;}
	static void _exp(const Scalar x, Scalar y[2]){Scalar e=exp(x); y[0]=e; y[1]=e;}
	static void _abs(const Scalar x, Scalar y[2]){y[0]=abs(x); y[1]=x>=0?Scalar(1):Scalar(-1);}
	static void _sin(const Scalar x, Scalar y[2]){y[0]=sin(x); y[1]=cos(x);}
	static void _cos(const Scalar x, Scalar y[2]){y[0]=cos(x); y[1]=-sin(x);}
	static void _tan(const Scalar x, Scalar y[2]){Scalar t=tan(x); y[0]=t; y[1]=Scalar(1)+t*t;}
	static void _arcsin(const Scalar x, Scalar y[2]){y[0]=arcsin(x); y[1]=Scalar(1)/sqrt(Scalar(1)-x*x);}
	static void _arccos(const Scalar x, Scalar y[2]){y[0]=arccos(x); y[1]=Scalar(-1)/sqrt(Scalar(1)-x*x);}
	static void _arctan(const Scalar x, Scalar y[2]){y[0]=arctan(x); y[1]=Scalar(1)/(Scalar(1)+x*x);}
	static void _sinh(const Scalar x, Scalar y[2]){y[0]=sinh(x); y[1]=cosh(x);}
	static void _cosh(const Scalar x, Scalar y[2]){y[0]=cosh(x); y[1]=sinh(x);}
	static void _tanh(const Scalar x, Scalar y[2]){Scalar t=tanh(x); y[0]=t; y[1]=Scalar(1)-t*t;}
	static void _arcsinh(const Scalar x, Scalar y[2]){y[0]=arcsinh(x); y[1]=Scalar(1)/sqrt(Scalar(1)+x*x);}
	static void _arccosh(const Scalar x, Scalar y[2]){y[0]=arccosh(x); y[1]=Scalar(1)/sqrt(x*x-Scalar(1));}
	static void _arctanh(const Scalar x, Scalar y[2]){y[0]=arctanh(x); y[1]=Scalar(1)/(Scalar(1)-x*x);}
};

template<typename Scalar> struct _Taylor2 { // second order Taylor expansions of elem functions
	static void _log(const Scalar x, Scalar y[3]){y[0]=log(x); const Scalar t=Scalar(1)/x; y[1]=t; y[2]=-t*t;}
	static void _exp(const Scalar x, Scalar y[3]){Scalar e=exp(x); y[0]=e; y[1]=e; y[2]=e;}
	static void _abs(const Scalar x, Scalar y[3]){y[0]=abs(x); y[1]=x>=0?Scalar(1):Scalar(-1); y[2]=Scalar(0);}
	static void _sin(const Scalar x, Scalar y[3]){y[0]=sin(x); y[1]=cos(x); y[2]=-y[0];}
	static void _cos(const Scalar x, Scalar y[3]){y[0]=cos(x); y[1]=-sin(x); y[2]=-y[0];}
	static void _tan(const Scalar x, Scalar y[3]){Scalar t=tan(x); y[0]=t; y[1]=Scalar(1)+t*t; y[2]=Scalar(2)*y[1]*t;}
	static void _arcsin(const Scalar x, Scalar y[3]){y[0]=arcsin(x); y[1]=Scalar(1)/sqrt(Scalar(1)-x*x); y[2]=x*y[1]*y[1]*y[1];}
	static void _arccos(const Scalar x, Scalar y[3]){y[0]=arccos(x); y[1]=Scalar(-1)/sqrt(Scalar(1)-x*x); y[2]=-x*y[1]*y[1]*y[1];}
	static void _arctan(const Scalar x, Scalar y[3]){y[0]=arctan(x); y[1]=Scalar(1)/(Scalar(1)+x*x); y[2]=Scalar(-2)*x*y[1]*y[1];}
	static void _sinh(const Scalar x, Scalar y[3]){y[0]=sinh(x); y[1]=cosh(x); y[2]=y[0];}
	static void _cosh(const Scalar x, Scalar y[3]){y[0]=cosh(x); y[1]=sinh(x); y[2]=y[0];}
	static void _tanh(const Scalar x, Scalar y[3]){Scalar t=tanh(x); y[0]=t; y[1]=Scalar(1)-t*t; y[2]=Scalar(-2)*y[1]*t;}
	static void _arcsinh(const Scalar x, Scalar y[3]){y[0]=arcsinh(x); y[1]=Scalar(1)/sqrt(Scalar(1)+x*x); y[2]=-x**y[1]*y[1]*y[1];}
	static void _arccosh(const Scalar x, Scalar y[3]){y[0]=arccosh(x); y[1]=Scalar(1)/sqrt(x*x-Scalar(1)); y[2]=-x**y[1]*y[1]*y[1];}
	static void _arctanh(const Scalar x, Scalar y[3]){y[0]=arctanh(x); y[1]=Scalar(1)/(Scalar(1)-x*x); y[2]=Scalar(2)*x*y[1]*y[1];}
};

/**
A class for first order dense automatic differentiation.

A element x = (a,v) of this class represents 
x = a + <v,h> + o(|h|), 
where h is an infinitesimal perturbation in dimension ndim. 

Note that a is a scalar, and v is a vector of dimension ndim.
*/
template<typename Scalar, int ndim>
struct Dense1 : DifferentiationTypeOperators<Dense1<Scalar,ndim>,Scalar> {
	typedef GeometryT<ndim> V;

	Scalar a;
	Scalar v[ndim];

	Dense1(){};
	Dense1(Scalar a_){a=a_;V::zero(v);}

	void operator += (const Dense1 & y){a+=y.a; V::add(y.v,v);}
	void operator -= (const Dense1 & y){a-=y.a; V::sub(y.v,v);}
	void operator *= (const Dense1 & y){
		for(int i=0; i<ndim;++i){v[i]=y.a*v[i]+a*y.v[i];}
		a = a*y.a;
	}
	void operator /= (const Dense1 & y){*this*=y.Inverse();}
	Dense1 Inverse() const {
		Dense1 r;
		r.a = Scalar(1)/a;
		V::mul(-r.a*r.a,v,r.v);
		return r;
	}
	Dense1 operator - () const {Dense1 y; y.a=-a; V::neg(v,y.v); return y; } 
	void operator += (const Scalar & y){a+=y;}
	void operator -= (const Scalar & y){a-=y;}
	void operator *= (const Scalar & y){a*=y; V::mul(y,v);}
	void operator /= (const Scalar & y){*this*=(1/y);}
    
    bool operator < (const Dense1 & y) const {return a < y.a;}
    bool operator < (const Scalar & y) const {return a < y;}

	static void Identity(Dense1 id[ndim]){
		for(int i=0; i<ndim; ++i){id[i].a=0; V::zero(id[i].v); id[i].v[i]=1;}
	}
	
	// in : Ah+b, out = -A^{-1}b
	static void solve(Dense1 in[ndim], Scalar out[ndim]){ 
		Scalar A[ndim][ndim], b[ndim];
		for(int i=0; i<ndim; ++i){
			b[i] = in[i].a;
			for(int j=0; j<ndim; ++j){A[i][j]=in[i].v[j];}
		}
		V::solve_av(A,b,out);
		V::neg(out,out);
	}

	void showself() const {
		printf("{"); 
		show(a);      printf(",");
		V::show_v(v); printf("}");
	}	
	void _math_helper(Scalar y[2]){a=y[0]; V::mul(y[1],v);}
};

template<typename Scalar,int ndim> void show(const Dense1<Scalar,ndim> & x){x.showself();}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> log(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_log(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> exp(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_exp(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> abs(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_abs(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> sin(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_sin(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> cos(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_cos(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arcsin(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arcsin(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arccos(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arccos(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arctan(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arctan(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> sinh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_sinh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> cosh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_cosh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> tanh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_tanh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arcsinh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arcsinh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arccosh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arccosh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense1<Scalar,ndim> arctanh(Dense1<Scalar,ndim> x){Scalar y[2]; _Taylor1<Scalar>::_arctanh(x.a,y); x._math_helper(y); return x;}


/**
A class for second order dense automatic differentiation.

A element x = (a,v,m) of this class represents
x = a + <v,h> + <h,m h>/2 + o(|h|^2),
where h is an infinitesimal perturbation in dimension ndim.

Note that a is a scalar, v is a vector of dimension ndim, and m is a symmetric matrix of shape
(ndim,ndim) which is stored in compact format with ndim*(ndim+1)/2 entries.
*/
template<typename Scalar, int ndim>
struct Dense2 : DifferentiationTypeOperators<Dense2<Scalar,ndim>,Scalar> {
    typedef GeometryT<ndim> V;
    static const int symdim = V::symdim;
    typedef GeometryT<symdim> M;
    typedef Dense1<Scalar,ndim> Dense1;

    Scalar a;
    Scalar v[ndim];
    Scalar m[symdim];

    Dense2(){};
    Dense2(Scalar a_){a=a_;V::zero(v);M::zero(m);}
    Dense2(const Dense1 & x){a=x.a; V::cast(x.v,v); M::zero(m);}

    void operator += (const Dense2 & y){a+=y.a; V::add(y.v,v); M::add(y.m,m);}
    void operator -= (const Dense2 & y){a-=y.a; V::sub(y.v,v); M::sub(y.m,m);}
    void operator *= (const Dense2 & y){
        for(int i=0,k=0; i<ndim; ++i){for(int j=0;j<=i; ++j,++k){
            m[k]=y.a*m[k]+a*y.m[k]+y.v[i]*v[j]+y.v[j]*v[i];}}
        for(int i=0; i<ndim;++i){v[i]=y.a*v[i]+a*y.v[i];}
        a = a*y.a;
    }
    void operator /= (const Dense2 & y){*this*=y.Inverse();}
    Dense2 Inverse() const {
        const Scalar ai = 1./a, ai2=ai*ai;
        Dense2 r;
        r.a = ai;
        V::mul(-ai2,v,r.v);
        V::self_outer(v,r.m);
        M::mul(2*ai,r.m);
        M::sub(m,r.m);
        M::mul(ai2,r.m);
        return r;
    }
    Dense2 operator - () const {Dense2 y; y.a=-a; V::neg(v,y.v); M::neg(m,y.m); return y; }
    void operator += (const Scalar & y){a+=y;}
    void operator -= (const Scalar & y){a-=y;}
    void operator *= (const Scalar & y){a*=y; V::mul(y,v); M::mul(y,m);}
    void operator /= (const Scalar & y){*this*=(1/y);}

    bool operator < (const Dense2 & y) const {return a < y.a;}
    bool operator < (const Scalar & y) const {return a < y;}

    static void Identity(Dense2 id[ndim]){
        for(int i=0; i<ndim; ++i){
            id[i].a=0;V::zero(id[i].v);M::zero(id[i].m);
            id[i].v[i]=1;
        }
    }
    /** Regard as a quadratic form, and find the stationnary point -dir (note the minus sign).
     The vector dir is the descent direction in the Newton method*/
    void solve_stationnary(Scalar dir[ndim]) const {
    // TODO : if positive definite, we could use cholesky factorization (simpler, cheaper)
        Scalar a[ndim][ndim], ai[ndim][ndim];
        V::copy_mA(m,a);
        V::inv_a(a,ai);
        V::dot_av(ai,v,dir);
    }
    
    
    void showself() const {
        printf("{");
        show(a);      printf(",");
        V::show_v(v); printf(",");
        V::show_m(m); printf("}");
    }
    void _math_helper(Scalar y[3]){
        for(int i=0,k=0; i<ndim; ++i){for(int j=0;j<=i; ++j,++k){
            m[k]=y[1]*m[k]+y[2]*v[i]*v[j];}}
        a=y[0]; V::mul(y[1],v);}
};

template<typename Scalar,int ndim> void show(const Dense2<Scalar,ndim> & x){x.showself();}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> log(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_log(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> exp(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_exp(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> abs(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_abs(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> sin(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_sin(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> cos(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_cos(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arcsin(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arcsin(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arccos(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arccos(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arctan(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arctan(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> sinh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_sinh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> cosh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_cosh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> tanh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_tanh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arcsinh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arcsinh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arccosh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arccosh(x.a,y); x._math_helper(y); return x;}
template<typename Scalar,int ndim> Dense2<Scalar,ndim> arctanh(Dense2<Scalar,ndim> x){Scalar y[3]; _Taylor2<Scalar>::_arctanh(x.a,y); x._math_helper(y); return x;}
