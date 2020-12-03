#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file implements efficient sorting methods, especially for small array sizes n<=32.
We use a network sort for n<=16, and merge sorts for larger n.
Note that these sorting methods are intended to be used on a single thread, hence
we do not attempt to extract parallelism.*/

template<int n,typename T> void fixed_length_sort(const T values[n], int order[n]);
// values, order, and tmp, must hold n elements
template<typename T> void variable_length_sort(const T * values, 
	int * order, int * tmp, int n);

namespace _NetworkSort {

#define ORDER_SWAP(i,j) { \
	const int _i = order[i], _j=order[j]; \
	const bool b = values[_i]<values[_j]; \
	order[i] = b ? _i : _j; \
	order[j] = b ? _j : _i; \
} \

template<typename T> void sort2(const T values[2],int order[2]){
ORDER_SWAP(0,1);
};

template<typename T> void sort3(const T values[3],int order[3]){
ORDER_SWAP(1,2);
ORDER_SWAP(0,2);
ORDER_SWAP(0,1);
};

template<typename T> void sort4(const T values[4],int order[4]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);
ORDER_SWAP(1,2);
};

template<typename T> void sort5(const T values[5],int order[5]){
ORDER_SWAP(0,1);ORDER_SWAP(3,4);
ORDER_SWAP(2,4);
ORDER_SWAP(2,3);ORDER_SWAP(1,4);
ORDER_SWAP(0,3);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);
ORDER_SWAP(1,2);
};

template<typename T> void sort6(const T values[6],int order[6]){
ORDER_SWAP(1,2);ORDER_SWAP(4,5);
ORDER_SWAP(0,2);ORDER_SWAP(3,5);
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);
ORDER_SWAP(2,4);ORDER_SWAP(1,3);
ORDER_SWAP(2,3);
};

template<typename T> void sort7(const T values[7],int order[7]){
ORDER_SWAP(1,2);ORDER_SWAP(3,4);ORDER_SWAP(5,6);
ORDER_SWAP(0,2);ORDER_SWAP(3,5);ORDER_SWAP(4,6);
ORDER_SWAP(0,1);ORDER_SWAP(4,5);ORDER_SWAP(2,6);
ORDER_SWAP(0,4);ORDER_SWAP(1,5);
ORDER_SWAP(0,3);ORDER_SWAP(2,5);
ORDER_SWAP(1,3);ORDER_SWAP(2,4);
ORDER_SWAP(2,3);
};

template<typename T> void sort8(const T values[8],int order[8]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);ORDER_SWAP(4,6);ORDER_SWAP(5,7);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(0,4);ORDER_SWAP(3,7);
ORDER_SWAP(1,5);ORDER_SWAP(2,6);
ORDER_SWAP(1,4);ORDER_SWAP(3,6);
ORDER_SWAP(2,4);ORDER_SWAP(3,5);
ORDER_SWAP(3,4);
};

template<typename T> void sort9(const T values[9],int order[9]){
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(6,7);
ORDER_SWAP(1,2);ORDER_SWAP(4,5);ORDER_SWAP(7,8);
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(6,7);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);ORDER_SWAP(5,8);
ORDER_SWAP(3,6);ORDER_SWAP(4,7);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);ORDER_SWAP(5,7);ORDER_SWAP(2,6);
ORDER_SWAP(1,3);ORDER_SWAP(4,6);
ORDER_SWAP(2,4);ORDER_SWAP(5,6);
ORDER_SWAP(2,3);
};

template<typename T> void sort10(const T values[10],int order[10]){
ORDER_SWAP(4,9);ORDER_SWAP(3,8);ORDER_SWAP(2,7);ORDER_SWAP(1,6);ORDER_SWAP(0,5);
ORDER_SWAP(1,4);ORDER_SWAP(6,9);ORDER_SWAP(0,3);ORDER_SWAP(5,8);
ORDER_SWAP(0,2);ORDER_SWAP(3,6);ORDER_SWAP(7,9);
ORDER_SWAP(0,1);ORDER_SWAP(2,4);ORDER_SWAP(5,7);ORDER_SWAP(8,9);
ORDER_SWAP(1,2);ORDER_SWAP(4,6);ORDER_SWAP(7,8);ORDER_SWAP(3,5);
ORDER_SWAP(2,5);ORDER_SWAP(6,8);ORDER_SWAP(1,3);ORDER_SWAP(4,7);
ORDER_SWAP(2,3);ORDER_SWAP(6,7);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);
ORDER_SWAP(4,5);
};

template<typename T> void sort11(const T values[11],int order[11]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);
ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(9,10);ORDER_SWAP(0,4);ORDER_SWAP(3,7);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(4,8);
ORDER_SWAP(5,9);ORDER_SWAP(2,6);ORDER_SWAP(0,4);ORDER_SWAP(3,8);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(2,3);ORDER_SWAP(8,9);
ORDER_SWAP(1,4);ORDER_SWAP(7,10);ORDER_SWAP(3,5);ORDER_SWAP(6,8);
ORDER_SWAP(2,4);ORDER_SWAP(7,9);ORDER_SWAP(5,6);
ORDER_SWAP(3,4);ORDER_SWAP(7,8);
};

template<typename T> void sort12(const T values[12],int order[12]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);
ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(9,10);ORDER_SWAP(0,4);ORDER_SWAP(7,11);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(3,7);ORDER_SWAP(4,8);
ORDER_SWAP(5,9);ORDER_SWAP(2,6);ORDER_SWAP(0,4);ORDER_SWAP(7,11);ORDER_SWAP(3,8);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(2,3);ORDER_SWAP(8,9);
ORDER_SWAP(1,4);ORDER_SWAP(7,10);ORDER_SWAP(3,5);ORDER_SWAP(6,8);
ORDER_SWAP(2,4);ORDER_SWAP(7,9);ORDER_SWAP(5,6);
ORDER_SWAP(3,4);ORDER_SWAP(7,8);
};

template<typename T> void sort13(const T values[13],int order[13]){
ORDER_SWAP(1,7);ORDER_SWAP(9,11);ORDER_SWAP(3,4);ORDER_SWAP(5,8);ORDER_SWAP(0,12);ORDER_SWAP(2,6);
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,6);ORDER_SWAP(8,11);ORDER_SWAP(7,12);ORDER_SWAP(5,9);
ORDER_SWAP(0,2);ORDER_SWAP(3,7);ORDER_SWAP(10,11);ORDER_SWAP(1,4);ORDER_SWAP(6,12);
ORDER_SWAP(7,8);ORDER_SWAP(11,12);ORDER_SWAP(4,9);ORDER_SWAP(6,10);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(1,7);
ORDER_SWAP(2,6);ORDER_SWAP(9,11);ORDER_SWAP(1,3);ORDER_SWAP(4,7);ORDER_SWAP(8,10);ORDER_SWAP(0,5);
ORDER_SWAP(2,5);ORDER_SWAP(6,8);ORDER_SWAP(9,10);
ORDER_SWAP(1,2);ORDER_SWAP(3,5);ORDER_SWAP(7,8);ORDER_SWAP(4,6);
ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);
};

template<typename T> void sort14(const T values[14],int order[14]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(3,7);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
};

template<typename T> void sort15(const T values[15],int order[15]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(12,14);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(10,14);ORDER_SWAP(3,7);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);ORDER_SWAP(6,14);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(13,14);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(11,14);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
};

template<typename T> void sort16(const T values[16],int order[16]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);ORDER_SWAP(14,15);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(12,14);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);ORDER_SWAP(13,15);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(10,14);ORDER_SWAP(3,7);ORDER_SWAP(11,15);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);ORDER_SWAP(6,14);ORDER_SWAP(7,15);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(13,14);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(11,14);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
};

} // Namespace _NetworkSort

namespace _VariableLengthSort {

template<typename T>
void dispatch(const T * __restrict__ values, int * __restrict__ order, int n){
	using namespace _NetworkSort;
	switch(n) {
		case 0: return; 
		case 1: return;
		case 2: return sort2(values,order);
		case 3: return sort3(values,order);
		case 4: return sort4(values,order);
		case 5: return sort5(values,order);
		case 6: return sort6(values,order);
		case 7: return sort7(values,order);
		case 8: return sort8(values,order);
		case 9: return sort9(values,order);
		case 10:return sort10(values,order);
		case 11:return sort11(values,order);
		case 12:return sort12(values,order);
		case 13:return sort13(values,order);
		case 14:return sort14(values,order);
		case 15:return sort15(values,order);
		case 16:return sort16(values,order);
		default:break; // Should not happen
	}
}

template<typename T>
void merge(const T * __restrict__ values, 
	const int * __restrict__ source, int * __restrict__ dest, int n0, int n1){
	const int n=n0+n1;
	const int * beg0 = source; const int *end0 = source+n0;
	const int * beg1 = end0;   const int *end1 = source+n;
	for(int i=0; i<n; ++i){
		if(beg0==end0 || (beg1!=end1 && values[*beg0]>values[*beg1]) ){
			  *dest=*beg1; ++beg1;}
		else {*dest=*beg0; ++beg0;}
		++dest;
	}
}

/*
// Apparently, cuda doesn't like recursive functions. I get illegal address error whereas 
this works fine on the cpu. A flattened version is implemented below, avoiding recursion.
(Reason : cuda compiler needs to upper bound recursion depth at compile time.)
template<typename T>
void recurse(const T * values,int * source, int * dest, int n, int rec){
	// rec even : source->source
	// rec odd  : source->dest
	if(rec==0) return dispach<T>(values, source, n);
	const int rrec = rec-1; const int n0=n/2; const int n1=n-n0;
	recurse<T>(values, source, dest, n0, rrec);
	recurse<T>(values, source+n0, dest+n0, n1, rrec);
	if(rec%2==0) {merge<T>(values,dest,source,n0,n1);}
	else         {merge<T>(values,source,dest,n0,n1);}
}*/

template<typename T>
void recurse(const T * __restrict__ values, 
	int * __restrict__ source, int * __restrict__ dest, int n, int rec){
	int step = 16; int nstep = n/step;
	for(int i=0; i<nstep; ++i){dispatch(values,source+i*step,16);}
	dispatch(values,source+nstep*step,n%step);
	for(int r=1; r<=rec; ++r){
		const int old_step = step;
		step*=2;
		const int final_step = nstep%2;
		nstep/=2;
		if(r%2==0){
			for(int i=0; i<nstep; ++i) merge<T>(values,dest+i*step,source+i*step,old_step,old_step);
			if(final_step) merge<T>(values,dest+nstep*step,source+nstep*step,old_step,(n%step)-old_step);
			else{for(int k=nstep*step; k<n; ++k){source[k]=dest[k];}}
		} else {
			for(int i=0; i<nstep; ++i) merge<T>(values,source+i*step,dest+i*step,old_step,old_step);
			if(final_step) merge<T>(values,source+nstep*step,dest+nstep*step,old_step,(n%step)-old_step);
			else{for(int k=nstep*step; k<n; ++k){dest[k]=source[k];}}
		}
	}
}

} // _VariableLengthSort

namespace _FixedLengthSort {
template<int n,typename T>
void dispatch(const T values[__restrict__ n],int order[__restrict__ n]){
	using namespace _NetworkSort;
	switch(n) { // Hopefully, no code generated thanks to dead branch removal
//		case 0: return; // Arrays of length zero are not permitted
		case 1: return;
		case 2: return sort2(values,order);
		case 3: return sort3(values,order);
		case 4: return sort4(values,order);
		case 5: return sort5(values,order);
		case 6: return sort6(values,order);
		case 7: return sort7(values,order);
		case 8: return sort8(values,order);
		case 9: return sort9(values,order);
		case 10:return sort10(values,order);
		case 11:return sort11(values,order);
		case 12:return sort12(values,order);
		case 13:return sort13(values,order);
		case 14:return sort14(values,order);
		case 15:return sort15(values,order);
		case 16:return sort16(values,order);
		default:break; // Should not happen
	}
}

template<int n0, int n1, typename T, int n=n0+n1>
void merge(const T values[__restrict__ n], 
	const int source[__restrict__ n], int dest[__restrict__ n]){
	const int * beg0 = source; const int *end0 = source+n0;
	const int * beg1 = end0;   const int *end1 = source+n;
	for(int i=0; i<n; ++i){
		if(beg0==end0 || (beg1!=end1 && values[*beg0]>values[*beg1]) ){
			  *dest=*beg1; ++beg1;}
		else {*dest=*beg0; ++beg0;}
		++dest;
	}
}


template<int n,int rec,typename T>
void recurse(const T values[__restrict__ n],
	int source[__restrict__ n],int dest[__restrict__ n]){
	// rec even : source->source
	// rec odd : source->dest
	if(rec==0) return dispatch<n,T>(values, source);
	const int rrec = rec-1; const int n0=n/2; const int n1=n-n0; //Natural parameters
	const int rrec_ = rrec >=0 ? rrec : 0, n0_ = n0>=1 ? n0 : 1, n1_ = n1>=1 ? n1 : 1;
	recurse<n0_,rrec_,T>(values, source, dest);
	recurse<n1_,rrec_,T>(values, source+n0, dest+n0);
	if(rec%2==0) {merge<n0,n1,T>(values,dest,source);}
	else         {merge<n0,n1,T>(values,source,dest);}
}

} // _FixedLenSort

template<typename T>
void variable_length_sort(const T * __restrict__ values, int * __restrict__ order, 
	int * __restrict__ tmp, int n){
	const int rec = max(0, 28-__clz((unsigned int) (n-1)));
	if(rec%2==0){
		for(int i=0; i<n; ++i) order[i]=i;
		return _VariableLengthSort::recurse<T>(values, order, tmp, n, rec);
	} else {
		for(int i=0; i<n; ++i) tmp[i]=i;
		return _VariableLengthSort::recurse<T>(values, tmp, order, n, rec);
	}
}


template<int n,typename T>
void fixed_length_sort(const T values[n], int order[n]){
	const int rec = (n>16)+(n>32)+(n>64)+(n>128)+(n>256)+(n>512)+(n>1024)+(n>2048)+(n>4096)+(n>8192);
	int tmp[n]; // Actually unused if n<=16
	if(rec%2==0){
		for(int i=0; i<n; ++i) order[i]=i;
		return _FixedLengthSort::recurse<n,rec,T>(values, order, tmp);
	} else {
		for(int i=0; i<n; ++i) tmp[i]=i;
		return _FixedLengthSort::recurse<n,rec,T>(values, tmp, order);
	}
}

/*

The following sorting networks are 'best' according to the following website
http://pages.ripco.net/~jgamble/nw.html
The following license information can be found on that website as well.

This software is copyright (c) 2018 by John M. Gamble <jgamble@cpan.org>.
--- The GNU General Public License, Version 1, February 1989 ---


--- 2
[[0,1]]
--- 3
[[1,2]]
[[0,2]]
[[0,1]]
--- 4
[[0,1],[2,3]]
[[0,2],[1,3]]
[[1,2]]
--- 5
[[0,1],[3,4]]
[[2,4]]
[[2,3],[1,4]]
[[0,3]]
[[0,2],[1,3]]
[[1,2]]
--- 6
[[1,2],[4,5]]
[[0,2],[3,5]]
[[0,1],[3,4],[2,5]]
[[0,3],[1,4]]
[[2,4],[1,3]]
[[2,3]]
--- 7
[[1,2],[3,4],[5,6]]
[[0,2],[3,5],[4,6]]
[[0,1],[4,5],[2,6]]
[[0,4],[1,5]]
[[0,3],[2,5]]
[[1,3],[2,4]]
[[2,3]]
--- 8
[[0,1],[2,3],[4,5],[6,7]]
[[0,2],[1,3],[4,6],[5,7]]
[[1,2],[5,6],[0,4],[3,7]]
[[1,5],[2,6]]
[[1,4],[3,6]]
[[2,4],[3,5]]
[[3,4]]
--- 9
[[0,1],[3,4],[6,7]]
[[1,2],[4,5],[7,8]]
[[0,1],[3,4],[6,7],[2,5]]
[[0,3],[1,4],[5,8]]
[[3,6],[4,7],[2,5]]
[[0,3],[1,4],[5,7],[2,6]]
[[1,3],[4,6]]
[[2,4],[5,6]]
[[2,3]]
--- 10
[[4,9],[3,8],[2,7],[1,6],[0,5]]
[[1,4],[6,9],[0,3],[5,8]]
[[0,2],[3,6],[7,9]]
[[0,1],[2,4],[5,7],[8,9]]
[[1,2],[4,6],[7,8],[3,5]]
[[2,5],[6,8],[1,3],[4,7]]
[[2,3],[6,7]]
[[3,4],[5,6]]
[[4,5]]
--- 11
[[0,1],[2,3],[4,5],[6,7],[8,9]]
[[1,3],[5,7],[0,2],[4,6],[8,10]]
[[1,2],[5,6],[9,10],[0,4],[3,7]]
[[1,5],[6,10],[4,8]]
[[5,9],[2,6],[0,4],[3,8]]
[[1,5],[6,10],[2,3],[8,9]]
[[1,4],[7,10],[3,5],[6,8]]
[[2,4],[7,9],[5,6]]
[[3,4],[7,8]]
--- 12
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
[[1,3],[5,7],[9,11],[0,2],[4,6],[8,10]]
[[1,2],[5,6],[9,10],[0,4],[7,11]]
[[1,5],[6,10],[3,7],[4,8]]
[[5,9],[2,6],[0,4],[7,11],[3,8]]
[[1,5],[6,10],[2,3],[8,9]]
[[1,4],[7,10],[3,5],[6,8]]
[[2,4],[7,9],[5,6]]
[[3,4],[7,8]]
--- 13
[[1,7],[9,11],[3,4],[5,8],[0,12],[2,6]]
[[0,1],[2,3],[4,6],[8,11],[7,12],[5,9]]
[[0,2],[3,7],[10,11],[1,4],[6,12]]
[[7,8],[11,12],[4,9],[6,10]]
[[3,4],[5,6],[8,9],[10,11],[1,7]]
[[2,6],[9,11],[1,3],[4,7],[8,10],[0,5]]
[[2,5],[6,8],[9,10]]
[[1,2],[3,5],[7,8],[4,6]]
[[2,3],[4,5],[6,7],[8,9]]
[[3,4],[5,6]]
--- 14
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
[[0,2],[4,6],[8,10],[1,3],[5,7],[9,11]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[3,7]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13]]
[[5,10],[6,9],[3,12],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]
--- 15
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
[[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14]]
[[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]
--- 16
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
[[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11],[13,15]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7],[11,15]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]
[[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]

*/

/* The functions of the _NetworkSort namespace are generated by the following Python 
code applied to the above networks.

def code_line(s):
    s=s.replace("[[","ORDER_SWAP(")
    s=s.replace("],[",");ORDER_SWAP(")
    s=s.replace("]]",");")
    return s

def code_network(s):
    s = s.split("\n")
    n = int(s[0])
    s = (f"template<typename T> void sort{n}(const T values[{n}],int order[{n}])" "{\n"+
         "\n".join(code_line(l) for l in s[1:])+
         "};\n")
    return s
def code_networks(s):
    s = s.split('--- ')[1:]
    return "\n".join(code_network(l) for l in s)

*/
