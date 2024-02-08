// Matrix determinant ! In dimension <=3 !
template<typename T>
T det_a(const T a[ndim][ndim]){
	if(ndim==1) {return a[0][0];}
	else if (ndim==2) {return a[0][0]*a[1][1] - a[1][0]*a[0][1];}
	else if (ndim==3) {
		T det=0;
		for (Int i = 0; i < 3; ++i) 
			det += a[i][0]*a[(i+1)%3][1]*a[(i+2)%3][2]-a[i][2]*a[(i+1)%3][1]*a[(i+2)%3][0];
		return det;
	} else {return 0;}

}

void div_kA(const Scalar k, Scalar out[ __restrict__ ndim][ndim]){
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j){out[i][j]/=k;}}}

//Matrix inversion
template<typename T>
void inv_a(const T a[ndim][ndim], Scalar out[__restrict__ ndim][ndim]){
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
		fill_kV(-1,i2j);; fill_kV(-1,j2i);
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
		// Solve remaining triangular system
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
	div_kA(det_a(a),out);
}