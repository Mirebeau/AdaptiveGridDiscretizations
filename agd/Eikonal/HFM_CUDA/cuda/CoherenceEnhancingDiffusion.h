__constant__ Scalar ced_alpha=-1,ced_gamma=1,
ced_cond_max=100,ced_cond_amplification_threshold=2;


// Polynomial (affine or constant) interpolating two values 
void interpolate2(const Scalar x[__restrict__ 2], const Scalar y[__restrict__ 2], 
	Scalar p[__restrict__ 2]){ 
	if(x[0]==x[1]){p[0]=y[0]; p[1]=0; return;}
	p[1] = (y[1]-y[0])/(x[1]-x[0]);
	p[0] = y[0]-p[1]*x[0];
}

// Polynomial (degree <= 2) interpolating three values 
void interpolate3(const Scalar x[__restrict__ 3], const Scalar y[__restrict__ 3], 
	Scalar p[__restrict__ 3]){
	if(x[0]==x[1]){p[2]=0; return interpolate2(x+1,y+1,p);}
	if(x[1]==x[2] or x[0]==x[2]){p[2]=0; return interpolate2(x,y,p);}
	const Scalar z0=y[0]*(x[1]-x[2]), z1=y[1]*(x[2]-x[0]), z2=y[2]*(x[0]-x[1]);
	p[0]=z0*x[1]*x[2]+z1*x[0]*x[2]+z2*x[0]*x[1];
	p[1]=-(z0*(x[1]+x[2])+z1*(x[0]+x[2])+z2*(x[0]+x[1]));
	p[2]=z0+z1+z2;
	const Scalar det = (x[0]-x[1])*(x[0]-x[2])*(x[1]-x[2]);
	for(int i=0; i<3; ++i) p[i]/=det;
}


#if ndim_macro==2
/// Change the eigenvalues of a symmetric matrix. lambda : original. mu : final.
void map_eigvalsh(Scalar m[__restrict__ symdim], 
	const Scalar lambda[__restrict__ ndim], const Scalar mu[__restrict__ ndim]){
	Scalar p[ndim]; interpolate2(lambda,mu,p);
	for(int i=0,k=0; i<ndim; ++i) for(int j=0; j<=i; ++j,++k) m[k] = p[0]*(i==j)+p[1]*m[k];
}

void ced(const Scalar lambda[__restrict__ ndim], Scalar mu[__restrict__ ndim]){
	const Scalar lambda_diff = lambda[1]-ced_cond_amplification_threshold*lambda[0];
    mu[0] = lambda_diff<=0 ? ced_alpha : 
    ced_alpha+(1.-ced_alpha)*exp(-ced_gamma/(lambda_diff*lambda_diff));
    mu[1] = max(ced_alpha,mu[0]/ced_cond_max);
}
#elif ndim_macro==3
/// Change the eigenvalues of a symmetric matrix. lambda : original. mu : final.
void map_eigvalsh(Scalar m[__restrict__ symdim], 
	const Scalar lambda[__restrict__ ndim], const Scalar mu[__restrict__ ndim]){
	Scalar p[ndim]; 
	interpolate3(lambda,mu,p);
	Scalar m_out[symdim];

	for(int i=0,k=0; i<ndim; ++i) {
		for(int j=0; j<=i; ++j,++k) {
			Scalar m2=0.; // Becomes the coefficient (i,j) of the matrix m**2
			for(int r=0; r<ndim; ++r) m2+=coef_m(m,i,r)*coef_m(m,r,j);
			m_out[k] = p[0]*(i==j)+p[1]*m[k]+p[2]*m2;
		}
	}
	copy_mM(m_out,m);
/*	if(blockIdx.x*blockDim.x + threadIdx.x==94531){
		printf("lambda %f,%f,%f \n",lambda[0],lambda[1],lambda[2]);
		printf("mu %f,%f,%f \n",mu[0],mu[1],mu[2]);
		printf("p %f,%f,%f\n",p[0],p[1],p[2]);
	}*/
}

void ced(const Scalar lambda[__restrict__ ndim], Scalar mu[__restrict__ ndim]){
	Scalar lambda_diff = lambda[2]-ced_cond_amplification_threshold*lambda[0];
	mu[0] = lambda_diff<=0 ? ced_alpha : 
	ced_alpha+(1.-ced_alpha)*exp(-ced_gamma/(lambda_diff*lambda_diff));
	const Scalar mu_min = mu[0]/ced_cond_max; // Minimum diffusivity
	lambda_diff = lambda[2]-ced_cond_amplification_threshold*lambda[1];
	mu[1] = max(mu_min,lambda_diff<=0 ? ced_alpha : 
		ced_alpha+(1.-ced_alpha)*exp(-ced_gamma/(lambda_diff*lambda_diff)));
	mu[2] = max(mu_min,ced_alpha);
}
#endif