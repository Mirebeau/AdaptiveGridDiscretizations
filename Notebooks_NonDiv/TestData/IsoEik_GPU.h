/* Assumptions : 
- the problem is solved on a square grid of size : side_ x side_.
- side_ is a multiple of 8.
- A block of threads deals with a subdomain of size : 8 x 8.

- variables ending with "_" correspond to the global grid.
- variables 
*/

__global__ void eikonal(float * u_, float * cost_, 
long * update_list, char * update_next, int xmax_, int ymax_){

// Expecting threadDim.x=64, threadDim.y=threadDim.z=blockDim.y=blockDim.z=1
const int tid = threadIdx.x;
const int bid = blockIdx.x;

const int Xmax = 1+(xmax_-1)/8, Ymax = 1+(ymax_-1)/8;
const float inf = 1./0.;

__shared__ int X,Y;
if(tid==0){
	const long n = update_list[bid];
	X = n/Ymax; Y = n%Ymax;
}

__shared__ float u[10][10];

for(int r=0; r<2; ++r){
	const int m = tid + r*threadDim.x;
	if(m>=100) break;
	const int x = m/10, y=m%10;

	const int x_ = 8*X+x-1, y_ = 8*Y+y-1;
	u[x][y] = (0<=x_ && x_<xmax_ && 0<=y_ && y_<ymax_) ? u_[side_*x_ + y_] : inf;
}

__syncthreads();

const int x  = (tid/8)+1, y=(tid%8)+1;
const int x_ = 8*X+x-1, y_=8*Y+y-1;
const float cost = (x_<xmax_ && y_<ymax_) ? cost_[ymax_*x_+y_] : inf;
const float u_old = u[x][y];

for(int iter=0; iter<8; ++iter){

	// Get the neighbor values
	const float 
	v0 = min(u[x+1][y], u[x-1][y]),
	v1 = min(u[x][y+1], u[x][y-1]);
	const float 
	w0 = min(v0,v1), 
	w1 = max(v0,v1);

	// Compute the update
	float u_new = w0 + cost;
	if(u_new>w1){
		delta = cost*cost - (w0-w1)*(w0-w1); // Non-negative, up to machine precision
		u_new = (w0+w1 + sqrt(max(0,delta)) )/2.;
	}

	// Set the new value, if smaller. 
	// (Guaranteed to decrease, up to machine precision, 
	// except at seed points which must be preserved)
	u[x][y] = min(u[x][y],u_new);

	__syncthreads();
}  

// Export the new values
if(x_<xmax_ && y_<ymax_) u_[ymax_*x_+y_] = u[x][y];

// Check if any value has substantially changed
__shared__ bool updated=false;
if(u_new < u_old-tol) updated=true;
__syncthreads();

// Mark block and neighbors for update, if adequate
if(tid==0 && updated){
	X/=8; Y/=8;

	update_next[bside*X+Y]=1; 
	if(X+1<Xmax)  update_next[Ymax*(X+1)+Y]=1;
	if(X-1>=0)    update_next[Ymax*(X-1)+Y]=1;
	if(Y+1<Ymax)  update_next[Ymax*X+(Y+1)]=1;
	if(Y-1>=0)    update_next[Ymax*X+(Y-1)]=1;
}

}