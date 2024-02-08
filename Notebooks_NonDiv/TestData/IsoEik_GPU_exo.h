const float tol = 1e-8; // Stopping criterion tolerance
const int niter = 8; // Iterations within a block

extern "C" { 
__global__ void eikonal_update(float * u_, float * cost_, 
int * update_list, char * update_next, int xmax_, int ymax_){

// Expecting threadDim.x=64, threadDim.y=threadDim.z=blockDim.y=blockDim.z=1
const int tid = threadIdx.x;
const int bid = blockIdx.x;

// Get the index and position of the current block
// Blocks are organized in a grid of size Xmax x Ymax
// The current block has position (X,Y) within this grid
const int Xmax = 1+(xmax_-1)/8, Ymax = 1+(ymax_-1)/8;
const float inf = 1./0.;

__shared__ int X,Y;
__shared__ bool updated; // Wether this block was significantly updated

if(tid==0){
	const long n = update_list[bid];
	X = n/Ymax; Y = n%Ymax;
	updated=false;
}
__syncthreads();

// Load the solution values at the points of the block and a layer of neighbors
__shared__ float u[10][10];

for(int r=0; r<2; ++r){
	const int m = tid + r*blockDim.x;
	if(m>=100) break;
	const int x = m/10, y=m%10;

	const int x_ = 8*X+x-1, y_ = 8*Y+y-1;
	u[x][y] = (0<=x_ && x_<xmax_ && 0<=y_ && y_<ymax_) ? u_[ymax_*x_ + y_] : inf;
}

__syncthreads();

// Compute the position (x,y) within this block
const int x  = (tid/8)+1, y=(tid%8)+1;
const float u_old = u[x][y];

// Compute the position within the global grid
const int x_ = 8*X+x-1, y_=8*Y+y-1;
const bool ingrid = x_<xmax_ && y_<ymax_;
const float cost = ingrid ? cost_[ymax_*x_+y_] : inf;

// Apply the local update operator a prescribed number of times
for(int iter=0; iter<niter; ++iter){

	// Get the neighbor values
	const float 
	v0 = // TODO
	v1 = // TODO
	const float 
	w0 = // TODO 
	w1 = // TODO

	// Compute the update
	float u_new = // TODO
	if(u_new>w1){
		const float delta = // TODO
		u_new = // TODO
	}

	// Set the new value, if smaller. 
	// (Guaranteed to decrease, up to machine precision, 
	// except at seed points which must be preserved)
	u[x][y] = min(u[x][y],u_new);

	__syncthreads();
}  

// Export the updated values of the solution
if(ingrid) // TODO

// Check if any value has substantially changed
if(u[x][y] < u_old-tol) updated=true;
__syncthreads();

// Mark this block and neighbors for update, if adequate
if(tid==0 && updated){
	update_next[Ymax*X+Y]=1; 
	if(X+1<Xmax)  // TODO
	if(X-1>=0)    // TODO
	if(Y+1<Ymax)  // TODO
	if(Y-1>=0)    // TODO
}

} // void eikonal_gpu_update
} // extern "C"