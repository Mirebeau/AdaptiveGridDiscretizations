#pragma once

/*Scalar and Int types must be defined in enclosing file.*/

/** Usage : to minimize a function f on interval a,b.
const Scalar bounds[2] = {a,b};
Scalar midpoints[2]; golden_search::init(bounds,midpoints);
Scalar values[2]={f(midpoints[0]),f(midpoints[1])};
for(Int i=0; i<niter; ++i){
	const Int k = golden_search::step(midpoints,values);
	values[k] = f(midpoints[k]); 
}
*/

namespace golden_search {

const Scalar phi = 0.6180339887498949; // (sqrt(5.)-1.)/2;
const Scalar psi = 0.3819660112501051; // 1.-phi;

void init(const Scalar x[2], Scalar y[2]){
	y[0] = phi*x[0]+psi*x[1];
	y[1] = psi*x[0]+phi*x[1];
}
// returns the position where value needs to be updated
Int step(Scalar x[2], Scalar v[2], const bool mix_is_min=true){
	const Scalar dx = x[1]-x[0];
	if(mix_is_min == (v[0]<v[1])){
		x[1]=x[0];
		v[1]=v[0];
		x[0]-=dx*phi;
		return 0; 
	} else {
		x[0]=x[1];
		v[0]=v[1];
		x[1]+=dx*phi;
		return 1;
	}
}

} // golden_search