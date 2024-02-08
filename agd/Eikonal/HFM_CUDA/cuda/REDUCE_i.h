#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** Reduction operation over a block.
Involves n_i, size_i, log2_size_i (thread identifier, number of threads, upper bound on log)
cmds should look like 
s[n_i] += s[m_i]
and the sum will be stored in s[0].
CAUTION : no syncing in the last iteration, so s[0] is only visible to thread 0.
*/
#define REDUCE_i(cmds) { \
	Int shift_=1; \
	for(Int k_=0; k_<log2_size_i; ++k_){ \
		const Int old_shift_=shift_; \
		shift_=shift_<<1; \
		if( (n_i%shift_)==0 ){ \
			const Int m_i = n_i+old_shift_; \
			if(m_i<size_i){ \
				cmds \
			} \
		} \
		if(k_<log2_size_i-1) {__syncthreads();} \
	} \
} \

