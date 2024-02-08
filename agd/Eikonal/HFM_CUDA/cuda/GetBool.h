#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** Extract a boolean value, defined as a single bit of an array, 
using "little" bit ordering.*/
bool GetBool(const BoolPack * arr, const Int n){
	HFM_DEBUG(assert(n>=0);)
	const Int m = 8*sizeof(BoolPack);
	const Int q = n/m, r=n%m;
	return (arr[q] >> r) & BoolPack(1);
}
