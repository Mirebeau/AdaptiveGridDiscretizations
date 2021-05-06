# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import os
import numpy as np
import cupy as cp
import numbers

hfm_debug_macro = False
# Use possibly in combination -lineinfo or -G and 
# <<< cuda-memcheck -- python MyCode.py

def _cupy_has_RawModule():
	"""
	RawModule appears in cupy 8. 
	"""
	from packaging.version import Version
	return Version(cp.__version__) >= Version("9") 

def getmtime_max(directory):
	"""
	Lists all the files in the given directory, and returns the last time one of them
	was modified. Information needed when compiling cupy modules, because they are cached.
	"""
	return max(os.path.getmtime(os.path.join(directory,file)) 
		for file in os.listdir(directory))

def GetModule(source,cuoptions):
	"""Returns a cupy raw module"""
	if _cupy_has_RawModule(): return cp.RawModule(code=source,options=cuoptions)
	else: return cp.core.core.compile_with_cache(source, 
		options=cuoptions, prepend_cupy_headers=False)


def SetModuleConstant(module,key,value,dtype):
	"""
	Sets a global constant in a cupy cuda module.
	"""
	if _cupy_has_RawModule(): 
		memptr = module.get_global(key)
	else: 
		#https://github.com/cupy/cupy/issues/1703
		b = cp.core.core.memory_module.BaseMemory()
		b.ptr = module.get_global_var(key)
		memptr = cp.cuda.MemoryPointer(b,0)

	value=cp.ascontiguousarray(cp.asarray(value,dtype=dtype))
	module_constant = cp.ndarray(value.shape, value.dtype, memptr)
	module_constant[...] = value

# cuda does not have int8_t, int32_t, etc
np2cuda_dtype = {
	np.int8:'char',
	np.uint8:'unsigned char',
	np.int16:'short',
	np.int32:'int',
	np.int64:'long long',
	np.float32:'float',
	np.float64:'double',
	}

def traits_header(traits,
	join=False,size_of_shape=False,log2_size=False,integral_max=False):
	"""
	Returns the source (mostly a preamble) for the gpu kernel code 
	for the given traits.
	- join (optional): return a multiline string, rather than a list of strings
	- size_of_shape: insert traits for the size of each shape.
	- log2_size: insert a trait for the ceil of the base 2 logarithm of previous size.
	- integral_max: declare max of integral typedefs
	"""
	traits.setdefault('hfm_debug_macro',hfm_debug_macro)

	def to_c(value): 
		if isinstance(value,bool): return str(value).lower()
		else: return value

	source = []
	for key,value in traits.items():
		if key.endswith('macro'):
			source.append(f"#define {key} {to_c(value)}")
			continue
		elif (key+'_macro') not in traits:
			source.append(f"#define {key}_macro")

		if isinstance(value,numbers.Integral):
			source.append(f"const int {key}={to_c(value)};")
		elif isinstance(value,tuple) and isinstance(value[1],type):
			val,dtype=value
			line = f"const {np2cuda_dtype[dtype]} {key} = "
			if   val== np.inf: line+="1./0."
			elif val==-np.inf: line+="-1./0."
			else: line+=str(val)
			source.append(line+";")

		elif isinstance(value,type):
			ctype = np2cuda_dtype[value]
			source.append(f"typedef {ctype} {key};")
			if integral_max and issubclass(value,numbers.Integral):
				source.append(f"const {ctype} {key}_Max = {np.iinfo(value).max};")
		elif all(isinstance(v,numbers.Integral) for v in value):
			source.append(f"const int {key}[{len(value)}] = "
				+"{"+",".join(str(to_c(s)) for s in value)+ "};")
		else: 
			raise ValueError(f"Unsupported trait {key}:{value}")

	# Special treatment for some traits
	for key,value in traits.items():
		if size_of_shape and key.startswith('shape_'):
			suffix = key[len('shape_'):]
			size = np.prod(value)
			source.append(f"const int size_{suffix} = {size};")
			if log2_size:
				log2 = int(np.ceil(np.log2(size)))
				source.append(f"const int log2_size_{suffix} = {log2};")

	return "\n".join(source) if join else source


