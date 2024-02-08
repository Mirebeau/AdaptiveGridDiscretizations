# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import numbers
import importlib
import importlib.util
import ast
import os

def SetInput(hfm,params):
	for key,val in params.items():
		if isinstance(val,np.ndarray) and val.ndim==0: val = val.flat[0]
		if isinstance(val,numbers.Number):
			hfm.set_scalar(key,val)
		elif isinstance(val,str):
			hfm.set_string(key,val)
		elif isinstance(val,np.ndarray):
			hfm.set_array(key,val)
		else:
			raise ValueError('Invalid type for key ' + key);

def GetOutput(hfm):
	comp=hfm.computed_keys()
	if comp[0]=='(': # Should be patched now
		comp=comp.replace('),)',"']]")
		comp=comp.replace('),(',')(')
		comp=comp.replace(',',"','")
		comp=comp.replace(')(',"'],['")
		comp=comp.replace('((',"[['")

	result = {}
	for key,t in ast.literal_eval(comp):
		if t=='float':
			result[key] = hfm.get_scalar(key)
		elif t=='string':
			result[key] = hfm.get_string(key)
		elif t=='array':
			result[key] = hfm.get_array(key)
		else:
			raise ValueError('Unrecognized type '+ t + ' for key '+ key)
	return result

def ListToNDArray(params):
	for (key,val) in params.items():
		if isinstance(val,list):
			params[key]=np.array(val)

def RunDispatch(params,bin_dir):
	modelName = params['model']
	ListToNDArray(params)
	if bin_dir is None:
		moduleName = 'HFMpy.HFM_'+modelName
		HFM = importlib.import_module(moduleName)
		hfm = HFM.HFMIO()
		SetInput(hfm,params)
		hfm.run()
		return GetOutput(hfm)
	else:
		from . import FileIO
		execName = 'FileHFM_'+modelName
		return FileIO.WriteCallRead(params, execName, bin_dir)

binary_dir={}

def GetBinaryDir(execName,libName):
	"""
	This function is used due to the special way the HamiltonFastMarching library is used:
	- either as a bunch of command line executables, whose name begins with FileHFM.
	- or as a python library, named HFMpy

	The function will look for a file named "FileHFM_binary_dir.txt" (or a global variable named FileHFM_binary_dir)
	- if it exists, the first line is read
	  - if the first line is None -> use the HFMpy library
	  - otherwise, check that the first line is a valid directory -> should contain the FileHFM executables
	- if file cannot be read -> use the HFMpy library
	"""
	if execName in binary_dir: return binary_dir[execName]
	dirName = execName + "_binary_dir"
	fileName = dirName + ".txt"
	pathExample = "path/to/"+execName+"/bin"
	set_directory_msg = f"""
You can set the path to the {execName} compiled binaries, as follows : \n
>>> {__name__}.binary_dir['{execName}']='{pathExample}'\n
\n
In order to do this automatically in the future, please set this path 
in the first line of a file named '{fileName}' in the current directory\n
>>> with open('{fileName}','w+') as file: file.write('{pathExample}')
"""
	fileName_parent = os.path.join('..',fileName)
	if os.path.isfile(fileName_parent) and not os.path.isfile(fileName):
		fileName = fileName_parent
		
	try:
		# Try reading the file
		with open(fileName,'r') as f:
			bin_dir = f.readline().replace('\n','')
			if bin_dir=="None":
				return None
			if not os.path.isdir(bin_dir):
				print(f"ERROR : the path to the {execName} binaries appears to be incorrect.\n")
				print("Current path : ", bin_dir, "\n")
				print(set_directory_msg)
			return bin_dir
	except OSError as e:
		# Try importing the library
		if libName is not None and importlib.util.find_spec(libName) is not None:
			return None
		error_msg = "ERROR :"
		if libName: error_msg+=f" the {libName} library is not found"
		if libName and execName: error_msg+=", and"
		if execName: error_msg+=f" the path to the {execName} binaries is not set"
		print(error_msg)
		print(set_directory_msg)
		raise


