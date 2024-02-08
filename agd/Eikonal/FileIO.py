# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

import numbers;
import numpy as np;
import os
from operator import mul
from functools import reduce
import subprocess
import platform


#These two methods export a dictonary to a (pair of) files, and conversely.
def RawToFiles(params,prefix='input'):
	"""
	Exports a dictionary to a pair of files, whose name begins with 'prefix'.
	The dictionnary elements must by strings, scalars, and numpy arrays.
	The resulting files are readable by the HFM library.
	"""
	assert isinstance(prefix,str)
	f = open(prefix+'_Format.txt','w')
	data=[]
	for key,val in params.items():
		if isinstance(val,numbers.Number):
			f.write(key+'\n0\n\n')
			data.append([val])
		elif isinstance(val,str):
			f.write(key+'\n-1\n'+val.encode("unicode_escape").decode("utf-8")+'\n\n')
		elif type(val)==np.ndarray:
			f.write(key+'\n'+str(val.ndim)+'\n')
			for dim in val.shape:
				f.write(str(dim)+'\n')
			f.write('\n')
			data.append(val.flatten())
		else:
			raise ValueError(f"Invalid type for key {key} : {type(val)}")
	f.close()
	np.concatenate(data).astype('d').tofile(prefix+'_Data.dat')
	
def FilesToRaw(prefix='output'):
	"""
	Imports a pair of files, whose name begins with 'prefix', into a dictionary.
	These files may be produced by the HFM library.
	"""
	data=np.fromfile(prefix+'_Data.dat')
	pos=0;
	f=open(prefix+'_Format.txt')
	dict={}
	while True:
		key=f.readline().strip()
		if not key: break
		keyType = int(f.readline())
		if keyType==-1:
			dict[key]=f.readline().strip().encode("utf-8").decode("unicode_escape")
		elif keyType==0:
			dict[key]=data[pos]
			pos+=1
		else:
			dims=[int(f.readline()) for i in range(keyType)]
			size=reduce(mul,dims)
			dict[key]=np.reshape(data.take(np.arange(pos,pos+size)),dims)
			pos+=size
		f.readline()
	return dict

def WriteCallRead(inputData,executable,binary_dir='',working_dir=None,
				  inputPrefix="input",outputPrefix="output"):
	if working_dir is None: working_dir=binary_dir
	if platform.system()=='Linux' and binary_dir=='.': binary_dir=''

	RawToFiles(inputData,os.path.join(working_dir,inputPrefix) ) # Export inputData
	
	process = subprocess.Popen([os.path.join(binary_dir,executable),inputPrefix,outputPrefix],
		stdout=subprocess.PIPE,stderr=subprocess.STDOUT, universal_newlines=True,cwd=working_dir)
	for stdout_line in iter(process.stdout.readline, ""):
		print(stdout_line,end='')
	retcode = process.wait() #returncode
	if retcode!=0:  print('Returned with exit code ', retcode)
	
	outputData = FilesToRaw(os.path.join(working_dir,outputPrefix)) # Import outputData
	outputData['retcode']=retcode
	return outputData