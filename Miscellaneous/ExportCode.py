import nbformat 
import json
import sys
import os
import difflib

from TestCode import ListNotebooks

help_str = """
This file export code from cells tagged 'ExportCode', from notebooks into files.

Arguments:
	* notebook filenames. (If none, assumes that executed from the root of the 
	AdaptiveGridDiscretizations repository.)
	* --update: wether to update the exported code.
	* --show: wether to show the exported code differences.
	* --root="mydirectory": Add a command to include that given directory. 
	(default : "../..")

"""

result_path = "ExportedCode"

def ExportCode(inFName,outFName,update=False,show=False):
	with open(inFName, encoding='utf8') as data_file:
		data = json.load(data_file)
	directory,filename = os.path.split(inFName)
	output = [
		f"# Code automatically exported from notebook {filename} in directory {directory}\n",
		"# Do not modify\n"]
	nAlgo = 0
	for c in data['cells']:
		if 'tags' in c['metadata'] and 'ExportCode' in c['metadata']['tags']:
			output.extend(c['source'])
			output.append('\n\n')
			nAlgo+=1
	if nAlgo==0: return
	for i,line in enumerate(output): 
		output[i] = line.replace('from agd import', 'from ... import')
	try:
		with open(outFName,'r',encoding='utf8') as output_file:
			output_previous = output_file.readlines()
	except FileNotFoundError:
		output_previous=[""]

	changes_found=False
	for prev,new in zip(output_previous,output):
		if prev.rstrip()!=new.rstrip(): 
			changes_found=True
			if show:
				print('--- Difference found ---')
				print([prev])
				print([new])
	if not changes_found: return
	else: print(f"Changes in code tagged for export in file {inFName}")

#if show:
#		for line in difflib.unified_diff(output_previous,output,
#			fromfile=inFName+'_previous',tofile=inFName):
#			print(line)
	if update:
		print("Exporting ", nAlgo, " code cells from notebook ", inFName, " in file ", outFName)
		with open(outFName,'w+', encoding='utf8') as output_file:
			output_file.write(''.join(output))

def Main(notebook_filenames=tuple(),**kwargs):
	if len(notebook_filenames)==0: notebook_filenames = ListNotebooks()
	for name in notebook_filenames:
		ExportCode(name+'.ipynb',os.path.join('agd/ExportedCode',name)+'.py',**kwargs)

if __name__ == '__main__':
	if "--help" in sys.argv[1:]:
		print(help_str)
		exit(0)

	kwargs = {key[2:]:True for key in sys.argv[1:] if key[:2]=='--' and '=' not in key}
	kwargs.update([key[2:].split('=') for key in sys.argv[1:] if key[:2]=='--' and '=' in key])
	notebook_filenames = [key for key in sys.argv[1:] if key[:2]!='--']

	Main(notebook_filenames,**kwargs)

