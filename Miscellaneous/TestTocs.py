import nbformat
import os
import json
import TocTools
import sys
from distutils.util import strtobool

"""
This file, when executed as a Python script, tests that the table of contents 
of the notebooks are all correct, and reports any inconsistency.

Optional arguments :
	--update, the tables of contents are updated.
	--show, the tables of contents are shown
	--GPU_config=True/False, turn on or off gpu config
	--check_raise, raise an exception if toc is incorrect
"""

def ListNotebookDirs():
	return [dirname for dirname in os.listdir() if dirname[:10]=="Notebooks_"]
def ListNotebookFiles(dirname):
	filenames_extensions = [os.path.splitext(f) for f in os.listdir(dirname)]
	return [filename for filename,extension in filenames_extensions 
	if extension==".ipynb" and filename!="Summary"]

def CheckTodo(filepath,data):
	if not CheckTodo.run: return
	for cell in data['cells']:
		if 'tags' in cell['metadata']:
			tags = cell['metadata']['tags']
			for tag in tags:
				if tag.lower()=='todo':
					print(f"TODO found in notebook {filepath}");
					print(tags)
					if CheckTodo.show: print(cell['source'])

CheckTodo.run = False
CheckTodo.show = False

def UpdateConfig(filepath,data):
	"""
	Updates the EikonalGPU_config cell (comment or uncomment).
	Returns : wether an update was performed.
	"""
	if UpdateConfig.GPU_config is None:return False
	for cell in data['cells']:
		if 'tags' in cell['metadata']:
			tags = cell['metadata']['tags']
			if 'EikonalGPU_config' in tags or 'GPU_config' in tags: break
	else: return False

	request = UpdateConfig.GPU_config 
	source = cell['source']
	present = not source[0].startswith('#')

	if request==present: return False
	if not UpdateConfig.silent: print(f"Inconsistent config found for Notebook {filepath}")
	if UpdateConfig.show: print(source)
	if request: source[0] = source[0][1:]
	else: source[0] = '#'+source[0]
	return True

UpdateConfig.show = False
UpdateConfig.GPU_config = None
UpdateConfig.silent = False

def UpdateToc(filepath,data,toc):
	"""
	Updates the table of contents.
	Returns : wether an update was performed
	"""
	for cell in data['cells']:
		if (('tags' in cell['metadata'] and 'TOC' in cell['metadata']['tags'])
			or (len(cell['source'])>0 and cell['source'][0]==toc[0])): break 
	else: raise ValueError(f"TOC not found for {filepath}")

	# A bit of cleanup
	while toc[-1]=="\n": toc=toc[:-1]
	toc[-1]=toc[-1].rstrip()
	cell['source'][-1] = cell['source'][-1].rstrip()

	if toc==cell['source']: return False # No need to update

	print(f"TOC of file {filepath} needs updating")
	if UpdateToc.show:
		print("------- Old toc -------\n",*cell['source'])
		print("------- New toc -------\n ",*toc)

	cell['source'] = toc
	return True

UpdateToc.show = False

def UpdateHeader(filepath,data):
	"""
	Update the first cell of the notebook, to follow the header conventions.
	-- Returns : wether an update is required.
	"""
	dirname,_ = os.path.split(filepath)
	s = data['cells'][0]['source']
	line0 = s[0].strip()
	line0_ref = (
		"# The HFM library - A fast marching solver with adaptive stencils"
		if dirname=="Notebooks_FMM" else
		"# Adaptive PDE discretizations on cartesian grids")

	if line0!=line0_ref:
		print(f"Notebook {filepath}, line0 : {line0}, differs from expexted {line0_ref}")
		s[0] = line0_ref

	line1 = s[1].strip()
	line1_ref = {
	'Notebooks_Algo':	"## Volume : Algorithmic tools",
	'Notebooks_Div':	"## Volume : Divergence form PDEs",
	'Notebooks_NonDiv':	"## Volume : Non-divergence form PDEs",
	'Notebooks_FMM':	"",
	'Notebooks_Repro':	"## Volume : Reproducible research",
#	'Notebooks_GPU':    "## Volume : GPU accelerated methods",
	}[dirname]

	if line1!=line1_ref:
		print(f"Notebook {filepath}, line1 : '{line1}' differs from expected '{line1_ref}'")
		s[1] = line1_ref

	return line0!=line0_ref or line1!=line1_ref

def Load(filepath):
	with open(filepath, encoding='utf8') as data_file: 
		return json.load(data_file)

def Dump(filepath,data):
	msg = f"-- Updates needed for file {filepath} --"
	if Dump.check_raise: raise ValueError(msg)
	if not Dump.update: print(msg); return
	print(f"Updating {filepath}")
	with open(filepath,'w', encoding='utf8') as f: json.dump(data,f,ensure_ascii=False,indent=1)

Dump.check_raise = False
Dump.update = False

def TestToc(dirname,filename):
	filepath = os.path.join(dirname,filename)+".ipynb"
	data = Load(filepath)
	toc = TocTools.displayTOC(dirname+"/"+filename,dirname[10:]).splitlines(True)

	CheckTodo(filepath,data)
	updates = [
		UpdateHeader(filepath,data),
		UpdateToc(filepath,data,toc),
		UpdateConfig(filepath,data), ]

	if any(updates): Dump(filepath,data)

def TestTocs(dirname):
	filepath = os.path.join(dirname,"Summary.ipynb")
	data = Load(filepath)
	toc = TocTools.displayTOCs(dirname[10:],dirname+"/").splitlines(True)
	if UpdateToc(filepath,data,toc): Dump(filepath,data)

def TestTocss():
	filepath = "Summary.ipynb"
	data = Load(filepath)
	toc = TocTools.displayTOCss().splitlines(True)
	if UpdateToc(filepath,data,toc): Dump(filepath,data)

def Main(update=False,check_raise=False,show=False,GPU_config=None,todo=False):
	Dump.update = update
	Dump.check_raise = check_raise
	UpdateToc.show = show
	UpdateConfig.show = show
	UpdateConfig.GPU_config = GPU_config
	CheckTodo.run = todo
	TestTocss()
	for dirname in ListNotebookDirs():
		TestTocs(dirname)
		for filename in ListNotebookFiles(dirname):
			TestToc(dirname,filename)

def SysArgs(argv,prefix='--',default=True,caster=None):
	kwargs = {}
	for keyval in argv:
		if '=' in keyval : 
			key,val = keyval.split('=')
			if caster is not None: val = caster(val)
		else: 
			key = keyval
			val = default
		assert key.startswith(prefix)
		kwargs[key[len(prefix):]]=val
	return kwargs

if __name__ == '__main__':
	Main(**SysArgs(sys.argv[1:],caster=strtobool))