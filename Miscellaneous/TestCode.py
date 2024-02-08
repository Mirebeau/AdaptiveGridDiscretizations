import nbformat 
from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError
import sys
import os
import subprocess
import time

help_str = """
This script runs the code of the specified notebook, or of all the notebooks 
in the current directory, and catches and reports any raised exceptions.

Optional arguments
  --CommandLine : needed on my windows machine, due to asyncio error

Note: you may need to first run TestTocs.py with the following option first
	--GPU_config=True/False, turn on or off gpu config (cell tagged GPU_config)
"""

# ------- Specific to this repository -----

result_path = "test_results"

# --------- Generic ---------

def ListNotebooks(dir=None):
	filenames_extensions = [os.path.splitext(f) for f in os.listdir(dir)]
	filenames = [filename for filename,extension in filenames_extensions 
		if extension==".ipynb"]
	subdirectories = [filename for filename,extension in filenames_extensions 
		if extension=="" and filename.startswith("Notebooks_")] 
		# and filename!="Notebooks_Repro"
	subfilenames = [os.path.join(subdir,file) for subdir in subdirectories 
		for file in ListNotebooks(subdir)]
	return filenames+subfilenames

def TestNotebook_CommandLine(notebook_filepath, result_path):
	"""
	Caling nbconvert via command line, 
	required on one of my windows laptops, due to apparent bug with asyncio (?)
	"""
	print("Testing notebook " + notebook_filename,end='',flush=True)
	filepath,extension = os.path.splitext(notebook_filepath)
	if extension=='': extension='.ipynb'
	subdir,filename = os.path.split(filepath)
	filepath_out = os.path.join(subdir,'test_results',filename+'_out')

	start = time.time()
	process = subprocess.Popen("jupyter nbconvert --to notebook --execute "
		f" {filename}{extension}",
		stdout=subprocess.PIPE,stderr=subprocess.STDOUT, universal_newlines=True,
		cwd=os.path.join(os.getcwd(),subdir))

	output=list(iter(process.stdout.readline, ""))
	retcode = process.wait() #returncode
	print(f" [Elapsed : {int(time.time()-start)}]")

	if retcode==0:
		os.replace(filepath+".nbconvert"+extension,filepath_out+extension)
	elif output[-2].startswith('DeliberateNotebookError:'):
		print("Notebook stopped deliberately")
	else:		
		print(*output)
		print(f"Error executing the notebook {notebook_filepath} :"
			f" Returned with exit code {retcode}")
		return False
	return True

def TestNotebook(notebook_filename, result_path):
	print("Testing notebook " + notebook_filename, end='',flush=True)
	filename,extension = os.path.splitext(notebook_filename)
	if extension=='': extension='.ipynb'
	filename_out = filename+"_out"
	with open(filename+extension, encoding='utf8') as f:
		nb = nbformat.read(f,as_version=4) # alternatively nbformat.NO_CONVERT
	ep = ExecutePreprocessor(timeout=600,kernel_name='python3')
	success = True
	start = time.time()
	try:
		out = ep.preprocess(nb,{}) 
	except CellExecutionError as e:
		if 'DeliberateNotebookError' in str(e):
			DeliberateMsg = str(e).split('\n')[-2]
			print(f" Notebook {notebook_filename} stopped deliberately -- {DeliberateMsg}")
		else:
			print(f" Error executing the notebook {notebook_filename}")
			print(f"See notebook {filename_out} for the traceback.")
			print(str(e))
			success=False
	finally:
		print(f" [Elapsed : {int(time.time()-start)}]")
		subdir,file = os.path.split(filename_out)
		os.makedirs(os.path.join(subdir,result_path),exist_ok=True)
		with open(os.path.join(subdir,result_path,file)+extension, 
			mode='wt', encoding='utf8') as f:
			nbformat.write(nb, f)
		return success

tags_known = ['--CommandLine','--nodefault','--exclude_long','--help']

def help():
	print(help_str)
	print(f"{tags_known=}")

if __name__ == '__main__':
#	if not os.path.exists(result_path): os.mkdir(result_path)
#	import asyncio
#	asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

	args = sys.argv[1:]
	included,excluded,tags = [],[],[]
	for arg in args: 
		if arg.startswith('--'): tags.append(arg)
		elif arg.startswith('-'): excluded.append(arg[1:])
		else: 
			if arg.endswith('.ipynb'): arg = arg[:-6]
			included.append(arg)
	for tag in tags:
		if not tag in tags_known:
			help()
			raise ValueError(f"{tag=} not in {tags_known=}")
	if '--help' in tags:
		help()
		exit(0)

	os.environ['PYDEVD_DISABLE_FILE_VALIDATION']='1' # Allow use of frozen modules (no breakpoints)

	#Ugly defaults for my computers (excluding excessively long tests...)
	if '--nodefault' not in tags:
		import platform
		node = platform.node()
		if node=="DESKTOP-ICQG8H1": # Windows with GPU
			excluded += ["MongeAmpere","Curvature3","PucciMongeAmpere"]
#			tags.append("--CommandLine") 
		elif node in ("w-155-1.wfer.ens-paris-saclay.fr",'MBPdeJeanMarie'):
			excluded.append("MongeAmpere")
	if '--exclude_long' in tags: 
		excluded += ["MongeAmpere","Curvature3","PucciMongeAmpere","EikonalRate","EikonalEulerian_Rate"]
	excluded = list(dict.fromkeys(excluded))
	excluded_found = []

	include_all = len(included)==0
	def keep(filepath):
		"""Wether to keep the file with specified file path"""
		split = os.path.split(filepath)
		if any(e==filepath or e in split for e in excluded): 
			excluded.remove(e)
			excluded_found.append(e)
			return False
		if include_all: return True
		for e in included:
			if e==filepath or e in split:
				included.remove(e)
				return True
		return False

	notebook_filenames = [f for f in ListNotebooks() if keep(f)]
	print(f"Notebooks to be tested : {notebook_filenames}, {excluded_found=}, excluded_notfound={excluded}")

	if len(included)>0: print(f"Warning ! Could not find the following notebooks : {included}")
	notebooks_failed = []
	Tester = TestNotebook_CommandLine if '--CommandLine' in tags else TestNotebook
	for notebook_filename in notebook_filenames:
		if not Tester(notebook_filename,result_path):
			notebooks_failed.append(notebook_filename)

	if len(notebooks_failed)>0:
		print("!!! Failure !!! The following notebooks raised errors:\n"
			+" ".join(notebooks_failed))
	else:
		print("Success ! All notebooks completed without errors.")
