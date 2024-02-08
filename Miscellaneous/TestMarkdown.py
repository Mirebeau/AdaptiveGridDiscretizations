import os
import sys
import json

help_str = """
This file tests for an annoying Markdown typesetting issue: 
a mathematical expression of the form
$$
	a
	+ b
$$
will render correctly on some markdown viewers (including the jupyter notebook editor), 
and not on others (including nbviewer).

The solution is to replace + with {+} in the start of a line, 
if the next character is a blank space. Likewise for - and *.
I have once encountered a rare similar bug with =, but was not able to reproduce.

---
Another issue was encountered in an equation of the form
$$
<n
$$

--- 
Another issue occurs when a line contains a single starting comment tag, 
and the previous line is not blank.

blabla.
<!---
my comment
--->

---
Another issue is that the command \\rm is not interpreted in the same manner depending
on the markdown front-end : it may apply to all the following characters, or only one. 
Prefer \\mathrm.

----
We also test for invalid links within the AdaptiveGridDiscretizations directory
"""

import os.path

def check_link(a,s,dirname,where,cell):
	# Exclude web links, or internal links
	if a.startswith('http') or a.startswith('#'): return 
	# Check link
	if os.path.isfile(os.path.join(dirname,a)): return 
	print(f"Found invalid link {a}, in file {where}, line contents : \n {s}")
	showcell(cell)
	 
def check_links(s,*args):
	for a in s.split("](")[1:]:
		check_link(a.split(")")[0],s,*args)

def showcell(cell,source_only=True,show=None,check_raise=None): 
	if show is None: show=showcell.show
	if check_raise is None: check_raise=showcell.check_raise

	if show: 
		if source_only: print("(Cell source) : \n", *cell["source"])
		else: print("'Cell contents) : ",cell)
	if check_raise: raise ValueError("Error found")

def TestMarkdownCell(where,cell,cache,dirname):
	eqn = None
	prevLine="\n"
	for line in cell['source']:
		if line in ("$$","$$\n", # Note : no space allowed after $$
			"\\begin{equation*}\n","\\end{equation*}\n",
			"\\begin{align*}\n","\\end{align*}\n"):
			eqn = "" if eqn is None else None
			continue
		if eqn is None:
			check_links(line,dirname,where,cell)
		else:
			eqn = eqn+line
			l = line.lstrip()
			if line[0]=='<' or (l[0] in ['+','-','*'] and l[1]==' '): # also '=' ? 
				print(f"--- Markdown displaymath issue ", where, " : ---")
				print(eqn)
				showcell(cell)
		if line==("<!---\n") and prevLine!="\n":
			print(f"--- Markdown comment issue ", where, ": ---")
			print([prevLine],line)
			showcell(cell)
		if "\\rm " in line:
			print(f"--- Mardown math issue ", where, 
				"prefer \\mathrm{bla} to {\\rm bla} ---")
			print(line)
			showcell(cell)

		prevLine = line

def TestCodeCell(where,cell,cache):
	if len(cell['source'])==0: return
	expected_count = cache['execution_count']
	cell_count = cell['execution_count']
	if cell_count is not None: cache['execution_count']=cell_count+1

	if cell_count is None and cache.get('allcount',True):
		print("Unexecuted cell ", where)
		showcell(cell,source_only=False)
		cache['allcount']=False
	elif expected_count!=cell_count and cache.get('csqcount',True):
		print("Non consecutive execution_count ", where)
		cache['csqcount']=False
		showcell(cell,check_raise=False,show=False)

def TestNotebook(dirname,filename):
	filepath = os.path.join(dirname,filename)
	with open(filepath, encoding='utf8') as data_file:
		data = json.load(data_file)
	cache={'execution_count':1}
	for cell in data["cells"]:
		where = f" in file {filepath}, expected cell number {cache['execution_count']}"
		if cell['cell_type']=='markdown': TestMarkdownCell(where,cell,cache,dirname)
#		if cell['cell_type']=='code': TestCodeCell(where,cell,cache)

def Main(show=False,check_raise=False):
	showcell.show = show
	showcell.check_raise = check_raise
	for dirname in os.listdir():
		if not dirname.startswith("Notebooks_"): continue
		for filename in os.listdir(dirname):
			if not filename.endswith(".ipynb"): continue
			TestNotebook(dirname,filename)

if __name__ == '__main__':
	if "--help" in sys.argv[1:]:
		print(help_str)
		exit(0)

	kwargs = {"show":False,"check_raise":False}
	for key in sys.argv[1:]:
		assert key[:2]=="--" and key[2:] in kwargs
		kwargs[key[2:]]=True

	Main(**kwargs)
