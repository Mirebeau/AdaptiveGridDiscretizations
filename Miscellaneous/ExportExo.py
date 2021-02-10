import json
import sys
import os

"""
Produce an exercise notebook, from a standard notebook with some special 
comments in the cell source, and optionally tags in the cell metadata.

Usage : 
	python ExportExo filename.ipynb


Notebook formatting : 
- In a markdown cell, introduce a comment of the following form.
It will create a new markdown cell, containing the comment contents, preceded with 
the exercise number.

<!---ExoFR
Statement of exercise.
--->

- In a markdown cell, introduce a comment of the following form.
It will create a new code cell, containing the comment contents. 
In addition, the next code cell is removed.

<!---ExoCode
def f(x):
	# TODO : complete the definition of f.
--->

- In an arbitrary cell, introduce the tag 'ExoRemove'.
The cell will be removed in the produced exercise.

"""

indexExo = 0
language = 'FR'

def SplitExo(c):
	text = []
	comment = []
	statementFR =[]
	statementEN=[]
	code = []
	current = text
#	print(c['source'],"\n\n\n")
	for line in c['source']:
#		print(line)
		if line == '<!---ExoFR\n':
			assert current is text
			current = statementFR
			continue
		elif line == '<!---ExoEN\n':
			assert current is text
			current = statementEN
			continue
		elif line == '<!---ExoCode\n':
			assert current is text
			current = code
			continue
		elif line == '<!---\n':
			assert current is text
			current = comment
			continue
		elif line == '--->' or line =='--->\n':
			assert current is not text
			current = text
			continue
		current.append(line)
	assert current is text
	c['source'] = text
	result = [c]
	global language,indexExo
	if len(statementFR)!=0 and language=='FR':
		indexExo+=1
		result.append({
			'cell_type':'markdown',
			'source':["*Question "+str(indexExo)+"*\n","===\n"]+statementFR,
			'metadata':{}
			})
	if len(statementEN)!=0 and language=='EN':
		indexExo+=1
		result.append({
			'cell_type':'markdown',
			'source':["*Question "+str(indexExo)+"*\n","===\n"]+statementEN,
			'metadata':{}
			})
	if len(code)!=0:
		result.append({		
		'cell_type':'code',
		'source':code,
		'execution_count':None,
		'outputs':[],
		'metadata':{},
		})
	return result

def HasTag(cell,tag):
	if 'metadata' in cell:
		metadata=cell['metadata']
		if 'tags' in metadata:
			tags = metadata['tags']
			return tag in tags
	return False

def MakeExo(FileName,ExoName):
	with open(FileName, encoding='utf8') as data_file:
		data=json.load(data_file)
	newcells = []
	removeCell = False
	for cell in data['cells']:
		if HasTag(cell,'ExoRemove') or removeCell:
			removeCell=False
			continue
		elif HasTag(cell,'ExoSplit') or cell['cell_type']=='markdown':
			if "<!---ExoCode\n" in cell['source']:
				removeCell=True # Remove next cell
			for subCell in SplitExo(cell):
				newcells.append(subCell)
			continue
		newcells.append(cell)
	data['cells']=newcells

	with open(ExoName,'w', encoding='utf8') as f:
		json.dump(data,f,ensure_ascii=False)

if __name__ == '__main__':
	for name in sys.argv[1:]:
		dir,FileName = os.path.split(name)
		prefix,ext = os.path.splitext(FileName)
		ExoName = os.path.join(dir,"Exo",prefix+'_Exo.ipynb')
		MakeExo(FileName,ExoName)