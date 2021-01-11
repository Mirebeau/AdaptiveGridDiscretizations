import nbformat 
from nbconvert.preprocessors import ExecutePreprocessor,CellExecutionError
import sys
import os

if __name__ == '__main__':
	notebook_filename = 'SternBrocot.ipynb'
	with open(notebook_filename) as f:
		nb = nbformat.read(f, as_version=4)
	ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
	ep.preprocess(nb, {'metadata': {'path': '.'}})
	with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
		nbformat.write(nb, f)