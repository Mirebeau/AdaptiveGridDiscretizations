# Adaptive Grid Discretizations using Lattice Basis Reduction (AGD-LBR)
## A set of tools for discretizing anisotropic PDEs on cartesian grids

This repository contains
- the agd library (Adaptive Grid Discretizations), written in Python&reg; and cuda&reg;
- a series of *jupyter notebooks* in the Python&reg; language, reproducing my research in Anisotropic PDE discretizations and their applications.
- a basic documentation, [view online](https://mirebeau.github.io/AdaptiveGridDiscretizations/agd.html),
generated with [pdoc](https://pdoc.dev/).


### The AGD library

The recommended ways to install are
```console
conda install agd -c agd-lbr
```
alternatively (required for using the GPU eikonal solver)
```console
pip install agd
```

### The notebooks

You may [visualize the notebooks online using nbviewer](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations_showcase/master/Summary.ipynb
), or experimentally [run and modify the notebooks online using GoogleColab](https://colab.research.google.com/notebook#fileId=1exIN-55tUG1LFlgoHM582k8o8zy6H46f&offline=true&sandboxMode=true).
You may need to turn on GPU acceleration in GoogleColab (typical error: cannot import cupy) : Modify->Notebook parameters->GPU.

The notebooks are intended as documentation and testing for the adg library. They encompass:
* Anisotropic fast marching methods, for shortest path computation.
* Non-divergence form PDEs, including non-linear PDEs such as Monge-Ampere.
* Divergence form anisotropic PDEs, often encountered in image processing.
* Algorithmic tools, related with lattice basis reduction methods, and automatic differentiation.

For offline consultation, please download and install [anaconda](https://www.anaconda.com) or [miniconda](https://conda.io/en/latest/miniconda.html).  
*Optionally*, you may create a dedicated conda environnement by typing the following in a terminal:
```console
conda env create --file agd-hfm.yaml
conda activate agd-hfm
```
In order to open the book summary, type in a terminal:
```console
jupyter notebook Summary.ipynb
```
Then use the hyperlinks to navigate within the notebooks.

### Matlab users

Recent versions of Matlab are able to call the Python interpreter, and thus to use the 
agd library. See Notebooks_FMM/Matlab for examples featuring the CPU and GPU eikonal solvers.
