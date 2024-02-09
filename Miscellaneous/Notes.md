## Make and serve slides

```console
jupyter nbconvert InteractiveStencils.ipynb --to slides --post serve
```
See [link](https://medium.com/@mjspeck/presenting-code-using-jupyter-notebook-slides-a8a3c3b59d67)

## Number of lines of code

```console
find . "(" -name "*.h" -or -name "*.hxx" -or -name "*.cxx" -or -name "*.hpp" ")" -print0 | xargs -0 wc -l
```

```console
find . "(" -name "*.py" ")" -print0 | xargs -0 wc -l
```

## cupy install

After the agd-hfm_gpu environnement is installed, type in command line:

```console
pip install cupy-cuda102
``` 

The suffix should be the cudatoolkit version number.

## documentation

In a terminal in AdaptiveGridDiscretizations directory

Build documentation (on a machine with cupy installed):
pip install pdoc
pdoc -t Miscellaneous/ -o ../AdaptiveGridDiscretizations_help/docs agd

View documentation in interactive mode : 
pdoc -t Miscellaneous/ agd
Quit the interactive mode : Ctrl+C (MacOs), Ctrl+fn+S (Windows, Dell keyboard)


<!---
All terminal commands presented here assume that the base directory is the directory containing this file.

### Anisotropic Fast Marching methods

In folder *Notebooks_FMM*. A series of notebooks illustrating the Hamilton-Fast-Marching (HFM) library, which is devoted to solving shortest path problems w.r.t. anisotropic metrics. These notebooks are intended as documentation, user's guide, and test cases for the HFM library.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_FMM/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_FMM/Summary.ipynb
```

In order to run these notebooks, you need the binaries of the HFM library. It is open source and available on the following [Github repository](https://github.com/mirebeau/AdaptiveGridDiscretizations)

### Non-linear second order PDEs in non-divergence form

In folder *Notebooks_NonDiv*. This collection of notebooks presents a series of general principles and reference implementations for *Non-linear  Partial Differential Equations (PDEs) in non-divergence form*, using *adaptive finite difference schemes on cartesian grids*.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_NonDiv/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_NonDiv/Summary.ipynb
```

### Anisotropic PDEs in divergence form

In folder *Notebooks_Div*. This collection of notebooks illustrates the discretization of *anisotropic PDEs in divergence form*, using non-negative discretizations which obey the discrete maximum principle.

You can view the summary of this series [online](http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations/master/Notebooks_Div/Summary.ipynb), or open it offline with the following terminal command:
```console
jupyter notebook Notebooks_Div/Summary.ipynb
```
--->
