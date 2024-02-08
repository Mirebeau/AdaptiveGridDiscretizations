This directory illustrates Matlab's use of the agd Python library, to solve various anisotropic eikonal equations, on the CPU and GPU (cuda capable device required).

Please see the Python notebooks for more detailed examples and more complete mathematical descriptions.

### Alternative approach (using MEX, and CPU only)

Another approach to solve anisotropic eikonal equations in Matlab is to compile the HamiltonFastMarching (hfm) library using  MEX, see 
https://github.com/Mirebeau/HamiltonFastMarching
and within that
Interfaces/MatlabHFM/ExampleFiles
However, the MEX approach only allows using the CPU eikonal solver. 