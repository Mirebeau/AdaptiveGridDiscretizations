% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay, 2020

InitPython % ! IMPORTANT ! : the "InitPython" script must be tuned to your machine, see source
pause('on'); % Disables pauses. (Good for testing, bad for viewing)

% This file is intended for testing, and solves a variety of eikonal
% equations on the CPU and GPU. 
% All the tests, see the loop below, can be run indepedently. Defaults
% arguments are provided. For example
% (matlab)>>> InitPython; IsotropicDemo.ConstantCost; % Runs 

% ! Warning ! : total execution time is several minutes 
% (Mostly due to the Pompidou test when running on the CPU)

% ! Warning ! : you need the hfm library installed to use the CPU eikonal
% solver. (terminal)>>> conda install hfm -c agd-lbr

% ! Warning ! : you need a cuda capable gpu to use the GPU eikonal solver,
% and the cupy python library installed. https://cupy.dev/
% In addition, only the mode "gpu_transfer" is currently supported 
% (CPU arrays are passed to Python, and are transferred to the GPU just 
% before solving the PDE)

% --- Mathematical details, enhanced examples ---
% Please see the Python notebooks for more explanations/illustrations
% http://nbviewer.jupyter.org/urls/rawgithub.com/Mirebeau/AdaptiveGridDiscretizations_showcase/master/Summary.ipynb

% --- Fast marching eikonal solver ---
for mode=["cpu","gpu_transfer"] % Chosen eikonal solver
    IsotropicDemo.ConstantCost(mode); pause()
    IsotropicDemo.SmoothCost(mode); pause()
    RiemannDemo.SmoothMetric(mode); pause()
    AsymmetricDemo.RanderMetric(mode); pause()
    AsymmetricDemo.AsymmetricQuadraticMetric(mode); pause()
    for model=["ReedsShepp2","ReedsSheppForward2","Elastica2","Dubins2"]
        CurvatureDemo.ConstantCost(mode,model); pause();
        CurvatureDemo.Pompidou(mode,model); pause();
    end
end

% --- MinCut solver ---
MinCutDemo.MinCut(["Euclidean","Isotropic","Riemannian","Randers","AsymIso","AsymQuad"],true,"cpu")
MinCutDemo.MinCut(["Euclidean","Isotropic","Riemannian","Randers","AsymIso","AsymQuad"],false,"gpu")