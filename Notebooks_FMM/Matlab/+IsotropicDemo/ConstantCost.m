% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay, 2020

% This file demonstrates the fast marching method, with an isotropic metric, 
% in dimension two. Higher dimensions are also supported.

% -------------------------------------------
% Recompute the Euclidean distance from a seed point
% -------------------------------------------
function [hfmIn,hfmOut] = ConstantCost(mode,dimx)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; dimx=100; end

%Setup
hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'}=mode;
hfmIn{'model'}='Isotropic2';
hfmIn{'seeds'}=[0,0; 0.5,1.5]; % two seed points for front propagation
hfmIn{'tips'}=[-0.2,-0.5; -0.8,1.7; 0.3,-0.6]; %three tips for backatracking
hfmIn{'cost'}=1; % Cost function, here constant

% Define a rectangular domain [-1,1]x[-1,2] on 50x75 grid
% Alternatively, one may directly set dims, gridScale, origin
hfmIn.SetRect([-1,1;-1,2],pyargs('dimx',dimx)) 

% Run
hfmIn{'exportValues'}=1; % Export the solution values
hfmOut = hfmIn.Run();

% Display
clf;
axes = hfmIn.Axes();
% double converts to matlab array
values = double(hfmOut{'values'});
% IMPORTANT : imagesc transposes XY axes, but the 
% Eikonal solver does not use this convention.
imagesc(double(axes{1}),double(axes{2}),values')
axis image %Same aspect ratio for both axes

for geodesic=hfmOut{'geodesics'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:))
end

% Basic tests
if mode=="gpu_transfer"
    for stop=hfmOut{'geodesic_stopping_criteria'}; assert(stop{1}=='AtSeed'); end
end

end


