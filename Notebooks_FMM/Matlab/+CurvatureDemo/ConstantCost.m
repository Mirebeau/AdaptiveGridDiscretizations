% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay, 2020

% This file demonstrates the fast marching method, with non-holonomic path
% models penalizing curvature. The extracted paths live in in R^2 x S^1

% The cost for these model takes the form cost(x,x') C(xi x'').
% The cost and xi are specified as a scalar or array
% The function C is model dependent: 
% - C(s)=sqrt(1+s^2) for the ReedsSheppForward2 model. The ReedsShepp2 model is similar but allows cusps.
% - C(s)=1+s^2 for Elastica2.
% - C(s)=1 if |s|<=1, infty otherwise for the Dubins2 model.

% ----------------------------------------------------------
% Minimal paths, with curvature penalization, in empty 2D space
% ----------------------------------------------------------
function [hfmIn,hfmOut] = ConstantCost(mode,model,dimx)
narginchk(0,3)
if nargin<1; mode="cpu"; end
if nargin<2; model="Dubins2"; end
if nargin<3; dimx=100; end

hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'} = mode;
hfmIn{'model'} = model;
hfmIn.SetRect([0,1; 0,1.1],pyargs('dimx',dimx))
hfmIn.nTheta = 60;
hfmIn{'cost'} = 1; % cost data, here a scalar
hfmIn{'xi'}=0.1; % Homogeneous to a curvature radius

%Set seeds (front propagation start) and tips (geodesic backtracking start)
hfmIn{'seed'} = [0.5,0.5,pi/3];
hfmIn.SetUniformTips([3,3,1])
tips = double(hfmIn{'tips'}); tips(:,3)=-pi/3; hfmIn{'tips'}=tips;

% Run
hfmIn{'exportValues'}=true; % distance table, of size [n,n,numberOfDirections]
hfmOut = hfmIn.Run();

%Display
axes = hfmIn.Axes();
values = double(hfmOut{'values'});
% Show minimal value among all directions (third axis)
values_min = min(values,[],3);
imagesc(double(axes{1}),double(axes{2}),values_min')
axis image
% Show projection of geodesic on first two axes
for geodesic=hfmOut{'geodesics'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:))
end
end