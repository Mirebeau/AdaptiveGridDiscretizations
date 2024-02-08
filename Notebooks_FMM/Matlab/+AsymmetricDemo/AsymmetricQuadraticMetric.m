% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay, 2020

% ------------ Asymmetric Quadratic metrics -----------
% A metric of AsymmetricQuadratic type takes the form
% F(v)^2 = <v,M v> + <omega,v>_+^2
% The symmetric matrix M and vector omega are parameters that may depend on
% the current point. The matrix M must be positive definite.

% The dual metric, defined in the co-tangent space, has a similar form.
% F^*(v)^2 = <v,D v> + <eta,v>_+^2,
% where D and eta have algebraic expressions in terms of M and omega

% This type of metric can be used to approximate 'HalfDisk models', whose unit ball 
% is a half disk, or half ellipse, by using a (reasonably) large vector v.

% There are three ways to specify the metric:
% - directly define M and omega
% - define the dual parameters D and eta
% - define a half disk model.

% The numerical example below is two dimensional. Three dimensional
% asymmetric quadratic metrics are supported as well. Note that the
% PDE discretization used in the CPU and GPU implementations is different.

function [hfmIn,hfmOut] = AsymmetricQuadraticMetric(mode,dimx)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; dimx=201; end

hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'}=mode;
hfmIn{'model'}='AsymmetricQuadratic2';

hfmIn.SetRect([-1,1;-1,1],pyargs('dimx',dimx))

hfmIn{'seed'}=[0,0];
hfmIn.SetUniformTips([6,6]);

% The metrics is constructed by providing 
% M with shape d x d x n1 x ... x nd (or d x d if constant over domain)
% omega with shape d x n1 x ... x nd (or d if constant over domain)
hfmIn{'metric'} = py.agd.Metrics.AsymQuad(eye(2),[1,1]);

hfmIn{'exportValues'}=true;
hfmOut = hfmIn.Run();

clf;
axes = hfmIn.Axes();
% double converts to matlab array
values = double(hfmOut{'values'});
% IMPORTANT : imagesc transposes XY axes, but the 
% Eikonal solver does not use this convention.
imagesc(double(axes{1}),double(axes{2}),values',[0,1])
axis image %Same aspect ratio for both axes

for geodesic=hfmOut{'geodesics'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:))
end

end



