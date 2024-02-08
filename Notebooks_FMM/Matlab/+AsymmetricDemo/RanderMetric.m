% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay, 2020

% ------------------ Rander metrics -------------------
% A Rander metric takes the form
% F(v) = sqrt(<v,M v>) + <omega,v>
% where M is symmetric positive definite, and <omega, M^-1 omega> < 1.
% Both M and omega may vary over the domain.

% They are arguably the simplest and smoothest class of asymmetric metrics.
% Rander metrics arise in Zermelo's navigation problem.

% The numerical examples presented here are two dimensional.
% Three dimensional Rander metrics are supported by the GPU eikonal solver,
% but not the CPU eikonal solver.


function [hfmIn,hfmOut] = RanderMetric(mode,dimx)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; dimx=201; end

hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'}=mode;
hfmIn{'model'}='Rander2';

hfmIn.SetRect([-0.5,0.5;-0.5,0.5],pyargs('dimx',dimx));
X = double(hfmIn.Grid());

% Set the seeds (front propagation start) 
% and tips (geodesic backtracking start)
hfmIn{'seed'}=[0,0];
hfmIn.SetUniformTips([6,6]);

% Generate the metric. The recommended way is 
% metric = py.agd.Metrics.Rander(M,omega);
% where M is a d x d x n1 x ... nd array of symmetric positive definite
% matrices, and omega is a d x n1 x ... x nd array of vectors.
% However we use here a specific constructor, related to Zermelo's
% navigation problem.
% Dummy metric to access from_Zermelo constructor
metric=py.agd.Metrics.Rander(py.None,py.None);
metric = metric.from_Zermelo(eye(2),DriftFunction(X));
hfmIn{'metric'}=metric;

% Run the solver
hfmIn{'exportValues'}=true;
hfmOut = hfmIn.Run();

% Display
clf;
axes = hfmIn.Axes();
values = double(hfmOut{'values'});
imagesc(double(axes{1}),double(axes{2}),values')
axis image %Same aspect ratio for both axes

for geodesic=hfmOut{'geodesics'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:))
end

end

function drift = DriftFunction(X)
R = sqrt(X(1,:,:).^2+X(2,:,:).^2);
driftMult = 0.9*sin(4*pi*X(1,:,:)).*sin(4.*pi*X(2,:,:));
drift = (driftMult./R) .* X;
end