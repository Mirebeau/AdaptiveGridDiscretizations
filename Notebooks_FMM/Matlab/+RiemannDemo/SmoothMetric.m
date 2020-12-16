% Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay, 2020

% This file demonstrates the fast marching method, with a Riemannian metric, 
% in dimension two. Three dimensional metrics are also supported.

function [hfmIn,hfmOut] = SmoothMetric(mode,dimx)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; dimx=100; end

hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'} = mode;
hfmIn{'model'}='Riemann2';
%Set domain, and get coordinate system
hfmIn.SetRect([-1,1;-1,1],pyargs('dimx',dimx))
X = double(hfmIn.Grid()); % Coordinate system (ndgrid like)

%Set seeds (front propagation start) and tips (geodesic backtracking start)
hfmIn{'seed'} = [0,0];
hfmIn.SetUniformTips([4,4])

% Metric construction
% Here, we create a diagonal metric, and then rotate it
% Alternatively, define the metric directly from an array of shape
% metric = Riemann(m); m with shape 2 x 2 x n1 x ... x nd
metric = py.agd.Metrics.Riemann(py.None); % dummy instance, to access class methods
metric = metric.from_diagonal([1,4]);
theta = squeeze(X(1,:,:));
metric = metric.rotate_by(theta);
hfmIn{'metric'} = metric;

%Run
hfmIn{'exportValues'}=true;
hfmOut = hfmIn.Run();

%Display
clf;
axes = hfmIn.Axes();
values = double(hfmOut{'values'});
imagesc(double(axes{1}),double(axes{2}),values') 
axis image 

for geodesic=hfmOut{'geodesics'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:))
end

end
