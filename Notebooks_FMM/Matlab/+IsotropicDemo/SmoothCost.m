
% --------------------------------
% Path length distance with a variable cost
% --------------------------------

function [hfmIn,hfmOut] = SmoothCost(mode,dimx)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; dimx=200; end

% Setup
hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'} = mode;
hfmIn{'model'} = 'Isotropic2';
hfmIn{'seed'}=[0,0];
hfmIn{'exportValues'}=true;
hfmIn.SetRect([-2*pi,2*pi;-2*pi,2*pi],pyargs('dimx',dimx))
X = double(hfmIn.Grid());
hfmIn{'cost'} = CostFunction(X);
hfmIn.SetUniformTips([4,4]) % tips on a 4x4 uniform sub-grid

% Run
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

function cost = CostFunction(x)
eps=0.1;
cost = squeeze(eps^2 + sin(x(1,:,:)).^2 .* cos(x(2,:,:)).^2 ...
    + cos(x(1,:,:)).^2 .* sin(x(2,:,:)).^2);
end