function [hfmIn,hfmOut] = Pompidou(mode,model)
narginchk(0,2)
if nargin<1; mode="cpu"; end
if nargin<2; model='ReedsShepp2'; end

hfmIn = py.agd.Eikonal.dictIn;
hfmIn{'mode'}=mode;
hfmIn{'model'}=model;
if model=="ReedsShepp2"; hfmIn{'projective'}=true; end
hfmIn{'xi'}=0.7; %Model parameter, typical radius of curvature.
hfmIn{'cost'}=1;

im = imread('..\TestImages\centre_pompidou_800x546.png');
im = ( (im(:,:,1)==255) | (im(:,:,3)==255)) & (im(:,:,2)==0);
walls = (im==0)';
hfmIn{'walls'} = py.numpy.array(walls);
hfmIn{'origin'}=[0,0];
hfmIn{'dims'}=size(walls);
hfmIn.nTheta = 60;
h=1./90;
hfmIn{'gridScale'}=h;

hfmIn{'seeds_Unoriented'}=[80,170;80,290]*h;
hfmIn{'tips_Unoriented'}=...
    [369.4, 252.2, 285., 418.6, 479.8, 687.2, 745.8, 740.4, 593.8, 558.6,...
    599.2, 497.2, 495.8, 427.2, 339., 264.6, 242.4, 354.6, 191.6, ...
    178.8, 105.8, 124., 127., 419.2; ...
    482.5, 354.5, 478., 488., 487.5, ...
    478., 502.5, 300., 225.5, 378., 475.5, 81., 127.5, 128., 111., 108.,...
    176.5, 290.5, 110., 252.5, 428.5, 494., 353., 421.]' * h;    
hfmIn{'geodesicVolumeBound'}=12;

%Run
hfmIn{'exportValues'}=true;
hfmOut = hfmIn.Run();

clf;
% Display minimal distance over all directions, with a cutoff due to
% inaccessible regions for e.g. the Dubins car
dist=min(double(hfmOut{'values'}),[],3);
dist(dist==Inf)=0;
if model=="Dubins2"
    cutoff = sort(dist(:)); 
    cutoff = cutoff(ceil(numel(dist)*0.95));
    dist = min(dist,cutoff);
end
axes = hfmIn.Axes();
imagesc(double(axes{1}),double(axes{2}),dist');
axis image 

for geodesic=hfmOut{'geodesics_Unoriented'}
    geo=double(geodesic{1});
    line(geo(1,:),geo(2,:));
end

end