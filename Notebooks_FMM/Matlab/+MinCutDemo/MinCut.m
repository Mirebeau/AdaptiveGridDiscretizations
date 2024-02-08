% Copyright Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay, 2020

% This file illustrates the numerical solution to the mincut problem in 
% non-Euclidean geometry, in dimension two. (Three dimensional data is also supported)

% The metrics are constructed based on the structure tensor and/or the
% smoothed image gradient. More elaborate constructions should yield better
% results. See the python notebook Notebooks_Div/Prox_MinCut for a
% discussion of the numerical method and of the metric constructions.


function MinCut(metric_kinds,show_metric,mode)
narginchk(0,3)
allkinds = ["Euclidean","Isotropic","Riemannian","Randers","AsymIso","AsymQuad"];
if nargin<1; metric_kinds=allkinds; end
if nargin<2; show_metric = true; end
if nargin<3; mode = 'cpu'; end

mod = py.importlib.import_module('agd.ExportedCode.Notebooks_Div.Prox_MinCut');
np = py.importlib.import_module('numpy');

% The following lines generate an image, with a given size and noise level
shape = [101,71]; % the image size in each dimension
corners = [0,0; 1,0.7];
noiselevel=0.7;
tup = mod.MinCut.stag_grids(py.tuple(int64(shape)),np.array(corners));
Xs=double(tup{1}); Xv=double(tup{2}); dx=double(tup{3});
image = TestImage(Xs,noiselevel);
imagesc(image'); title('Image to be segmented'); pause();

% Our metric constructions use the image structure tensor S and the 
% smoothed gradient eta (normalized)
% Caution : the dimension is 1 pixel less in each direction than the image
tup = mod.structure_tensor(np.array(image));
g_=tup{1}; gx=double(g_{1}); gy=double(g_{2}); % Smoothed gradient
g_=tup{2}; gxx=double(g_{1}); gxy=double(g_{2}); gyy=double(g_{3});

S = zeros(2,2,shape(1)-1,shape(2)-1); % structure tensor
S(1,1,:,:) = gxx; S(1,2,:,:) = gxy; S(2,1,:,:) = gxy; S(2,2,:,:) = gyy;

eta = zeros(2,shape(1)-1,shape(2)-1); % smoothed gradient
eta(1,:,:) = gx; eta(2,:,:) = gy;

gmax = max(gxx+gyy,[],"all"); % normalization factor
S = S./gmax; eta = eta./sqrt(gmax); 
S(1,1,:,:) = S(1,1,:,:) + 0.05; S(2,2,:,:) = S(2,2,:,:) + 0.05; % Ensure positive definiteness
epsilon = 0.1; % Small parameter used in the AsymIso and AsymQuad constructions


for kind=metric_kinds
    if kind=="Euclidean"
        metric = py.agd.Metrics.Isotropic(1.).with_cost(0.01);
    elseif kind=="Isotropic"
        edge_detect = squeeze(S(1,1,:,:)+S(2,2,:,:)); % Should be large close to image edges
        metric = py.agd.Metrics.Isotropic(1./edge_detect).with_cost(0.01);
    elseif kind=="Riemannian"
        metric = py.agd.Metrics.Riemann(S).dual().with_cost(0.01);
    elseif kind=="Randers"
        metric = py.agd.Metrics.Rander(S,-eta).dual().with_cost(0.01);
    elseif kind=="AsymIso"
        metric = py.agd.Metrics.asym_iso.AsymIso(1.,eta/epsilon).with_cost(0.01);
    elseif kind=="AsymQuad"
        metric = py.agd.Metrics.AsymQuad(eye(2),-eta/epsilon).dual().with_cost(0.01);
    else
        fprintf("Unrecognized metric kind %s\nSupported : ",kind)
        disp(all_kinds)
    end

    if show_metric
        py.agd.Plotting.Tissot(metric,np.array(Xv),int64(100),int64(4));
        py.matplotlib.pyplot.title(sprintf("Unit balls of the %s metric (Tissot indicatrix)",kind));
        py.matplotlib.pyplot.show();
    end

    if mode=="cpu"
        res = mod.mincut_cpu(np.array(image),metric,np.array(dx));
    elseif mode=="gpu"
        res = mod.mincut_gpu(np.array(image),metric,np.array(dx));
    else
        fprintf("Urecognized computation mode %s. Supported : cpu/gpu",mode)
        return
    end

    imagesc(double(res{'ϕ'})'); 
    title(sprintf("Mincut solution, using %s metric",kind)); 
    pause();
end

end % Main function

function im = TestImage(X,noiselevel)
%im = squeeze(X(1,:,:)>=0.5 | X(2,:,:)>=0.3);
im = ((X(1,:,:)>=0.15) & (X(2,:,:)>=0.15) & (X(1,:,:)+2*X(2,:,:)<=1.1)) | ((X(1,:,:)-0.7).^2+(X(2,:,:)-0.45).^2<=0.15^2);
im = squeeze(2*im-1);
rng(42); % Fix the random seed for reproducibility
rnd = rand(size(im));
im(rnd<noiselevel)=1;
im(rnd<noiselevel/2)=-1;
end
