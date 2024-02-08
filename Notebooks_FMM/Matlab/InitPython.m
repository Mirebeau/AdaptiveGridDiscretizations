% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay

% -------- IMPORTANT -------
% Please tune the following values to your computer (see description below)
pythonw_executable='C:\\Users\\jmmir\\miniconda3\\envs\\agd-hfm_cuda_310\\pythonw.exe';
path_to_agd="C:\\Users\\jmmir\\Documents\\GitHub\\AdaptiveGridDiscretizations"; % "../..";
FileHFM_binary_dir = "C:\\Users\\jmmir\\Documents\\bin\\FileHFM\\Release";
allow_duplicate_mkl=true;
% --------------------------

% This file prepares Matlab for calling the agd python library
% https://github.com/Mirebeau/AdaptiveGridDiscretizations
% which contains a GPU eikonal solver and other related tools

% Warning : calling python from matlab on apple M1 requires taking care of 
% x86/arm64 incompatibilities, see https://stackoverflow.com/a/70307576

% Arguments : 

% - pythonw_executable (string or NaN) : 
% We recommend using a conda environment featuring
% all the required libraries, see the provided yaml files for inspiration.
% pythonw_executable should be the full path to the pythonw.exe file of
% that environment. To obtain it (in a python script within appropriate env) 
% (python)>>> import sys; sys.executable 
% Set to NaN to ignore

% - path_to_agd (string or NaN) : 
% path to where the agd python library can be found
% Set to NaN to ignore, if the agd library is installed as a Python package 
% pip install (GPU compatible): (terminal)>>> pip install agd
% conda install (Not GPU compatible): (terminal)>>> conda install agd -c agd-lbr


% - FileHFM_binary_dir (string or NaN)
% path to where the FileHFM binaries can be found (CPU eikonal solver,
% interfacing with files rather than a direct Python link)
% Set to NaN to ignore, if the hfm library is installed as a package
% conda install : (terminal)>>> conda install hfm -c agd-lbr

% - allow_duplicate_mkl : The python numpy library will be used. On my 
% machine Matlab crashes complaining about duplicate instances of the 
% mkl library, unless this is explicitly allowed by the python environment.
% Note that this way of doing things is apparently not recommended
% https://stackoverflow.com/a/59119273


%pythonw_executable='/Users/mirebeau/opt/miniconda3/envs/agd-hfm_dev/bin/python'; path_to_agd="/Users/mirebeau/Dropbox/Programmes/Github/AdaptiveGridDiscretizations"; FileHFM_binary_dir = "/Users/mirebeau/bin/HamiltonFastMarching/FileHFM/Release";
%pythonw_executable='C:\\Users\\Shadow\\Miniconda3\\envs\\agd-hfm_cuda\\pythonw.exe';path_to_agd="C:\\Users\\Shadow\\Documents\\GitHub\\AdaptiveGridDiscretizations";FileHFM_binary_dir = "C:\\Users\\Shadow\\Documents\\bin\\FileHFM";
%pythonw_executable='/Users/jean-mariemirebeau/opt/miniconda3/envs/agd-hfm_dev_310/bin/python';path_to_agd='/Users/jean-mariemirebeau/Dropbox/Programmes/GithubM1/AdaptiveGridDiscretizations'; FileHFM_binary_dir = '/Users/jean-mariemirebeau/bin/FileHFM/Release'
%"/usr/bin/python3"

tmp = pyenv; 
if tmp.Status=="Loaded"
    disp("InitPython aborted : Python is already loaded") 
    return
end
    

if ischar(pythonw_executable)
%Code copied from
%https://fr.mathworks.com/matlabcentral/answers/443558-matlab-crashes-when-using-conda-environment-other-than-base#answer_486374
py_root_useFromMATLAB = fileparts(pythonw_executable);
ENV = getenv('PATH');
ENV = strsplit(ENV, ';');
items_to_add_to_path = {
    fullfile(py_root_useFromMATLAB, 'Library', 'mingw-w64', 'bin')
    fullfile(py_root_useFromMATLAB, 'Library', 'usr', 'bin')
    fullfile(py_root_useFromMATLAB, 'Library', 'bin')
    fullfile(py_root_useFromMATLAB, 'Scripts')
    };
ENV = [items_to_add_to_path(:); ENV(:)];
ENV = unique(ENV, 'stable');
ENV = strjoin(ENV, ';');
setenv('PATH', ENV);

pyenv('Version',pythonw_executable);
else
    assert(isnan(pythonw_executable),...
    "pythonw_executable must be string or NaN")
end

if allow_duplicate_mkl
% The python numpy library will be used. 
% On my machine this line is needed.
    tmp=py.os.environ; tmp{'KMP_DUPLICATE_LIB_OK'}='TRUE';
end


if isstring(path_to_agd)
% The agd library must be visible from the python path
% (This line is useless if agd is installed as a conda or pip packaged)
    tmp=py.sys.path; tmp.append(path_to_agd)
else
    assert(isnan(path_to_agd),"path_to_agd must be string or NaN")
end

if isstring(FileHFM_binary_dir)
    tmp=py.agd.Eikonal.LibraryCall.binary_dir; 
    tmp{"FileHFM"}=FileHFM_binary_dir;
else
    assert(isnan(FileHFM_binary_dir),"FileHFM_binary_dir must be string or NaN")
end


