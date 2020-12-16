% Copyright Jean-Marie Mirebeau, ENS Paris-Saclay, CNRS, University Paris-Saclay

% -------- IMPORTANT -------
% Please tune the following values to your computer (see description below)
pythonw_executable='C:\\Users\\jmmir\\miniconda3\\envs\\agd-hfm_cuda\\pythonw.exe';
path_to_agd="../..";
allow_duplicate_mkl=true;
% --------------------------

% This file prepares Matlab for calling the agd python library
% https://github.com/Mirebeau/AdaptiveGridDiscretizations
% which contains a GPU eikonal solver and other related tools

% Arguments : 

% - pythonw_executable (string or NaN) : 
% We recommend using a conda environment featuring
% all the required libraries, see the provided yaml files for inspiration.
% pythonw_executable should be the full path to the pythonw.exe file of
% that environment. To obtain it (in python script within appropriate env) 
% >>> import sys; sys.executable 
% Set to NaN to ignore

% - path_to_agd (string or NaN) : 
% full path to where the agd python library can be found
% Set to NaN to ignore, if e.g. the agd library is installed as a pip or
% conda package

% - allow_duplicate_mkl : The python numpy library will be used. On my 
% machine Matlab crashes complaining about duplicate instances of the 
% mkl library, unless this is explicitly allowed by the python environment.
% Note that this way of doing things is apparently not recommended
% https://stackoverflow.com/a/59119273

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

if isstring(path_to_agd)
% The agd library must be visible from the python path
% (This line is useless if agd is installed as a conda or pip packaged)
    tmp=py.sys.path; tmp.append(path_to_agd)
else
    assert(isnan(path_to_agd),"path_to_agd must be string or NaN")
end


% If you use the HFM library (CUP eikonal solver), locally compiled 
% from sources, then the path to the executables must be provided
%FileHFM_binary_dir = 'C:\Users\jmmir\Documents\bin\FileHFM\Release';
%py.agd.Eikonal.LibraryCall.FileHFM_binary_dir = FileHFM_binary_dir;

if allow_duplicate_mkl
% The python numpy library will be used. 
% On my machine this line is needed.
    tmp=py.os.environ; tmp{'KMP_DUPLICATE_LIB_OK'}='TRUE';
end

