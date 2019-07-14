function ConvertModels(modeldir)
% CONVERTMODELS
%
%   This script loads the weights from the trained models and then
%   converts them to MATLAB matracies, saving them to .mat files.
%   These .mat files are then loaded by the Decoder class which
%   apply the weights during prediction.
%
%   Note: Keras saves weights as 32-bit floats (single) but the
%   decoder implementation in MATLAB operates on doubles at the moment
%   so we cast to a double here. In the future this may be changed for
%   greater efficiency, but it is not currently a concern.
%

% get filenames for all .h5 model weights
models = dir(fullfile(modeldir,'*.h5'));

fprintf('Found %d models in %s\n', length(models), modeldir)

% load each set of model weights
for i=1:length(models)
	h5path  = fullfile(modeldir, models(i).name);
	matpath = fullfile('assets', strrep(models(i).name, '.h5', '.mat')); 
	LoadModel(h5path, matpath);
end
 