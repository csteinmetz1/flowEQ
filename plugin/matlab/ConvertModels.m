function ConvertModels(modeldir)
% CONVERTMODELS Create mat files with model weights from Keras hdf5 files.
%
%   MODELDIR (required) - Path to directory containing trained Keras models
%
%   Running the trainfull.py script will train all 12 models and save out 
%   each model architecture (.json) along with the learned weights (.h5).
%   Once complete, passing the directory to this function will iterate 
%   over the models and convert each one to a mat file.
% 
%   See also LoadModel 

% get filenames for all .h5 model weights
models = dir(fullfile(modeldir,'autoencoders','*.h5'));

fprintf('Found %d models in %s\n', length(models), modeldir)

% load each set of model weights
for i=1:length(models)
    h5path  = fullfile(modeldir, 'autoencoders', models(i).name);
    matpath = fullfile('assets', strrep(models(i).name, '.h5', '.mat')); 
    LoadModel(h5path, matpath);
end
 