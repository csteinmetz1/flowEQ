function buildPlugin
% BUILDPLUGIN Generate a VST or AU of flowEQ and flowEQ-lite for your platform.
%
%   This adds a few folders to the path that are required to compile
%   the plugin. It also checks if you have the 'assets' directory with the
%   trained models and latent codes. If you want to target a different type
%   of plugin library see the usage of generateAudioPlugin and run from the 
%   Command Window.
%   
%   This whole process takes about 3 minutes on my 2018 MacBook Pro.
%
%   On error make sure to complete the following steps:
%
%   1. Download the archive from the project website with the assets.
%                                                (https://flowEQ.ml)
%   or
%
%   1. Run trainfull.py script to train all 12 models.
%   2. Afterwards, run embeddings.py to generate codes 
%       (this creates the assets directory).
%   3. Run the ConvertModels function to finally convert
%       the trained models to mat files.
%
%   See the README for more details on the process.
%
%   See also audioTestBench, ConvertModels, generateAudioPlugin
% 

% check if the models and codes have been generated
if ~exist('assets', 'dir')
    fprintf('assets directory was not found. Downloading assets...\n');
    assetsurl = 'https://docs.google.com/uc?export=download&id=1bwp0B2xMQl7u3sKalO5AmNkVLN9Zxi0g';
    archive = 'assets.zip';
    outfile = websave(archive,assetsurl);       % download assets if folder not found
    unzip(outfile);                             % extract files from archive
    delete(outfile);                            % remove archive
    fprintf('Download complete.\n');
end
    
% add important directories to path
addpath enums
addpath assets

% first validate the plugins
validateAudioPlugin flowEQ.m
validateAudioPlugin flowEQ_lite.m

% compile the plugins based on platform
fprintf('Generating flowEQ plugin... ')
generateAudioPlugin flowEQ.m
fprintf('Done.\n')

fprintf('Generating flowEQ-lite plugin... ')
generateAudioPlugin flowEQ_lite.m
fprintf('Done.\n')



