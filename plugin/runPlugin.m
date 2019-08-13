function runPlugin(pluginType)
% RUN Run flowEQ plugin as MATLAB code
%
%   pluginType (optional) - string specifying type of the plugin to run
%   Use 'full' for the full featured plugin, or 'lite' for the simple GUI.
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
%   See also build, audioTestBench
% 

% check if the models and codes have been generated
if ~exist('assets', 'dir')
    error('assets directory not found. Refer the to the README for help.')
end
    
% add important directories to path
addpath enums
addpath assets

% default to full plugin
if ~exist('pluginType', 'var')
    pluginType = 'full';
end

% start up the audioTestBench
if     strcmp(pluginType,'lite')
    audioTestBench(flowEQ_lite)
elseif strcmp(pluginType,'full')
    audioTestBench(flowEQ)
else
    error('Invalid pluginType: %s\nTry ''lite'' or ''full''...', pluginType)
end

