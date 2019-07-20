function run
% RUN Run flowEQ plugin as MATLAB code
%
%	On error make sure to complete the following steps:
%
%	1. Run trainfull.py script to train all 12 models.
%	2. Afterwards, run embeddings.py to generate codes 
%	   (this creates the assets directory).
%	3. Run the ConvertModels function to finally convert
%	   the trained models to mat files.
%	
%	See the README for more details on the process.
%
%   See also build, audioTestBench
% 
addpath enums
addpath assets

% check if the models and codes have been generated
if exist('assets', 'dir')
    audioTestBench(flowEQ)
else
    error('assets directory not found.')
end
