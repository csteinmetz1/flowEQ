function build(modeldir)
% BUILD Generate a VST or AU flowEQ for your platform.
%
%   MODELDIR (optional) - Pass path to models if they need to be converted 
%
%   This adds a few folders to the path that are required to compile
%   the plugin. It also runs some functions which create files that 
%   are needed. If you want to target a different type or platform 
%   see the usage of generateAudioPlugin and run from the Command Window. 
%
%   To run the plugin as MATLAB code using audioTestBench use the run script.
%
%   Note: This takes about 60 seconds on my 2018 MacBook Pro.
%
%   See also ConvertModels, generateAudioPlugin
% 
addpath enums
addpath assets

if exist('modeldir', 'var')
    %% generate assets for compilation
    ConvertModels(modeldir);
end

%% compile the plugin based on platform
if ismac
    generateAudioPlugin -au flowEQ.m
else
    generateAudioPlugin -vst flowEQ.m
end
