function build
% BUILD Generate a VST or AU flowEQ for your platform.
%
%   This adds a few folder to the path which is required to compile
%   the plugin. It also runs some functions which create files that 
%   are needed. If you want to target a different type or platform 
%   see the usage of generateAudioPlugin and run from the Command Window. 
%
%   Note: This takes about 60 seconds on my 2018 MacBook Pro.
%

addpath enums
addpath matmodels

%% Generate assets for compilation
HDF52mat            % Convert the latest Keras models (hdf5) to mat files
SaveLatentCodes     % Create mat file with the semantic embeddings

%% Compile the plugin based on platform
if ismac
    generateAudioPlugin -vst flowEQ.m
else
    generateAudioPlugin -vst flowEQ.m
end
