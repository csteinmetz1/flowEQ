function SaveLatentCodes
% BUILD Generate a VST or AU flowEQ for your platform.
%
%   This adds a few folder to the path which is required to compile
%   the plugin. It also runs some functions which create files that 
%   are needed. If you want to target a different type or platform 
%   see the usage of generateAudioPlugin and run from the Command Window. 
%
%   Note: This takes about 60 seconds on my 2018 MacBook Pro.
%

latentCodes1d = {...
     [ 0.2],... % warm
	 [-0.2],... % bright
	 [-1.2]};   % sharp

latentCodes2d = {...
     [ 0.2  0.1],... % warm
	 [-0.2  0.5],... % bright
	 [-1.2  1.2]};   % sharp

latentCodes3d = {...
     [ 0.2  0.1  1.2],... % warm
	 [-0.2  0.5 -2.1],... % bright
	 [-1.2  1.2  0.1]};   % sharp

codes = {latentCodes1d, latentCodes2d, latentCodes3d};

%codes = [0.2 0.0 0.0; -0.2]

save('matmodels/codes.mat', 'codes')