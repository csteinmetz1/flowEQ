function ConvertModels
% CONVERTMODELS
%
%   This adds a few folder to the path which is required to compile
%   the plugin. It also runs some functions which create files that 
%   are needed. If you want to target a different type or platform 
%   see the usage of generateAudioPlugin and run from the Command Window. 
%
%   Note: This takes about 60 seconds on my 2018 MacBook Pro.
%

filename1d = '../models/vae1d.h5';
filename2d = '../models/vae2d.h5';
filename3d = '../models/vae3d.h5';

W1 = double(h5read(filename1d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename1d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename1d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename1d,'/decoder/dense_4/bias:0').');

save('assets/decoder1d.mat', 'W1', 'b1', 'W2', 'b2')

W1 = double(h5read(filename2d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename2d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename2d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename2d,'/decoder/dense_4/bias:0').');

save('assets/decoder2d.mat', 'W1', 'b1', 'W2', 'b2')

W1 = double(h5read(filename3d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename3d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename3d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename3d,'/decoder/dense_4/bias:0').');

save('assets/decoder3d.mat', 'W1', 'b1', 'W2', 'b2')
