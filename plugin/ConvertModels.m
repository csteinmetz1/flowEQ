function ConvertModels
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

%% One dimensional model
filename1d = '../models/vae1d.h5';

W1 = double(h5read(filename1d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename1d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename1d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename1d,'/decoder/dense_4/bias:0').');

save('assets/decoder1d.mat', 'W1', 'b1', 'W2', 'b2')

%% Two dimensional model
filename2d = '../models/vae2d.h5';

W1 = double(h5read(filename2d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename2d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename2d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename2d,'/decoder/dense_4/bias:0').');

save('assets/decoder2d.mat', 'W1', 'b1', 'W2', 'b2')

%% Three dimensional model
filename3d = '../models/vae3d.h5';

W1 = double(h5read(filename3d,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename3d,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename3d,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename3d,'/decoder/dense_4/bias:0').');

save('assets/decoder3d.mat', 'W1', 'b1', 'W2', 'b2')
