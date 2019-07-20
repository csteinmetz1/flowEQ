function LoadModel(h5path, matpath)
% LOADMODEL Load a Keras model (autoencoder) and extract the decoder weights.
%
%   H5PATH  (required) - Path to Keras model weights file
%   MATPATH (required) - Path to output mat file with weights
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
    
    W1 = double(h5read(h5path,'/decoder/dense_3/kernel:0').');
    b1 = double(h5read(h5path,'/decoder/dense_3/bias:0').');
    W2 = double(h5read(h5path,'/decoder/dense_4/kernel:0').');
    b2 = double(h5read(h5path,'/decoder/dense_4/bias:0').');
    
    save(matpath, 'W1', 'b1', 'W2', 'b2')