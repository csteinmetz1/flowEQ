xmax = [ 12.0, 1000.0,  12.0, 3900.0, 10.0,  12.0, 4700.0, 10.0,  12.0, 10000.0, 10.0,  12.0, 20000.0];
xmin = [-12.0,   22.0, -12.0,   82.0,  0.1, -12.0,  180.0,  0.1, -12.0,   220.0,  0.1, -12.0,   580.0];

filename = '../models/vae.h5';
W1 = double(h5read(filename,'/decoder/dense_3/kernel:0').');
b1 = double(h5read(filename,'/decoder/dense_3/bias:0').');
W2 = double(h5read(filename,'/decoder/dense_4/kernel:0').');
b2 = double(h5read(filename,'/decoder/dense_4/bias:0').');

save 'decoder.mat' W1 b1 W2 b2 xmax xmin