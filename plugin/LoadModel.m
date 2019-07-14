function LoadModel(h5path, matpath)
	% LOADMODEL
	%
	%
	
	W1 = double(h5read(h5path,'/decoder/dense_3/kernel:0').');
	b1 = double(h5read(h5path,'/decoder/dense_3/bias:0').');
	W2 = double(h5read(h5path,'/decoder/dense_4/kernel:0').');
	b2 = double(h5read(h5path,'/decoder/dense_4/bias:0').');
	
	save(matpath, 'W1', 'b1', 'W2', 'b2')