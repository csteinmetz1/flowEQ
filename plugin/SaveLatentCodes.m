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

codes = [0.2 0.0 0.0; -0.2]

save('matmodels/codes.mat', 'codes')