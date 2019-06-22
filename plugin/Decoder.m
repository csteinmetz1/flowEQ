classdef Decoder
    properties
        W1, b1, W2, b2;
        xmax, xmin;
    end
    
    methods
        
        function obj = Decoder(filename)
            % parameter vectors for norm/denormalization
            obj.xmax = [ 12.0 1000.0  12.0 3900.0 10.0  12.0 4700.0 10.0  12.0 10000.0 10.0  12.0 20000.0];
            obj.xmin = [-12.0   22.0 -12.0   82.0  0.1 -12.0  180.0  0.1 -12.0   220.0  0.1 -12.0   580.0];

            % load trained model weights from .mat file
            weights = coder.load(filename);
            obj.W1 = weights.W1;
            obj.b1 = weights.b1;
            obj.W2 = weights.W2;
            obj.b2 = weights.b2;
        end
        
        function y_hat = predict(obj, z)
            z1 = (z * obj.W1) + obj.b1;
            a1 = obj.ReLU(z1);
            z2 = (a1 * obj.W2) + obj.b2;
            a2 = obj.sigmoid(z2);
            y_hat = a2;
        end
        
        function x_a = ReLU(~, x)
            x_a = max(x, 0);
        end
        
        function x_a = sigmoid(~, x)
            x_a = 1 ./ (1 + exp(-x));
        end
        
        function xNorm = normalize(obj, x)
            xNorm = (x - obj.xmin) ./ (obj.xmax - obj.xmin);
        end
        
        function xDenorm = denormalize(obj, x)
           xDenorm = (x .* (obj.xmax - obj.xmin)) + obj.xmin;
        end

   end
end
    