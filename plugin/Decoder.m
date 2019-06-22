classdef Decoder
    properties
        W1, b1, W2, b2;
        xmax, xmin;
    end
    
    methods
        
        function obj = Decoder(filename)
            weights = coder.load(filename);
            obj.W1 = weights.W1;
            obj.b1 = weights.b1;
            obj.W2 = weights.W2;
            obj.b2 = weights.b2;
            obj.xmax = weights.xmax;
            obj.xmin = weights.xmin; 
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
    