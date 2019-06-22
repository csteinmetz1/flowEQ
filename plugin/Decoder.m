classdef Decoder
    properties
        W1, b1, W2, b2;
    end
    
    methods
        
        function obj = Decoder(filename)
            obj.W1 = h5read(filename,'/decoder/dense_3/kernel:0').';
            obj.b1 = h5read(filename,'/decoder/dense_3/bias:0').';
            obj.W2 = h5read(filename,'/decoder/dense_4/kernel:0').';
            obj.b2 = h5read(filename,'/decoder/dense_4/bias:0').';
            
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
        
        function normalize_params(x)
        end
        
        function denormalize_params(x)
        end

   end
end
    