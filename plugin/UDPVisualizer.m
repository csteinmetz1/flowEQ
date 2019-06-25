function UDPVisualizer
% Provide visual front-end for the flowEQ plugin
%
% This function reads UDP packets from the plugin
% and then creates a variety of visualizations.
%
% Data format
% ----------------------------------------------
% 01-15 : (b)   Numerator biquad coefficients
% 16-30 : (a)   Denominator biquad coefficients
% 31-43 : (p)   Parametric EQ parameters
% 44-46 : (z)   Latent vector
%    47laten : (dim) Latent dimension
% 

%% Configure UDP Receiver
UDPReceive = dsp.UDPReceiver('LocalIPPort', 20000, ...
                             'MessageDataType', 'double', ...
                             'MaximumMessageLength', 49, ...
                             'ReceiveBufferSize', 49*80);
c = onCleanup(@()release(UDPReceive));

v = dsp.DynamicFilterVisualizer(8192, 44100, [20 20e3], 'XScale','Log');
show(v);

%% Graphics
figure;

while true        
    udpr = UDPReceive();
    if ~isempty(udpr)
        size(udpr)
        b = reshape(udpr(1:15),3,5).';
        a = reshape(udpr(16:30),3,5).';
        p = udpr(31:43);
        z = udpr(44:46);
        dim = udpr(47);
        
        v(b,a);
        
        if     dim == 1
            
        elseif dim == 2
            scatter(z(1), z(2));
            xlim([-4 4])
            ylim([-4 4])
        else
            scatter3(z(1), z(2), z(3));
            xlim([-4 4])
            ylim([-4 4])
            zlim([-4 4])
        end
            
    end
end

