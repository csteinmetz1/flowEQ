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
%    47 : (dim) Latent dimension
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
close all;
latentPlot = figure('Position', [10 10 900 600]);

while true        
    udpr = UDPReceive();
    if ~isempty(udpr)
        b = reshape(udpr(1:15),3,5).';
        a = reshape(udpr(16:30),3,5).';
        p = udpr(31:43);
        z = udpr(44:46);
        dim = udpr(47);
        
        param_text = {sprintf('Lowshelf Gain:    %0.2f dB', p(1))...
                      sprintf('Lowshelf Freq:    %0.2f Hz', p(2))...
                      sprintf('First band Gain:  %0.2f dB', p(3))...
                      sprintf('First band Freq:  %0.2f Hz', p(4))...
                      sprintf('First band Q:     %0.2f',    p(5))...
                      sprintf('Second band Gain: %0.2f dB', p(6))...
                      sprintf('Second band Freq: %0.2f Hz', p(7))...
                      sprintf('Second band Q:    %0.2f',    p(8))...
                      sprintf('Third band Gain:  %0.2f dB', p(9))...
                      sprintf('Third band Freq:  %0.2f Hz', p(10))...
                      sprintf('Third band Q:     %0.2f',    p(11))...
                      sprintf('Highshelf Gain:   %0.2f dB', p(12))...
                      sprintf('Highshelf Freq:   %0.2f Hz', p(13))};
                      
        param_text = sprintf('%s\n',param_text{:});
        
        annotation('textbox',[.7 .5 .5 .2],'String',param_text,'EdgeColor','none','FontName', 'Source Code Variable')
        
        v(b,a);
        
        clf;
        if     dim == 1
            latent_text = sprintf('x = %+0.2f', z(1));
            
        elseif dim == 2
            latent_text = sprintf('x = %+0.2f y = %+0.2f', z(1:2));
            scatter(z(1), z(2));
            xlim([-4 4])
            ylim([-4 4])
        else
            latent_text = sprintf('x = %+0.2f y = %+0.2f z = %+0.2f', z);
            scatter3(z(1), z(2), z(3));
            xlim([-4 4])
            ylim([-4 4])
            zlim([-4 4])
        end  
        
        title(latent_text);
        pos=get(gca,'position');  % retrieve the current values
        pos(3)=0.7*pos(3);        % try reducing width 30%
        set(gca,'position',pos);  % write the new values
    end
end

