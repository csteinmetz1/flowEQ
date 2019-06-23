function UDPVisualizer
%HelperUDPPluginVisualizer Visualize plug-in as it runs in DAW
%   HelperUDPPluginVisualizer launches a magnitude response visualizer for
%   an equalizer being executed in a DAW.
%
%   This file  is used only in support of UDPDeployedAudioPluginExample and
%   may be modified or removed in the future.

% Copyright 2015-2018 The MathWorks, Inc.

%% Setup
UDPReceive = dsp.UDPReceiver('LocalIPPort', 20000, ...
                             'MessageDataType', 'double', ...
                             'MaximumMessageLength', 30, ...
                             'ReceiveBufferSize', 30*80);
c = onCleanup(@()release(UDPReceive));
udpRec = UDPReceive(); %#ok

v = dsp.DynamicFilterVisualizer(8192, 44100, [20 20e3], 'XScale','Log');
show(v);

%% Streaming
tic
while toc < 1000 % Visualize for 60 seconds        
    udpRec = UDPReceive();
    if ~isempty(udpRec)
        B = reshape(udpRec(1:15),3,5).';
        A = reshape(udpRec(16:end),3,5).';
        v(B,A);
    end
end

