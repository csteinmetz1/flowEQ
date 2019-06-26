classdef flowEQ < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        net1d;  % one dimensional latent space model
        net2d;  % two dimensional latent space model
        net3d;  % three dimensional latent space model
        udpvst = false; % set to this true if you want VST with UDP support
        udpsend;        % UDP sender object to be used
    end     
    
    properties 
        % In/Out Gain Parameters
        inputGain      =     0.00;   
        outputGain     =     0.00;
        % Neural Network Parameters
        xDim           =      0.0;
        yDim           =      0.0;
        zDim           =      0.0;
        decoderMode    = 'manual';
        latentDim      =      '1';
        firstTerm      =   'warm';
        interpolate    =      0.0;
        secondTerm     = 'bright';
        strength       =      1.0;
        autoMode        =    true;
        % Parametric EQ Parameters (manual)
        lowShelfGain   =     0.00;
        lowShelfFreq   =   150.00;
        firstBandGain  =     0.00;
        firstBandFreq  =   650.00;
        firstBandQ     =     0.71;
        secondBandGain =     0.00;
        secondBandFreq =  1000.00;
        secondBandQ    =     0.71;
        thirdBandGain  =     0.00;
        thirdBandFreq  =  3300.00;
        thirdBandQ     =     0.71;
        highShelfGain  =     0.00;
        highShelfFreq  =   8200.0;
        % Parametric EQ Parameters (auto)
        autoEqState        = struct(...
        'lowShelfGain',      0.00,...
        'lowShelfFreq',    650.00,...
        'firstBandGain',     0.00,...
        'firstBandFreq',   650.00,...
        'firstBandQ',        0.71,...
        'secondBandGain',    0.00,...
        'secondBandFreq', 1000.00,...
        'secondBandQ',       0.71,...
        'thirdBandGain',     0.00,...
        'thirdBandFreq',  3300.00,...
        'thirdBandQ',        0.71,...
        'highShelfGain',     0.00,...
        'highShelfFreq',  8200.0);
    end
    
    properties (Constant)
        PluginInterface = audioPluginInterface(...
            'InputChannels',2,...
            'OutputChannels',2,...
            'PluginName','flowEQ',...
            ... % Model Parameters
            audioPluginParameter('latentDim',      'DisplayName','Latent Dimension',  'Label','',   'Mapping',{'enum','1','2','3'}),...
            audioPluginParameter('xDim',           'DisplayName','x',                 'Label','',   'Mapping',{'lin', -4, 4}),...
            audioPluginParameter('yDim',           'DisplayName','y',                 'Label','',   'Mapping',{'lin', -4, 4}),...
            audioPluginParameter('zDim',           'DisplayName','z',                 'Label','',   'Mapping',{'lin', -4, 4}),...
            audioPluginParameter('decoderMode',    'DisplayName','Decoder Mode',      'Label','',   'Mapping',{'enum', 'manual', 'semantic'}),...    
            audioPluginParameter('firstTerm',      'DisplayName','Embedding A',       'Label','',   'Mapping',{'enum', 'warm', 'bright', 'tight', 'deep'}),...
            audioPluginParameter('interpolate',    'DisplayName','Interpolate',       'Label','',   'Mapping',{'lin', -1, 1}),...
            audioPluginParameter('secondTerm',     'DisplayName','Embedding B',       'Label','',   'Mapping',{'enum', 'warm', 'bright', 'tight', 'deep'}),...
            audioPluginParameter('strength',       'DisplayName','Strength',          'Label','',   'Mapping',{'lin', 0, 1}),...
            audioPluginParameter('autoMode',       'DisplayName','Automatic',         'Label','',   'Mapping',{'enum', 'Manual', 'Auto'}),...
            ... % Parametric EQ Parameters 
            audioPluginParameter('inputGain',      'DisplayName','Input Gain',        'Label','dB', 'Mapping',{'pow', 1/3, -80, 12}),...
            audioPluginParameter('outputGain',     'DisplayName','Output Gain',       'Label','dB', 'Mapping',{'pow', 1/3, -80, 12}),...
            audioPluginParameter('lowShelfGain',   'DisplayName','Lowshelf Gain',     'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('lowShelfFreq',   'DisplayName','Lowshelf Freq.',    'Label','Hz', 'Mapping',{'lin', 22, 1000}),...
            audioPluginParameter('firstBandGain',  'DisplayName','First Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('firstBandFreq',  'DisplayName','First Band Freq.',  'Label','Hz', 'Mapping',{'lin', 82, 3900}),...
            audioPluginParameter('firstBandQ',     'DisplayName','First Band Q',      'Label','Q',  'Mapping',{'lin', 0.1, 10}),...
            audioPluginParameter('secondBandGain', 'DisplayName','Second Band Gain',  'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('secondBandFreq', 'DisplayName','Second Band Freq.', 'Label','Hz', 'Mapping',{'lin', 180, 4700}),...
            audioPluginParameter('secondBandQ',    'DisplayName','Second Band Q',     'Label','Q',  'Mapping',{'lin', 0.1, 10}),...
            audioPluginParameter('thirdBandGain',  'DisplayName','Third Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('thirdBandFreq',  'DisplayName','Third Band Freq.',  'Label','Hz', 'Mapping',{'lin', 220, 10000}),...
            audioPluginParameter('thirdBandQ',     'DisplayName','Third Band Q',      'Label','Q',  'Mapping',{'lin', 0.1, 10}),...
            audioPluginParameter('highShelfGain',  'DisplayName','Highshelf Gain',    'Label','dB', 'Mapping',{'lin', -12, 12}),... 
            audioPluginParameter('highShelfFreq',  'DisplayName','Highshelf Freq.',   'Label','Hz', 'Mapping',{'lin', 580, 20000}))
    end
    %----------------------------------------------------------------------
    % private properties
    %----------------------------------------------------------------------
    properties(Access = private)
        % Lowshelf filter coefs
        lowShelfb, lowShelfa;
        lowShelfState = zeros(2);

        % First band filter coefs
        firstBandb, firstBanda;
        firstBandState = zeros(2);

        % Second band filter coefs
        secondBandb, secondBanda;
        secondBandState = zeros(2);

        % Third band filter coefs
        thirdBandb, thirdBanda;
        thirdBandState = zeros(2);

        % Highself filter coefs
        highShelfb, highShelfa;
        highShelfState = zeros(2);

        % Paramter update flags 
        updateAutoEqState = false;  
        updateLowShelf    = false;
        updateFirstBand   = false;
        updateSecondBand  = false;
        updateThirdBand   = false;
        updateHighShelf   = false;

    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateAutoEqState && plugin.autoMode
                % pass latent vector through decoder
                if     strcmp(plugin.latentDim,'1')
                    x_hat = plugin.net1d.predict([plugin.xDim]);
                elseif strcmp(plugin.latentDim,'2')
                    x_hat = plugin.net2d.predict([plugin.xDim plugin.yDim]);
                else   
                    x_hat = plugin.net3d.predict([plugin.xDim plugin.yDim plugin.zDim]);
                end
                % denormalize 1x13 param vector
                x_hat = plugin.net1d.denormalize(x_hat);
                % update autoEqState to match new params
                plugin.storeEqState(x_hat);
                % request coefficient update for all filters
                plugin.fullFilterReset();
                % turn the parameter update flag off since we are finished
                setUpdateAutoEqState(plugin, false);
            end
            if plugin.updateLowShelf
                fs = getSampleRate(plugin);
                if plugin.autoMode
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(fs, plugin.autoEqState.lowShelfFreq, 0.71, 10.^(plugin.autoEqState.lowShelfGain/20));
                else
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(fs, plugin.lowShelfFreq, 0.71, 10.^(plugin.lowShelfGain/20));
                end
                setUpdateLowShelf(plugin, false);
            end
            if plugin.updateFirstBand
                fs = getSampleRate(plugin);
                if plugin.autoMode
                    [plugin.firstBandb, plugin.firstBanda] = plugin.makePeakFilter(fs, plugin.autoEqState.firstBandFreq, plugin.autoEqState.firstBandQ, 10.^(plugin.autoEqState.firstBandGain/20));
                else
                    [plugin.firstBandb, plugin.firstBanda] = plugin.makePeakFilter(fs, plugin.firstBandFreq, plugin.firstBandQ, 10.^(plugin.firstBandGain/20));
                end
                setUpdateFirstBand(plugin, false);
            end
            if plugin.updateSecondBand
                fs = getSampleRate(plugin);
                if plugin.autoMode
                    [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(fs, plugin.autoEqState.secondBandFreq, plugin.autoEqState.secondBandQ, 10.^(plugin.autoEqState.secondBandGain/20));
                else
                    [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(fs, plugin.secondBandFreq, plugin.secondBandQ, 10.^(plugin.secondBandGain/20));
                end
                setUpdateSecondBand(plugin, false);
            end
            if plugin.updateThirdBand
                fs = getSampleRate(plugin);
                if plugin.autoMode
                    [plugin.thirdBandb, plugin.thirdBanda] = plugin.makePeakFilter(fs, plugin.autoEqState.thirdBandFreq, plugin.autoEqState.thirdBandQ, 10.^(plugin.autoEqState.thirdBandGain/20));
                else
                    [plugin.thirdBandb, plugin.thirdBanda] = plugin.makePeakFilter(fs, plugin.thirdBandFreq, plugin.thirdBandQ, 10.^(plugin.thirdBandGain/20));
                end
                setUpdateThirdBand(plugin, false);
            end
            if plugin.updateHighShelf
                fs = getSampleRate(plugin);
                if plugin.autoMode
                    [plugin.highShelfb, plugin.highShelfa] = plugin.makeHighShelf(fs, plugin.autoEqState.highShelfFreq, 0.71, 10.^(plugin.autoEqState.highShelfGain/20));
                else
                    [plugin.highShelfb, plugin.highShelfa] = plugin.makeHighShelf(fs, plugin.highShelfFreq, 0.71, 10.^(plugin.highShelfGain/20));
                end
                setUpdateHighShelf(plugin, false);
            end
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            u = 10.^(plugin.inputGain/20)*u;

            % Apply biquad filters one-by-one
            [u, plugin.lowShelfState]   = filter(plugin.lowShelfb,   plugin.lowShelfa,   u, plugin.lowShelfState);
            [u, plugin.firstBandState]  = filter(plugin.firstBandb,  plugin.firstBanda,  u, plugin.firstBandState);
            [u, plugin.secondBandState] = filter(plugin.secondBandb, plugin.secondBanda, u, plugin.secondBandState);
            [u, plugin.thirdBandState]  = filter(plugin.thirdBandb,  plugin.thirdBanda,  u, plugin.thirdBandState); 
            [u, plugin.highShelfState]  = filter(plugin.highShelfb,  plugin.highShelfa,  u, plugin.highShelfState);

            % Apply output gain
            y = 10.^(plugin.outputGain/20)*u;
            
            % ------------------------ UDP Comms --------------------------
            if coder.target('MATLAB') || plugin.udpvst
                b = [plugin.lowShelfb plugin.firstBandb...  
                     plugin.secondBandb plugin.thirdBandb...  
                     plugin.highShelfb];
                 
                a = [plugin.lowShelfa plugin.firstBanda...
                     plugin.secondBanda plugin.thirdBanda...
                     plugin.highShelfa];
                 
                p = [plugin.autoEqState.lowShelfGain...
                     plugin.autoEqState.lowShelfFreq...
                     plugin.autoEqState.firstBandGain...
                     plugin.autoEqState.firstBandFreq...
                     plugin.autoEqState.firstBandQ...
                     plugin.autoEqState.secondBandGain...
                     plugin.autoEqState.secondBandFreq...
                     plugin.autoEqState.secondBandQ...
                     plugin.autoEqState.thirdBandGain...
                     plugin.autoEqState.thirdBandFreq...
                     plugin.autoEqState.thirdBandQ...
                     plugin.autoEqState.highShelfGain...
                     plugin.autoEqState.highShelfFreq];
                 
                z = [plugin.xDim plugin.yDim plugin.zDim];
                dim = str2double(plugin.latentDim);
                plugin.udpsend([b, a, p, z, dim])
            end
        end
        
        function setupImpl(plugin, ~)    
            % Initialize filters based on mode (default is 'Auto')
            fs = getSampleRate(plugin);
            if strcmp(plugin.autoMode, 'Auto')
                [plugin.lowShelfb,   plugin.lowShelfa]   = plugin.makeLowShelf  (fs, plugin.autoEqState.lowShelfFreq, 0.71, 10.^(plugin.autoEqState.lowShelfGain/20));
                [plugin.firstBandb,  plugin.firstBanda]  = plugin.makePeakFilter(fs, plugin.autoEqState.firstBandFreq, plugin.autoEqState.firstBandQ, 10.^(plugin.autoEqState.firstBandGain/20));
                [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(fs, plugin.autoEqState.secondBandFreq, plugin.autoEqState.secondBandQ, 10.^(plugin.autoEqState.secondBandGain/20));
                [plugin.thirdBandb,  plugin.thirdBanda]  = plugin.makePeakFilter(fs, plugin.autoEqState.thirdBandFreq, plugin.autoEqState.thirdBandQ, 10.^(plugin.autoEqState.thirdBandGain/20));
                [plugin.highShelfb,  plugin.highShelfa]  = plugin.makeHighShelf (fs, plugin.autoEqState.highShelfFreq, 0.71, 10.^(plugin.autoEqState.highShelfGain/20));
            else
                [plugin.lowShelfb,   plugin.lowShelfa]   = plugin.makeLowShelf  (fs, plugin.lowShelfFreq, 0.71, 10.^(plugin.lowShelfGain/20));
                [plugin.firstBandb,  plugin.firstBanda]  = plugin.makePeakFilter(fs, plugin.firstBandFreq, plugin.firstBandQ, 10.^(plugin.firstBandGain/20));
                [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(fs, plugin.secondBandFreq, plugin.secondBandQ, 10.^(plugin.secondBandGain/20));
                [plugin.thirdBandb,  plugin.thirdBanda]  = plugin.makePeakFilter(fs, plugin.thirdBandFreq, plugin.thirdBandQ, 10.^(plugin.thirdBandGain/20));
                [plugin.highShelfb,  plugin.highShelfa]  = plugin.makeHighShelf (fs, plugin.highShelfFreq, 0.71, 10.^(plugin.highShelfGain/20));
            end
            
            % construct decoder objects
            plugin.net1d = Decoder('decoder1d.mat');
            plugin.net2d = Decoder('decoder2d.mat');
            plugin.net3d = Decoder('decoder3d.mat');
            
            if coder.target('MATLAB') || plugin.udpvst
                % setup UDP sender for comm with DAW
                plugin.udpsend = dsp.UDPSender('RemoteIPPort', 20000);
            end
        end

        function resetImpl(plugin)
            % Request full filter reset
            fullFilterReset(plugin);
            % Request update to Auto EQ params
            setUpdateAutoEqState(plugin, true);

        end
    end    
    %----------------------------------------------------------------------
    % private methods
    %----------------------------------------------------------------------
    methods (Access = private)
        function storeEqState(plugin, x_hat)
            plugin.autoEqState.lowShelfGain   = x_hat(1);
            plugin.autoEqState.lowShelfFreq   = x_hat(2);
            plugin.autoEqState.firstBandGain  = x_hat(3);
            plugin.autoEqState.firstBandFreq  = x_hat(4);
            plugin.autoEqState.firstBandQ     = x_hat(5);
            plugin.autoEqState.secondBandGain = x_hat(6);
            plugin.autoEqState.secondBandFreq = x_hat(7);
            plugin.autoEqState.secondBandQ    = x_hat(8);
            plugin.autoEqState.thirdBandGain  = x_hat(9);
            plugin.autoEqState.thirdBandFreq  = x_hat(10);
            plugin.autoEqState.thirdBandQ     = x_hat(11);
            plugin.autoEqState.highShelfGain  = x_hat(12);
            plugin.autoEqState.highShelfFreq  = x_hat(13);
        end
        %------------------- Full Filter Reset ----------------------------
        function fullFilterReset(plugin)
            % Reset intial conditions for filters
            %plugin.lowShelfState   = zeros(2);
            %plugin.firstBandState  = zeros(2);
            %plugin.secondBandState = zeros(2);
            %plugin.thirdBandState  = zeros(2);
            %plugin.highShelfState  = zeros(2);

            % Request update of manual filters
            setUpdateLowShelf  (plugin, true);
            setUpdateFirstBand (plugin, true);
            setUpdateSecondBand(plugin, true);
            setUpdateThirdBand (plugin, true);
            setUpdateHighShelf (plugin, true);
        end
        %----------------- Parameter Change Flags -------------------------
        function setUpdateAutoEqState(plugin, flag)
            plugin.updateAutoEqState = flag;
        end
        function setUpdateLowShelf(plugin, flag)
            plugin.updateLowShelf = flag;
        end
        function setUpdateFirstBand(plugin, flag)
            plugin.updateFirstBand = flag;
        end
        function setUpdateSecondBand(plugin, flag)
            plugin.updateSecondBand = flag;
        end
        function setUpdateThirdBand(plugin, flag)
            plugin.updateThirdBand = flag;
        end
        function setUpdateHighShelf(plugin, flag)
            plugin.updateHighShelf = flag;
        end
        %--------------------- Coefficient Cooking ------------------------
        % These functions are based on the JUCE DSP library in an
        % effort to match the implementation in the SAFE EQ plugin.
        % The JUCE lib is based upon the Cookbook forumlae by RBJ:
        % https://www.w3.org/2011/audio/audio-eq-cookbook.html
        %------------------------------------------------------------------
        function [b, a] = makeLowShelf(~, fs, cutOffFrequency, Q, gainFactor)
            % initial values
            A = max(0.0, sqrt(gainFactor));
            aminus1 = A - 1;
            aplus1 = A + 1;
            omega = (2 * pi * max(cutOffFrequency, 2.0)) / fs;
            coso = cos(omega);
            beta_ = sin(omega) * sqrt(A) / Q; 
            aminus1TimesCoso = aminus1 * coso;

            % coefs calculation
            b0 = A * (aplus1 - aminus1TimesCoso + beta_);
            b1 = A * 2 * (aminus1 - aplus1 * coso);
            b2 = A * (aplus1 - aminus1TimesCoso - beta_);
            a0 = aplus1 + aminus1TimesCoso + beta_;
            a1 = -2 * (aminus1 + aplus1 * coso);
            a2 = aplus1 + aminus1TimesCoso - beta_;

            % output coefs (not sure to normalize by a0 or not?)
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = makeHighShelf(~, fs, cutOffFrequency, Q, gainFactor)
            % initial values
            A = max(0.0, sqrt(gainFactor));
            aminus1 = A - 1;
            aplus1 = A + 1;
            omega = (2 * pi * max(cutOffFrequency, 2.0)) / fs;
            coso = cos(omega);
            beta_ = sin(omega) * sqrt(A) / Q; 
            aminus1TimesCoso = aminus1 * coso;

            % coefs calculation
            b0 = A * (aplus1 + aminus1TimesCoso + beta_);
            b1 = A * -2 * (aminus1 + aplus1 * coso);
            b2 = A * (aplus1 + aminus1TimesCoso - beta_);
            a0 = aplus1 - aminus1TimesCoso + beta_;
            a1 = 2 * (aminus1 - aplus1 * coso);
            a2 = aplus1 - aminus1TimesCoso - beta_;

            % output coefs
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = makePeakFilter(~, fs, frequency, Q, gainFactor)
            % initial values
            A = max(0.0, sqrt(gainFactor));
            omega = (2 * pi * max(frequency, 2.0)) / fs;
            alpha_ = sin(omega) / (Q * 2);
            c2 = -2 * cos(omega);
            alphaTimesA = alpha_ * A;
            alphaOverA = alpha_ / A;

            % coefs calculation
            b0 = 1 + alphaTimesA;
            b1 = c2;
            b2 = 1 - alphaTimesA;
            a0 = 1 + alphaOverA;
            a1 = c2;
            a2 = 1 - alphaOverA;

            % output coefs
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
    end
    %----------------------------------------------------------------------
    % setter and getter methods
    %----------------------------------------------------------------------
    methods 
        %---------------------------- Decoder -----------------------------
        function set.latentDim(plugin, val)
            plugin.latentDim = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.latentDim(plugin)
            val = plugin.latentDim;
        end
        function set.xDim(plugin, val)
            plugin.xDim = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.xDim(plugin)
            val = plugin.xDim;
        end
        function set.yDim(plugin, val)
            plugin.yDim = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.yDim(plugin)
            val = plugin.yDim;
        end
        function set.zDim(plugin, val)
            plugin.zDim = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.zDim(plugin)
            val = plugin.zDim;
        end
        %-------------------------- Mode Control --------------------------
        function set.autoMode(plugin, val)
            plugin.autoMode = val;
            fullFilterReset(plugin);
        end
        function val = get.autoMode(plugin)
            val = plugin.autoMode;
        end
        %---------------------------- Lowshelf ----------------------------
        function set.lowShelfGain(plugin, val)
            plugin.lowShelfGain = val;
            setUpdateLowShelf(plugin, true);
        end
        function val = get.lowShelfGain(plugin)
            val = plugin.lowShelfGain;
        end
        function set.lowShelfFreq(plugin, val)
            plugin.lowShelfFreq = val;
            setUpdateLowShelf(plugin, true);
        end
        function val = get.lowShelfFreq(plugin)
            val = plugin.lowShelfFreq;
        end
        %---------------------------- First band ----------------------------
        function set.firstBandGain(plugin, val)
            plugin.firstBandGain = val;
            setUpdateFirstBand(plugin, true);
        end
        function val = get.firstBandGain(plugin)
            val = plugin.firstBandGain;
        end
        function set.firstBandQ(plugin, val)
            plugin.firstBandQ = val;
            setUpdateFirstBand(plugin, true);
        end
        function val = get.firstBandQ(plugin)
            val = plugin.firstBandQ;
        end
        function set.firstBandFreq(plugin, val)
            plugin.firstBandFreq = val;
            setUpdateFirstBand(plugin, true);
        end
        function val = get.firstBandFreq(plugin)
            val = plugin.firstBandFreq;
        end
        %---------------------------- Second band ----------------------------
        function set.secondBandGain(plugin, val)
            plugin.secondBandGain = val;
            setUpdateSecondBand(plugin, true);
        end
        function val = get.secondBandGain(plugin)
            val = plugin.secondBandGain;
        end
        function set.secondBandQ(plugin, val)
            plugin.secondBandQ = val;
            setUpdateSecondBand(plugin, true);
        end
        function val = get.secondBandQ(plugin)
            val = plugin.secondBandQ;
        end
        function set.secondBandFreq(plugin, val)
            plugin.secondBandFreq = val;
            setUpdateSecondBand(plugin, true);
        end
        function val = get.secondBandFreq(plugin)
            val = plugin.secondBandFreq;
        end
        %---------------------------- Third band ----------------------------
        function set.thirdBandGain(plugin, val)
            plugin.thirdBandGain = val;
            setUpdateThirdBand(plugin, true);
        end
        function val = get.thirdBandGain(plugin)
            val = plugin.thirdBandGain;
        end
        function set.thirdBandQ(plugin, val)
            plugin.thirdBandQ = val;
            setUpdateThirdBand(plugin, true);
        end
        function val = get.thirdBandQ(plugin)
            val = plugin.thirdBandQ;
        end
        function set.thirdBandFreq(plugin, val)
            plugin.thirdBandFreq = val;
            setUpdateThirdBand(plugin, true);
        end
        function val = get.thirdBandFreq(plugin)
            val = plugin.thirdBandFreq;
        end
        %---------------------------- Highself ----------------------------
        function set.highShelfGain(plugin, val)
            plugin.highShelfGain = val;
            setUpdateHighShelf(plugin, true);
        end
        function val = get.highShelfGain(plugin)
            val = plugin.highShelfGain;
        end
        function set.highShelfFreq(plugin, val)
            plugin.highShelfFreq = val;
            setUpdateHighShelf(plugin, true);
        end
        function val = get.highShelfFreq(plugin)
            val = plugin.highShelfFreq;
        end
    end  
end