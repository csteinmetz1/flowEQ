classdef flowEQ < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % Public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        net1d;  % 1D latent space model
        net2d;  % 2D latent space model
        net3d;  % 3D latent space model
        
        % Latent space embeddings for semantic descriptors
        codes = load('codes.mat', 'codes');
                        
        udpvst = false; % Set to this true if you want VST with UDP support
        udpsend;        % UDP sender object to be used
    end     
    
    properties 
        % In/Out Gain Parameters
        inputGain        =     0.00;   
        outputGain       =     0.00;
        gainCompensation =    false;
        gainUpdateRate   =  GainUpdateRate.medium;
        gainRange        =  GainRange.medium;
        % Neural Network Parameters
        xDim             =      0.0;
        yDim             =      0.0;
        zDim             =      0.0;
        interpolate      =      0.0;
        strength         =      1.0;
        firstCode        =  Semantic.warm;
        secondCode       =  Semantic.bright;
        latentDim        =  LatentDim.two;
        eqMode           =  OperatingMode.traverse;
        disentanglement  =  DisentangleControl.less;
        extend           =    false;
        % Parametric EQ Parameters (manual)
        lowShelfActive   =     true;
        lowShelfGain     =     0.00;
        lowShelfFreq     =   150.00;
        firstBandActive  =     true;
        firstBandGain    =     0.00;
        firstBandFreq    =   650.00;
        firstBandQ       =     0.71;
        secondBandActive =     true;
        secondBandGain   =     0.00;
        secondBandFreq   =  1000.00;
        secondBandQ      =     0.71;
        thirdBandActive  =     true;
        thirdBandGain    =     0.00;
        thirdBandFreq    =  3300.00;
        thirdBandQ       =     0.71;
        highShelfActive  =     true;
        highShelfGain    =     0.00;
        highShelfFreq    =   8200.0;
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
        % Generate plugin interface object based on MATLAB version
        PluginInterface = GeneratePluginInterface('1.0.3');
        end
    %----------------------------------------------------------------------
    % Private properties
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

        % Loudness compenstation
        loudnessGain;
        loudnessFrames
        preEqLoudnessMeter;
        postEqLoudnessMeter; 

        % bound gain value for safety (otherwise ouch...)
        minGain = -12.0;
        maxGain =  12.0;

        % Latent vector
        x = 0; y = 0; z = 0;

    end
    %----------------------------------------------------------------------
    % Public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function out = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateAutoEqState && plugin.eqMode ~= OperatingMode.manual
                
                % Determine latent code based on operating mode
                if plugin.eqMode == OperatingMode.traverse
                    plugin.x = plugin.xDim;
                    plugin.y = plugin.yDim;
                    plugin.z = plugin.zDim;
                elseif plugin.eqMode == OperatingMode.semantic
                    aIdx = plugin.firstCode;
                    bIdx = plugin.secondCode;
                    A = reshape(plugin.codes.codes(plugin.latentDim, plugin.disentanglement, aIdx, :), 3, 1);
                    B = reshape(plugin.codes.codes(plugin.latentDim, plugin.disentanglement, bIdx, :), 3, 1);
                    latentCode = A + (plugin.interpolate * (B - A));
                    plugin.x = latentCode(1);
                    plugin.y = latentCode(2);
                    plugin.z = latentCode(3);
                end

                % Extend the area of the latent space that is reachable
                % (This is only applicable in Traverse mode)
                if plugin.extend && plugin.eqMode == OperatingMode.traverse
                    plugin.x = plugin.x * 2;
                    plugin.y = plugin.y * 2;
                    plugin.z = plugin.z * 2;
                end

                % Pass latent vector through decoder
                switch (plugin.latentDim)
                    case LatentDim.one
                        x_hat = plugin.net1d{plugin.disentanglement}.predict([plugin.x]);
                    case LatentDim.two
                        x_hat = plugin.net2d{plugin.disentanglement}.predict([plugin.x plugin.y]);
                    otherwise 
                        x_hat = plugin.net3d{plugin.disentanglement}.predict([plugin.x plugin.y plugin.z]);
                end 
                
                x_hat = plugin.net1d{1}.denormalize(x_hat); % Denormalize 1x13 param vector
                plugin.storeEqState(x_hat);                 % Update autoEqState to match new params
                plugin.fullFilterReset();                   % Request coefficient update for all filters
                setUpdateAutoEqState(plugin, false);        % Turn the param update flag off, we are done :)
            end
            if plugin.updateLowShelf
                fs = getSampleRate(plugin);
                if ~(plugin.eqMode == OperatingMode.manual)
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                           fs,...
                                                           plugin.autoEqState.lowShelfFreq,...
                                                           0.71,...
                                                           10.^((plugin.strength * plugin.autoEqState.lowShelfGain)/20));
                else
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                           fs,...
                                                           plugin.lowShelfFreq,...
                                                           0.71,...
                                                           10.^(plugin.lowShelfGain/20));
                end
                setUpdateLowShelf(plugin, false);
            end
            if plugin.updateFirstBand
                fs = getSampleRate(plugin);
                if ~(plugin.eqMode == OperatingMode.manual)
                    [plugin.firstBandb, plugin.firstBanda] = plugin.makePeakFilter(...
                                                             fs,...
                                                             plugin.autoEqState.firstBandFreq,...
                                                             plugin.autoEqState.firstBandQ,...
                                                             10.^((plugin.strength * plugin.autoEqState.firstBandGain)/20));
                else
                    [plugin.firstBandb, plugin.firstBanda] = plugin.makePeakFilter(...
                                                             fs,...
                                                             plugin.firstBandFreq,...
                                                             plugin.firstBandQ,...
                                                             10.^(plugin.firstBandGain/20));
                end
                setUpdateFirstBand(plugin, false);
            end
            if plugin.updateSecondBand
                fs = getSampleRate(plugin);
                if ~(plugin.eqMode == OperatingMode.manual)
                    [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(...
                                                               fs,...
                                                               plugin.autoEqState.secondBandFreq,...
                                                               plugin.autoEqState.secondBandQ,...
                                                               10.^((plugin.strength * plugin.autoEqState.secondBandGain)/20));
                else
                    [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(...
                                                               fs,...
                                                               plugin.secondBandFreq,...
                                                               plugin.secondBandQ, 10.^(plugin.secondBandGain/20));
                end
                setUpdateSecondBand(plugin, false);
            end
            if plugin.updateThirdBand
                fs = getSampleRate(plugin);
                if ~(plugin.eqMode == OperatingMode.manual)
                    [plugin.thirdBandb, plugin.thirdBanda] = plugin.makePeakFilter(...
                                                             fs,...
                                                             plugin.autoEqState.thirdBandFreq,...
                                                             plugin.autoEqState.thirdBandQ,...
                                                             10.^((plugin.strength * plugin.autoEqState.thirdBandGain)/20));
                else
                    [plugin.thirdBandb, plugin.thirdBanda] = plugin.makePeakFilter(...
                                                             fs,...
                                                             plugin.thirdBandFreq,...
                                                             plugin.thirdBandQ,...
                                                             10.^(plugin.thirdBandGain/20));
                end
                setUpdateThirdBand(plugin, false);
            end
            if plugin.updateHighShelf
                fs = getSampleRate(plugin);
                if ~(plugin.eqMode == OperatingMode.manual)
                    [plugin.highShelfb, plugin.highShelfa] = plugin.makeHighShelf(...
                                                             fs,...
                                                             plugin.autoEqState.highShelfFreq,...
                                                             0.71,...
                                                             10.^((plugin.strength * plugin.autoEqState.highShelfGain)/20));
                else
                    [plugin.highShelfb, plugin.highShelfa] = plugin.makeHighShelf(...
                                                             fs,...
                                                             plugin.highShelfFreq,...
                                                             0.71, 10.^(plugin.highShelfGain/20));
                end
                setUpdateHighShelf(plugin, false);
            end
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            u = 10.^(plugin.inputGain/20)*u;

            % Measure pre EQ short-term loudness
            [~,preEqShortTermLoudness,~,~] = plugin.preEqLoudnessMeter(u);

            % Apply biquad filters one-by-one, only if active
            if plugin.lowShelfActive
                [u, plugin.lowShelfState]   = filter(plugin.lowShelfb,   plugin.lowShelfa,   u, plugin.lowShelfState);
            end
            if plugin.firstBandActive
                [u, plugin.firstBandState]  = filter(plugin.firstBandb,  plugin.firstBanda,  u, plugin.firstBandState);
            end
            if plugin.secondBandActive
                [u, plugin.secondBandState] = filter(plugin.secondBandb, plugin.secondBanda, u, plugin.secondBandState);
            end
            if plugin.thirdBandActive
                [u, plugin.thirdBandState]  = filter(plugin.thirdBandb,  plugin.thirdBanda,  u, plugin.thirdBandState); 
            end
            if plugin.highShelfActive
                [u, plugin.highShelfState]  = filter(plugin.highShelfb,  plugin.highShelfa,  u, plugin.highShelfState);
            end            
            
            % Measure post-EQ short-term loudness
            [~,postEqShortTermLoudness,~,~] = plugin.postEqLoudnessMeter(u);

            % Determine loudness gain compensation
            if plugin.gainCompensation
                plugin.loudnessFrames = plugin.loudnessFrames + 1;

                % Only update the gain compensation after set number of frames
                if plugin.loudnessFrames > plugin.gainUpdateRate
                    plugin.loudnessFrames = 1;

                    % Determine the difference between input and output loudness
                    gainComp = mean(preEqShortTermLoudness, 'all') - mean(postEqShortTermLoudness, 'all');

                    % If -Inf in loudness values set to unity gain
                    if any(isnan(gainComp))
                        gainComp = 0.0;
                    end

                    % Bound gain value for safety (otherwise, could be ouch...)
                    if     plugin.gainRange == GainRange.low
                        plugin.minGain = -6.0;
                        plugin.maxGain =  6.0;
                    elseif plugin.gainRange == GainRange.medium
                        plugin.minGain = -12.0;
                        plugin.maxGain =  12.0;
                    elseif   plugin.gainRange == GainRange.high
                        plugin.minGain = -24.0;
                        plugin.maxGain =  24.0;
                    end
                    
                    if gainComp > plugin.maxGain 
                        gainComp = plugin.maxGain;
                    elseif gainComp < plugin.minGain
                        gainComp = plugin.minGain;
                    end
                    
                    % Set the gain compensation value
                    plugin.loudnessGain = gainComp;
                end
                % Apply loudness compensation output gain
                out = 10.^(plugin.loudnessGain/20)*u;
            else
                % Apply user-set output gain
                out = 10.^(plugin.outputGain/20)*u;
            end
            
            % ------------------------ UDP Comms --------------------------
            if coder.target('MATLAB') || plugin.udpvst
                b = [plugin.lowShelfb plugin.firstBandb...  
                     plugin.secondBandb plugin.thirdBandb...  
                     plugin.highShelfb];
                 
                a = [plugin.lowShelfa plugin.firstBanda...
                     plugin.secondBanda plugin.thirdBanda...
                     plugin.highShelfa];
                
                if plugin.eqMode ~= OperatingMode.manual
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
                else
                    p = [plugin.lowShelfGain...
                         plugin.lowShelfFreq...
                         plugin.firstBandGain...
                         plugin.firstBandFreq...
                         plugin.firstBandQ...
                         plugin.secondBandGain...
                         plugin.secondBandFreq...
                         plugin.secondBandQ...
                         plugin.thirdBandGain...
                         plugin.thirdBandFreq...
                         plugin.thirdBandQ...
                         plugin.highShelfGain...
                         plugin.highShelfFreq];
                end
                code = [plugin.x, plugin.y, plugin.z];
                dim  = cast(plugin.latentDim, 'double');
                plugin.udpsend([b, a, p, code, dim])
            end
        end
        
        function setupImpl(plugin, ~)               
            % Initialize filters based on mode (default is 'Traverse')
            fs = getSampleRate(plugin);
            if plugin.eqMode == OperatingMode.traverse
                [plugin.lowShelfb,   plugin.lowShelfa]   = plugin.makeLowShelf(...
                                                           fs,...
                                                           plugin.autoEqState.lowShelfFreq,...
                                                           0.71,...
                                                           10.^((plugin.strength * plugin.autoEqState.lowShelfGain)/20));

                [plugin.firstBandb,  plugin.firstBanda]  = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.autoEqState.firstBandFreq,...
                                                           plugin.autoEqState.firstBandQ,...
                                                           10.^((plugin.strength * plugin.autoEqState.firstBandGain)/20));

                [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.autoEqState.secondBandFreq,...
                                                           plugin.autoEqState.secondBandQ,...
                                                           10.^(plugin.autoEqState.secondBandGain/20));

                [plugin.thirdBandb,  plugin.thirdBanda]  = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.autoEqState.thirdBandFreq,...
                                                           plugin.autoEqState.thirdBandQ,...
                                                           10.^((plugin.strength * plugin.autoEqState.thirdBandGain)/20));

                [plugin.highShelfb,  plugin.highShelfa]  = plugin.makeHighShelf(...
                                                           fs,...
                                                           plugin.autoEqState.highShelfFreq,...
                                                           0.71,...
                                                           10.^((plugin.strength * plugin.autoEqState.highShelfGain)/20));
            else
                [plugin.lowShelfb,   plugin.lowShelfa]   = plugin.makeLowShelf(...
                                                           fs,...
                                                           plugin.lowShelfFreq,...
                                                           0.71,...
                                                           10.^(plugin.lowShelfGain/20));

                [plugin.firstBandb,  plugin.firstBanda]  = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.firstBandFreq,...
                                                           plugin.firstBandQ,...
                                                           10.^(plugin.firstBandGain/20));

                [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.secondBandFreq,...
                                                           plugin.secondBandQ,...
                                                           10.^(plugin.secondBandGain/20));

                [plugin.thirdBandb,  plugin.thirdBanda]  = plugin.makePeakFilter(...
                                                           fs,...
                                                           plugin.thirdBandFreq,...
                                                           plugin.thirdBandQ,...
                                                           10.^(plugin.thirdBandGain/20));

                [plugin.highShelfb,  plugin.highShelfa]  = plugin.makeHighShelf(...
                                                           fs,...
                                                           plugin.highShelfFreq,...
                                                           0.71,...
                                                           10.^(plugin.highShelfGain/20));
            end
            
            % Setup loudness compensation
            plugin.loudnessGain         =   0.0;
            plugin.loudnessFrames       =     1;
            plugin.preEqLoudnessMeter   = loudnessMeter;
            plugin.postEqLoudnessMeter  = loudnessMeter;

            % Construct decoder objects (these are stored in 'assets' dir
            plugin.net1d = {Decoder('vae1d_beta_0.0000.mat'),Decoder('vae1d_beta_0.0010.mat'),Decoder('vae1d_beta_0.0100.mat'),Decoder('vae1d_beta_0.0200.mat')};
            plugin.net2d = {Decoder('vae2d_beta_0.0000.mat'),Decoder('vae2d_beta_0.0010.mat'),Decoder('vae2d_beta_0.0100.mat'),Decoder('vae2d_beta_0.0200.mat')};
            plugin.net3d = {Decoder('vae3d_beta_0.0000.mat'),Decoder('vae3d_beta_0.0010.mat'),Decoder('vae3d_beta_0.0100.mat'),Decoder('vae3d_beta_0.0200.mat')};
            
            if coder.target('MATLAB') || plugin.udpvst
                % Setup UDP sender for comm with DAW
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
    % Private methods
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
            % Initial values
            A = max(0.0, sqrt(gainFactor));
            aminus1 = A - 1;
            aplus1 = A + 1;
            omega = (2 * pi * max(cutOffFrequency, 2.0)) / fs;
            coso = cos(omega);
            beta_ = sin(omega) * sqrt(A) / Q; 
            aminus1TimesCoso = aminus1 * coso;

            % Coefs calculation
            b0 = A * (aplus1 - aminus1TimesCoso + beta_);
            b1 = A * 2 * (aminus1 - aplus1 * coso);
            b2 = A * (aplus1 - aminus1TimesCoso - beta_);
            a0 = aplus1 + aminus1TimesCoso + beta_;
            a1 = -2 * (aminus1 + aplus1 * coso);
            a2 = aplus1 + aminus1TimesCoso - beta_;

            % Output coefs (not sure to normalize by a0 or not?)
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = makeHighShelf(~, fs, cutOffFrequency, Q, gainFactor)
            % Initial values
            A = max(0.0, sqrt(gainFactor));
            aminus1 = A - 1;
            aplus1 = A + 1;
            omega = (2 * pi * max(cutOffFrequency, 2.0)) / fs;
            coso = cos(omega);
            beta_ = sin(omega) * sqrt(A) / Q; 
            aminus1TimesCoso = aminus1 * coso;

            % Coefs calculation
            b0 = A * (aplus1 + aminus1TimesCoso + beta_);
            b1 = A * -2 * (aminus1 + aplus1 * coso);
            b2 = A * (aplus1 + aminus1TimesCoso - beta_);
            a0 = aplus1 - aminus1TimesCoso + beta_;
            a1 = 2 * (aminus1 - aplus1 * coso);
            a2 = aplus1 - aminus1TimesCoso - beta_;

            % Output coefs
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = makePeakFilter(~, fs, frequency, Q, gainFactor)
            % Initial values
            A = max(0.0, sqrt(gainFactor));
            omega = (2 * pi * max(frequency, 2.0)) / fs;
            alpha_ = sin(omega) / (Q * 2);
            c2 = -2 * cos(omega);
            alphaTimesA = alpha_ * A;
            alphaOverA = alpha_ / A;

            % Coefs calculation
            b0 = 1 + alphaTimesA;
            b1 = c2;
            b2 = 1 - alphaTimesA;
            a0 = 1 + alphaOverA;
            a1 = c2;
            a2 = 1 - alphaOverA;

            % Output coefs
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
    end
    %----------------------------------------------------------------------
    % Setter and Getter methods
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
        function set.disentanglement(plugin, val)
            plugin.disentanglement = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.disentanglement(plugin)
            val = plugin.disentanglement;
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
        function set.strength(plugin, val)
            plugin.strength = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.strength(plugin)
            val = plugin.strength;
        end
        function set.firstCode(plugin, val)
            plugin.firstCode = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.firstCode(plugin)
            val = plugin.firstCode;
        end
        function set.secondCode(plugin, val)
            plugin.secondCode = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.secondCode(plugin)
            val = plugin.secondCode;
        end
        function set.interpolate(plugin, val)
            plugin.interpolate = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.interpolate(plugin)
            val = plugin.interpolate;
        end
        %-------------------------- Mode Control --------------------------
        function set.eqMode(plugin, val)
            plugin.eqMode = val;
            if plugin.eqMode == OperatingMode.manual
                fullFilterReset(plugin)
            else
                setUpdateAutoEqState(plugin, true);
            end
        end
        function val = get.eqMode(plugin)
            val = plugin.eqMode;
        end
        function set.extend(plugin, val)
            plugin.extend = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.extend(plugin)
            val = plugin.extend;
        end
        %---------------------------- Lowshelf ----------------------------
        function set.lowShelfActive(plugin, val)
            plugin.lowShelfActive = val;
        end
        function val = get.lowShelfActive(plugin)
            val = plugin.lowShelfActive;
        end
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
        function set.firstBandActive(plugin, val)
            plugin.firstBandActive = val;
        end
        function val = get.firstBandActive(plugin)
            val = plugin.firstBandActive;
        end
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
        function set.secondBandActive(plugin, val)
            plugin.secondBandActive = val;
        end
        function val = get.secondBandActive(plugin)
            val = plugin.secondBandActive;
        end
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
        function set.thirdBandActive(plugin, val)
            plugin.thirdBandActive = val;
        end
        function val = get.thirdBandActive(plugin)
            val = plugin.thirdBandActive;
        end
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
        function set.highShelfActive(plugin, val)
            plugin.highShelfActive = val;
        end
        function val = get.highShelfActive(plugin)
            val = plugin.highShelfActive;
        end
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