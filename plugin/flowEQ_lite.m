classdef flowEQ_lite < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        net3d;          % three dimensional latent space model                        
        udpvst = false; % set to this true if you want VST with UDP support
        udpsend;        % UDP sender object to be used
    end     
    
    properties 
        % In/Out Gain Parameters
        inputGain        =     0.00;   
        outputGain       =     0.00;
        gainCompensation =    false;
        % Neural Network Parameters
        xDim             =      0.0;
        yDim             =      0.0;
        zDim             =      0.0;
        strength         =      1.0;
        extend           =    false;
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
            'PluginName','flowEQ-lite',...
            'VendorName','Christian Steinmetz',...
            'VendorVersion','0.9.0',...
            ... % GUI Configuration
            audioPluginGridLayout('RowHeight',[30 30 25 25 25 25 25],...
                                  'ColumnWidth',[30 30 100 30 30 25 80 25 25],...
                                  'RowSpacing',15,...
                                  'ColumnSpacing',15,...
                                  'Padding',[20 20 20 20]),...
            ...%'BackgroundColor',[19/255, 22/255, 26/255],...
            ... % Model Parameters
            audioPluginParameter('xDim',            'DisplayName','x',                 'Label','',   'Mapping',{'lin', -2, 2},                            'Layout',[1,5; 1,9],   'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('yDim',            'DisplayName','y',                 'Label','',   'Mapping',{'lin', -2, 2},                            'Layout',[2,5; 2,9],   'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('zDim',            'DisplayName','z',                 'Label','',   'Mapping',{'lin', -2, 2},                            'Layout',[3,5; 3,9],   'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('extend',          'DisplayName','Extend (x2)',       'Label','',   'Mapping',{'enum', 'Off', 'On'},                     'Layout',[4,8; 4,9],   'DisplayNameLocation', 'below', 'Style', 'vtoggle'),...
            audioPluginParameter('strength',        'DisplayName','Strength',          'Label','',   'Mapping',{'lin', 0, 1},                             'Layout',[4,5; 5,6],   'DisplayNameLocation', 'right', 'Style', 'rotaryknob', 'EditBoxLocation', 'below'),...           
            audioPluginParameter('gainCompensation','DisplayName','Auto Gain Comp.',   'Label','',   'Mapping',{'enum', 'Off', 'On'},                     'Layout',[5,1; 5,2],   'DisplayNameLocation', 'right', 'Style', 'vtoggle'),...          
            audioPluginParameter('inputGain',       'DisplayName','In Gain',           'Label','dB', 'Mapping',{'pow', 1/3, -80, 12},                     'Layout',[1,1; 2,2],   'DisplayNameLocation', 'right', 'Style', 'rotaryknob'),...
            audioPluginParameter('outputGain',      'DisplayName','Out Gain',          'Label','dB', 'Mapping',{'pow', 1/3, -80, 12},                     'Layout',[3,1; 4,2],   'DisplayNameLocation', 'right', 'Style', 'rotaryknob'));
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

        % Loudness compenstation
        loudnessGain;
        loudnessFrames
        preEqLoudnessMeter;
        postEqLoudnessMeter; 

        % Latent vector
        x = 0; y = 0; z = 0;

    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function out = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateAutoEqState
                
                plugin.x = plugin.xDim;
                plugin.y = plugin.yDim;
                plugin.z = plugin.zDim;

                % extend the area of the latent space that is reachable
                if plugin.extend
                    plugin.x = plugin.x * 2;
                    plugin.y = plugin.y * 2;
                    plugin.z = plugin.z * 2;
                end
            
                x_hat = plugin.net3d.predict([plugin.x plugin.y plugin.z]); % pass the latent code through the decoder
                x_hat = plugin.net3d.denormalize(x_hat);                 % denormalize 1x13 param vector
                plugin.storeEqState(x_hat);                                 % update autoEqState to match new params
                plugin.fullFilterReset();                                   % request coefficient update for all filters
                setUpdateAutoEqState(plugin, false);                        % turn the param update flag off (we are done)
            end
            if plugin.updateLowShelf
                fs = getSampleRate(plugin);
                [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                        fs,...
                                                        plugin.autoEqState.lowShelfFreq,...
                                                        0.71,...
                                                        10.^((plugin.strength * plugin.autoEqState.lowShelfGain)/20));
                setUpdateLowShelf(plugin, false);
            end
            if plugin.updateFirstBand
                fs = getSampleRate(plugin);
                [plugin.firstBandb, plugin.firstBanda] = plugin.makePeakFilter(...
                                                            fs,...
                                                            plugin.autoEqState.firstBandFreq,...
                                                            plugin.autoEqState.firstBandQ,...
                                                            10.^((plugin.strength * plugin.autoEqState.firstBandGain)/20));
                setUpdateFirstBand(plugin, false);
            end
            if plugin.updateSecondBand
                fs = getSampleRate(plugin);
                [plugin.secondBandb, plugin.secondBanda] = plugin.makePeakFilter(...
                                                            fs,...
                                                            plugin.autoEqState.secondBandFreq,...
                                                            plugin.autoEqState.secondBandQ,...
                                                            10.^((plugin.strength * plugin.autoEqState.secondBandGain)/20));
                setUpdateSecondBand(plugin, false);
            end
            if plugin.updateThirdBand
                fs = getSampleRate(plugin);
                [plugin.thirdBandb, plugin.thirdBanda] = plugin.makePeakFilter(...
                                                            fs,...
                                                            plugin.autoEqState.thirdBandFreq,...
                                                            plugin.autoEqState.thirdBandQ,...
                                                            10.^((plugin.strength * plugin.autoEqState.thirdBandGain)/20));
                setUpdateThirdBand(plugin, false);
            end
            if plugin.updateHighShelf
                fs = getSampleRate(plugin);
                [plugin.highShelfb, plugin.highShelfa] = plugin.makeHighShelf(...
                                                            fs,...
                                                            plugin.autoEqState.highShelfFreq,...
                                                            0.71,...
                                                            10.^((plugin.strength * plugin.autoEqState.highShelfGain)/20));
                setUpdateHighShelf(plugin, false);
            end
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            u = 10.^(plugin.inputGain/20)*u;

            % measure pre EQ short-term loudness
            [~,preEqShortTermLoudness,~,~] = plugin.preEqLoudnessMeter(u);

            % Apply biquad filters one-by-one
            [u, plugin.lowShelfState]   = filter(plugin.lowShelfb,   plugin.lowShelfa,   u, plugin.lowShelfState);
            [u, plugin.firstBandState]  = filter(plugin.firstBandb,  plugin.firstBanda,  u, plugin.firstBandState);
            [u, plugin.secondBandState] = filter(plugin.secondBandb, plugin.secondBanda, u, plugin.secondBandState);
            [u, plugin.thirdBandState]  = filter(plugin.thirdBandb,  plugin.thirdBanda,  u, plugin.thirdBandState); 
            [u, plugin.highShelfState]  = filter(plugin.highShelfb,  plugin.highShelfa,  u, plugin.highShelfState); 
            
            % measure post EQ short-term loudness
            [~,postEqShortTermLoudness,~,~] = plugin.postEqLoudnessMeter(u);

            % determine loudness gain compensation
            if plugin.gainCompensation
                plugin.loudnessFrames = plugin.loudnessFrames + 1;

                % only update the gain compensation after set number of frames
                if plugin.loudnessFrames > 4
                    plugin.loudnessFrames = 1;

                    % determine the difference between input and output loudness
                    gainComp = mean(preEqShortTermLoudness, 'all') - mean(postEqShortTermLoudness, 'all');

                    % if -Inf in loudness values set unity gain
                    if any(isnan(gainComp))
                        gainComp = 0.0;
                    end

                    % bound gain value for safety (otherwise ouch...)
                    minGain = -12.0;
                    maxGain =  12.0;

                    if gainComp > maxGain 
                        gainComp = maxGain;
                    elseif gainComp < minGain
                        gainComp = minGain;
                    end
                    
                    %gainComp
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

                code = [plugin.x, plugin.y, plugin.z];
                dim  = 3;
                plugin.udpsend([b, a, p, code, dim])
            end
        end
        
        function setupImpl(plugin, ~)               
            % Initialize filters
            fs = getSampleRate(plugin);

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
          
            % setup loudness compensation
            plugin.loudnessGain         =   0.0;
            plugin.loudnessFrames       =     1;
            plugin.preEqLoudnessMeter   = loudnessMeter;
            plugin.postEqLoudnessMeter  = loudnessMeter;

            % construct decoder objects ( try to do this programatically )
            plugin.net3d = Decoder('vae3d_beta_0.0010.mat');
            
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
        %-------------------------- Mode Control --------------------------
        function set.extend(plugin, val)
            plugin.extend = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.extend(plugin)
            val = plugin.extend;
        end
    end  
end