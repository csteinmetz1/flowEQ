classdef myPlugin < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % constructor
    %----------------------------------------------------------------------
    methods
        function plugin = myPlugin
            % load network weights here
        end
    end
    %----------------------------------------------------------------------
    % public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        sample = ones(1, 13, 1);
        net; % define the network object   
    end     
    
    properties 
        % In/Out Gain Parameters
        inputGain      =     0.00;   
        outputGain     =     0.00;
        % Neural Network Parameters
        xDim           =      0.0;
        yDim           =      0.0;
        zDim           =      0.0;
        decoderMode    =    'xyz';
        latentDim      =      '1';
        firstTerm      =   'warm';
        interpolate    =      0.0;
        secondTerm     = 'bright';
        % Parametric EQ Parameters
        eqMode         =   'Auto';
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
        
        eqState        = struct(...
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
            'PluginName','myPlugin',...
            ... % Model Parameters
            audioPluginParameter('latentDim',      'DisplayName','Latent Dimension',  'Label','',   'Mapping',{'enum','1','2','3'}),...
            audioPluginParameter('xDim',           'DisplayName','x',                 'Label','',   'Mapping',{'lin', -1, 1}),...
            audioPluginParameter('yDim',           'DisplayName','y',                 'Label','',   'Mapping',{'lin', -1, 1}),...
            audioPluginParameter('zDim',           'DisplayName','z',                 'Label','',   'Mapping',{'lin', -1, 1}),...
            audioPluginParameter('decoderMode',    'DisplayName','Decoder Mode',      'Label','',   'Mapping',{'enum', 'xyz', 'semantic'}),...    
            audioPluginParameter('firstTerm',      'DisplayName','Embedding A',       'Label','',   'Mapping',{'enum', 'warm', 'bright', 'tight', 'deep'}),...
            audioPluginParameter('interpolate',    'DisplayName','Interpolate',       'Label','',   'Mapping',{'lin', -1, 1}),...
            audioPluginParameter('secondTerm',     'DisplayName','Embedding B',       'Label','',   'Mapping',{'enum', 'warm', 'bright', 'tight', 'deep'}),...
            ... % Parametric EQ Parameters 
            audioPluginParameter('inputGain',      'DisplayName','Input Gain',        'Label','dB', 'Mapping',{'pow', 1/3, -140, 12}),...
            audioPluginParameter('outputGain',     'DisplayName','Output Gain',       'Label','dB', 'Mapping',{'pow', 1/3, -140, 12}),...
            audioPluginParameter('eqMode',         'DisplayName','EQ Mode',           'Label','',   'Mapping',{'enum', 'Manual', 'Auto'}),...
            audioPluginParameter('lowShelfGain',   'DisplayName','Lowshelf Gain',     'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('lowShelfFreq',   'DisplayName','Lowshelf Freq.',    'Label','Hz', 'Mapping',{'lin', 150, 1000}),...
            audioPluginParameter('firstBandGain',  'DisplayName','First Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('firstBandFreq',  'DisplayName','First Band Freq.',  'Label','Hz', 'Mapping',{'lin', 560, 3900}),...
            audioPluginParameter('firstBandQ',     'DisplayName','First Band Q',      'Label','Q',  'Mapping',{'lin', 0.71, 10}),...
            audioPluginParameter('secondBandGain', 'DisplayName','Second Band Gain',  'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('secondBandFreq', 'DisplayName','Second Band Freq.', 'Label','Hz', 'Mapping',{'lin', 1000, 4700}),...
            audioPluginParameter('secondBandQ',    'DisplayName','Second Band Q',     'Label','Q',  'Mapping',{'lin', 0.71, 10}),...
            audioPluginParameter('thirdBandGain',  'DisplayName','Third Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12}),...
            audioPluginParameter('thirdBandFreq',  'DisplayName','Third Band Freq.',  'Label','Hz', 'Mapping',{'lin', 3300, 10000}),...
            audioPluginParameter('thirdBandQ',     'DisplayName','Third Band Q',      'Label','Q',  'Mapping',{'lin', 0.71, 10}),...
            audioPluginParameter('highShelfGain',  'DisplayName','Highshelf Gain',    'Label','dB', 'Mapping',{'lin', -12, 12}),... 
            audioPluginParameter('highShelfFreq',  'DisplayName','Highshelf Freq.',   'Label','Hz', 'Mapping',{'lin', 8200, 20000}))
    end
    %----------------------------------------------------------------------
    % private properties
    %----------------------------------------------------------------------
    properties(Access = private)

        % Lowshelf filter coefs
        lowShelfa, lowShelfb;
        lowShelfState = zeros(2);

        % First band filter coefs
        firstBandCoefs;
        firstBandState = zeros(2);

        % Second band filter coefs
        secondBandCoefs;
        secondBandState = zeros(2);

        % Third band filter coefs
        thirdBandCoefs;
        thirdBandState = zeros(2);

        % Highself filter coefs
        highShelfCoefs;
        highShelfState = zeros(2);

        % Paramter update flags 
        updateEqState    = false;  
        updateLowShelf   = false;
        updateFirstBand  = false;
        updateSecondBand = false;
        updateThirdBand  = false;
        updateHighShelf  = false;

    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateLowShelf
                if strcmp(plugin.eqMode, 'Auto')
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                           getSampleRate(plugin),...
                                                           plugin.lowShelfFreq,...
                                                           0.71,...
                                                           plugin.lowShelfGain);
                else
                    [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                           getSampleRate(plugin),...
                                                           plugin.lowShelfFreq,...
                                                           0.71,...
                                                           plugin.lowShelfGain);
                end
                setUpdateLowShelf(plugin, false);
            end
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            u = 10.^(plugin.inputGain/20)*u;

            % Apply biquad filters one-by-one
            [u, plugin.lowShelfState]   = filter(plugin.lowShelfb,   plugin.lowShelfa,   u, plugin.lowShelfState);
            %[u, plugin.firstBandState]  = filter(plugin.firstBandCoefs(1),  plugin.firstBandCoefs(2),  u, plugin.firstBandState)
            %[u, plugin.secondBandState] = filter(plugin.secondBandCoefs(1), plugin.secondBandCoefs(2), u, plugin.secondBandState)
            %[u, plugin.thirdBandState]  = filter(plugin.thirdBandCoefs(1),  plugin.thirdBandCoefs(2),  u, plugin.thirdBandState) 
            %[u, plugin.highShelfState]  = filter(plugin.highShelfCoefs(1),  plugin.highShelfCoefs(2),  u, plugin.lowShelfState)

            % Apply output gain
            y = 10.^(plugin.outputGain/20)*u;
        end

        function setupImpl(plugin, ~)    
            % Initialize filters
            [plugin.lowShelfb, plugin.lowShelfa] = plugin.makeLowShelf(...
                                                   getSampleRate(plugin),...
                                                   plugin.lowShelfFreq,...
                                                   0.71,...
                                                   plugin.lowShelfGain);
        end

        function resetImpl(plugin)
            % Reset intial conditions for filters
            plugin.lowShelfState  = zeros(2);
            plugin.highShelfState = zeros(2);
        end
    end    
    %----------------------------------------------------------------------
    % private methods
    %----------------------------------------------------------------------
    methods (Access = private)
        %----------------- Parameter Change Flags -------------------------
        function setUpdateEqState(plugin, flag)
            plugin.updateEqState = flag;
        end
        function setUpdateLowShelf(plugin, flag)
            plugin.updateLowShelf = flag;
        end
        %--------------------- Coefficient Cooking ------------------------
        % These functions are based on the JUCE DSP library in an
        % effort to match the implementation in the SAFE EQ plugin
        %------------------------------------------------------------------
        function [b, a] = makeLowShelf(plugin, fs, cutOffFrequency, Q, gainFactor)
            % initial values
            A = max(0.0, gainFactor);
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
            b = [b0/a0, b1/a0, b2/a0]
            a = [a0/a0, a1/a0, a2/a0]
        end
        function [b, a] = makeHighShelf(fs, cutOffFrequency, Q, gainFactor)
            % initial values
            A = max(0.0, gainFactor);
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
            a0 = aplus1 + aminus1TimesCoso + beta_;
            a1 = 2 * (aminus1 - aplus1 * coso);
            a2 = aplus1 - aminus1TimesCoso - beta_;

            % output coefs
            b = [b0, b1, b2];
            a = [a0, a1, a2];
        end
        function [b, a] = makePeakFilter(fs, frequency, Q, gainFactor)
            % initial values
            A = max(0.0, gainFactor)
            omega = (2 * pi * max(frequency, 2.0)) / fs;
            alpha_ = sin(omega) / (Q * 2);
            c2 = -2 * cos(omega);
            alphaTimesA = alpha_ * A;
            alphaOverA = alpha_ / A;

            % coefs calculation
            b0 = 1 + alphaTimesA;
            b1 = c2;
            b2 = 1 - alphaOverA;
            a0 = 1 + alphaOverA;
            a1 = c2
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
        function set.latentDim(plugin, val)
            plugin.latentDim = val;
            setUpdateEqState(plugin, true);
        end
        function val = get.latentDim(plugin)
            val = plugin.latentDim;
        end
        function set.xDim(plugin, val)
            plugin.xDim = val;
            setUpdateEqState(plugin, true);
        end
        function val = get.xDim(plugin)
            val = plugin.xDim;
        end
        function set.yDim(plugin, val)
            plugin.yDim = val;
            setUpdateEqState(plugin, true);
        end
        function val = get.yDim(plugin)
            val = plugin.yDim;
        end
        function set.zDim(plugin, val)
            plugin.zDim = val;
            setUpdateEqState(plugin, true);
        end
        function val = get.zDim(plugin)
            val = plugin.zDim;
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
    end  
end