classdef flowEQ < audioPlugin & matlab.System
    %----------------------------------------------------------------------
    % public properties
    %----------------------------------------------------------------------
    properties(Nontunable)
        net1d;  % one dimensional latent space model
        net2d;  % two dimensional latent space model
        net3d;  % three dimensional latent space model
        
        % latent space embeddings for semantic descriptors
        codes = struct2cell(load('codes.mat', 'codes'));
                        
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
        interpolate      =      0.0;
        strength         =      1.0;
        secondTerm       =  Semantic.warm;
        latentDim        =  LatentDim.two;
        firstTerm        =  Semantic.warm;
        eqMode           =  OperatingMode.traverse;
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
        PluginInterface = audioPluginInterface(...
            'InputChannels',2,...
            'OutputChannels',2,...
            'PluginName','flowEQ',...
            'VendorName','Christian Steinmetz',...
            'VendorVersion','0.5.0',...
            ... % GUI Configuration
            audioPluginGridLayout('RowHeight',[25 25 25 25 25 25 25 25 25 60 25 60 25 60 25],...
                                  'ColumnWidth',[60 60 60 100 100 100 100 100],...
                                  'RowSpacing',15,...
                                  'ColumnSpacing',15,...
                                  'Padding',[20 20 20 20]),...
            ...%'BackgroundColor',[19/255, 22/255, 26/255],...
            ... % Model Parameters
            audioPluginParameter('latentDim',       'DisplayName','Latent',            'Label','',   'Mapping',{'enum','1','2','3'},                        'Layout',[6,5; 6,6], 'DisplayNameLocation', 'left'),...
            audioPluginParameter('eqMode',          'DisplayName','EQ Mode',           'Label','',   'Mapping',{'enum', 'Automatic', 'Semantic', 'Manual'}, 'Layout',[1,5; 1,8], 'DisplayNameLocation', 'left'),...
            audioPluginParameter('xDim',            'DisplayName','x',                 'Label','',   'Mapping',{'lin', -2, 2},                              'Layout',[2,4; 2,5], 'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('yDim',            'DisplayName','y',                 'Label','',   'Mapping',{'lin', -2, 2},                              'Layout',[3,4; 3,5], 'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('zDim',            'DisplayName','z',                 'Label','',   'Mapping',{'lin', -2, 2},                              'Layout',[4,4; 4,5], 'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('extend',          'DisplayName','Extend (x2)',       'Label','',   'Mapping',{'enum', 'On', 'Off'},                       'Layout',[5,5; 5,5], 'DisplayNameLocation', 'left'),...
            audioPluginParameter('firstTerm',       'DisplayName','Embedding A',       'Label','',   'Mapping',{'enum', 'Warm', 'Bright', 'Sharp'},         'Layout',[2,7; 2,8], 'DisplayNameLocation', 'left'),...
            audioPluginParameter('secondTerm',      'DisplayName','Embedding B',       'Label','',   'Mapping',{'enum', 'Warm', 'Bright', 'Sharp'},         'Layout',[3,7; 3,8], 'DisplayNameLocation', 'left'),...
            audioPluginParameter('interpolate',     'DisplayName','Interpolate',       'Label','',   'Mapping',{'lin', 0, 1},                               'Layout',[4,7; 4,8], 'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('strength',        'DisplayName','Strength',          'Label','',   'Mapping',{'lin',  0, 1},                              'Layout',[7,5; 7,8], 'DisplayNameLocation', 'left', 'EditBoxLocation', 'right'),...
            audioPluginParameter('gainCompensation','DisplayName','Compensation',      'Label','',   'Mapping',{'enum', 'Off', 'On'},                       'Layout',[10,1; 10,1], 'DisplayNameLocation', 'below', 'Style', 'checkbox'),...          
            audioPluginParameter('inputGain',       'DisplayName','In Gain',           'Label','dB', 'Mapping',{'pow', 1/3, -80, 12},                       'Layout',[1,1; 7,1], 'DisplayNameLocation', 'below', 'Style', 'vslider'),...
            audioPluginParameter('outputGain',      'DisplayName','Out Gain',          'Label','dB', 'Mapping',{'pow', 1/3, -80, 12},                       'Layout',[1,2; 7,2], 'DisplayNameLocation', 'below', 'Style', 'vslider'),...
            ... % Parametric EQ Parameters 
            audioPluginParameter('lowShelfActive',  'DisplayName','Active',            'Label','',   'Mapping',{'enum', 'Bypassed', 'Active'},              'Layout',[9,4; 9,4],   'DisplayNameLocation', 'none',  'Style', 'checkbox'),...
            audioPluginParameter('lowShelfGain',    'DisplayName','Lowshelf Gain',     'Label','dB', 'Mapping',{'lin', -12, 12},                            'Layout',[10,4; 10,4], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('lowShelfFreq',    'DisplayName','Lowshelf Freq.',    'Label','Hz', 'Mapping',{'lin', 22, 1000},                           'Layout',[12,4; 12,4], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('firstBandActive', 'DisplayName','Active',            'Label','',   'Mapping',{'enum', 'Bypassed', 'Active'},              'Layout',[9,5; 9,5],   'DisplayNameLocation', 'none',  'Style', 'checkbox'),...
            audioPluginParameter('firstBandGain',   'DisplayName','First Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12},                            'Layout',[10,5; 10,5], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('firstBandFreq',   'DisplayName','First Band Freq.',  'Label','Hz', 'Mapping',{'lin', 82, 3900},                           'Layout',[12,5; 12,5], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('firstBandQ',      'DisplayName','First Band Q',      'Label','',   'Mapping',{'lin', 0.1, 10},                            'Layout',[14,5; 14,5], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('secondBandActive','DisplayName','Active',            'Label','',   'Mapping',{'enum', 'Bypassed', 'Active'},              'Layout',[9,6; 9,6],   'DisplayNameLocation', 'none',  'Style', 'checkbox'),...
            audioPluginParameter('secondBandGain',  'DisplayName','Second Band Gain',  'Label','dB', 'Mapping',{'lin', -12, 12},                            'Layout',[10,6; 10,6], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('secondBandFreq',  'DisplayName','Second Band Freq.', 'Label','Hz', 'Mapping',{'lin', 180, 4700},                          'Layout',[12,6; 12,6], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('secondBandQ',     'DisplayName','Second Band Q',     'Label','',   'Mapping',{'lin', 0.1, 10},                            'Layout',[14,6; 14,6], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('thirdBandActive', 'DisplayName','Active',            'Label','',   'Mapping',{'enum', 'Bypassed', 'Active'},              'Layout',[9,7; 9,7],   'DisplayNameLocation', 'none',  'Style', 'checkbox'),...
            audioPluginParameter('thirdBandGain',   'DisplayName','Third Band Gain',   'Label','dB', 'Mapping',{'lin', -12, 12},                            'Layout',[10,7; 10,7], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('thirdBandFreq',   'DisplayName','Third Band Freq.',  'Label','Hz', 'Mapping',{'lin', 220, 10000},                         'Layout',[12,7; 12,7], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('thirdBandQ',      'DisplayName','Third Band Q',      'Label','',   'Mapping',{'lin', 0.1, 10},                            'Layout',[14,7; 14,7], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),...
            audioPluginParameter('highShelfActive', 'DisplayName','Active',            'Label','',   'Mapping',{'enum', 'Bypassed', 'Active'},              'Layout',[9,8; 9,8],   'DisplayNameLocation', 'none',  'Style', 'checkbox'),...
            audioPluginParameter('highShelfGain',   'DisplayName','Highshelf Gain',    'Label','dB', 'Mapping',{'lin', -12, 12},                            'Layout',[10,8; 10,8], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'),... 
            audioPluginParameter('highShelfFreq',   'DisplayName','Highshelf Freq.',   'Label','Hz', 'Mapping',{'lin', 580, 20000},                         'Layout',[12,8; 12,8], 'DisplayNameLocation', 'below', 'Style', 'rotaryknob'));
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
        maxGain;
        minGain;
        loudnessGain;
        timeConstant;
        loudnessFrames;
        preEqLoudnessMeter;
        postEqLoudnessMeter; 

    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateAutoEqState && plugin.eqMode ~= OperatingMode.manual

                % initialize temporary latent vector values 
                x = 0; y = 0; z = 0;
                
                if plugin.eqMode == OperatingMode.traverse
                    x = plugin.xDim;
                    y = plugin.yDim;
                    z = plugin.zDim;
                elseif plugin.eqMode == OperatingMode.semantic
                    A = plugin.codes{1}{plugin.latentDim}{plugin.firstTerm};
                    B = plugin.codes{1}{plugin.latentDim}{plugin.secondTerm};
                    latentCode = A + (plugin.interpolate * (B - A));
                    
                    if     (plugin.latentDim) == 1
                        x = latentCode(1);
                        y = 0;
                        z = 0;
                    elseif (plugin.latentDim) == 2
                        x = latentCode(1);
                        y = latentCode(2);
                        z = 0;      
                    elseif (plugin.latentDim) == 3
                        x = latentCode(1);
                        y = latentCode(2);
                        z = latentCode(3);
                    end
                end

                % extend the area of the latent space that is reachable
                % this is only applicable in Traverse mode
                if plugin.extend && plugin.eqMode == OperatingMode.traverse
                    x = x * 2;
                    y = y * 2;
                    z = z * 2;
                end

                % pass latent vector through decoder
                if     plugin.latentDim == LatentDim.one
                    x_hat = plugin.net1d.predict([x]);
                elseif plugin.latentDim == LatentDim.two
                    x_hat = plugin.net2d.predict([x y]);
                else
                    x_hat = plugin.net3d.predict([x y z]);
                end

                x_hat = plugin.net1d.denormalize(x_hat);    % denormalize 1x13 param vector
                plugin.storeEqState(x_hat);                 % update autoEqState to match new params
                plugin.fullFilterReset();                   % request coefficient update for all filters
                setUpdateAutoEqState(plugin, false);        % turn the param update flag off (we are done)
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

            % measure pre EQ short-term loudness
            [~,preEqShortTermLoudness,~,~] = plugin.preEqLoudnessMeter(u);

            % Apply biquad filters one-by-one only if active
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
            
            % measure post EQ short-term loudness
            [~,postEqShortTermLoudness,~,~] = plugin.postEqLoudnessMeter(u);

            % determine loudness gain compensation
            if plugin.gainCompensation
                plugin.loudnessFrames = plugin.loudnessFrames + 1;

                % only update the gain compensation after set number of frames
                if plugin.loudnessFrames > plugin.timeConstant
                    plugin.loudnessFrames = 1;

                    % determine the difference between input and output loudness
                    gainComp = mean(preEqShortTermLoudness, 'all') - mean(postEqShortTermLoudness, 'all')

                    % if -Inf in loudness values set unity gain
                    if any(isnan(gainComp))
                        gainComp = 0.0;
                    end
                    % bound gain value for safety (otherwise ouch...)
                    if gainComp > plugin.maxGain
                        gainComp = plugin.maxGain;
                    elseif gainComp < plugin.minGain
                        gainComp = plugin.minGain;
                    end
                    plugin.loudnessGain = gainComp;
                end
                % Apply loudness compensation output gain
                y = 10.^(plugin.loudnessGain/20)*u;
            else
                % Apply user-set output gain
                y = 10.^(plugin.outputGain/20)*u;
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
                 
                z = [plugin.xDim plugin.yDim plugin.zDim];
                dim = str2double(plugin.latentDim);
                plugin.udpsend([b, a, p, z, dim])
            end
        end
        
        function setupImpl(plugin, ~)               
            % Initialize filters based on mode (default is 'Autommatic')
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
            
            % setup loudness compensation
            plugin.loudnessGain         =   0.0;
            plugin.timeConstant         =     4;
            plugin.loudnessFrames       =     1;
            plugin.maxGain              =  12.0;
            plugin.minGain              = -12.0;
            plugin.preEqLoudnessMeter   = loudnessMeter;
            plugin.postEqLoudnessMeter  = loudnessMeter;

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
        function set.strength(plugin, val)
            plugin.strength = val;
            setUpdateAutoEqState(plugin, true);
        end
        function val = get.strength(plugin)
            val = plugin.strength;
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
            setUpdateAutoEqState(plugin, true);
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