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
        operator       =      '+';
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
            audioPluginParameter('decoderMode',    'DisplayName','EQ Mode',           'Label','',   'Mapping',{'enum', 'xyz', 'semantic'}),...    
            audioPluginParameter('firstTerm',      'DisplayName','Descriptor',        'Label','',   'Mapping',{'enum', 'warm', 'bright'}),...
            audioPluginParameter('operator',       'DisplayName','Operator',          'Label','',   'Mapping',{'enum', '+', '-', 'x', '÷', 'interpolate'}),...
            audioPluginParameter('secondTerm',     'DisplayName','Descriptor',        'Label','',   'Mapping',{'enum', 'warm', 'bright'}),...
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
        % Paramter update flags 
        updateEqState = false;  % update EQ parameters
    end
    %----------------------------------------------------------------------
    % public methods
    %----------------------------------------------------------------------
    methods(Access = protected)
        function y = stepImpl(plugin,u)
            % -------------------- Parameter Updates ----------------------
            if plugin.updateEqState
                plugin.eqState.lowShelfGain = randi([-12,12]);
                setUpdateEqState(plugin, false);
            end
            % -------------------- Audio Processing -----------------------
            % Apply input gain
            dry = 10.^(plugin.inputGain/20)*u;
                        
            % mix wet and dry signals together
            y = dry;
        end

        function setupImpl(plugin, ~)    
            % initialize supported sample rate converters
            
        end

        function resetImpl(plugin)
        
        end
    end    
    %----------------------------------------------------------------------
    % private methods
    %----------------------------------------------------------------------
    methods (Access = private)
        %----------------- Parameter Change Flags -------------------------
        function setUpdateEqState(plugin,flag)
            plugin.updateEqState = flag;
        end
    end
    %----------------------------------------------------------------------
    % setter and getter methods
    %----------------------------------------------------------------------
    methods 
        function set.latentDim(plugin, val)
            plugin.latentDim = val;
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
    end  
end