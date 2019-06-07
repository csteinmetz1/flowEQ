classdef myBasicPlugin < audioPlugin
    % myBasicPlugin is a template basic plugin. Use this template to create
    % your own basic plugin.
    
    properties
        net;
        % Use this section to initialize properties that the end-user interacts
        % with.
    end
    properties (Access = private)
        % Use this section to initialize properties that the end-user does not
        % interact with directly.
    end
    properties (Constant)
        % This section contains instructions to build your audio plugin
        % interface. The end-user uses the interface to adjust tunable
        % parameters. Use audioPluginParameter to associate a public property
        % with a tunable parameter.
    end
    methods
        function plugin = myBasicPlugin
            plugin.net = coder.loadDeepLearningNetwork('ae.mat');
            coder.DeepLearningConfig('mkldnn');
        end
        
        function out = process(plugin, in)
            out = in;
        end
        function reset(plugin)
            % This section contains instructions to reset the plugin between
            % uses or if the environment sample rate changes.
        end
    end
end