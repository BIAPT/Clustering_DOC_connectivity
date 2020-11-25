%% Charlotte Maschke November 11 2020
% This script goal is to generate the time-resolved dpli matrices 
% The matrices will be generated twice: once with overlapping and
% once with non-overlapping windows in the alpha bandwidth.  

%FREQUENCY = "alpha";
%FREQUENCY = "theta";
FREQUENCY = "delta";


% Remote Source Setup
%
INPUT_DIR = '/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/BASELINE_5min_250Hz';
OUTPUT_DIR = strcat("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/", FREQUENCY, "/dpli/");
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/NeuroAlgo";
addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it

%just to test
%INPUT_DIR = 'C:/Users/BIAPT/Desktop/DATA_BASELINE_5min_250Hz';
%OUTPUT_DIR = 'C:/Users/BIAPT/Desktop/';

% This list contains all participant IDs
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17',...
    'WSAS02', 'WSAS05', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13','WSAS15','WSAS16','WSAS17',...
    'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22','WSAS23',...
    'AOMW03','AOMW04','AOMW08','AOMW22','AOMW28','AOMW31','AOMW34','AOMW36'};

%% dpli Parameters:
p_value = 0.05;
number_surrogates = 10;


if FREQUENCY == "alpha"
    low_frequency = 8;
    high_frequency = 13;
elseif FREQUENCY == "theta"
    low_frequency = 4;
    high_frequency = 8;
elseif FREQUENCY == "delta"
    low_frequency = 1;
    high_frequency = 4;
end


% Size of the cuts for the data
window_size = 10; % in seconds
%this parameter is set to 1 (overlapping windows)and 10(non-overlapping windows).
step_sizes = ["01", "10"]; % in seconds


%% loop over all particiopants and stepsizes and calculate dpli
for s = 1:length(step_sizes)
    step = step_sizes{s};
    for p = 1:length(P_IDS)
        p_id = P_IDS{p};
        
        fprintf("Analyzing dpli of participant '%s' with stepsize '%s' \n", p_id, step);
        
        participant_in = strcat(p_id, '_Base_5min.set');
        participant_out_path = strcat(OUTPUT_DIR,'step',step,'/dPLI_',FREQUENCY,'_step',step,'_',p_id,'.mat');            
        participant_channel_path = strcat(OUTPUT_DIR,'step',step,'/dPLI_',FREQUENCY,'_step',step,'_',p_id,'_channels.mat');            

        %% Load data
        recording = load_set(participant_in,INPUT_DIR);
        sampling_rate = recording.sampling_rate;
        
        if step == "01"
            step_size = 1;
        elseif step == "10"
            step_size = 10;
        end
        
        % calculate dPLI with NEUROALGO
        %the following part is the same content as in this function: 
        %result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
        % but here, it is parralelized
        
        %% Getting the configuration
        configuration = get_configuration();
        frequency_band = [low_frequency high_frequency]; % This is in Hz

        %% Setting Result
        result = Result('dpli', recording);
        result.parameters.frequency_band = frequency_band;
        result.parameters.window_size = window_size;
        result.parameters.step_size = step_size;
        result.parameters.number_surrogate = number_surrogates;
        result.parameters.p_value = p_value;

    
        %% Filtering the data
        print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
        recording = recording.filter_data(recording.data, frequency_band);

        % Here we init the sliding window slicing 
        recording = recording.init_sliding_window(window_size, step_size);
        number_window = recording.max_number_window;
        %create windows with filtered data 
        windowed_data = create_sliding_window(recording.filt_data, window_size, step_size, sampling_rate);
        
        %% initialize empty 3d dpli matrix and fill it in a parallized way
        dpli_tofill = zeros(number_window, recording.number_channels, recording.number_channels);
        
        parfor win_i = 1:number_window
            disp(strcat("dpli at window: ",string(win_i)," of ", string(number_window))); 
            segment_data = squeeze(windowed_data(win_i,:,:));
            dpli_tofill(win_i,:,:) = dpli(segment_data, number_surrogates, p_value);
        end
        
        
        channels = struct2cell(result.metadata.channels_location);

        %% Average dpli
        %result.data.avg_dpli = squeeze(mean(result.data.dpli,1));
        save(participant_out_path,'dpli_tofill')
        save(participant_channel_path,'channels')
    end
end

