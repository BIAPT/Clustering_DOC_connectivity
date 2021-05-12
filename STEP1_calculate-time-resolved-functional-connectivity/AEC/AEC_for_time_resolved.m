%% Charlotte Maschke November 11 2020
% This script goal is to generate the time-resolved AEC matrices 
% The matrices will be generated twice: once with overlapping and
% once with non-overlapping windows in the alpha bandwidth.  

FREQUENCY = "alpha";
%FREQUENCY = "theta";
%FREQUENCY = "delta";


% Remote Source Setup
%
INPUT_DIR = '/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/BASELINE_5min_250Hz';
OUTPUT_DIR = strcat("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/new_", FREQUENCY, "/AEC/");
mkdir_if_not_exist(OUTPUT_DIR)
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/NeuroAlgo";
addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it

% This list contains all participant IDs
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17',...
    'WSAS02', 'WSAS05', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13','WSAS15','WSAS16','WSAS17',...
    'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22','WSAS23',...
    'AOMW03','AOMW04','AOMW08','AOMW22','AOMW28','AOMW31','AOMW34','AOMW36'};


%% AEC Parameters:
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

% cut_amount: amount of points from hilbert transform to remove from the start and end.
% the goal is to not keep cut_amount from the start and cut_amount from the end.
cut_amount = 10;
% Size of the cuts for the data
window_size = 10; % in seconds
%this parameter is set to 1 (overlapping windows)and 10(non-overlapping windows).
%step_sizes = ["01", "10"]; % in seconds
step_sizes = ["10"]; % in seconds


%% loop over all particiopants and stepsizes and calculate AEC
for s = 1:length(step_sizes)
    step = step_sizes{s};
    for p = 1:length(P_IDS)
        p_id = P_IDS{p};
        
        fprintf("Analyzing AEC of participant '%s' with stepsize '%s' \n", p_id, step);
        
        participant_in = strcat(p_id, '_Base_5min.set');
        participant_out_path = strcat(OUTPUT_DIR,'step',step,'/AEC_',FREQUENCY,'_step',step,'_',p_id,'.mat');            
        participant_channel_path = strcat(OUTPUT_DIR,'step',step,'/AEC_',FREQUENCY,'_step',step,'_',p_id,'_channels.mat');            

        %% Load data
        recording = load_set(participant_in,INPUT_DIR);
        sampling_rate = recording.sampling_rate;
        num_channels = recording.number_channels;
        
        if step == "01"
            step_size = 1;
        elseif step == "10"
            step_size = 10;
        end
                
        %% Getting the configuration
        configuration = get_configuration();
        frequency_band = [low_frequency high_frequency]; % This is in Hz

        %% Setting Result
        result = Result('aec', recording);
        result.parameters.frequency_band = frequency_band;
        result.parameters.window_size = window_size;
        result.parameters.step_size = step_size;
        result.parameters.cut_amount = cut_amount;
    
        %% Filtering the data and initiate window
        print_message(strcat("Filtering Data from ",string(frequency_band(1)), "Hz to ", string(frequency_band(2)), "Hz."),configuration.is_verbose);
        recording = recording.filter_data(recording.data, frequency_band);

        % Here we init the sliding window slicing 
        recording = recording.init_sliding_window(window_size, step_size);
        number_window = recording.max_number_window;
        %create windows with filtered data 
        windowed_data = create_sliding_window(recording.filt_data, window_size, step_size, sampling_rate);
        
        %% initialize empty 3d aec matrix and fill it in a parallized way
        aec_tofill = zeros(number_window, recording.number_channels, recording.number_channels);
        
        parfor win_i = 1:number_window
            disp(strcat("AEC at window: ",string(win_i)," of ", string(number_window))); 
            segment_data = squeeze(windowed_data(win_i,:,:));
            aec_tofill(win_i,:,:) = aec_pairwise_corrected(segment_data, num_channels, cut_amount);
        end
        
        channels = struct2cell(result.metadata.channels_location);

        %% Save AEC
        save(participant_out_path,'aec_tofill')
        save(participant_channel_path,'channels')
    end
end
