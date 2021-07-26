%% Charlotte Maschke November 11 2020
% This script goal is to generate the time-resolved wPLI matrices 
% The matrices will be generated twice: once with overlapping and
% once with non-overlapping windows in the alpha bandwidth.  

% Load the Paricipant IDS: 
info = readtable('/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/DOC_Cluster_participants.txt');
P_IDS = info.Patient_ID;

FREQUENCIES = ["alpha","theta","delta"];

for f = 1:length(FREQUENCIES) 
    FREQUENCY = FREQUENCIES{f};
    % Remote Source Setup
    %
    INPUT_DIR = '/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/DATA_BASELINE_5min_250Hz';
    OUTPUT_DIR = strcat("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/", FREQUENCY, "/wpli/");
    NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/NeuroAlgo";
    addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it


    %% wPLI Parameters:
    p_value = 0.05;
    number_surrogates = 20;

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


    %% loop over all particiopants and stepsizes and calculate wPLI
    for s = 1:length(step_sizes)
        step = step_sizes{s};
        for p = 1:length(P_IDS)
            p_id = P_IDS{p};

            fprintf("Analyzing '%s' wPLI of participant '%s' with stepsize '%s' \n", FREQUENCY, p_id, step);

            participant_in = strcat(p_id, '_Base_5min.set');
            participant_out_path = strcat(OUTPUT_DIR,'step',step,'/wpli_',FREQUENCY,'_step',step,'_',p_id,'.mat');            
            participant_channel_path = strcat(OUTPUT_DIR,'step',step,'/wpli_',FREQUENCY,'_step',step,'_',p_id,'_channels.mat');            

            %% Load data
            recording = load_set(participant_in,INPUT_DIR);
            sampling_rate = recording.sampling_rate;

            if step == "01"
                step_size = 1;
            elseif step == "10"
                step_size = 10;
            end

            % calculate wPLI with NEUROALGO
            %the following part is the same content as in this function: 
            %result_wpli = na_wpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
            % but here, it is parralelized

            %% Getting the configuration
            configuration = get_configuration();
            frequency_band = [low_frequency high_frequency]; % This is in Hz

            %% Setting Result
            result = Result('wpli', recording);
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

            %% initialize empty 3d wpli matrix and fill it in a parallized way
            wpli_tofill = zeros(number_window, recording.number_channels, recording.number_channels);

            parfor win_i = 1:number_window
                disp(strcat("wPLI at window: ",string(win_i)," of ", string(number_window))); 
                segment_data = squeeze(windowed_data(win_i,:,:));
                wpli_tofill(win_i,:,:) = wpli(segment_data, number_surrogates, p_value);
            end

            channels = struct2cell(result.metadata.channels_location);

            %% Average wPLI
            %result.data.avg_wpli = squeeze(mean(result.data.wpli,1));
            save(participant_out_path,'wpli_tofill')
            save(participant_channel_path,'channels')
        end
    end
end
