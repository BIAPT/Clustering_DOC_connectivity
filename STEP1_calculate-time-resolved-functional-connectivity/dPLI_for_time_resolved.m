%% Charlotte Maschke November 11 2020
% This script goal is to generate the time-resolved dPLI matrices 
% The matrices will be generated twice: once with overlapping and
% once with non-overlapping windows in the alpha bandwidth.  

FREQUENCY = "alpha";
%FREQUENCY = "theta";
%FREQUENCY = "delta";


% Remote Source Setup
%
INPUT_DIR = '/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/BASELINE_5min_250Hz';
OUTPUT_DIR = strcat("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/", FREQUENCY, "/dpli/");
NEUROALGO_PATH = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/NeuroAlgo";
addpath(genpath(NEUROALGO_PATH)); % Add NA library to our path so that we can use it

%just to test
%INPUT_DIR = 'C:/Users/BIAPT/Desktop/DATA_BASELINE_5min_250Hz';

% This list contains all participant IDs
P_IDS = {'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17',...
    'WSAS02', 'WSAS05', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13','WSAS15','WSAS16','WSAS17',...
    'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22','WSAS23',...
    'AOMW03','AOMW04','AOMW08','AOMW22','AOMW28','AOMW31','AOMW34','AOMW36'};

%% dPLI Parameters:
p_value = 0.05;
number_surrogate = 10;

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


%% loop over all particiopants and stepsizes and calculate dPLI
for s = 1:length(step_sizes)
    step = step_sizes{s};
    for p = 1:length(P_IDS)
        p_id = P_IDS{p};
        
        fprintf("Analyzing dPLI of participant '%s' with stepsize '%s' \n", p_id, step);
        
        participant_in = strcat(p_id, '_Base_5min.set');
        participant_out_path = strcat(OUTPUT_DIR,'step',step,'/dPLI_',FREQUENCY,'_step',step,'_',p_id,'.mat');            

        %% Load data
        recording = load_set(participant_in,INPUT_DIR);
        
        if step == "01"
            step_size = 1;
        elseif step == "10"
            step_size = 10;
        end
        
        % calculate dPLI with NEUROALGO
        frequency_band = [low_frequency high_frequency]; % This is in Hz
        result_dpli = na_dpli(recording, frequency_band, window_size, step_size, number_surrogate, p_value);
        save(participant_out_path,'result_dpli')

    end
end

