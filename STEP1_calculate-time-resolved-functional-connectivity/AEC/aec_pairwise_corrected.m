function [aec] = aec_pairwise_corrected(data, num_channels, cut_amount)
%% AEC PAIRWISE CORRECTED helper function to calculate the pairwise corrected aec
%
% input:
% data: the data segment to calculate pairwise corrected aec on
% num_channels: number of channels
% cut_amount: the amount we need to remove from the hilbert transform
%
% output:
% aec: a num_region*num_region matrix which has the amplitude envelope
% correlation between two regions
% orthogonilization?

    % start by turning matrix to fit to code below
    data = data';
    aec = zeros(num_channels, num_channels);
        
    %% Pairwise leakage correction in window for AEC
    % Loops around all possible ROI pairs
    for region_i = 1:num_channels
        y = data(:, region_i);
        for region_j =  1:num_channels
            
            % Skip the correlation between itself
            if region_i == region_j
               continue 
            end
            
            x = data(:, region_j);
            
            % Leakage Reduction
            beta_leak = pinv(y)*x;
            xc = x - y*beta_leak;            
                       
            ht = hilbert([xc,y]);
            ht = ht(cut_amount+1:end-cut_amount,:);
            ht = bsxfun(@minus,ht,mean(ht,1));
            
            % Envelope
            env = abs(ht);
            c = corr(env);
            
            aec(region_i,region_j) = abs(c(1,2));
        end
    end
    
    % Set the diagonal to 0 
    aec(:,:) = aec(:,:).*~eye(num_channels);
end

