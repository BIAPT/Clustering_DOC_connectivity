"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP2 to extract the features
It only extracts the upper half of a FC matrix
"""

import numpy as np

def extract_single_features(X_step, channels, selection_1, selection_2, time):
    '''
    :param X_step: FC data on a specific time t.
            matrix size(nr_electrodes x nr_electrodes), must be diagonal
    :param channels: list of channels in this dataset (corresponds to order of FC matirx col and rownames)
    :param selection_1: channels from the area of interest (source area)
    :param selection_2: channels from the area of interest (target area)
    :param time: recent time step
    :return: averaged FC between source and target region
             and list of missing electrodes in this participant
    '''

    missing = []
    selected_1 = []
    selected_2 = []

    # Step1: check which electrodes from the source area are in this dataset
    for i in range(0, len(selection_1)):
        try:
            selected_1.append(np.where(channels == selection_1[i])[0][0])
        except:
            if time == 1:
                missing.append(str(selection_1[i]))

    # Step2: check which electrodes from the target area are in this dataset
    for i in range(0, len(selection_2)):
        try:
            selected_2.append(np.where(channels == selection_2[i])[0][0])
        except:
            if time == 1:
                missing.append(str(selection_2[i]))

    # initialize empty FC list
    FC = []

    # loop over all possible electrode- electrode combinations in the
    # two selected areas and fill the list with corresponding values
    # keep track of completed values to not list redundant connections
    done = []

    for a in selected_1:
        for b in selected_2:
            # only take pairs of different channels
            if a != b:
                done.append(str(a)+'_'+str(b))
                # if this electrode pair is not in the data already
                # this is done here because we just take the upper triangle from the matrix
                if done.__contains__(str(b)+'_'+str(a)) == False:
                    # with the min max, we select only the upper triangel values
                    FC.append(X_step[min(a, b), max(a, b)])

    return np.mean(FC), missing




