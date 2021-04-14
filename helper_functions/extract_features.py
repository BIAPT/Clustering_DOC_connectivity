"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP2 to extract the features
It only extracts the upper half of a FC matrix
"""

import numpy as np

def extract_single_features(X_step, channels, selection_1, selection_2, time,mode):
    #X_step: FC data
    # channels: all channels in this dataset (col and rownames)
    # selection1 and 2: channels from the area of interest
    #time: timestep

    missing = []
    selected_1 = []
    selected_2 = []
    for i in range(0, len(selection_1)):
        try:
            selected_1.append(np.where(channels == selection_1[i])[0][0])
        except:
            if time == 1:
                missing.append(str(selection_1[i]))


    for i in range(0, len(selection_2)):
        try:
            selected_2.append(np.where(channels == selection_2[i])[0][0])
        except:
            if time == 1:
                missing.append(str(selection_2[i]))

    PLI = []
    done = []

    for a in selected_1:
        for b in selected_2:
            if a != b:
                done.append(str(a)+'_'+str(b))
                # if the inverse connectivity is not in the data already:
                if done.__contains__(str(b)+'_'+str(a)) == False:
                    PLI.append(X_step[min(a, b), max(a, b)])

    #if mode == 'dpli':
        # return the absolute strength of direction, but not the direction itself
        #PLI = abs(np.array(PLI) - 0.5)

    return np.mean(PLI), missing




