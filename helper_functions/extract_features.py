import pandas as pd
import numpy as np


def extract_single_features(X_step, channels, selection_1, selection_2, time):

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
                    #X_step[min(a, b), max(a, b)] = 200  # !! Just activate for test purpose
                    PLI.append(X_step[min(a, b), max(a, b)])
                    #print(channels[a],channels[b],X_step[min(a, b), max(a, b)])

    return np.mean(PLI), missing




