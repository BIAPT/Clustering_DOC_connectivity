"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to compute all dynamic properties of the dynamic time series of
functional connectivity states
"""

import pandas as pd
import numpy as np

def calculate_occurence(AllPart,k,P_kmc,data, partnames, groupnames):
    # this function calculates the occurence rate of one cluster in all participants and groups
    # allpart: List of all groups and which participants belonf to which group (see general information)
    # k: number of Clusters/ actual cluster number
    # data: time resolved functional conn data
    # partnames: List of all participants (see general information)
    # groupames: List of names of groups (see general information)

    # initialize empty dataframe with k columns + ID+ Group
    occurence = pd.DataFrame(np.empty((len(AllPart["Part"]), k+2)))

    # name the columns of the dataframe
    names=["group", "ID"]
    for i in range(k):
        names.append(str(i))
    occurence.columns = names

    # compute the time spent in one phase
    for s in range(k):
        c = 0
        for t in AllPart["Part"]:
            # insert group and ID
            occurence.loc[c, 'ID'] = t

            if np.isin(t, AllPart[partnames[0]]):
                occurence.loc[c, 'group'] = groupnames[0]

            elif np.isin(t, AllPart[partnames[1]]):
                occurence.loc[c, 'group'] = groupnames[1]

            elif np.isin(t, AllPart[partnames[2]]):
                occurence.loc[c, 'group'] = groupnames[2]

            elif np.isin(t, AllPart[partnames[3]]):
                occurence.loc[c, 'group'] = groupnames[3]

            elif np.isin(t, AllPart[partnames[4]]):
                occurence.loc[c, 'group'] = groupnames[4]

            # calculate and insert occurence
            occurence.loc[c,str(s)] = (len(np.where((P_kmc == s) & (data['ID'] == t))[0]))\
                                      /len(np.where(data['ID'] == t)[0])
            c += 1

    return occurence

def calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames):
    # this function calculates the switching probability
    # allpart: List of all groups and which participants belonf to which group (see general information)
    # P_kmc: time series of clusters
    # data: time resolved functional conn data
    # partnames: List of all participants (see general information)
    # groupames: List of names of groups (see general information)

    # create empty dataframe with ID, group, p_switch
    dynamic = pd.DataFrame(np.empty((len(AllPart["Part"]), 3)))
    names = ["ID", "group","p_switch"]
    dynamic.columns=names
    c=0

    for t in AllPart["Part"]:
        # fill in group and participant
        dynamic.loc[c, 'ID'] = t

        if  np.isin(t,AllPart[partnames[0]]):
            dynamic.loc[c, 'group'] = groupnames[0]

        elif np.isin(t,AllPart[partnames[1]]):
            dynamic.loc[c, 'group'] = groupnames[1]

        elif np.isin(t,AllPart[partnames[2]]):
            dynamic.loc[c, 'group'] = groupnames[2]

        elif np.isin(t,AllPart[partnames[3]]):
            dynamic.loc[c, 'group'] = groupnames[3]

        elif np.isin(t,AllPart[partnames[4]]):
            dynamic.loc[c, 'group'] = groupnames[4]

        part_cluster = P_kmc[data['ID'] == t]
        switch = len(np.where(np.diff(part_cluster) != 0)[0])/(len(part_cluster)-1)
        switch = switch*100

        dynamic.loc[c, "p_switch"] = switch
        c += 1
    return dynamic

def calculate_dwell_time(AllPart, P_kmc, data,k, partnames, groupnames):
    # this function calculates the dwell time
    # allpart: List of all groups and which participants belonf to which group (see general information)
    # P_kmc: time series of clusters
    # data: time resolved functional conn data
    # partnames: List of all participants (see general information)
    # groupames: List of names of groups (see general information)

    # initializ empty dataframe with k columns + ID + Group
    dwelltime = pd.DataFrame(np.empty((len(AllPart["Part"]), k+2)))

    # name the columns of the dataframe
    names=["group","ID"]
    for i in range(k):
        names.append(str(i))
    dwelltime.columns=names

    c=0
    for t in AllPart["Part"]:
        # insert ID and Group
        dwelltime.loc[c, 'ID'] = t
        if  np.isin(t,AllPart[partnames[0]]):
            dwelltime.loc[c, 'group'] = groupnames[0]

        elif np.isin(t,AllPart[partnames[1]]):
            dwelltime.loc[c, 'group'] = groupnames[1]

        elif np.isin(t,AllPart[partnames[2]]):
            dwelltime.loc[c, 'group'] = groupnames[2]

        elif np.isin(t,AllPart[partnames[3]]):
            dwelltime.loc[c, 'group'] = groupnames[3]

        elif np.isin(t,AllPart[partnames[4]]):
            dwelltime.loc[c, 'group'] = groupnames[4]

        # extract cluster of this participant
        part_cluster = P_kmc[data['ID'] == t]

        # compute the time spent in one phase
        for s in range(k):
            staytime = []
            tmp=0
            # for all time steps by ignoring the first and last one
            for l in range(1, len(part_cluster)-1):
                # if the first
                if l == 1 and part_cluster[1] == s:
                    tmp += 1
                # if the previous is not the same cluster add one timestep
                elif part_cluster[l] == s and part_cluster[l-1] != s:
                    tmp += 1
                # if the previous is the same add one time step
                elif part_cluster[l] == s and part_cluster[l-1] == s:
                    tmp += 1
                # if the one is different from this before, finish step and continue counting again
                elif part_cluster[l] != s and part_cluster[l-1] == s:
                    if tmp > 0:
                        staytime.append(tmp)
                        tmp = 0
                if l == len(part_cluster)-2 and part_cluster[l] == s:
                    if tmp > 0:
                        staytime.append(tmp)

            if len(staytime) == 0:
                dwelltime.loc[c,str(s)] = 0
            else:
                # averaged staytime from all visits divided by length of recording
                dwelltime.loc[c,str(s)] = np.mean(staytime) #/len(part_cluster)

        c += 1
    return dwelltime

def get_transition_matrix(states,n_states):
    # this function calculates the transition probability matrix
    # states: time series of clusters
    # n_states: maximal number of clusters (k)
    # returns a procentual transition probability matrix

    n = n_states

    #create empty matrix
    M = [[0]*n for _ in range(n)]

    # fill with all transitions
    for (i,j) in zip(states, states[1:]):
        M[i][j] += 1

    #now convert to probabilities:
    M_per = np.array(M)/sum(sum(np.array(M)))

    return M_per
