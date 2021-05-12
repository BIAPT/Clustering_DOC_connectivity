"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to load the data and prepare the variables for the analysis
"""

import pandas as pd
import numpy as np

def get_data(mode, frequency, step):
    try:
        data = pd.read_csv("../data/features/33_Part_{}_10_{}_{}.csv".format(mode, step, frequency))
    except:
        data = pd.read_csv("data/features/33_Part_{}_10_{}_{}.csv".format(mode, step, frequency))

    if data.columns[0] != 'Name':
        del data[data.columns[0]]

    # only use Baseline
    data = data.query("Phase=='Base'")

    if step == '10':
        # only use timesteps up to 29
        data = data[data['Time'] <= 29]

    if step == '01':
        # only use timesteps up to 299
        data = data[data['Time'] <= 299]
    return data

def load_data(mode, frequency, step):
    # this function takes all analysis parameters and returns the data neede to perform the clustering

    AllPart= {}

    AllPart["Part"] = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                        'S18', 'S19', 'S20', 'S22', 'S23',
                        'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36',
                        'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

    AllPart["Part_heal"] = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

    AllPart["Part_nonr"] = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S22', 'S23', 'W04', 'W36']

    AllPart["Part_ncmd"] = ['S19', 'W03', 'W08', 'W28', 'W31', 'W34']

    AllPart["Part_reco"] = ['S02', 'S07', 'S09', 'S20',  'W22']

    data = get_data(mode, frequency, step)

    # only keep the participants in AllPart[Part]
    data = data[data['ID'].isin(AllPart["Part"])]

    # extract only the X- values
    X = data.iloc[:, 4:]

    # Assign outcome
    Y_out = np.empty(len(X))
    Y_out[data['ID'].isin(AllPart["Part_nonr"])] = 0
    Y_out[data['ID'].isin(AllPart["Part_ncmd"])] = 1
    Y_out[data['ID'].isin(AllPart["Part_reco"])] = 2
    Y_out[data['ID'].isin(AllPart["Part_heal"])] = 3

    groupnames=["Nonr_Patients", "CMD_Patients", "Reco_Patients", "Healthy control"]
    partnames = ["Part_nonr", "Part_ncmd", "Part_reco", "Part_heal"]

    # Additional Info ID
    CRSR_ID=['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                            'S18', 'S19', 'S20', 'S22', 'S23',
                            'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36']

    CRSR_value = [4, 10, 12, 4, 5, 6, 11, 5, 8, 0, 0, 5, 0, 3, 5, 5, 6, 6, 8, 7, 6, 5, 5, 4]

    # Time-Since injury (according to CRSR-ID)
    TSI = [0.3, 9, 0.3, 0.3, 0.3, 1, 2, 8, 21, 0.3, 0.3,
               0.7, 0.3, 0.3, 0.3, 0.3,
               14.5, 5, 6, 0.3, 3.5, 6.5, 3.5, 1]

    # Age (according to CRSR-ID)
    Age = [29, 28, 35, 50, 75, 28, 36, 24, 54, 56, 74,
           53, 40, 42, 18, 56,
           36, 27, 32, 19, 20, 52, 35, 65]


    # Status Chronic/Acute (according to ID above)
    Status = ['A', 'C', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'A', 'A',
              'C', 'A', 'A', 'A', 'A',
              'C', 'C', 'C', 'A', 'C', 'C', 'C', 'C',
              'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H']

    # Diagnosis (according to ID above)
    Diag = ['UWS', 'MCS', 'UWS', 'UWS', 'UWS', 'UWS', 'MCS', 'UWS', 'UWS', 'Coma', 'Coma',
            'UWS', 'CMD', 'UWS', 'UWS', 'UWS',
            'CMD', 'UWS', 'CMD', 'UWS', 'CMD', 'CMD', 'CMD', 'UWS',
            'H','H','H','H','H','H','H','H','H']



    return AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames, Status, Diag, TSI, Age

