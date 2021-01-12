"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to load the data and prepare the variables for the analysis
"""

import pandas as pd
import numpy as np

def get_data(mode, frequency, step):
    # import data
    try:
        data = pd.read_csv("../data/33_Part_{}pli_10_{}_{}.csv".format(mode[0], step, frequency))
    except:
        data = pd.read_csv("data/33_Part_{}pli_10_{}_{}.csv".format(mode[0], step, frequency))

    if data.columns[0] != 'Name':
        del data[data.columns[0]]

    # combine both sets
    data = data.query("Phase=='Base'")
    return data

def load_data(mode, frequency, step, healthy, value):
    # this function takes all analysis parameters and returns the data neede to perform the clustering

    AllPart= {}

    if healthy=='Yes':
        AllPart["Part"] = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                            'S18', 'S19', 'S20', 'S22', 'S23',
                            'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36',
                            'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
    elif healthy=='No':
        AllPart["Part"] = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                            'S18', 'S19', 'S20', 'S22', 'S23',
                            'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36']

    if value=="Prog":
        if healthy == 'Yes':
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
        if healthy == 'Yes':
            Y_out[data['ID'].isin(AllPart["Part_heal"])] = 3

        if healthy == 'Yes':
            groupnames=["Nonr_Patients", "CMD_Patients", "Reco_Patients", "Healthy control"]
            partnames = ["Part_nonr", "Part_ncmd", "Part_reco", "Part_heal"]

        elif healthy == 'No':
            groupnames=["Nonr_Patients", "CMD_Patients", "Reco_Patients"]
            partnames = ["Part_nonr", "Part_ncmd", "Part_reco"]

    if value == "Diag":
        if healthy == 'Yes':
            AllPart["Part_heal"] = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']

        AllPart["Part_Coma"] = ['S16', 'S17','S19']

        AllPart["Part_UWS"] = ['S02', 'S07', 'S09', 'S20',  'W22', 'S18', 'S22', 'S23', 'W04', 'W36',
                               'S10', 'S11',  'S13', 'S15', 'W03', 'W28', 'W31', 'W34']

        AllPart["Part_MCS"] = ['W08', 'S05', 'S12']

        data = get_data(mode, frequency, step)

        # only keep the participants in AllPart[Part]
        data = data[data['ID'].isin(AllPart["Part"])]

        # extract only the X- values
        X = data.iloc[:, 4:]

        # Assign outcome
        Y_out = np.empty(len(X))
        Y_out[data['ID'].isin(AllPart["Part_Coma"])] = 0
        Y_out[data['ID'].isin(AllPart["Part_UWS"])] = 1
        Y_out[data['ID'].isin(AllPart["Part_MCS"])] = 2
        if healthy == 'Yes':
            Y_out[data['ID'].isin(AllPart["Part_heal"])] = 3

        if healthy == 'Yes':
            groupnames = ["Coma_Patients", "UWS_Patients", "MCS_Patients", "Healthy control"]
            partnames = ["Part_Coma", "Part_UWS", "Part_MCS", "Part_heal"]
        elif healthy == 'No':
            groupnames = ["Coma_Patients", "UWS_Patients", "MCS_Patients"]
            partnames = ["Part_Coma", "Part_UWS", "Part_MCS"]



    CRSR_ID=['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                            'S18', 'S19', 'S20', 'S22', 'S23',
                            'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36']
    CRSR_value=[4, 10, 12, 4, 5, 6, 11, 5, 8, 0, 0, 5, 0, 3, 5, 5, 6, 6, 8, 7, 6, 5, 5, 4]

    return AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames

