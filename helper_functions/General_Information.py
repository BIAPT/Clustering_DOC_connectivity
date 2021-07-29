"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to load the data and prepare the variables for the analysis
"""
import pandas as pd
import numpy as np


def get_data(mode, frequency, step):

    data = pd.read_csv("../data/features/74_Part_{}_10_{}_{}.csv".format(mode, step, frequency))

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

    # First load the participant info file
    info = pd.read_table("../data/DOC_Cluster_information.txt".format(mode, step, frequency))

    AllPart = {}
    AllPart["Part"] = list(info["ID"])
    AllPart["Part_heal"] = list(info.query("Healthy==1")["ID"])
    AllPart["Part_nonr"] = list(info.query("NonReco==1")["ID"])
    AllPart["Part_reco"] = list(info.query("Healthy==1")["ID"])
    AllPart["Part_ncmd"] = list(info.query("NCMD==1")["ID"])

    data = get_data(mode, frequency, step)

    # only keep the participants in AllPart[Part]
    data = data[(data['ID'].apply(str)).isin(AllPart["Part"])]

    # extract only the X- values
    X = data.iloc[:, 4:]

    # Assign outcome
    Y_out = np.empty(len(X))
    Y_out[data['ID'].isin(AllPart["Part_nonr"])] = 0
    Y_out[data['ID'].isin(AllPart["Part_ncmd"])] = 1
    Y_out[data['ID'].isin(AllPart["Part_reco"])] = 2
    Y_out[data['ID'].isin(AllPart["Part_heal"])] = 3

    #groupnames = ["Nonr_Patients", "CMD_Patients", "Reco_Patients", "Healthy control"]
    #partnames = ["Part_nonr", "Part_ncmd", "Part_reco", "Part_heal"]

    return AllPart, data, X, Y_out, info

