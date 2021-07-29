"""
written by CHARLOTTE MASCHKE for: DOC Dimensionality reduction 2021
This code contains a list of participants and a list of electrodes which belong to one areas of interest
It will go through the individual wPLI and dPLI files and select the features (functional connectivity for the regions
of interest).

It will output a pickle and csv with the features for the ML pipeline
"""

import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
sys.path.append('../')
from helper_functions import extract_features

def extract_average_features(data, channels, p_id):
    ROI = ['LF_LC', 'LF_LP', 'LF_LO', 'LF_LT',
           'LT_LO', 'LT_LC', 'LT_LP',
           'LP_LO', 'LP_LC',
           'LC_LO',
           'LF_LF', 'LC_LC', 'LP_LP', 'LT_LT', 'LO_LO',

           'RF_RC', 'RF_RP', 'RF_RO', 'RF_RT',
           'RT_RO', 'RT_RC', 'RT_RP',
           'RP_RO', 'RP_RC',
           'RC_RO',
           'RF_RF', 'RC_RC', 'RP_RP', 'RT_RT', 'RO_RO',

           'LF_RC', 'LF_RP', 'LF_RO', 'LF_RT',
           'LT_RO', 'LT_RC', 'LT_RP',
           'LP_RO', 'LP_RC',
           'LC_RO',

           'RF_LC', 'RF_LP', 'RF_LO', 'RF_LT',
           'RT_LO', 'RT_LC', 'RT_LP',
           'RP_LO', 'RP_LC',
           'RC_LO',

           'LF_RF', 'LC_RC', 'LP_RP', 'LT_RT', 'LO_RO']

    # change the channel notation from 'Fp2 to E9'
    if p_id != 'WSAS02':
        channels[np.where(channels == 'Fp2')] = 'E9'
        channels[np.where(channels == 'Fz')] = 'E11'
        channels[np.where(channels == 'F3')] = 'E24'
        channels[np.where(channels == 'F7')] = 'E33'
        channels[np.where(channels == 'Fp1')] = 'E22'
        channels[np.where(channels == 'C3')] = 'E36'
        channels[np.where(channels == 'T7')] = 'E25'
        channels[np.where(channels == 'P3')] = 'E52'
        channels[np.where(channels == 'LM')] = 'E57'
        channels[np.where(channels == 'P7')] = 'E58'
        channels[np.where(channels == 'Pz')] = 'E62'
        channels[np.where(channels == 'O1')] = 'E70'
        channels[np.where(channels == 'Oz')] = 'E75'
        channels[np.where(channels == 'O2')] = 'E83'
        channels[np.where(channels == 'P4')] = 'E92'
        channels[np.where(channels == 'P8')] = 'E96'
        channels[np.where(channels == 'RM')] = 'E100'
        channels[np.where(channels == 'C4')] = 'E104'
        channels[np.where(channels == 'T8')] = 'E108'
        channels[np.where(channels == 'F8')] = 'E122'
        channels[np.where(channels == 'F4')] = 'E124'

    # initialize a dict of regions and referring electrodes
    regions = {}

    regions["LF"] = ['E15', 'E32', 'E22', 'E16', 'E18', 'E23', 'E26', 'E11', 'E19', 'E24', 'E27', 'E33',
                     'E12', 'E20', 'E28', 'E34']
    regions["LC"] = ['E6', 'E13', 'E29', 'E35', 'E7', 'E30', 'E36', 'E41', 'Cz', 'E31', 'E37', 'E42',
                     'E55', 'E54', 'E47', 'E53']
    regions["LP"] = ['E52', 'E51', 'E61', 'E62', 'E60', 'E67', 'E59', 'E72', 'E58', 'E71', 'E66']
    regions["LO"] = ['E75', 'E70', 'E65', 'E64', 'E74', 'E69']
    regions["LT"] = ['E38', 'E44', 'E39', 'E40', 'E46', 'E45', 'E50', 'E57']

    regions["RF"] = ['E15', 'E1', 'E9', 'E16', 'E10', 'E3', 'E2', 'E11', 'E4', 'E124', 'E123', 'E122',
                     'E5', 'E118', 'E117', 'E116']
    regions["RC"] = ['E6', 'E112', 'E111', 'E110', 'E106', 'E105', 'E104', 'E103', 'Cz', 'E80', 'E87',
                     'E93', 'E55', 'E79', 'E98', 'E86']
    regions["RP"] = ['E92', 'E97', 'E78', 'E62', 'E85', 'E77', 'E91', 'E72', 'E96', 'E76', 'E84']
    regions["RO"] = ['E75', 'E83', 'E90', 'E95', 'E82', 'E89']
    regions["RT"] = ['E121', 'E114', 'E115', 'E109', 'E102', 'E108', 'E101', 'E100']

    if p_id == 'WSAS02':
        regions = {}
        regions["LF"] = ['Fp1', 'Fpz', 'AF3', 'AF7', 'Afz', 'Fz', 'F1', 'F3', 'F5', 'F7']
        regions["LC"] = ['Cz', 'C1', 'C3', 'C5', 'FCz', 'FC1', 'FC3', 'FC5']
        regions["LP"] = ['Pz', 'P1','P3', 'P5', 'P7', 'CP1', 'CP3', 'CP5', 'CPz']
        regions["LO"] = ['POz', 'PO3', 'PO7', 'Oz', 'O1']
        regions["LT"] = ['FT7', 'T7', 'TP7']

        regions['RF'] = ['Fp2', 'Fpz', 'AF4', 'AF8', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8']
        regions['RC'] = ['Cz', 'C2', 'C4', 'C6', 'FCz', 'FC2', 'FC4', 'FC6']
        regions['RP'] = ['Pz', 'P2', 'P4', 'P6', 'P8', 'CP2', 'CP4', 'CP6', 'CPz']
        regions['RO'] = ['POz', 'PO4', 'PO8', 'Oz', 'O2']
        regions['RT'] = ['FT8', 'T8', 'TP8']

    missingel = []
    df_feature = pd.DataFrame(np.zeros((1, len(ROI))),columns=ROI)

    for r in ROI:
        r1=r[0:2]
        r2=r[3:5]

        conn, missing = extract_features.extract_single_features(X_step=data.copy(), channels=channels,
                                                              selection_1=regions[r1], selection_2=regions[r2],time = 1)
        missingel.extend(missing)
        df_feature[r] = conn

    return df_feature, missingel
