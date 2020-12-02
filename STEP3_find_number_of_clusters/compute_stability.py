import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import joblib
import numpy as np
import pandas as pd
import os

scriptpath = "."
sys.path.append(os.path.abspath(scriptpath))

from helper_functions import stability_measure
import helper_functions.General_Information as general

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

# Parse the parameters
(mode, frequency, healthy, step, r) = analysis_param.split("_")

# this parameter won't change anything is this part of the analysis
value = 'Diag'

OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/stability/"

_, data, X, Y_out, _, _, _, _ = general.load_data(mode, frequency, step, healthy, value)

#random data with same characteristics as X
#data_random = np.random.normal(np.mean(X), np.std(X), size=X.shape)
#Y_ID_random = data['ID']

Y_ID = data['ID']

"""
Stability Index
"""
P = [3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate

SI = stability_measure.compute_stability_index(X, Y_ID, P, K, r)


pd.DataFrame(SI).to_csv(OUTPUT_DIR + "SI_healthy_{}_{}_10_{}_{}_rep_{}.txt".format(healthy, mode, step, frequency,r))
print('Stability index finished')

"""
Silhouette Score
(only in the first run, no repetition needed)
"""
if r == 1:
    SIS = stability_measure.compute_silhouette_score(X, P, K)
    pd.DataFrame(SIS).to_csv(OUTPUT_DIR+"SIS_healthy_{}_{}_10_{}_{}.txt".format(healthy, mode, step, frequency))


print('THE END')

