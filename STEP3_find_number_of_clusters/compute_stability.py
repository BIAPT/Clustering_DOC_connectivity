"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
This code will be executed by the generate_jobs_step3.sl and will compute the stability index and silhouette score
for the later analysis.
It will output many txt files (one per repetition and condition)
"""
import sys
sys.path.append('../')
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
(mode, frequency, step, r) = analysis_param.split("_")

# this parameter won't change anything is this part of the analysis

OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/new_stability/"

_, data, X, Y_out, _, _, _, _, _, _, _ = general.load_data(mode, frequency, step)

Y_ID = data['ID']

"""
Stability Index
"""


P = [3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate

SI = stability_measure.compute_stability_index(X, Y_ID, P, K, r)


pd.DataFrame(SI).to_csv(OUTPUT_DIR + "SI_{}_10_{}_{}_rep_{}.txt".format(mode, step, frequency,r))
print('Stability index finished')

"""
Silhouette Score
(only in the first run, no repetition needed)
"""

if r == "1":
    SIS = stability_measure.compute_silhouette_score(X, P, K)
    pd.DataFrame(SIS).to_csv(OUTPUT_DIR+"SIS_{}_10_{}_{}.txt".format(mode, step, frequency))
    print('Silhouette score finished')

print('THE END')

