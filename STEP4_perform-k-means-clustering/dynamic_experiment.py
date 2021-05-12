"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
This code executes the clustering analysis, visualizes all results, performs the statistic and
saves all results into a single PDF.
This code relies on the parameters provided in helper function/generalInformation
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from hmmlearn import hmm
from statsmodels.stats.multicomp import MultiComparison
import helper_functions.General_Information as general
import helper_functions.statistics as stats
from helper_functions import visualize
import helper_functions.process_properties as prop
from scikit_posthocs import posthoc_dunn


# define the analysis parameters

mode = 'dpli' # can be dPLI
frequency = 'alpha' # can be theta and delta
step = '10' #can be '1' (stepsize)
healthy ='Yes' # can be 'No' (analysis with and without healthy participants)
value = 'Prog' # can be 'Diag' (prognostic value and diagnostic value)
palett = "muted" # to have different colors for prognosis and diagnosis
#value = 'Diag'
#palett = "Spectral_r"

OUTPUT_DIR= ""

AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames, Status, Diag, TSI,_ = general.load_data(mode,frequency, step)
areas = X.columns

diff = pd.DataFrame(columns=['ID','outcome','diff'])
diff['ID'] = AllPart['Part']

for i, p in enumerate(AllPart['Part']):
    #diff['outcome'][i] = p
    if AllPart['Part_heal'].__contains__(p) :
        diff['outcome'][i] = 'healthy'
    if AllPart['Part_reco'].__contains__(p) :
        diff['outcome'][i] = 'recovered'
    if AllPart['Part_nonr'].__contains__(p) :
        diff['outcome'][i] = 'non-recovered'
    if AllPart['Part_ncmd'].__contains__(p) :
        diff['outcome'][i] = 'CMD'

    data_part = data.query("ID=='{}'".format(p))[areas]
    part_diff = abs(np.diff(data_part,axis=0))
    part_diff = np.mean(np.mean(part_diff, axis=0))

    diff['diff'][i]=part_diff

sns.boxplot(x="outcome", y="diff",
            data=diff, palette=palett)
sns.scatterplot(x="outcome", y="diff",
            data=diff, palette=palett)
plt.show()