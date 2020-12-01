import sys
sys.path.append('../')
from helper_functions import stability_measure
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import helper_functions.General_Information as general
import joblib
import numpy as np
import pandas as pd

# This will be given by the srun in the bash file
# Get the argument
analysis_param = sys.argv[1]

# Parse the parameters
(mode, frequency, healthy, step) = analysis_param.split("_")

# this parameter won't change anything is this part of the analysis
value = 'Diag'

OUTPUT_DIR = "/home/lotte/projects/def-sblain/lotte/Cluster_DOC/results/stability/"

_, data, X, Y_out, _, _, _, _ = general.load_data(mode, frequency, step, healthy, value)

pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_DIR+"SI_SIS_healthy_{}_{}_10_{}_{}.pdf".format(healthy, mode, step, frequency))

# random data with same characteristics as X
data_random= np.random.normal(np.mean(X), np.std(X), size=X.shape)
Y_ID_random = data['ID']
Y_ID = data['ID']

"""
Stability Index
"""
#P=[3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
#K=[2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate

P=[3, 4]          #number of Principal components to iterate
K=[2, 3]       #number of K-clusters to iterate
Rep=1                                #number of Repetitions (Mean at the end)

SI_M_rand, SI_SD_rand = stability_measure.compute_stability_index(data_random, Y_ID_random, P, K, Rep)
SI_M_Base, SI_SD_Base = stability_measure.compute_stability_index(X, Y_ID, P, K, Rep)

fig,a = plt.subplots(2, 2)
plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10'],
        yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0][0].imshow(np.transpose(SI_M_rand))
a[0][0].set_title('Stability Index Mean: Random')
a[0][0].set_xlabel("")
plt.colorbar(im,ax=a[0,0])

im=a[0][1].imshow(np.transpose(SI_SD_rand))
a[0][1].set_xlabel("")
a[0][1].set_title('Stability Index SD: Random')
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[0,1])

im=a[1][0].imshow(np.transpose(SI_M_Base))
a[1][0].set_title('Stability Index Mean: Baseline')
a[1][0].set_xlabel("")
im.set_clim(0.2,0.4)
plt.colorbar(im,ax=a[1,0])

im=a[1][1].imshow(np.transpose(SI_SD_Base))
a[1][1].set_title('Stability Index SD: Baseline')
a[1][1].set_xlabel("")
im.set_clim(0.01,0.1)
plt.colorbar(im,ax=a[1,1])

fig.set_figheight(17)
fig.set_figwidth(10)
plt.show()

pdf.savefig(fig)

pd.DataFrame(SI_M_Base).to_pickle(OUTPUT_DIR + "SI_MEAN_healthy_{}_{}_10_{}_{}.pdf".format(healthy, mode, step, frequency))

pd.DataFrame(SI_SD_Base).to_pickle(OUTPUT_DIR + "SI_SD_healthy_{}_{}_10_{}_{}.pdf".format(healthy, mode, step, frequency))

print('Stability index finished')

"""
Silhouette Score
"""
P=[3, 4, 5, 6, 7, 8, 9, 10]        #number of Principal components to iterate
K=[2, 3, 4, 5, 6, 7, 8, 9, 10]     #number of K-clusters to iterate

with joblib.parallel_backend('loky'):
    SIS_Rand = stability_measure.compute_silhouette_score(data_random, P, K)
    SIS_Base = stability_measure.compute_silhouette_score(X, P, K)

fig, a = plt.subplots(1, 2)
plt.setp(a, xticks=[0,1,2,3,4,5,6,7,8,9] , xticklabels=['2','3','4','5','6','7','8','9','10'],
        yticks=[0,1,2,3,4,5,6,7,8], yticklabels= ['3','4','5','6','7','8','9','10'],
         xlabel= 'K-Clusters',ylabel='Principle Components')

im=a[0].imshow(np.transpose(SIS_Rand),cmap='viridis_r')
a[0].set_title('Silhouette Score  : Random')
a[0].set_xlabel("")
plt.colorbar(im,ax=a[0])

im=a[1].imshow(np.transpose(SIS_Base),cmap='viridis_r')
a[1].set_title('Silhouette Score : Baseline')
a[1].set_xlabel("")
#im.set_clim(0.1,0.45)
plt.colorbar(im,ax=a[1])

print('Silhouette score finished')

fig.set_figheight(3)
fig.set_figwidth(10)
plt.show()
pdf.savefig(fig)
pdf.close()

print('PDF Closed')


pd.DataFrame(SIS_Base).to_csv(OUTPUT_DIR+"SI_SIS_healthy_{}_{}_10_{}_{}.csv".format(healthy, mode, step, frequency))

print('THE END')

#pd.DataFrame(SIS_Rand).to_pickle('SIS_rand_33part_wPLI_30_10_allfr.pickle')
