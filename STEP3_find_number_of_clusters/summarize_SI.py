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

#pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_DIR+"SI_SIS_healthy_{}_{}_10_{}_{}.pdf".format(healthy, mode, step, frequency))

#random data with same characteristics as X
#data_random= np.random.normal(np.mean(X), np.std(X), size=X.shape)
#Y_ID_random = data['ID']


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

