"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
This code will summarize all individual stability .txt and output
a pdf with the summarized figures.
"""

import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
import pandas as pd

healthy = "Yes"
mode = ["dpli", "wpli"]
step = "10"
#frequency = ["alpha"]
frequency = ["alpha", "theta", "delta"]
Rep = 50
saveimg = True

pdf = matplotlib.backends.backend_pdf.PdfPages("SI_SIS_healthy_{}_10_{}_all_frequency_uncut.pdf".format(healthy, step))

P = [3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate

for f in frequency:
    for m in mode:
        SI = np.empty([Rep, len(K), len(P)])  # Collection of stability index over Repetitions

        for r in range(Rep):
            data = pd.read_csv("../data/new_stability/SI_healthy_{}_{}_10_{}_{}_rep_{}.txt".format(healthy, m, step, f,r+1))
            SI[r,:,:] = data.iloc[:,1:]

        fig,a = plt.subplots(3,figsize=(5,10))
        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2','3','4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')

        im=a[0].imshow(np.transpose(np.mean(SI,axis = 0)),vmin=0.2,vmax=0.5,cmap='viridis')
        a[0].set_title('Stability Index Mean {} {}'.format(m, f))
        a[0].set_xlabel("")
        plt.colorbar(im,ax=a[0])

        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2','3','4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')

        im=a[1].imshow(np.transpose(np.std(SI,axis = 0)),vmin=0,vmax=0.1,cmap='viridis')
        a[1].set_title('Stability Index Std {} {}'.format(m, f))
        a[1].set_xlabel("")
        plt.colorbar(im,ax=a[1])

        SIS = pd.read_csv("../data/new_stability/SIS_healthy_{}_{}_10_{}_{}.txt".format(healthy, m, step, f, r + 1))
        SIS = SIS.iloc[:, 1:]

        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2','3','4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')

        im=a[2].imshow(np.transpose(SIS),vmin=0.1,vmax=0.4, cmap = 'magma_r')
        a[2].set_title('Silhouette score {} {}'.format(m, f))
        a[2].set_xlabel("")
        plt.colorbar(im,ax=a[2])
        pdf.savefig(fig)
        if saveimg:
            plt.savefig("SI_SIS_{}_{}_step_{}.jpeg".format(f, m, step))

        # write a summary csv:
        summary = np.hstack((np.mean(SI, axis = 0),np.std(SI, axis = 0),SIS))

        #pd.DataFrame(summary).to_csv("SIS_healthy_{}_{}_10_{}_{}.csv".format(healthy, m, step, f))

pdf.close()
print('PDF Closed')

