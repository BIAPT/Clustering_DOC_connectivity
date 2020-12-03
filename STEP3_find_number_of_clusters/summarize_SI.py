import sys
sys.path.append('../')
from matplotlib import pyplot as plt
import matplotlib.backends.backend_pdf
import joblib
import numpy as np
import pandas as pd
import os

healthy = "Yes"
mode = ["dpli", "wpli"]
step = "10"
frequency = ["alpha", "theta", "delta"]
Rep = 50

pdf = matplotlib.backends.backend_pdf.PdfPages("SI_SIS_healthy_{}_10_{}_all_frequency.pdf".format(healthy, step))

P = [3, 4, 5, 6, 7, 8, 9, 10]          #number of Principal components to iterate
K = [2, 3, 4, 5, 6, 7, 8, 9, 10]       #number of K-clusters to iterate

for m in mode:
    for f in frequency:

        SI = np.empty([Rep, len(K), len(P)])  # Collection of stability index over Repetitions

        for r in range(Rep):
            data = pd.read_csv("data/stability/SI_healthy_{}_{}_10_{}_{}_rep_{}.txt".format(healthy, m, step, f,r+1))
            SI[r,:,:] = data.iloc[:len(K),1:len(P)+1]

        fig,a = plt.subplots(3)
        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')

        im=a[0].imshow(np.transpose(np.mean(SI,axis = 0)))
        a[0].set_title('Stability Index Mean {} {}'.format(m, f))
        a[0].set_xlabel("")
        plt.colorbar(im,ax=a[0])

        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')

        im=a[1].imshow(np.transpose(np.std(SI,axis = 0)))
        a[1].set_title('Stability Index Std {} {}'.format(m, f))
        a[1].set_xlabel("")
        plt.colorbar(im,ax=a[1])

        SIS = pd.read_csv("data/stability/SIS_healthy_{}_{}_10_{}_{}.txt".format(healthy, m, step, f, r + 1))

        plt.setp(a, xticks=[0, 1, 2, 3, 4, 5, 6, 7, 8], xticklabels=['2', '3', '4', '5', '6', '7', '8', '9', '10'],
                yticks=[0, 1, 2, 3, 4, 5, 6, 7], yticklabels= ['3', '4', '5', '6', '7', '8', '9', '10'],
                 xlabel= 'K-Clusters',ylabel='Principle Components')
        im=a[1].imshow(np.transpose(np.std(SI,axis = 0)))
        a[1].set_title('Silhouette score {} {}'.format(m, f))
        a[1].set_xlabel("")
        plt.colorbar(im,ax=a[1])
        plt.show()




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

