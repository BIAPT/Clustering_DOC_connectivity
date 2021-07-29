import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
from helper_functions import visualize
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import helper_functions.General_Information as general

MODES = ['wpli','dpli'] # type of functional connectivity: can be dpli/ wpli / AEC
frequencies = ['theta', 'alpha', 'delta'] # frequency band: can be alpha/ theta/ delta
conditions = ['Base']
STEP = '10' # stepsize: can be '01' or '10'

#info = pd.read_table("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/DOC_Cluster_participants.txt")
info = pd.read_table("../data/DOC_Cluster_participants.txt")
P_IDS = info['Patient_ID']
info = pd.read_table("../data/DOC_Cluster_information.txt")

for frequency in frequencies:
    for mode in MODES:

        INPUT_DIR = "C:/Users/User/Documents/GitHub/Clustering_DOC_connectivity/data/DOC_FC_results/results/{}/{}/step{}/".format(frequency, mode, STEP)
        AllPart, data, X, Y_out, info = general.load_data(mode, frequency, STEP)

        pdf = matplotlib.backends.backend_pdf.PdfPages(
            "PLOT_{}_{}_{}.pdf".format(frequency, mode, STEP))

        for p_id in P_IDS:

            # define path for data
            data_path = INPUT_DIR + "{}_{}_step{}_{}.mat".format(mode, frequency, STEP, p_id)
            channels_path = INPUT_DIR + "{}_{}_step{}_{}_channels.mat".format(mode, frequency, STEP,p_id)

            # load .mat and extract data
            data_fc = loadmat(data_path)
            data_fc = data_fc["{}_tofill".format(mode)]

            channel = scipy.io.loadmat(channels_path)['channels'][0][0]

            print('Load data comlpete {}'.format(p_id))

            # extract channels from the weird format
            channels = []
            for a in range(0, len(channel)):
                channels.append(channel[a][0])
            channels = np.array(channels)

            data_fc = np.mean(data_fc, axis=0)

            # plot connectivity matrix
            figure = plt.figure(figsize=(10,10))
            plt.imshow(data_fc, cmap='jet')
            plt.title(p_id + '  Baseline')
            if mode == 'wpli':
                plt.clim(0,0.25)
            if mode == 'dpli':
                plt.clim(0.3,0.7)
            if mode == 'aec':
                plt.clim(0,0.3)
            plt.colorbar()
            pdf.savefig(figure)
            plt.close()
            plt.show()

            x_id = data.query("Name=='{}_Base'".format(p_id)).iloc[:, 4:]
            x_id = np.mean(x_id)

            visualize.plot_connectivity(x_id,mode)
            pdf.savefig()
            plt.close()

        print('Mode finished {}  {}'.format(mode, frequency))
        pdf.close()