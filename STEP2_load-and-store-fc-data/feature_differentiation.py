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
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial import distance


MODES = ['wpli','dpli'] # type of functional connectivity: can be dpli/ wpli / AEC
frequencies = ['theta', 'alpha', 'delta'] # frequency band: can be alpha/ theta/ delta
conditions = ['Base']
STEP = '01' # stepsize: can be '01' or '10'
saveimg = False

#info = pd.read_table("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/DOC_Cluster_participants.txt")
info = pd.read_table("../data/DOC_Cluster_information.txt")
P_IDS = info['ID']

pdf = matplotlib.backends.backend_pdf.PdfPages("Dynamic_all_{}.pdf".format( STEP))

for frequency in frequencies:
    for mode in MODES:

        # define features to extract from raw data
        Means = []
        # variance over time, space and all on and off
        Var_time = []
        Var_space = []
        Difference = []
        Distances = []

        INPUT_DIR = "C:/Users/User/Documents/GitHub/Clustering_DOC_connectivity/data/DOC_FC_results/results/{}/{}/step{}/".format(frequency, mode, STEP)
        AllPart, data, X, Y_out, info = general.load_data(mode, frequency, STEP)

        for p_id in P_IDS:
            """
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
            """
            x_id = data.query("ID=='{}'".format(p_id)).iloc[:, 4:]

            # Mean FC
            x_mean = np.mean(np.mean(x_id))
            Means.append(x_mean)

            # variance over time
            x_var_t = np.sum(np.var(x_id))
            Var_time.append(x_var_t)

            # variance over space
            x_var_s = np.mean(np.var(x_id,axis=1))
            Var_space.append(x_var_s)

            # variance over space
            difference = sum(sum(abs(np.diff(x_id,axis=0))))
            Difference.append(difference)

            # Distance
            n_time = x_id.shape[0]
            dist = np.sum(distance.cdist(x_id, x_id, 'euclidean'))/2
            Distances.append(dist/n_time)

        toplot = pd.DataFrame()
        toplot['ID'] = P_IDS
        toplot['outcome'] = info['Outcome']
        # mean
        toplot['Mean'] = Means
        # variance
        toplot['Variance time'] = Var_time
        toplot['Variance space'] = Var_space
        toplot['Difference'] = Difference
        toplot['Distances'] = Distances

        # 0 = Non-recovered
        # 1 = CMD
        # 2 = Recovered
        # 3 = Healthy

        toplot_DOC = toplot.query("outcome == 0")
        toplot_DOC['CRSR'] = info.query("Outcome == 0")['CRSR'].astype(int)

        for i in toplot.columns[2:]:
            plt.figure()
            sns.boxplot(x='outcome', y=i, data=toplot)
            sns.stripplot(x='outcome', y=i, size=4, color=".3", data=toplot)
            plt.xticks([0, 1, 2, 3], ['Nonreco', 'CMD', 'Reco','Healthy'])
            plt.title("{} for {} in {}".format(i,mode,frequency))
            pdf.savefig()
            if saveimg:
                plt.savefig(i + ".jpeg")
            plt.close()


            # plot CRSR-transition probability
            fig = plt.figure()
            corr = pearsonr(toplot_DOC["CRSR"], toplot_DOC[i])
            sns.regplot(x='CRSR', y= i , data=toplot_DOC)
            plt.title("r = " + str(corr[0]) + "\n p = " + str(corr[1]))
            if saveimg:
                plt.savefig("{}_corr_CRSR.jpeg".format(mode))
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

        print('Mode finished {}  {}'.format(mode, frequency))
pdf.close()