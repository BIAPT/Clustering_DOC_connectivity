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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from yellowbrick.cluster.elbow import kelbow_visualizer
from scipy.spatial import distance


MODES = ['wpli','dpli'] # type of functional connectivity: can be dpli/ wpli / AEC
frequencies = ['theta', 'alpha', 'delta'] # frequency band: can be alpha/ theta/ delta
conditions = ['Base']
STEP = '10' # stepsize: can be '01' or '10'
saveimg = False

#info = pd.read_table("/home/lotte/projects/def-sblain/lotte/Cluster_DOC/data/DOC_Cluster_participants.txt")
info = pd.read_table("../data/DOC_Cluster_information.txt")
P_IDS = info['ID']

pdf = matplotlib.backends.backend_pdf.PdfPages("2Ind_cluster_all_{}.pdf".format( STEP))

for frequency in frequencies:
    for mode in MODES:

        Opt_cluster = []
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
            model = kelbow_visualizer(KMeans(random_state=4), x_id, k=(2, 20),locate_elbow=True, visualize = False)
            #pdf.savefig(model)
            #plt.close()
            optimal = model.elbow_value_

            kmeans = KMeans(n_clusters=optimal, n_init=100)
            kmeans.fit(x_id)  # fit the classifier

            centroids = kmeans.cluster_centers_
            dist = np.sum(distance.cdist(centroids, centroids, 'euclidean'))/2
            Distances.append(dist/optimal)

            """
            SIL = []
            clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            for k in clusters:
                kmeans = KMeans(n_clusters=k, n_init=100)
                kmeans.fit(x_id)  # fit the classifier
                Y_pred = kmeans.predict(x_id)
                silhouette = silhouette_score(x_id, Y_pred)
                SIL.append(silhouette)

            print("k-means finished for {}".format(p_id))

            optimal = clusters[np.where(np.diff(SIL) == min(np.diff(SIL)))[0][0]]
            optimal = optimal +1

            plt.plot(SIL)
            plt.title("{} optimal {} clusters".format(p_id, optimal))
            pdf.savefig()
            plt.close()
            """
            # reduce to 2 dimensions
            #pca = PCA(n_components=2)
            #X_pca2 = pca.fit_transform(X=x_id)
            #plt.scatter(X_pca2[:,0],X_pca2[:,1])
            #plt.show()

            Opt_cluster.append(optimal)

        toplot = pd.DataFrame()
        toplot['ID'] = P_IDS
        toplot['outcome'] = info['Outcome']
        # mean
        toplot['Opt_cluster'] = Opt_cluster
        toplot['Distances'] = Distances

        # 0 = Non-recovered
        # 1 = CMD
        # 2 = Recovered
        # 3 = Healthy

        toplot_DOC = toplot.query("outcome != 3")
        toplot_DOC['CRSR'] = info.query("Outcome != 3")['CRSR'].astype(int)

        for i in toplot.columns[2:]:
            plt.figure()
            sns.violinplot(x='outcome', y=i, data=toplot)
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