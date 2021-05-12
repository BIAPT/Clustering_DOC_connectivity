import scipy
import numpy as np
import sys
from scipy.io import loadmat
import pandas as pd
import helper_functions.p_entropy as entropy
sys.path.append('../')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer


mode = 'dpli' # type of functional connectivity: can be dpli/ wpli
frequency = 'alpha' # frequency band: can be alpha/ theta/ delta
step = '01' # stepsize: can be '01' or '10'

IDS = ['WSAS05', 'WSAS19','WSAS02', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13', 'WSAS15',
       'WSAS16', 'WSAS17', 'WSAS18', 'WSAS20', 'WSAS22', 'WSAS23',
       'AOMW03', 'AOMW04', 'AOMW08', 'AOMW22', 'AOMW28', 'AOMW31', 'AOMW34', 'AOMW36',
       'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17']

outcome = ['0', '1', '2', '2', '2', '0', '0', '0', '0', '0', '0', '0',
                        '0', '2', '0', '0',
                        '1', '0', '1', '2', '1', '1', '1', '0',
                        '3', '3', '3', '3', '3', '3', '3', '3', '3']


INPUT_DIR = "../data/connectivity/new_{}/{}/step{}/".format(frequency, mode, step)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    "Independent_Cluster_{}_{}_{}.pdf".format(frequency, mode, step))

# define features to extract from raw data
Opt_cluster = []

for p_id in IDS:
    """
    1)    IMPORT DATA
    """
    # define path for data ON and OFF
    data_path = INPUT_DIR + "{}PLI_{}_step{}_{}.mat".format(mode[0], frequency, step, p_id)
    channels_path = INPUT_DIR + "{}PLI_{}_step{}_{}_channels.mat".format(mode[0], frequency, step, p_id)

    # load .mat and extract data
    data = loadmat(data_path)
    data = data["{}pli_tofill".format(mode[0])]
    channel = scipy.io.loadmat(channels_path)['channels'][0][0]
    print('Load data comlpete {}'.format(p_id))

    # extract channels from the weird format
    channels = []
    for a in range(0, len(channel)):
        channels.append(channel[a][0])
    channels = np.array(channels)

    # reduce channels to avoid nonbrain channels
    nonbrain = ["E127", "E126", "E17", "E21", "E14", "E25", "E8", "E128", "E125", "E43", "E120", "E48", "E119", "E49",
                "E113", "E81", "E73", "E88", "E68", "E94", "E63", "E99", "E56", "E107"]
    common_channels = np.intersect1d(channels, nonbrain)
    # only channels which are not in nobrain
    select = np.invert(pd.Series(channels).isin(common_channels))
    channels = channels[select]

    data = data[:, select, :]
    data = data[:, :, select]

    # extract the upper triangle and put the data into the 2D format
    nr_features = len(data[0][np.triu_indices(len(channels),k=1)])
    nr_timesteps = len(data)

    data_2d = np.empty((nr_timesteps, nr_features))

    for i in range(nr_timesteps):
        data_2d[i] = data[i][np.triu_indices(len(channels),k=1)]

    """
    Reduce Data PCA
    """
    pca = PCA(n_components = 10)
    X_reduced = pca.fit_transform(X = data_2d)

    """
    Calculate Silhouette Score
    """
    # Instantiate the clustering model and visualizer
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 12))

    visualizer.fit(data_2d)  # Fit the data to the visualizer
    visualizer.show()  # Finalize and render the figure

    """
    SIL =[]
    distortions = []

    clusters = [2,3,4,5,6,7,8,9,10]
    for k in clusters:
        kmeans = KMeans(n_clusters=k, n_init=100)
        kmeans.fit(X_reduced)  # fit the classifier
        Y_pred = kmeans.predict(X_reduced)
        silhouette = silhouette_score(X_reduced, Y_pred)
        SIL.append(silhouette)
        distortions.append(kmeans.inertia_)

    plt.plot(SIL)
    plt.plot(distortions)
    plt.show()


    optimal1 = clusters[np.where(np.diff(SIL) == min(np.diff(SIL)))[0][0]]
    optimal2 = clusters[np.where(np.diff(distortions) == min(np.diff(distortions)))[0][0]]

    #reduce to 2 dimensions
    pca = PCA(n_components = 2)
    X_pca2 = pca.fit_transform(X = data_2d)

    # k-means with optimal k
    kmeans = KMeans(n_clusters=optimal, n_init=100)
    kmeans.fit(X_reduced)  # fit the classifier
    Y_pred = kmeans.predict(X_reduced)

    for i in range(optimal):
        n = np.where(Y_pred == i)
        plt.scatter(X_pca2[n,0],X_pca2[n,1])

    plt.title(p_id + " n_clusters: " + str(optimal))
    pdf.savefig()
    plt.close()

    Opt_cluster.append(optimal)
    """



toplot = pd.DataFrame()
toplot['ID'] = IDS
toplot['outcome'] = outcome
#mean
toplot['Optimal Cluster Number'] = Opt_cluster

# 0 = Non-recovered
# 1 = CMD
# 2 = Recovered
# 3 = Healthy

for i in toplot.columns[2:]:
    plt.figure()
    sns.boxplot(x='outcome', y=i, data=toplot)
    sns.stripplot(x='outcome', y=i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1, 2, 3], ['NonReco','CMD', 'Reco', 'Healthy'])
    plt.title(i)
    pdf.savefig()
    plt.close()

"""
# plot averaged weights
PC1_weights = pd.DataFrame(PC1_weights)
nonr_weight = PC1_weights[np.array(outcome) == '0']
reco_weight = PC1_weights[np.array(outcome) == '1']
heal_weight = PC1_weights[np.array(outcome) == '2']

# plot average weights normalized
areas = avg_features.columns

#RECOVERED
mean_reco = np.array(np.mean(reco_weight))
mean_reco_norm = (mean_reco - np.min(mean_reco)) / (np.max(mean_reco) - np.min(mean_reco))
features_reco = pd.DataFrame(mean_reco_norm.reshape(-1, len(areas)), columns=areas)
visualize.plot_features(features_reco)
plt.savefig("Feature_Reco.jpeg")
pdf.savefig()
plt.close()

# NON_Recovered
mean_nonr = np.array(np.mean(nonr_weight))
mean_nonr_norm = (mean_nonr - np.min(mean_nonr)) / (np.max(mean_nonr) - np.min(mean_nonr))
features_nonr = pd.DataFrame(mean_nonr_norm.reshape(-1, len(areas)), columns=areas)
visualize.plot_features(features_nonr)
plt.savefig("Feature_Nonreco.jpeg")
pdf.savefig()
plt.close()

# HEALTHY
mean_heal = np.array(np.mean(heal_weight))
mean_heal_norm = (mean_heal - np.min(mean_heal)) / (np.max(mean_heal) - np.min(mean_heal))
features_heal = pd.DataFrame(mean_heal_norm.reshape(-1, len(areas)), columns=areas)
visualize.plot_features(features_heal)
plt.savefig("Feature_Healthy.jpeg")
pdf.savefig()
plt.close()
"""

#toplot.to_csv("Differentiation_{}_{}_{}.csv".format(frequency, mode, step), index=False, sep=';')

pdf.close()