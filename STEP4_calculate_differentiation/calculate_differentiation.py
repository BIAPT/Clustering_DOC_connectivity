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

mode = 'dpli' # type of functional connectivity: can be dpli/ wpli / AEC
frequency = 'alpha' # frequency band: can be alpha/ theta/ delta
step = '01' # stepsize: can be '01' or '10'
saveimg = False

INPUT_DIR = "../data/connectivity/{}/{}/step{}/".format(frequency, mode, step)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    "Variance_{}_{}_{}.pdf".format(frequency, mode, step))

info = pd.read_table("../data/DOC_Cluster_information.txt")
P_IDS = info['Patient']

# define features to extract from raw data
Mean = []
# variance over time, space and all on and off
Var_time = []
Var_space = []
Var_spacetime = []

Diff_time = []

for p_id in P_IDS:
    """
    1)    IMPORT DATA
    """
    data_path = INPUT_DIR + "{}_{}_step{}_{}.mat".format(mode, frequency, step, p_id)
    channels_path = INPUT_DIR + "{}_{}_step{}_{}_channels.mat".format(mode, frequency, step, p_id)

    # load .mat and extract data
    data = loadmat(data_path)

    data = data["{}_tofill".format(mode)]
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
    Calculate Mean and Variance
    """
    Mean.append(np.mean(data_2d))

    var_time = np.mean(np.var(data_2d,axis=0))
    Var_time.append(var_time)

    var_space = np.mean(np.var(data_2d,axis=1))
    Var_space.append(var_space)

    Var_spacetime.append(var_time + var_space)

    """
        Calculate Difference
    """
    # calculate the absolute difference between 2 timesteps
    diff = np.abs(np.diff(np.transpose(data_2d)))
    # mean them over time and space:
    Diff_time.append(np.mean(diff))


toplot = pd.DataFrame()
toplot['ID'] = P_IDS
toplot['outcome'] = info['Outcome']
#mean
toplot['Mean'] = Mean
#variance
toplot['Variance time'] = Var_time
toplot['Variance space'] = Var_space
toplot['Variance spacetime'] = Var_spacetime
#Difference
toplot['Differnce time'] = Diff_time

# 0 = Non-recovered
# 1 = CMD
# 2 = Recovered
# 3 = Healthy

for i in toplot.columns[2:]:
    plt.figure()
    sns.boxplot(x='outcome', y = i, data=toplot)
    sns.stripplot(x='outcome', y = i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1, 2, 3], ['Nonreco', 'CMD', 'Reco', 'Healthy'])
    plt.title(i)
    pdf.savefig()
    if saveimg:
        plt.savefig( i + ".jpeg")
    plt.close()


toplot.to_csv("Differentiation_{}_{}_{}.csv".format(frequency, mode, step), index=False, sep=';')

pdf.close()