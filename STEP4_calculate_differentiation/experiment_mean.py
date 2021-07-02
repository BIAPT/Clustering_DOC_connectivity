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

mode = 'wpli' # type of functional connectivity: can be dpli/ wpli / AEC
frequency = 'theta' # frequency band: can be alpha/ theta/ delta
step = '10' # stepsize: can be '01' or '10'
saveimg = False

AllPart ={}

AllPart["Part"] = ['WSAS02', 'WSAS05', 'WSAS07', 'WSAS09', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13',
                   'WSAS15', 'WSAS16', 'WSAS17',
                   'WSAS18', 'WSAS19', 'WSAS20', 'WSAS22', 'WSAS23',
                   'AOMW03', 'AOMW04', 'AOMW08', 'AOMW22', 'AOMW28', 'AOMW31', 'AOMW34', 'AOMW36',
                   'MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17']

AllPart["Part_heal"] = ['MDFA03', 'MDFA05', 'MDFA06', 'MDFA07', 'MDFA10', 'MDFA11', 'MDFA12', 'MDFA15', 'MDFA17']

AllPart["Part_nonr"] = ['WSAS05', 'WSAS10', 'WSAS11', 'WSAS12', 'WSAS13', 'WSAS15', 'WSAS16', 'WSAS17', 'WSAS18',
                        'WSAS22', 'WSAS23', 'AOMW04', 'AOMW36']

AllPart["Part_ncmd"] = ['WSAS19', 'AOMW03', 'AOMW08', 'AOMW28', 'AOMW31', 'AOMW34']

AllPart["Part_reco"] = ['WSAS02', 'WSAS07', 'WSAS09', 'WSAS20', 'AOMW22']

IDS = AllPart["Part"]

INPUT_DIR = "../data/connectivity/new_{}/{}/step{}/".format(frequency, mode, step)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    "Test_{}_{}_{}.pdf".format(frequency, mode, step))

# define features to extract from raw data
Mean = []
Vars = []
outcome = []
groups = []

for p_id in IDS:
    if AllPart["Part_nonr"].__contains__(p_id):
        out = 1
        group = "nonr"
    if AllPart["Part_ncmd"].__contains__(p_id):
        out = 2
        group = "ncmd"
    if AllPart["Part_reco"].__contains__(p_id):
        out = 0
        group = "reco"
    if AllPart["Part_heal"].__contains__(p_id):
        out = 3
        group = "heal"

    """
    1)    IMPORT DATA
    """
    if mode == 'AEC':
        # define path for data ON and OFF
        data_path = INPUT_DIR + "AEC_{}_step{}_{}.mat".format(frequency, step, p_id)
        channels_path = INPUT_DIR + "AEC_{}_step{}_{}_channels.mat".format(frequency, step, p_id)
    else:
        # define path for data ON and OFF
        data_path = INPUT_DIR + "{}PLI_{}_step{}_{}.mat".format(mode[0], frequency, step, p_id)
        channels_path = INPUT_DIR + "{}PLI_{}_step{}_{}_channels.mat".format(mode[0], frequency, step, p_id)

    # load .mat and extract data
    data = loadmat(data_path)
    if mode == "AEC":
        data = data["aec_tofill"]
    else:
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
        groups.append(group)
        outcome.append(out)
        Mean.append(np.mean(data_2d[i]))
        Vars.append(np.var(data_2d[i]))


toplot = pd.DataFrame()
toplot['outcome'] = outcome
toplot['group'] = groups
#mean
toplot['Mean'] = Mean
toplot['Variance'] = Vars

# 0 = Non-recovered
# 1 = CMD
# 2 = Recovered
# 3 = Healthy

for i in toplot.columns[2:]:
    plt.figure()
    sns.boxplot(x='outcome', y = i, data=toplot)
    sns.stripplot(x='outcome', y = i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1, 2, 3], ['Reco','NonReco','CMD', 'Healthy'])
    plt.title(i)
    pdf.savefig()
    if saveimg:
        plt.savefig( i + ".jpeg")
    plt.close()

pdf.close()