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

mode = 'AEC' # type of functional connectivity: can be dpli/ wpli / AEC
frequency = 'alpha' # frequency band: can be alpha/ theta/ delta
step = '01' # stepsize: can be '01' or '10'
n = 5
saveimg = True

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
outcome = np.empty(len(AllPart["Part"]))
group = []

for i,p in enumerate(AllPart["Part"]):
    if AllPart["Part_nonr"].__contains__(p):
        outcome[i] = 1
        group.append("nonr")
    if AllPart["Part_ncmd"].__contains__(p):
        outcome[i] = 2
        group.append("ncmd")
    if AllPart["Part_reco"].__contains__(p):
        outcome[i] = 0
        group.append("reco")
    if AllPart["Part_heal"].__contains__(p):
        outcome[i] = 3
        group.append("heal")



INPUT_DIR = "../data/connectivity/new_{}/{}/step{}/".format(frequency, mode, step)
pdf = matplotlib.backends.backend_pdf.PdfPages(
    "Variance_{}_{}_{}.pdf".format(frequency, mode, step))

# define features to extract from raw data
Mean = []
# variance over time, space and all on and off
Var_time = []
Var_space = []
Var_spacetime = []

P_Ent_time = []
Comp_time = []

Diff_time = []

for p_id in IDS:
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
        data = data["aec_tofill".format(mode[0])]
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

    """
    Calculate Mean and Variance
    """
    Mean.append(np.mean(data_2d))

    Var_time.append(np.mean(np.var(data_2d,axis=0)))

    Var_space.append(np.mean(np.var(data_2d,axis=1)))

    Var_spacetime.append(Var_time + Var_space)

    """
        Calculate Enthropy and Complexity
    """
    ent3 = []
    comp3= []
    """
    ent4 = []
    ent5 = []
    ent6 = []
    ent7 = []
    """

    for i in range(0,nr_features):
        op = entropy.ordinal_patterns(data_2d[:,i], n, 1)
        ent3.append(entropy.p_entropy(op))
        comp3.append(entropy.complexity(op))
        """
        op = entropy.ordinal_patterns(data_2d[:,i], 4, 1)
        ent4.append(entropy.p_entropy(op))
        op = entropy.ordinal_patterns(data_2d[:,i], 5, 1)
        ent5.append(entropy.p_entropy(op))
        op = entropy.ordinal_patterns(data_2d[:,i], 6, 1)
        ent6.append(entropy.p_entropy(op))
        op = entropy.ordinal_patterns(data_2d[:,i], 7, 1)
        ent7.append(entropy.p_entropy(op))
        """

    """
    plt.plot([np.mean(ent3),np.mean(ent4),np.mean(ent5),np.mean(ent6),np.mean(ent7)])
    plt.xticks((1,2,3,4,5),(3,4,5,6,7))

    plt.boxplot([ent3,ent4,ent5])
    plt.xticks((1,2,3),(3,4,5))
    plt.title("Permutation_Entropy  " + p_id )
    plt.show()
    """

    P_Ent_time.append(np.mean(ent3))
    Comp_time.append(np.mean(comp3))


    """
        Calculate Difference
    """
    # calculate the absolute difference between 2 timesteps
    diff = np.abs(np.diff(np.transpose(data_2d)))
    # mean them over time and space:
    Diff_time.append(np.mean(diff))


toplot = pd.DataFrame()
toplot['ID'] = IDS
toplot['outcome'] = outcome
toplot['group'] = group
#mean
toplot['Mean'] = Mean
#variance
toplot['Variance time'] = Var_time
toplot['Variance space'] = Var_space
toplot['Variance spacetime'] = Var_spacetime
#Entropy
toplot['Permutation Entropy time'] = P_Ent_time
#Complexity
toplot['Complexity time'] = Comp_time
#Difference
toplot['Differnce time'] = Diff_time

# 0 = Non-recovered
# 1 = CMD
# 2 = Recovered
# 3 = Healthy

for i in toplot.columns[3:]:
    plt.figure()
    sns.boxplot(x='outcome', y=i, data=toplot)
    sns.stripplot(x='outcome', y=i, size=4, color=".3", data=toplot)
    plt.xticks([0, 1, 2, 3], ['Reco','NonReco','CMD', 'Healthy'])
    plt.title(i)
    pdf.savefig()
    if saveimg:
        plt.savefig( i + "_.jpeg")
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

toplot.to_csv("Differentiation_{}_{}_{}.csv".format(frequency, mode, step), index=False, sep=';')

pdf.close()