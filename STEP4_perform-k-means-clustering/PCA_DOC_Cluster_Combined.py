"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
This code executes the clustering analysis, visualizes all results, performs the statistic and
saves all results into a single PDF.
This code relies on the parameters provided in helper function/generalInformation
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import helper_functions.General_Information as general
import helper_functions.statistics as stats
from helper_functions import visualize
import helper_functions.process_properties as prop
from scipy.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests


"""
Analysis Parameters
"""
mode = 'AEC' # type of functional connectivity: can be dpli/ wpli
frequency = 'alpha' # frequency band: can be alpha/ theta/ delta
step = '01' # stepsize: can be '01'
palett = "muted" # color palette
saveimg = True # if you want to save all images as seperate files

# number of Clusters/ Phases to explore
k = 6   # numbser of k-clustes
PC = 8   # number of PC principal components

# load the data
AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames, Status, Diag, TSI, Age = \
    general.load_data(mode,frequency, step)

# set up an empty pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("{}_{}_P{}_{}_K{}.pdf"
                                               .format(frequency, mode,str(PC), step, str(k)))

"""
    1) Run PCA with 3 PCS for visualization only
"""
pca = PCA(n_components=3)
pca.fit(X)
X3 = pca.transform(X)

visualize.plot_pca_results(pdf, X3, Y_out, groupnames)
print("#######  visualization PCA Completed ")

"""
   2) Run K_means on PC principal components and k Clusters
"""
# Fit PCA and transform X
pca = PCA(n_components=PC)
pca.fit(X)
X_reduced = pca.transform(X)

# fit K-means on reduced X and k clusters
kmc=KMeans(n_clusters=k, random_state=0, n_init=1000)
kmc.fit(X_reduced)
P_kmc=kmc.predict(X_reduced)
print("#######  K-means with k={} done".format(k))

# visualize the explained variance and the clustering result in a 3D space
visualize.plot_explained_variance(pdf,pca)
visualize.plot_clustered_pca(pdf,X3,Y_out,P_kmc,k,groupnames)

# visualize the distribution and pie-chart for every single participant
for part in AllPart["Part"]:
    part_cluster = P_kmc[data['ID']==part]
    visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

# visualize the time-series for all participants
visualize.plot_all_timeseries(pdf, AllPart,P_kmc, data, step, saveimg )


"""
    3) Calculate Cluster Occurence
"""
occurence = prop.calculate_occurence(AllPart, k, P_kmc, data, partnames, groupnames)
occurence_melt = pd.melt(occurence, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                       value_name="occurence", var_name="State")

# Plot the results
plt.figure()
sns.boxplot(x="State", y="occurence", hue="group",
                 data=occurence_melt,palette=palett)
plt.title('Cluster_Occurence_K-Means')
pdf.savefig()
plt.legend([],[], frameon=False)
pdf.savefig()
if saveimg:
    plt.savefig("k_{}_p_{}_occurence.jpeg".format(k, PC))
plt.close()

occurence.to_csv("occurence_state_{}.csv".format(mode), index=False, sep=';')

for k_tmp in range(k):
    # split the groups for statistics:
    N = occurence[str(k_tmp)][(occurence["group"] == groupnames[0])]
    C = occurence[str(k_tmp)][(occurence["group"] == groupnames[1])]
    R = occurence[str(k_tmp)][(occurence["group"] == groupnames[2])]
    H = occurence[str(k_tmp)][(occurence["group"] == groupnames[3])]


    H_N, H_N_p = stats.run_comparison(H,N)
    H_R, H_R_p = stats.run_comparison(H,R)
    H_C, H_C_p = stats.run_comparison(H,C)

    p_values = [H_N_p, H_R_p, H_C_p]
    p_adjusted = multipletests(p_values, alpha=0.05, method='bonferroni')[1]

    # only for plotting reasons set nan values to 99
    p_adjusted[np.isnan(p_adjusted)] = 99.999

    tmp = occurence_melt.copy()
    tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
    plt.figure()
    sns.boxplot(x="group", y="occurence", data=tmp,
                whis=[0, 100], width=.6,palette = palett)
    sns.stripplot(x="group", y="occurence", data=tmp,
                  size=4, color=".3", linewidth=0)
    plt.title("occurence state {}  ".format(k_tmp)+
              "H_N: {} p = {:.3f} \n".format(H_N,p_adjusted[0])+
              "H_R: {} p = {:.3f}  ".format(H_R,p_adjusted[1])+
              "H_C: {} p = {:.3f}  ".format(H_C,p_adjusted[2]))

    plt.yticks(fontsize=14)
    pdf.savefig()
    if saveimg:
        plt.savefig("k_{}_p_{}_occurence_state{}.jpeg".format(k,PC,str(k_tmp)))
    plt.close()

"""
    4) Calculate Dwell Time
"""
dwelltime = prop.calculate_dwell_time(AllPart, P_kmc, data, k, partnames, groupnames)
dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                             value_name="dwell_time", var_name="State")

# plot dwell-time
plt.figure()
sns.boxplot(x="State", y="dwell_time", hue="group",
                 data=dwelltime_melt, palette = palett)
plt.title('Dwell_Time_K-Means')
plt.yticks(fontsize=14)
pdf.savefig()
plt.legend([], [], frameon=False)
pdf.savefig()
if saveimg:
    plt.savefig("k_{}_p_{}_dwelltime.jpeg".format(k, PC))
plt.close()

dwelltime.to_csv("dwelltime_state_{}.csv".format(mode), index=False, sep=';')

for k_tmp in range(k):
    # statistics:
    N = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[0])]
    C = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[1])]
    R = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[2])]
    H = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[3])]

    H_N, H_N_p = stats.run_comparison(H,N)
    H_R, H_R_p = stats.run_comparison(H,R)
    H_C, H_C_p = stats.run_comparison(H,C)

    p_values = [H_N_p, H_R_p, H_C_p]
    p_adjusted = multipletests(p_values, alpha=0.05, method='bonferroni')[1]
    # only for plotting reasons set nan values to 99
    p_adjusted[np.isnan(p_adjusted)] = 99.999


    tmp = dwelltime_melt.copy()
    tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
    plt.figure()
    sns.boxplot(x="group", y="dwell_time", data=tmp,
                whis=[0, 100], width=.6, palette = palett)
    sns.stripplot(x="group", y="dwell_time", data=tmp,
                  size=4, color=".3", linewidth=0)
    plt.title("dwell_time state {}  ".format(k_tmp)+
              "H_N: {} p = {:.3f}\n".format(H_N,p_adjusted[0])+
              "H_R: {} p = {:.3f}  ".format(H_R,p_adjusted[1])+
              "H_C: {} p = {:.3f}  ".format(H_C,p_adjusted[2]))
    plt.yticks(fontsize=14)
    pdf.savefig()
    if saveimg:
        plt.savefig("k_{}_p_{}_dwelltime_state{}.jpeg".format(k, PC, str(k_tmp)))
    plt.close()

"""
    5) Calculate Switching Prob
"""
dynamic = prop.calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames)

dynamic.to_csv("dynamic_{}.csv".format(mode), index=False, sep=';')

# statistics:
N = dynamic['p_switch'][(dynamic["group"] == groupnames[0])]
C = dynamic['p_switch'][(dynamic["group"] == groupnames[1])]
R = dynamic['p_switch'][(dynamic["group"] == groupnames[2])]
H = dynamic['p_switch'][(dynamic["group"] == groupnames[3])]

H_N, H_N_p = stats.run_comparison(H, N)
H_R, H_R_p = stats.run_comparison(H, R)
H_C, H_C_p = stats.run_comparison(H, C)

p_values = [H_N_p, H_R_p, H_C_p]
p_adjusted = multipletests(p_values, alpha=0.05, method='bonferroni')[1]
# only for plotting reasons set nan values to 99
p_adjusted[np.isnan(p_adjusted)] = 99.999

plt.figure()
sns.boxplot(x='p_switch', y="group", data=dynamic,
            whis=[0, 100], width=.6, palette = palett)
sns.stripplot(x='p_switch', y="group", data=dynamic,
              size=4, color=".3", linewidth=0)
plt.title("switch_prob     " +
          "H_N: {} p = {:.3f}\n".format(H_N,p_adjusted[0])+
          "H_R: {} p = {:.3f}  ".format(H_R,p_adjusted[1])+
          "H_C: {} p = {:.3f}  ".format(H_C,p_adjusted[2]))

plt.yticks(fontsize=14)
pdf.savefig()
if saveimg:
    plt.savefig("k_{}_p_{}_switchingprob.jpeg".format(k, PC))
plt.close()

# plot with status
dynamic['Status'] = Status
dynamic['Diag'] = Diag

fig = plt.figure()
sns.boxplot(x='p_switch', y="Status", data=dynamic,
            whis=[0, 100], width=.6, palette="Spectral_r")
if saveimg:
    plt.savefig("k_{}_p_{}_switchingprob_state.jpeg".format(k, PC))
sns.stripplot(x='p_switch', y="Status", hue='Diag', data=dynamic,
              size=6, linewidth=0.2, palette=sns.color_palette("bright", 5))

plt.title("switch_prob state chronic_acute_points")
plt.yticks(fontsize=14)
if saveimg:
    plt.savefig("k_{}_p_{}_switchingprob_chronic_acute_points.jpeg".format(k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

"""
    6) Calculate and plot Centroid and Average
"""
centroids = pd.DataFrame(pca.inverse_transform(kmc.cluster_centers_))
centroids.columns = X.columns

# create average connectivity image
for s in range(k):
    X_conn = np.mean(X.iloc[np.where(P_kmc == s)[0]])
    visualize.plot_connectivity(X_conn, mode)
    pdf.savefig()
    plt.close()

    visualize.plot_connectivity(centroids.iloc[s], mode)
    pdf.savefig()
    if saveimg:
        plt.savefig("cluster_{}".format(s))
    plt.close()

"""
    7) Calculate Phase Transition
"""
# individual Phase Transition
for group in partnames:
    fig, ax = plt.subplots(len(AllPart["{}".format(group)]),1, figsize=(5, 50))
    fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
    c = 0
    for part in AllPart["{}".format(group)]:
        part_cluster = P_kmc[data['ID'] == part]
        TPM_part = prop.get_transition_matrix(part_cluster, k)
        sns.heatmap(TPM_part, annot=True, cbar=False, ax=ax[c], fmt='.1g')
        ax[c].set_title('Participant: '+part)
        c +=1
    pdf.savefig(fig)
    plt.close()

# group averaged Phase Transition
visualize.plot_group_averaged_TPM(AllPart,P_kmc,k,pdf,data,partnames,groupnames)

pd.DataFrame(np.vstack((P_kmc,data['ID']))).to_csv("mode_{}_Pkmc_K_{}_P_{}.txt".format(mode, k, PC))

"""
    8) Run and plot the Correlation analysis
"""

dyn_DOC = dynamic[np.isin(dynamic['ID'], CRSR_ID)]
dyn_DOC.loc[:, ('CRSR')]= CRSR_value
dyn_DOC['TSI']= TSI
dyn_DOC['Age']= Age

# plot CRSR-transition probability
fig = plt.figure()
corr = pearsonr(dyn_DOC["CRSR"], dyn_DOC["p_switch"])
sns.regplot(x='CRSR', y='p_switch', data=dyn_DOC)
plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
plt.xlim(-0.2,12.2)
if saveimg:
    plt.savefig("{}_corr_CRSR.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

# plot TSI-transition probability
fig = plt.figure()
corr = pearsonr(dyn_DOC["TSI"], dyn_DOC["p_switch"])
sns.regplot(x='TSI', y='p_switch', data=dyn_DOC)
plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
plt.xlim(0,21.5)
if saveimg:
    plt.savefig("{}_corr_TSI.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

# plot Age-transition probability
fig = plt.figure()
corr = pearsonr(dyn_DOC["Age"], dyn_DOC["p_switch"])
sns.regplot(x='Age', y='p_switch', data=dyn_DOC)
plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
plt.xlim(20,80)
if saveimg:
    plt.savefig("{}_corr_Age.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

pdf.close()

print('THE END')

