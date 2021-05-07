import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import scipy.stats as stats
from statsmodels.stats.multicomp import MultiComparison
from scikit_posthocs import posthoc_dunn


wPLI = pd.read_csv("STEP4_perform-k-means-clustering/mode_wpli_Pkmc_K_4_P_7.txt")
dPLI = pd.read_csv("STEP4_perform-k-means-clustering/mode_dpli_Pkmc_K_5_P_7.txt")

pdf = matplotlib.backends.backend_pdf.PdfPages("Merged_wPLI_dPLI.pdf")

if (wPLI.iloc[1]==dPLI.iloc[1]).all():
    print('ID identic')

P_wPLI = np.array(wPLI.iloc[0][1:])
P_dPLI = np.array(dPLI.iloc[0][1:])

cf_matrix = confusion_matrix(P_wPLI,P_dPLI)
cf_matrix2 = (cf_matrix.T / np.sum(cf_matrix,axis=1))*100

cf_matrix3 = cf_matrix.T/ np.sum(np.sum(cf_matrix,axis=1))*100

fig = plt.figure()
sns.heatmap(cf_matrix2,annot=True,cbar_kws={'format': '%.0f%%'})
plt.xlabel("wPLI")
plt.ylabel("dPLI")
plt.show()

pdf.savefig(fig)
plt.savefig("wPLI_dPLI_confusion.jpg")
pdf.close()
plt.show()
nr_wPLI = []
nr_dPLI = []
Part= dPLI.iloc[1,1:].unique()

for p in Part:
    tmp = P_wPLI[np.where(wPLI.iloc[1,1:] == p)]
    nr_wPLI.append(len(np.unique(tmp)))
    tmp = P_dPLI[np.where(dPLI.iloc[1,1:] == p)]
    nr_dPLI.append(len(np.unique(tmp)))

min(nr_dPLI)
np.mean(nr_dPLI)
np.std(nr_dPLI)

min(nr_wPLI)
np.mean(nr_wPLI)
np.std(nr_wPLI)

AllPart = {}
AllPart["Part"] = ['S02', 'S05', 'S07', 'S09', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17',
                            'S18', 'S19', 'S20', 'S22', 'S23',
                            'W03', 'W04', 'W08', 'W22', 'W28', 'W31', 'W34', 'W36',
                            'A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
AllPart["Part_heal"] = ['A03', 'A05', 'A06', 'A07', 'A10', 'A11', 'A12', 'A15', 'A17']
AllPart["Part_nonr"] = ['S05', 'S10', 'S11', 'S12', 'S13', 'S15', 'S16', 'S17', 'S18', 'S22', 'S23', 'W04', 'W36']
AllPart["Part_ncmd"] = ['S19', 'W03', 'W08', 'W28', 'W31', 'W34']
AllPart["Part_reco"] = ['S02', 'S07', 'S09', 'S20', 'W22']

occurence = pd.DataFrame(np.empty((len(AllPart["Part"]), 4)))

# name the columns of the dataframe
names=["group", "ID", "wPLI", "dPLI"]
occurence.columns = names
c=0

for t in AllPart["Part"]:
    occurence.loc[c, 'ID'] = t

    if np.isin(t, AllPart["Part_reco"]):
        occurence.loc[c, 'group'] = "Part_reco"

    elif np.isin(t, AllPart["Part_nonr"]):
        occurence.loc[c, 'group'] = "Part_nonr"

    elif np.isin(t, AllPart["Part_heal"]):
        occurence.loc[c, 'group'] = "Part_heal"

    elif np.isin(t, AllPart["Part_ncmd"]):
        occurence.loc[c, 'group'] = "Part_ncmd"

    tmp = P_wPLI[np.where(wPLI.iloc[1,1:] == t)]
    occurence.loc[c,"wPLI"]= len(np.unique(tmp))
    tmp = P_dPLI[np.where(dPLI.iloc[1,1:] == t)]
    occurence.loc[c,"dPLI"]=len(np.unique(tmp))
    c += 1

occurence_melt = pd.melt(occurence, id_vars=['group'], value_vars=["wPLI","dPLI"],
                         value_name="States", var_name="Condition")

# one with legend
plt.figure()
sns.boxplot(x="Condition", y="States", hue="group",
            data=occurence_melt, palette="muted")
plt.show()

for k_tmp in range(k):
    # statistics:
    N = occurence[str(k_tmp)][(occurence["group"] == groupnames[0])]
    C = occurence[str(k_tmp)][(occurence["group"] == groupnames[1])]
    R = occurence[str(k_tmp)][(occurence["group"] == groupnames[2])]
    H = occurence[str(k_tmp)][(occurence["group"] == groupnames[3])]

    fvalue, pvalue, test = stats.ANOVA_assumptions_test(R, N, C, H)

    tmp = occurence_melt.copy()
    tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
    plt.figure()
    sns.boxplot(x="group", y="occurence", data=tmp,
                whis=[0, 100], width=.6, palette="muted")
    sns.stripplot(x="group", y="occurence", data=tmp,
                  size=4, color=".3", linewidth=0)
    plt.title("occurence state {} \n {} f-value {}, p-value {}\n".format(str(k_tmp), str(fvalue)[0:5], test,
                                                                         str(pvalue)[0:5]))
    plt.yticks(fontsize=14)
    pdf.savefig()
    plt.savefig("{}_k_{}_p_{}_occurence_state{}.jpeg".format(model, k, PC, str(k_tmp)))
    plt.close()

    occurence_tmp = occurence_melt[(occurence_melt["State"] == str(k_tmp))]

    # POSTHOC TEST
    if test == 'kruskal':
        toprint = posthoc_dunn(occurence_tmp, val_col='occurence', group_col='group', p_adjust='bonferroni')
        title = "DUNN Test"

    if test == 'ANOVA':
        # perform multiple pairwise comparison (Tukey's HSD)
        MultiComp = MultiComparison(occurence_tmp['occurence'],
                                    occurence_tmp['group'])
        toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())
        title = "TUKEY Test"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
    plt.title('{} clutser {}'.format(title, str(k_tmp)))
    pdf.savefig(fig, bbox_inches='tight')


all_DOC=P_dPLI[np.isin(wPLI.iloc[1,1:],AllPart["Part_heal"], invert=True)]


len(np.where(all_DOC == '1')[0])/len(all_DOC)*100

len(np.where(P_wPLI == '3')[0])/len(P_wPLI)*100


26.56+2.49+56.22+9.75+4.98
54.25+11.83+4.46+29.46