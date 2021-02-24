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
import scipy
import numpy as np
import pandas as pd
from hmmlearn import hmm
from statsmodels.stats.multicomp import MultiComparison
import helper_functions.General_Information as general
import helper_functions.statistics as stats
from helper_functions import visualize
import helper_functions.process_properties as prop
from scikit_posthocs import posthoc_dunn


# define the analysis parameters

model = 'K-means'
mode = 'wpli' # can be dPLI
frequency = 'alpha' # can be theta and delta
step = '10' #can be '1' (stepsize)
healthy ='Yes' # can be 'No' (analysis with and without healthy participants)
#value = 'Prog' # can be 'Diag' (prognostic value and diagnostic value)
#palett = "muted" # to have different colors for prognosis and diagnosis
value = 'Diag'
palett = "Spectral_r"

k = 6
PC = 7

OUTPUT_DIR= ""

AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames, Status, Diag, TSI = general.load_data(mode,frequency, step, healthy, value)

pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_DIR+"{}_{}_{}_P{}_{}_{}_{}_K{}_results_part3.pdf"
                                               .format(frequency, mode, model,str(PC), healthy, step, value, str(k)))

"""
    K_means 7 PC
"""
pca = PCA(n_components=PC)
pca.fit(X)
X7 = pca.transform(X)

if model == "K-means":
    kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
    kmc.fit(X7)
    P_kmc=kmc.predict(X7)

"""
    Switching Prob
"""
dynamic = prop.calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames)
dynamic['group'] = Status

# statistics:
C = dynamic['p_switch'][(dynamic["group"] == 'C')]
A = dynamic['p_switch'][(dynamic["group"] == 'A')]
H = dynamic['p_switch'][(dynamic["group"] == 'H')]

fvalue, pvalue, test = stats.ANOVA_assumptions_test_state(C,A,H)

dynamic['Diag'] = Diag

fig = plt.figure()
sns.boxplot(x='p_switch', y="group", data=dynamic,
            whis=[0, 100], width=.6, palette=palett)
sns.stripplot(x='p_switch', y="group", hue='Diag', data=dynamic,
              size=9, linewidth=0.8, jitter=0.25)

plt.title("switch_prob state chronic_acute_points")
plt.yticks(fontsize=14)
plt.show()

plt.savefig("{}_k_{}_p_{}_switchingprob_TSI_DIAG.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()


fig = plt.figure()
sns.boxplot(x='p_switch', y="group", data=dynamic,
            whis=[0, 100], width=.6, palette=palett)
sns.stripplot(x='p_switch', y="group", data=dynamic,
              size=4,linewidth=0.8, jitter=0.25,color = 'black')

plt.title("switch_prob state chronic_acute_points")
plt.yticks(fontsize=14)
plt.savefig("{}_k_{}_p_{}_switchingprob_TSI_DIAh.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()
plt.show()

# POST_HOC TEST
if test == 'kruskal':
    toprint = posthoc_dunn(dynamic, val_col='p_switch', group_col='group', p_adjust='bonferroni')
    title = "DUNN Test"

if test == 'ANOVA':
    # perform multiple pairwise comparison (Tukey's HSD)
    MultiComp = MultiComparison(dynamic['p_switch'],
                                dynamic['group'])
    toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())
    title = "TUKEY Test"

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('tight')
ax.axis('off')
the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
plt.title('{} '.format(title))
pdf.savefig(fig, bbox_inches='tight')

"""
run and plot the correlation 
"""
dyn_DOC = dynamic[np.isin(dynamic['ID'], CRSR_ID)]
dyn_DOC['CRSR']= CRSR_value
dyn_DOC['TSI']= TSI


# use the function regplot to make a scatterplot
fig = plt.figure()
corr = scipy.stats.pearsonr(dyn_DOC["CRSR"], dyn_DOC["p_switch"])
sns.regplot(x='CRSR', y='p_switch', data=dyn_DOC)
plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
plt.xlim(-0.2,12.2)
plt.savefig("{}_corr_CRSR.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

# use the function regplot to make a scatterplot
fig = plt.figure()
corr = scipy.stats.pearsonr(dyn_DOC["TSI"], dyn_DOC["p_switch"])
sns.regplot(x='TSI', y='p_switch', data=dyn_DOC)
plt.title("r = "+str(corr[0])+ "\n p = "+ str(corr[1]))
plt.xlim(0,21.5)
plt.savefig("{}_corr_TSI.jpeg".format(mode, k, PC))
pdf.savefig(fig, bbox_inches='tight')
plt.close()

pdf.close()
print('finished all extra analyisiisisis')

print('THE END')
