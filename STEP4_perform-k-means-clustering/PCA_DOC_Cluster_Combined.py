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

# number of Clusters/ Phases to explore
KS = [6]
PCs = [7]

OUTPUT_DIR= ""

AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames, Status, Diag, TSI = general.load_data(mode,frequency, step, healthy, value)

for PC in PCs:

    pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_DIR+"{}_{}_{}_P{}_{}_{}_{}_K{}_test.pdf"
                                                   .format(frequency, mode, model,str(PC), healthy, step, value, str(KS)))

    """
        PCA - all_participants
    """
    pca = PCA(n_components=3)
    pca.fit(X)
    X3 = pca.transform(X)

    visualize.plot_pca_results(pdf, X3, Y_out, groupnames, healthy)
    print("#######  PCA Completed ")

    """
        K_means 7 PC
    """
    pca = PCA(n_components=PC)
    pca.fit(X)
    X7 = pca.transform(X)

    for k in KS:
        if model == "K-means":
            kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
            kmc.fit(X7)
            P_kmc=kmc.predict(X7)

        if model == "HMM":
            # create HMM
            scores = []
            models = []

            for i in range(10):
                model_tmp = hmm.GaussianHMM(n_components=k, covariance_type="full", n_iter=100)
                model_tmp.fit(X7)
                scores.append(model_tmp.score(X7))
                models.append(model_tmp)

            # select model with highest score
            model_best = models[np.where(scores == max(scores))[0][0]]

            if model_best.monitor_.converged == False:
                print('Model not Converged Error')
                break

            P_kmc = model_best.predict(X7)

        visualize.plot_explained_variance(pdf,pca)
        visualize.plot_clustered_pca(pdf,X3,Y_out,P_kmc,k,groupnames, healthy)

        for part in AllPart["Part"]:

            part_cluster = P_kmc[data['ID']==part]
            visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

        print("#######  {} with k={} started".format(model,k))

        for group in partnames:
            fig, ax = plt.subplots(len(AllPart["{}".format(group)]), 1, figsize=(5, 40))
            fig.suptitle('{}; \n {}_Clusters_wholeBrain_alpha'.format(group, k), size=16)
            c = 0
            for part in AllPart["{}".format(group)]:
                #part=AllPart["{}".format(group)][part]
                part_cluster = P_kmc[data['ID'] == part]

                piedata = []
                clusternames = []
                for i in range(k):
                    piedata.append(list(part_cluster).count(i))
                    clusternames.append('c ' + str(i))

                ax[c].pie(piedata, labels=clusternames, startangle=90)
                ax[c].set_title('Participant: '+part)
                c +=1
            pdf.savefig(fig)
            plt.close()

        #plot all part dynamics
        all_dyn = np.zeros((len(AllPart["Part"]),35))
        #part_order=np.hstack((AllPart["Part_heal"],AllPart["Part_reco"],AllPart["Part_ncmd"],AllPart["Part_nonr"]))
        if healthy == "Yes":
            part_order=np.hstack((AllPart[partnames[0]],AllPart[partnames[1]],AllPart[partnames[2]],AllPart[partnames[3]]))
        if healthy == "No":
            part_order=np.hstack((AllPart[partnames[0]],AllPart[partnames[1]],AllPart[partnames[2]]))

        for i, part in enumerate(part_order):
            part_cluster = P_kmc[data['ID'] == part]
            all_dyn[i,:len(part_cluster)]=part_cluster+1


        my_cmap = plt.get_cmap('viridis', k)
        my_cmap.set_under('lightgrey')
        plt.imshow(all_dyn,cmap=my_cmap, vmin =0.001, vmax = k+.5, alpha=0.7)
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 35, 2))
        ax.set_yticks(np.arange(.5, len(part_order), 1))
        ax.set_xticklabels(np.arange(0, 35, 2))
        ax.set_yticklabels(part_order)
        plt.colorbar()
        plt.clim(0.5,k + 0.5)

        pdf.savefig()
        plt.savefig("{}_alldynamics.jpeg".format(model))
        plt.close()

        """
        Cluster Occurence
        """
        occurence = prop.calculate_occurence(AllPart,k,P_kmc,data,partnames, groupnames)
        occurence_melt=pd.melt(occurence, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                               value_name="occurence", var_name="State")

        # one with legend
        plt.figure()
        sns.boxplot(x="State", y="occurence", hue="group",
                         data=occurence_melt,palette=palett)
        plt.title('Cluster_Occurence_K-Means')
        pdf.savefig()
        plt.savefig("{}_k_{}_p_{}_occurence.jpeg".format(model, k, PC))
        plt.close()

        # one without legend
        plt.figure()
        sns.color_palette("colorblind")
        sns.boxplot(x="State", y="occurence", hue="group",
                         data=occurence_melt,palette=palett)
        plt.legend([],[], frameon=False)
        plt.title('Cluster_Occurence_K-Means')
        pdf.savefig()
        plt.savefig("{}_k_{}_p_{}_occurence2.jpeg".format(model, k, PC))
        plt.close()

        for k_tmp in range(k):
            # statistics:
            N = occurence[str(k_tmp)][(occurence["group"] == groupnames[0])]
            C = occurence[str(k_tmp)][(occurence["group"] == groupnames[1])]
            R = occurence[str(k_tmp)][(occurence["group"] == groupnames[2])]
            H = occurence[str(k_tmp)][(occurence["group"] == groupnames[3])]
            if value == 'Diag':
                NC = occurence[str(k_tmp)][(occurence["group"] == groupnames[4])]
                fvalue, pvalue, test = stats.ANOVA_assumptions_test_Diag(R, N, C, H, NC)
            else:
                fvalue, pvalue, test = stats.ANOVA_assumptions_test(R, N, C, H)

            tmp = occurence_melt.copy()
            tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
            plt.figure()
            sns.boxplot(x="group", y="occurence", data=tmp,
                        whis=[0, 100], width=.6,palette = palett)
            sns.stripplot(x="group", y="occurence", data=tmp,
                          size=4, color=".3", linewidth=0)
            plt.title("occurence state {} \n {} f-value {}, p-value {}\n".format(str(k_tmp),str(fvalue)[0:5],test, str(pvalue)[0:5]))
            plt.yticks(fontsize=14)
            pdf.savefig()
            plt.savefig("{}_k_{}_p_{}_occurence_state{}.jpeg".format(model,k,PC,str(k_tmp)))
            plt.close()

            occurence_tmp = occurence_melt[(occurence_melt["State"] == str(k_tmp))]

            #POSTHOC TEST
            if test =='kruskal':
                toprint = posthoc_dunn(occurence_tmp, val_col='occurence', group_col='group', p_adjust='bonferroni')
                title = "DUNN Test"

            if test =='ANOVA':
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

        """
        Dwell Time
        """
        dwelltime = prop.calculate_dwell_time(AllPart, P_kmc, data, k, partnames, groupnames)
        dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                                     value_name="dwell_time", var_name="State")
        plt.figure()
        sns.boxplot(x="State", y="dwell_time", hue="group",
                         data=dwelltime_melt, palette = palett)
        plt.title('Dwell_Time_K-Means')
        plt.yticks(fontsize=14)
        pdf.savefig()
        plt.savefig("{}_k_{}_p_{}_dwelltime.jpeg".format(model, k, PC))
        plt.close()

        plt.figure()
        sns.boxplot(x="State", y="dwell_time", hue="group",
                         data=dwelltime_melt, palette = palett)
        plt.title('Dwell_Time_K-Means')
        plt.yticks(fontsize=14)
        plt.legend([], [], frameon=False)
        pdf.savefig()
        plt.savefig("{}_k_{}_p_{}_dwelltime2.jpeg".format(model, k, PC))
        plt.close()

        for k_tmp in range(k):
            # statistics:
            # statistics:
            N = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[0])]
            C = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[1])]
            R = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[2])]
            H = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[3])]
            if value == 'Diag':
                NC = dwelltime[str(k_tmp)][(dwelltime["group"] == groupnames[4])]
                fvalue, pvalue, test = stats.ANOVA_assumptions_test_Diag(R, N, C, H, NC)
            else:
                fvalue, pvalue, test = stats.ANOVA_assumptions_test(R, N, C, H)

            tmp = dwelltime_melt.copy()
            tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
            plt.figure()
            sns.boxplot(x="group", y="dwell_time", data=tmp,
                        whis=[0, 100], width=.6, palette = palett)
            sns.stripplot(x="group", y="dwell_time", data=tmp,
                          size=4, color=".3", linewidth=0)
            plt.title("dwell_time state {} \n {} f-value {}, p-value {}\n".format(str(k_tmp), str(fvalue)[0:5], test, str(pvalue)[0:5]))
            plt.yticks(fontsize=14)
            pdf.savefig()
            plt.savefig("{}_k_{}_p_{}_dwelltime_state{}.jpeg".format(model, k, PC, str(k_tmp)))
            plt.close()

            # POST_HOC TEST
            dwelltime_tmp = dwelltime_melt[(dwelltime_melt["State"] == str(k_tmp))]

            if test == 'kruskal':
                toprint = posthoc_dunn(dwelltime_tmp, val_col='dwell_time', group_col='group', p_adjust='bonferroni')
                title = "DUNN Test"

            if test == 'ANOVA':
                # perform multiple pairwise comparison (Tukey's HSD)
                MultiComp = MultiComparison(dwelltime_tmp['dwell_time'],
                                            dwelltime_tmp['group'])
                toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())
                title = "TUKEY Test"

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
            plt.title('{} clutser {}'.format(title, str(k_tmp)))
            pdf.savefig(fig, bbox_inches='tight')

        """
            Switching Prob
        """
        dynamic = prop.calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames)

        # statistics:
        N = dynamic['p_switch'][(dynamic["group"] == groupnames[0])]
        C = dynamic['p_switch'][(dynamic["group"] == groupnames[1])]
        R = dynamic['p_switch'][(dynamic["group"] == groupnames[2])]
        H = dynamic['p_switch'][(dynamic["group"] == groupnames[3])]
        if value == 'Diag':
            NC = dynamic['p_switch'][(dynamic["group"] == groupnames[4])]
            fvalue, pvalue, test = stats.ANOVA_assumptions_test_Diag(R, N, C, H, NC)
        else:
            fvalue, pvalue, test = stats.ANOVA_assumptions_test(R, N, C, H)


        plt.figure()
        sns.boxplot(x='p_switch', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette = palett)
        sns.stripplot(x='p_switch', y="group", data=dynamic,
                      size=4, color=".3", linewidth=0)

        plt.title("switch_prob state \n {} f-value {}, p-value {}\n".format(test, str(fvalue)[0:5], str(pvalue)[0:5]))
        plt.yticks(fontsize=14)
        pdf.savefig()
        plt.savefig("{}_k_{}_p_{}_switchingprob.jpeg".format(model, k, PC))

        plt.close()

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

        dynamic['Status'] = Status
        fig = plt.figure()
        sns.boxplot(x='p_switch', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette=palett)
        sns.stripplot(x='p_switch', y="group", hue='Status', data=dynamic,
                      size=6, linewidth=0.2, palette=sns.color_palette("bright", 3))

        plt.title("switch_prob state chronic_acute_points")
        plt.yticks(fontsize=14)
        plt.savefig("{}_k_{}_p_{}_switchingprob_chronic_acute_points.jpeg".format(model, k, PC))
        pdf.savefig(fig, bbox_inches='tight')
        plt.show()

        """
            Centroid and Average
        """
        if model == "K-means":
            centroids = pd.DataFrame(pca.inverse_transform(kmc.cluster_centers_))
            centroids.columns = X.columns

        # create average connectivity image
        for s in range(k):
            X_conn = np.mean(X.iloc[np.where(P_kmc == s)[0]])
            visualize.plot_connectivity(X_conn, mode)
            pdf.savefig()
            plt.close()

            if model == "K-means":
                visualize.plot_connectivity(centroids.iloc[s], mode)
                pdf.savefig()
                plt.savefig("{}_cluster_{}".format(model, s))
                plt.close()

        """
            Phase Transition
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
        visualize.plot_group_averaged_TPM(AllPart,P_kmc,Y_out,k,pdf,data,partnames,groupnames, healthy)

        pd.DataFrame(np.vstack((P_kmc,data['ID']))).to_csv("mode_{}_Pkmc_K_{}_P_{}.txt".format(mode, k, PC))

    pdf.close()
    print('finished all with PC {}'.format(PC))

print('THE END')

# library & dataset
import seaborn as sns

dyn_DOC = dynamic[np.isin(dynamic['ID'], CRSR_ID)]
dyn_DOC['CRSR']= CRSR_value
#dyn_DOC['TSI']= TSI

#dyn_DOC = dyn_DOC[dyn_DOC['Status']!='C']

# use the function regplot to make a scatterplot
sns.regplot(x=dyn_DOC["CRSR"], y=dyn_DOC["p_switch"])
plt.show()

# use the function regplot to make a scatterplot
#sns.regplot(x=dyn_DOC["TSI"], y=dyn_DOC["p_switch"])
#plt.show()
