import matplotlib
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.backends.backend_pdf
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy
import numpy as np
import pandas as pd
import sys
import os
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import helper_functions.General_Information as general
from helper_functions import visualize
import helper_functions.process_properties as prop
import fpdf

model = 'K-means'
#model = 'HMM'
mode = 'wpli'
frequency = 'alpha'
step = '10'
healthy ='Yes'
value = 'Prog'

# number of Clusters/ Phases to explore
KS = [6]
PCs = [7]

OUTPUT_DIR= ""

AllPart, data, X, Y_out, CRSR_ID, CRSR_value, groupnames, partnames = general.load_data(mode,
                                                                                        frequency, step, healthy, value)

for PC in PCs:

    pdf = matplotlib.backends.backend_pdf.PdfPages(OUTPUT_DIR+"{}_{}_{}_P{}_{}_{}_{}.pdf"
                                                   .format(frequency, mode, model,str(PC), healthy, step, value))

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
        kmc=KMeans(n_clusters=k, random_state=0,n_init=1000)
        kmc.fit(X7)
        P_kmc=kmc.predict(X7)

        visualize.plot_explained_variance(pdf,pca)
        visualize.plot_clustered_pca(pdf,X3,Y_out,P_kmc,k,groupnames, healthy)

        for part in AllPart["Part"]:

            part_cluster = P_kmc[data['ID']==part]
            visualize.plot_pie_and_distribution(pdf, part, part_cluster, k)

        print("#######  K_means with k={} started".format(k))

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
        part_order=np.hstack((AllPart["Part_heal"],AllPart["Part_reco"],AllPart["Part_ncmd"],AllPart["Part_nonr"]))

        for i, part in enumerate(part_order):
            part_cluster = P_kmc[data['ID'] == part]
            all_dyn[i,:len(part_cluster)]=part_cluster+1

        my_cmap = plt.get_cmap('PiYG_r', k)
        my_cmap.set_under('lightgrey')
        plt.imshow(all_dyn,cmap=my_cmap, vmin =0.001, vmax = k+.5)
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 35, 2))
        ax.set_yticks(np.arange(.5, 33, 1))
        ax.set_xticklabels(np.arange(0, 35, 2))
        ax.set_yticklabels(part_order)
        plt.colorbar()
        plt.clim(0.5,k + 0.5)
        pdf.savefig()
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
                         data=occurence_melt,palette="muted")
        plt.title('Cluster_Occurence_K-Means')
        pdf.savefig()
        plt.close()

        # one without legend
        plt.figure()
        sns.color_palette("colorblind")
        sns.boxplot(x="State", y="occurence", hue="group",
                         data=occurence_melt,palette="muted")
        plt.legend([],[], frameon=False)
        plt.title('Cluster_Occurence_K-Means')
        pdf.savefig()
        plt.close()

        for k_tmp in range(k):
            # statistics:
            R = occurence[str(k_tmp)][(occurence["group"] == 'Reco_Patients')]
            N = occurence[str(k_tmp)][(occurence["group"] == 'Nonr_Patients')]
            C = occurence[str(k_tmp)][(occurence["group"] == 'CMD_Patients')]
            H = occurence[str(k_tmp)][(occurence["group"] == 'Healthy control')]

            # stats f_oneway functions takes the groups as input and returns F and P-value
            fvalue, pvalue = stats.f_oneway(R, N, C, H)

            tmp = occurence_melt.copy()
            tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
            plt.figure()
            sns.boxplot(x="group", y="occurence", data=tmp,
                        whis=[0, 100], width=.6,palette="muted")
            sns.stripplot(x="group", y="occurence", data=tmp,
                          size=4, color=".3", linewidth=0)
            plt.title("occurence state {} \n f-value {}, p-value {}".format(str(k_tmp),str(fvalue)[0:5],str(pvalue)[0:5]))
            plt.ylim(0,1)
            pdf.savefig()
            plt.close()

            # perform multiple pairwise comparison (Tukey's HSD)
            occurence_tmp = occurence_melt[(occurence_melt["State"] == str(k_tmp))]
            MultiComp = MultiComparison(occurence_tmp['occurence'],
                                        occurence_tmp['group'])
            toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
            plt.title('Tukeys HSD clutser {}'.format(str(k_tmp)))
            pdf.savefig(fig, bbox_inches='tight')

        """
        Dwell Time
        """
        dwelltime = prop.calculate_dwell_time(AllPart, P_kmc, data, k, partnames, groupnames)
        dwelltime_melt = pd.melt(dwelltime, id_vars=['group'], value_vars=[str(i) for i in np.arange(k)],
                                     value_name="dwell_time", var_name="State")
        plt.figure()
        sns.boxplot(x="State", y="dwell_time", hue="group",
                         data=dwelltime_melt, palette="muted")
        plt.title('Dwell_Time_K-Means')
        pdf.savefig()
        plt.close()

        plt.figure()
        sns.boxplot(x="State", y="dwell_time", hue="group",
                         data=dwelltime_melt, palette="muted")
        plt.title('Dwell_Time_K-Means')
        plt.legend([], [], frameon=False)
        pdf.savefig()
        plt.close()

        for k_tmp in range(k):
            # statistics:
            R = dwelltime[str(k_tmp)][(dwelltime["group"] == 'Reco_Patients')]
            N = dwelltime[str(k_tmp)][(dwelltime["group"] == 'Nonr_Patients')]
            C = dwelltime[str(k_tmp)][(dwelltime["group"] == 'CMD_Patients')]
            H = dwelltime[str(k_tmp)][(dwelltime["group"] == 'Healthy control')]

            # stats f_oneway functions takes the groups as input and returns F and P-value
            fvalue, pvalue = stats.f_oneway(R, N, C, H)

            tmp = dwelltime_melt.copy()
            tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
            plt.figure()
            sns.boxplot(x="group", y="dwell_time", data=tmp,
                        whis=[0, 100], width=.6, palette="muted")
            sns.stripplot(x="group", y="dwell_time", data=tmp,
                          size=4, color=".3", linewidth=0)
            plt.title("dwell_time state {} \n f-value {}, p-value {}".format(str(k_tmp), str(fvalue)[0:5], str(pvalue)[0:5]))
            pdf.savefig()
            plt.close()

            # perform multiple pairwise comparison (Tukey's HSD)
            dwelltime_tmp = dwelltime_melt[(dwelltime_melt["State"] == str(k_tmp))]
            MultiComp = MultiComparison(dwelltime_tmp['dwell_time'],
                                        dwelltime_tmp['group'])
            toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.axis('tight')
            ax.axis('off')
            the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
            plt.title('Tukeys HSD clutser {}'.format(str(k_tmp)))
            pdf.savefig(fig, bbox_inches='tight')

            fig = plt.figure()
            tmp = dwelltime_melt.copy()
            tmp = tmp.iloc[np.where(tmp["State"] == str(k_tmp))]
            plt.figure()
            sns.boxplot(x="group", y="dwell_time", data=tmp,
                        whis=[0, 100], width=.6,palette="muted")
            sns.stripplot(x="group", y="dwell_time", data=tmp,
                          size=4, color=".3", linewidth=0)
            plt.title("dwelltime state {}".format(str(k_tmp)))
            pdf.savefig(fig)
            plt.close()

        """
            Switching Prob
        """
        dynamic = prop.calculate_dynamics(AllPart, P_kmc, data, partnames, groupnames)

        # statistics:
        R = dynamic['p_switch'][(dynamic["group"] == 'Reco_Patients')]
        N = dynamic['p_switch'][(dynamic["group"] == 'Nonr_Patients')]
        C = dynamic['p_switch'][(dynamic["group"] == 'CMD_Patients')]
        H = dynamic['p_switch'][(dynamic["group"] == 'Healthy control')]

        # stats f_oneway functions takes the groups as input and returns F and P-value
        fvalue, pvalue = stats.f_oneway(R, N, C, H)
        plt.figure()
        sns.boxplot(x='p_switch', y="group", data=dynamic,
                    whis=[0, 100], width=.6, palette="muted")
        sns.stripplot(x='p_switch', y="group", data=dynamic,
                      size=4, color=".3", linewidth=0)

        plt.title("switch_prob state \n f-value {}, p-value {}".format(str(fvalue)[0:5], str(pvalue)[0:5]))
        pdf.savefig()
        plt.close()

        # perform multiple pairwise comparison (Tukey's HSD)
        MultiComp = MultiComparison(dynamic['p_switch'],
                                    dynamic['group'])
        toprint = pd.DataFrame(MultiComp.tukeyhsd().summary())

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('tight')
        ax.axis('off')
        the_table = ax.table(cellText=toprint.values, colLabels=toprint.columns, loc='center')
        plt.title('Tukeys HSD clutser')
        pdf.savefig(fig, bbox_inches='tight')


        """
            Centroid and Average
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
