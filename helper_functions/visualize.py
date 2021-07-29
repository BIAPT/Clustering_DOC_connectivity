"""
written by CHARLOTTE MASCHKE: DOC Clustering 2020/2021
this code is used by STEP3 to visualize all steps of the analysis
"""

import matplotlib
import numpy as np
import pandas as pd
from nilearn import plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import helper_functions.process_properties as prop

def plot_connectivity(X_conn, mode):
    regions = ['LF','LC','LP','LO','LT','RF','RC','RP','RO','RT']
    try:
        coords = np.loadtxt('../helper_functions/coordinates.txt')
    except:
        coords = np.loadtxt('helper_functions/coordinates.txt')

    for t in range(len(X_conn)):
        tmp = X_conn
        conn_tmp = pd.DataFrame(np.zeros((len(regions), len(regions))))
        conn_tmp.columns = regions
        conn_tmp.index = regions

        for i in regions:
            for a in regions:
                try:
                    conn_tmp.loc[i, a] = tmp[i + '_' + a]
                except:
                    conn_tmp.loc[i, a] = tmp[a + '_' + i]

    conn_matrix = np.array(conn_tmp)

    if mode == 'wpli':
        #colormap = matplotlib.cm.get_cmap('OrRd')
        colormap = matplotlib.cm.get_cmap('jet')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.25)
        fig = plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0, edge_vmax=0.25,
                                       edge_cmap=colormap, colorbar=True, edge_threshold=None,
                                       node_color=colormap(norm(conn_matrix.diagonal())),
                                       display_mode='lzr')
    if mode == 'aec':
        colormap = matplotlib.cm.get_cmap('OrRd')
        norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)
        fig = plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0, edge_vmax=0.3,
                                       edge_cmap=colormap, colorbar=True, edge_threshold=None,
                                       node_color=colormap(norm(conn_matrix.diagonal())),
                                       display_mode='lzr')

    if mode == 'dpli':
        colormap = matplotlib.cm.get_cmap('jet')
        norm = matplotlib.colors.Normalize(vmin=0.3, vmax=0.7)
        fig = plotting.plot_connectome(conn_matrix, node_coords=coords, edge_vmin=0.3, edge_vmax=0.7,
                                       edge_cmap=colormap, colorbar=True, edge_threshold=None,
                                       node_color=colormap(norm(conn_matrix.diagonal())),
                                       display_mode='lzr')
    return fig

def plot_pca_results(pdf,X3,Y_out,groupnames):
    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    n = np.where(Y_out == 0)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='blue', label=groupnames[0])
    n = np.where(Y_out == 1)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='green', label=groupnames[1])
    n = np.where(Y_out == 2)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='red', label=groupnames[2])
    n = np.where(Y_out == 3)
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2], color='orange', label=groupnames[3])
    plt.title('PCA_allPart_wholeBrain_alpha')
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('PCA_allPart_wholeBrain_alpha', size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 0)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='blue', label=groupnames[0])
    n = np.where(Y_out == 1)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='green', label=groupnames[1])
    n = np.where(Y_out == 2)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='red', label=groupnames[2])
    n = np.where(Y_out == 3)
    ax[0].scatter(X3[n, 0], X3[n, 1], color='orange', label=groupnames[3])

    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out == 0)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='blue', label=groupnames[0])
    n = np.where(Y_out == 1)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='green', label=groupnames[1])
    n = np.where(Y_out == 2)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='red', label=groupnames[2])
    n = np.where(Y_out == 3)
    ax[1].scatter(X3[n, 1], X3[n, 2], color='orange', label=groupnames[3])

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out == 0)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='blue', label=groupnames[0])
    n = np.where(Y_out == 1)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='green', label=groupnames[1])
    n = np.where(Y_out == 2)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='red', label=groupnames[2])
    n = np.where(Y_out == 3)
    ax[2].scatter(X3[n, 0], X3[n, 2], color='orange', label=groupnames[3])

    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()


def plot_clustered_pca(pdf,X3,Y_out,P,k,groupnames):
    # visualize in 3D
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig)
    n = np.where(Y_out==0)[0]
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='o',c=P[n],label=groupnames[0])
    n= np.where(Y_out==1)[0]
    ax.scatter(X3[n, 0], X3[n, 1],X3[n, 2],marker='x',c=P[n],label=groupnames[1])
    n = np.where(Y_out == 2)[0]
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2],marker='.', c=P[n], label=groupnames[2])
    n = np.where(Y_out == 3)[0]
    ax.scatter(X3[n, 0], X3[n, 1], X3[n, 2],marker='v', c=P[n], label=groupnames[3])
    plt.title('{}_Clusters_allPart_wholeBrain_alpha'.format(str(k)))
    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle('{}_Clusters_allPart_wholeBrain_alpha'.format(str(k)), size=16)

    ax[0].set_title('PC 0 and 1')
    n = np.where(Y_out == 0)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='o', c=P[n], label=groupnames[0])
    n = np.where(Y_out == 1)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='x', c=P[n], label=groupnames[1])
    n = np.where(Y_out == 2)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='.', c=P[n], label=groupnames[2])
    n = np.where(Y_out == 3)[0]
    ax[0].scatter(X3[n, 0], X3[n, 1], marker='v', c=P[n], label=groupnames[3])

    ax[1].set_title('PC 1 and 2')
    n = np.where(Y_out==0)[0]
    ax[1].scatter(X3[n, 1],X3[n, 2],marker='o',c=P[n],label=groupnames[0])
    n= np.where(Y_out==1)[0]
    ax[1].scatter(X3[n, 1],X3[n, 2],marker='x',c=P[n],label=groupnames[1])
    n = np.where(Y_out == 2)[0]
    ax[1].scatter(X3[n, 1], X3[n, 2],marker='.', c=P[n], label=groupnames[2])
    n = np.where(Y_out == 3)[0]
    ax[1].scatter(X3[n, 1], X3[n, 2],marker='v', c=P[n], label=groupnames[3])

    ax[2].set_title('PC 0 and 2')
    n = np.where(Y_out==0)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='o',c=P[n],label= groupnames[0])
    n= np.where(Y_out==1)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='x',c=P[n],label=groupnames[1])
    n = np.where(Y_out == 2)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='.', c=P[n], label=groupnames[2])
    n = np.where(Y_out == 3)[0]
    ax[2].scatter(X3[n, 0], X3[n, 2], marker='v', c=P[n], label=groupnames[3])

    plt.legend(loc='lower right')
    pdf.savefig(fig)
    plt.close()


def plot_explained_variance(pdf,pca):
    # PLot explained Variance
    fig = plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.title('Explained_Variance_allPart_wholeBrain_alpha \n ex_var = {}'
              .format(np.cumsum(pca.explained_variance_ratio_)[-1]))
    pdf.savefig(fig)
    plt.close()


def plot_pie_and_distribution(pdf,part,part_cluster,k):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Part {}; {}_Clusters_wholeBrain_alpha'.format(part, k), size=16)

    ax[0].plot(part_cluster)
    ax[0].set_ylim(0, k - 1)
    ax[0].set_title('Part {}; {}_Clusters_wholeBrain_alpha'.format(part, k))
    ax[0].set_ylabel('cluaster_Number')
    ax[0].set_xlabel('time')

    piedata = []
    clusternames = []
    for i in range(k):
        piedata.append(list(part_cluster).count(i))
        clusternames.append('cluster ' + str(i))

    ax[1].pie(piedata, labels=clusternames, autopct='%1.1f%%', startangle=90)
    pdf.savefig(fig)
    plt.close()

def plot_group_averaged_TPM(AllPart, P, k, pdf, data, partnames, groupnames):

    P_0 = np.empty((len(AllPart[partnames[0]]),k,k))
    P_1 = np.empty((len(AllPart[partnames[1]]),k,k))
    P_2 = np.empty((len(AllPart[partnames[2]]),k,k))
    P_3 = np.empty((len(AllPart[partnames[3]]),k,k))

    for c,part in enumerate(AllPart[partnames[0]]):
        part_cluster = P[data['ID'] == part]
        P_0[c,:,:] = prop.get_transition_matrix(part_cluster, k)

    for c,part in enumerate(AllPart[partnames[1]]):
        part_cluster = P[data['ID'] == part]
        P_1[c,:,:] = prop.get_transition_matrix(part_cluster, k)

    for c,part in enumerate(AllPart[partnames[2]]):
        part_cluster = P[data['ID'] == part]
        P_2[c,:,:] = prop.get_transition_matrix(part_cluster, k)

    for c,part in enumerate(AllPart[partnames[3]]):
        part_cluster = P[data['ID'] == part]
        P_3[c,:,:] = prop.get_transition_matrix(part_cluster, k)

    TPM_0 = np.mean(P_0,axis=0)
    TPM_1 = np.mean(P_1,axis=0)
    TPM_2 = np.mean(P_2,axis=0)
    TPM_3 = np.mean(P_3,axis=0)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24,5))
    g1 = sns.heatmap(TPM_0, annot=True,cbar=False, ax = ax1, fmt='.1g')
    g1.set_title(groupnames[0])
    g2 = sns.heatmap(TPM_1, annot=True,cbar=False, ax = ax2, fmt='.1g')
    g2.set_title(groupnames[1])
    g3 = sns.heatmap(TPM_2, annot=True,cbar=False, ax= ax3, fmt='.1g')
    g3.set_title(groupnames[2])
    g4 = sns.heatmap(TPM_3, annot=True,cbar=False, ax= ax4, fmt='.1g')
    g4.set_title(groupnames[3])
    pdf.savefig(f)
    plt.close()

def plot_all_timeseries(pdf, AllPart,P_kmc, data, step, saveimg ):
    # plot all participants dynamics
    if step == "01":
        time = 300
        plt.figure(figsize=(37,4))
    if step == "10":
        time = 30

    all_dyn = np.zeros((len(AllPart["Part"]), time))
    part_order = np.hstack((AllPart["Part_heal"], AllPart["Part_reco"], AllPart["Part_ncmd"], AllPart["Part_nonr"]))

    k = max(P_kmc)+1

    for i, part in enumerate(part_order):
        part_cluster = P_kmc[data['ID'] == part]
        all_dyn[i, :len(part_cluster)] = part_cluster + 1

    my_cmap = plt.get_cmap('viridis', k)
    my_cmap.set_under('lightgrey')
    plt.imshow(all_dyn, cmap=my_cmap, vmin=0.001, vmax=k + .5, alpha=0.7)
    ax = plt.gca()
    ax.set_xticks(np.arange(0, time, 2))
    ax.set_yticks(np.arange(.5, len(part_order), 1))
    ax.set_xticklabels(np.arange(1, time+1, 2))
    ax.set_yticklabels(part_order)
    plt.colorbar()
    plt.clim(0.5, k + 0.5)
    pdf.savefig()

    if saveimg:
        plt.savefig("alldynamics.jpeg")

    plt.close()

