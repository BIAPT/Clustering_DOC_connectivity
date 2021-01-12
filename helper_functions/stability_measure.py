'''
This code provides the methods is to
calculate the stability index of a K-Means
Clustering Solution (propoosed by Lange et al 2004)
'''
# General Import
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from numpy import unravel_index
import multiprocessing as mp
import os
import sys

def compute_silhouette_score(X,P,K):
    SIL = np.zeros([len(K), len(P)])

    for p in P:
        pca = PCA(n_components=p)
        pca.fit(X)
        X_LD = pca.transform(X)

        for k in K:
            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
            kmeans.fit(X_LD)  # fit the classifier on all X_LD
            S = kmeans.predict(X_LD)
            silhouette = silhouette_score(X_LD, S)
            SIL[K.index(k), P.index(p)] = silhouette
            print('Silhouette_score with P = {} and k = {}'.format(p, k))

    return SIL


def compute_stability_index(X,Y_ID,P,K,r):
    # for all repetitions
    SI = np.empty([len(K), len(P)])  # Collection of stability index over Repetitions
    for p in P:
        print("Start: r = {}, p = {}".format(r, p))
        x_complete = X.copy()  # complete input set for PCA-fit
        # divide the participants into two groups temp and test
        part = np.unique(Y_ID)
        nr_part = len(part)
        rand_temp = np.random.choice(part, nr_part // 2, replace=False)
        rand_test = np.setdiff1d(part, rand_temp)

        X_temp = X[np.isin(Y_ID, rand_temp)]
        X_test = X[np.isin(Y_ID, rand_test)]

        # fit the pca on the complete dataset and transfor the divided sets
        pca = PCA(n_components=p)
        pca.fit(x_complete)
        X_temp_LD = pca.transform(X_temp)  # get a low dimension version of X_temp
        X_test_LD = pca.transform(X_test)  # and X_test
        print("PCA FINISHED: r = {}, p = {}".format(r, p))

        for k in K:
            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
            kmeans.fit(X_temp_LD)  # fit the classifier on X_template
            S_temp = kmeans.predict(X_test_LD)

            kmeans = KMeans(n_clusters=k, max_iter=1000, n_init=100)
            kmeans.fit(X_test_LD)  # fit the classifier on X_test
            S_test = kmeans.predict(X_test_LD)

            # now we would need to define which clusters correspond to each other
            # IMPORTANT: The label alone can not be used in this case
            overlap = np.empty([k, k])

            for c_test in range(k):
                for c_temp in range(k):
                    # get common points of groups
                    common = np.intersect1d(np.where(S_test == c_test),
                                            np.where(S_temp == c_temp)).shape[0]
                    overlap[c_test, c_temp] = common

            common_val = 0

            for i in range(k):
                maxval = unravel_index(overlap.argmax(), overlap.shape)
                common_val += overlap[maxval[0], maxval[1]]
                # save overlap value and set row and col to -1
                # only continue with finding other electrode pairs
                overlap[maxval[0], :] = -1
                overlap[:, maxval[1]] = -1

            # compute the hamming distance between the 2 solutions
            # equal to the percantage amount of unequal digits.
            unequal = 1 - (common_val / S_test.shape[0])
            print("END: p = {}, k = {}".format(p, k))

            SI[K.index(k), P.index(p)] = unequal

    return SI



