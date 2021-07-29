"""
Written by: Charlotte Maschke
Last update: July 28 2021

This code provides the methods to calculate the stability index and silhouette score
of a K-Means clustering algorithm. It provides the methods to do a grid search over
several numbers of K (nr. of clusters) and P (nr. of principal components)
The Stability index was proposed by (Lange et al 2004)
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def compute_silhouette_score(X, P, K):
    '''
    :param X: data frame containing FC data for areas of interest
    :param P: List of integer containing all Hyperparametes P (numbers principle components) to explore
    :param K: List of integer containing all Hyperparametes K (numbers of Clusters) to explore
    :return: Silhouette score Matrix(P x K) containing the Silhouette score for every hyperparameter pair
    '''

    # initialize empty frame for silhouette score
    sil = np.zeros([len(K), len(P)])

    for p in P:
        print("Start Silhouette score with : p = {}".format(p))
        # reduce dimension of data to get X_LD = "Lower Dimension"
        pca = PCA(n_components=p)
        pca.fit(X)
        x_ld = pca.transform(X)

        for k in K:
            # initialize k-means with k clusters
            kmeans = KMeans(n_clusters = k, n_init=100)
            kmeans.fit(x_ld)

            # fit the classifier on all X_LD
            s = kmeans.predict(x_ld)

            # predict the corresponding clusters
            silhouette = silhouette_score(x_ld, s)

            # calculate silhouette
            sil[K.index(k), P.index(p)] = silhouette
            print('Finished Silhouette_score with P = {} and k = {}'.format(p, k))

    return sil


def compute_stability_index(X, Y_ID, P, K):
    '''
    :param X: data frame containing FC data for areas of interest
    :param Y_ID: List of Participant IDS corresponding to every instance of x
    :param P: List of integer containing all Hyperparametes P (numbers principle components) to explore
    :param K: List of integer containing all Hyperparametes K (numbers of Clusters) to explore
    :param r:
    :return: Silhouette score Matrix(P x K) containing the Silhouette score for every hyperparameter pair
    '''

    # create empty frame for all repetitions to fill in the stability index
    si = np.empty([len(K), len(P)])

    # keep complete input set used later for PCA-fit
    x_complete = X.copy()

    # divide the participants into two groups temp and test
    part = np.unique(Y_ID)
    nr_part = len(part)
    rand_temp = np.random.choice(part, nr_part // 2, replace=False)
    rand_test = np.setdiff1d(part, rand_temp)

    X_temp = X[np.isin(Y_ID, rand_temp)]
    X_test = X[np.isin(Y_ID, rand_test)]

    for p in P:
        print("Start Stability index with : p = {}".format(p))
        # fit the pca on the complete dataset and transform the divided sets
        pca = PCA(n_components=p)
        pca.fit(x_complete)
        X_temp_LD = pca.transform(X_temp)  # get a low dimension version of X_temp
        X_test_LD = pca.transform(X_test)  # and X_test

        for k in K:
            kmeans = KMeans(n_clusters=k, n_init=100)
            kmeans.fit(X_temp_LD)  # fit the classifier on X_template
            S_temp = kmeans.predict(X_test_LD)

            kmeans = KMeans(n_clusters=k, n_init=100)
            kmeans.fit(X_test_LD)  # fit the classifier on X_test
            S_test = kmeans.predict(X_test_LD)

            # now we would need to define which clusters correspond to each other
            # IMPORTANT: The label alone can not be used in this case
            # We are therefore searching the most overlapping clusters to assign them to be one cluster
            overlap = np.empty([k, k])

            for c_test in range(k):
                for c_temp in range(k):
                    # get common points of groups
                    common = np.intersect1d(np.where(S_test == c_test),
                                            np.where(S_temp == c_temp)).shape[0]
                    overlap[c_test, c_temp] = common

            common_val = 0

            for i in range(k):
                maxval = np.unravel_index(overlap.argmax(), overlap.shape)
                common_val += overlap[maxval[0], maxval[1]]
                # save overlap value and set row and col to -1
                # only continue with finding other electrode pairs
                overlap[maxval[0], :] = -1
                overlap[:, maxval[1]] = -1

            # compute the hamming distance between the 2 solutions
            # equal to the percantage amount of unequal cluster assignments
            unequal = 1 - (common_val / S_test.shape[0])
            print("END: p = {}, k = {}".format(p, k))

            # Save the stability index in the corresponding index
            si[K.index(k), P.index(p)] = unequal

    return si



