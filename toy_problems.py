# -*- coding: utf-8 -*-
import sklearn as skl
import sklearn.cluster
import sklearn.neighbors
import sklearn.metrics

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def createGaussianData(*args):
    clusters = []
    for centroid in args:
        random_var = multivariate_normal(mean=centroid[:-1])
        #cov matriax is identity
        clust = random_var.rvs(centroid[-1])
        clusters.append(clust)
    return np.concatenate(clusters)
    
def produceSilhouetteDiag(data, clustering_algorithm, n_clust_list):
    silhouettes = []
    clust_ticks = np.arange(len(n_clust_list)) + 1
    for n in n_clust_list:
        algo = clustering_algorithm(n_clusters = n)
        labels = algo.fit_predict(data)
        silhouettes.append(skl.metrics.silhouette_score(data, labels, sample_size = 3000)) 
    plt.plot(clust_ticks, silhouettes, "bo", markersize=10)
    plt.xlim([0, len(n_clust_list) + 1])    
    plt.title("Silhouette Index (KMeans Algorithm)")   
    plt.xticks(clust_ticks, n_clust_list)
    plt.xlabel("No. of clusters")

COLORS = np.array([x for x in "bgrcmykbgrcmyk"])
def plotClustering(data, clustering_alg, n_clust):
    plt.title("Clustering")
    clustering_alg = clustering_alg(n_clusters=n_clust).fit(data)
    labels = clustering_alg.labels_
    plt.scatter(data[:,0],data[:,1], color=COLORS[labels])
    
def plotClusters(data, labels):
    plt.title("Data")
    plt.scatter(data[:,0],data[:,1], color=COLORS[labels])

def ward_linkage(n_clusters = 2):
    return skl.cluster.AgglomerativeClustering(n_clusters=n_clusters,
                                               linkage="ward")

def create_labels(n_clust, n_samples_in_clust):
    labels = []
    for i in xrange(n_clust):
        labels += [i] * n_samples_in_clust
    return labels

#if __name__ == "__main__":                                               
class Data:                                               
    gaussy2 = createGaussianData((5, 5, 500), (10, 10, 500))
    gaussy3 = createGaussianData((5, 5, 500), (10, 10, 500), (10, 5, 500))
    gaussy6 = createGaussianData((5, 5, 500), (10, 10, 500), (10, 5, 500),
                                 (5, 10, 500), (5, 15, 500), (10, 15, 500))
                                 
    ov_gaussy2 = createGaussianData((5, 5, 500), (7, 7, 500))
    ov_gaussy4 = createGaussianData((5, 5, 500), (7, 7, 500),
                                    (9, 9, 500), (11, 11, 500))
    ov_gaussy4_2 = createGaussianData((5, 5, 500), (8, 8, 500),
                                    (11, 11, 500), (14, 14, 500))

#Silhouette index 6gauss
#produceSilhouetteDiag(Data.gaussy6, skl.cluster.MiniBatchKMeans, range(2,11))
#plotClusters(Data.gaussy6, create_labels(6,500))
#plotClustering(Data.gaussy6, skl.cluster.MiniBatchKMeans, 6)

##Silhouette index 4gauss_ov
#produceSilhouetteDiag(Data.ov_gaussy4, skl.cluster.MiniBatchKMeans, range(2,11))
#plotClusters(Data.ov_gaussy4, create_labels(4,500))
#plotClustering(Data.ov_gaussy4, skl.cluster.MiniBatchKMeans, 4)




#connectivity = skl.neighbors.kneighbors_graph(a, n_neighbors=10,
#                                              include_self=False)
#connectivity = 0.5 * (connectivity + connectivity.T)
#
#average_linkage = skl.cluster.AgglomerativeClustering(
#        linkage="ward", affinity="euclidean", n_clusters=6,
#        connectivity=connectivity)
#average_linkage.fit(a)        
#labels = average_linkage.labels_
#            
##mb_kmeans = skl.cluster.MiniBatchKMeans(n_clusters = 2)
##mb_kmeans.fit(a)
##labels = mb_kmeans.labels_
