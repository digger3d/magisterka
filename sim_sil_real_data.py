import numpy as np
from toy_problems import produceSilhouetteDiag
from my_ward import myAgglomerativeClustering
from sklearn import cluster
from sklearn import metrics

original = np.load("datasets/spines2.npz")
shapes = np.mean(original["shapes_n"], axis=2)

k_means = cluster.MiniBatchKMeans
ward = myAgglomerativeClustering

def multiSilhouetteArray(data, clustering_algorithm, n_clust_list, n_iter):
    silhouettes = np.zeros((len(n_clust_list), n_iter))
    clust_ticks = np.arange(len(n_clust_list)) + 1
    euclidian_metric = metrics.pairwise.euclidean_distances(data)
    for i, n in enumerate(n_clust_list):
        for j in range(n_iter):
            print i + j
            algo = clustering_algorithm(n_clusters = n)
            labels = algo.fit_predict(data)
            silhouettes[i,j] = metrics.silhouette_score(euclidian_metric, labels, 
                metric="precomputed", sample = 3000)
    return silhouettes

## plotting and saving to file 

#a = multiSilhouetteArray(shapes, k_means, [3, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100], 1)
produceSilhouetteDiag(shapes, k_means, [3, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100])
#print a

#TODO reszata w stability.py


