##TODO : prediction method for ward clustering, 
## absolute and stability measures for labeled data
## prepare data

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

original = np.load("spines2.npz")
both_label_vector = np.logical_and(np.not_equal(original["kl1"], None),
                                   np.not_equal(original["kl2"], None))

both_match_vector = np.logical_and(both_label_vector,
                                   original["kl1"] == original["kl2"])

N_CLUSTERS = 9
kmeans = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS)

#transforming original normalised data to line shape
shapes = np.sum(original["shapes_n"], axis=2)

clust_labels = kmeans.fit_predict(shapes)

plt.figure()
for cluster_label in range(N_CLUSTERS):
    plt.subplot(3,3,cluster_label)
    plt.gca().invert_yaxis()
    one_cluster = original["shapes_n"][clust_labels == cluster_label, ...]
    mean_image = np.mean(one_cluster, axis=0)
    plt.pcolor(mean_image)

plt.show()