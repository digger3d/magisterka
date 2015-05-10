##TODO : prediction method for ward clustering, 
## absolute and stability measures for labeled data
## prepare data

import numpy as np
from sklearn import cluster

original = np.load("spines2.npz")
both_label_vector = np.logical_and(np.not_equal(original["kl1"], None),
                                   np.not_equal(original["kl2"], None))

both_match_vector = np.logical_and(both_label_vector,
                                   original["kl1"] == original["kl2"])

kmeans = cluster.MiniBatchKMeans(n_clusters=10)

#transforming original normalised data to line shape
shapes = np.sum(original["shapes_n"], axis=2)

print kmeans.fit_predict(shapes)