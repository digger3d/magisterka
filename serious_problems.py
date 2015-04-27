# -*- coding: utf-8 -*-
import numpy as np
import sklearn.cluster
import stability
import toy_problems as tp
from sklearn.feature_selection import VarianceThreshold

M = stability.Munkres()
all_data = np.load("./zaczyn/spines.npz")
all_norm = all_data["shapes_n"]
all_shape_extr = np.sum(all_norm, axis=2)
all_norm = all_norm.reshape(all_norm.shape[0], -1)

selector = VarianceThreshold(threshold=(16000))
pruned_all_norm = selector.fit_transform(all_norm)
print pruned_all_norm.shape


#data = np.load("smaller_1000.npz")
#norm = data["shapes_n"]
#shape_extr = np.sum(norm, axis=2)
#
#small_data = np.load("smaller_100.npz")
#small_norm = small_data["shapes_n"]
#small_shape_extr = np.sum(small_norm,axis=2)
#
KMEANS = sklearn.cluster.MiniBatchKMeans
#tp.produceSilhouetteDiag(small_shape_extr, sklearn.cluster.MiniBatchKMeans,
#                         [2, 4, 8])
#tp.produceSilhouetteDiag(all_shape_extr, KMEANS,
#                         [4, 8, 15, 30, 50])
#                         
#stability.plotSimilarityMeasure(all_shape_extr, KMEANS, [4, 8, 15, 30, 50],
#                                M, 50)