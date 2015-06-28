#coding: utf8
import numpy as np
from my_ward import myAgglomerativeClustering
#from toy_problems import Data
from sklearn import cluster
import matplotlib.pylab as plt
from new_stability import createTransDict
original = np.load("datasets/spines2.npz")
shapes = np.sum(original["shapes_n"], axis=2)
 ##at least one classifier chose label for a spine
both_label_vector = np.logical_and(np.not_equal(original["kl1"], None),
                                   np.not_equal(original["kl2"], None))
##both classyfiers agreed on spine
both_match_vector = np.logical_and(both_label_vector,
                                   original["kl1"] == original["kl2"])
##both stubby, long & thin or mushroom                                   
longs = original["kl1"] == "Long & thin"
mushrooms = original["kl1"] == "Mushroom"
stubbies = original["kl1"] == "Stubby"
##roznice w populacjach poszczegolnych typow
#print sum(longs), sum(mushrooms), sum(stubbies)
both_match_long_stubby_mush = lsm = both_match_vector & (longs | mushrooms | stubbies)

##clust algorithms
ward = myAgglomerativeClustering
k_means = cluster.MiniBatchKMeans 

def plotClusters(clust_labels, n_clusters, original_data, title):
    d = createTransDict(clust_labels, 100)
    plt.figure()
    plt.suptitle(title, fontsize=16)
    for cluster_label in range(n_clusters):
        plt.subplot(10,10,cluster_label + 1) ## lol watch out for +1 !!!!
        plt.xlim([0, 31])
        plt.ylim([0, 63])
        plt.gca().invert_yaxis()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)        
        plt.title(d[cluster_label])
        one_cluster = original_data["shapes_n"][clust_labels == cluster_label, ...]
        mean_image = np.mean(one_cluster, axis=0)
        plt.pcolor(mean_image)
    plt.subplots_adjust(hspace=0.45)
    
def plotClusters2(clust_labels, n_clusters, original_data):
    plt.figure()
    for cluster_label in range(n_clusters):
        plt.subplot(10,10,cluster_label + 1) ## lol watch out for +1 !!!!
        plt.title(cluster_label + 1)
        plt.gca().invert_yaxis()
        one_cluster = original_data["shapes_n"][clust_labels == cluster_label, ...]
        mean_image = np.mean(one_cluster, axis=0)
        plt.pcolor(mean_image)



def plotClustersPL(clust_labels, n_clusters, original_data, trans_dict):
    d = createTransDict(clust_labels, 100)
    plt.figure()
    for cluster_label in range(n_clusters):
        plt.subplot(10,10,cluster_label + 1) ## lol watch out for +1 !!!!
        plt.xlim([0, 31])
        plt.ylim([0, 63])
        plt.gca().invert_yaxis()
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)        
        plt.title(trans_dict[d[cluster_label]])
        one_cluster = original_data["shapes_n"][clust_labels == cluster_label, ...]
        mean_image = np.mean(one_cluster, axis=0)
        plt.pcolor(mean_image)
    plt.subplots_adjust(hspace=0.45)
def simplyClusterData(data_to_cluster,n_clusters, clust_algo,
                      original_data=original):
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    labels = clust_algo_instance.fit_predict(data_to_cluster)
    return labels

trans_dict = {"Long & thin" : u"Długie",
              "Mushroom" : "Grzybkowate",
              "Stubby" : "Przysadziste",
              "84" : "Przysadziste"}

#labels_k_means = simplyClusterData(shapes, 100, k_means)
labels_ward = simplyClusterData(shapes, 100, ward)

#plotClustersPL(labels_k_means, 100, original, trans_dict)
plotClustersPL(labels_ward, 100, original, trans_dict)

#
#plotClusters(labels_k_means, 100, original,
#             u"Średnia sylwetka kolca w obrębie klastra. Algorytm: K-Means")
#plotClusters(labels_ward, 100, original,
#             u"Średnia sylwetka kolca w obrębie klastra. Algorytm: Ward")
             
plt.show()