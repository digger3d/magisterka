#coding: utf8
import numpy as np
from my_ward import myAgglomerativeClustering
from sklearn import cluster
import matplotlib.pylab as plt
from labeled_stability import adjustedNeighbourCeoff
from sklearn import metrics
from new_stability import *

original = np.load("datasets/spines2.npz")
ward = myAgglomerativeClustering
k_means = cluster.MiniBatchKMeans    
shapes = np.mean(original["shapes_n"], axis=2) 

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
#
#def checkCrossStability(n_clusters, data_to_cluster, clust_algo, n_iter = 5):
#    """selection vector is applied after clustering, and throw away
#    vector before"""
#    ##TODO - selection vector version
#    clust_algo_instance = clust_algo(n_clusters = n_clusters)
#    label_matrix = np.zeros((n_iter, len(data_to_cluster)))
#    euclidian_metric = metrics.pairwise.euclidean_distances(data_to_cluster)
#    for i in range(n_iter):
#        label_matrix[i,...] = clust_algo_instance.fit_predict(data_to_cluster)
#    neighbour_coeff_list = []
#    silhouettes_list = []
#    for i in xrange(n_iter):
#        silhouettes_list.append(metrics.silhouette_score(euclidian_metric, label_matrix[i], 
#                metric="precomputed", sample = 3000))
#        for j in xrange(n_iter):
#            if i < j:
#                coeff = adjustedNeighbourCeoff(label_matrix[i], label_matrix[j])
#                neighbour_coeff_list.append(coeff)
#    return silhouettes_list, neighbour_coeff_list
#
#a, b = checkCrossStability(3, shapes, k_means, 2)
#print a, b
#def multiCrossStability(list_n_clusters, data_to_cluster, clust_algo,
#                        path, n_iter = 10):
#    data_file = open(path + "data.txt", "w")
#    data_file.write("algorithm\t{0}\n"
#    "list_of_cluster_n\t{1}\n"
#    "n_iter\t{2}\n".format(str(clust_algo), list_n_clusters, n_iter, throw_away))    
#    accordance_indices = []    
#    neighbour_coeffs = []
#    for i, n_cluster in enumerate(list_n_clusters):
#        print path, i, n_cluster
#        accordance, neighbour = checkCrossStability(n_cluster, data_to_cluster, clust_algo,
#            n_iter, original_data, throw_away)
#        neighbour_coeffs.append(neighbour)
#        accordance_indices.append(accordance)
#    accordance_indices = np.array(accordance_indices)
#    neighbour_coeffs = np.array(neighbour_coeffs)
#    np.save(path + "silhouettes.npy", accordance_indices)
#    np.save(path + "neigh.npy", neighbour_coeffs)
#    print path, "finished!"
#    return np.array(accordance_indices), np.array(neighbour_coeffs)
    
def checkCrossStability(n_clusters, data_to_cluster, clust_algo,
                       n_iter = 5,  original_data = original,
                         throw_away = 20):
    """selection vector is applied after clustering, and throw away
    vector before"""
    selection_vec = np.ones(len(data_to_cluster), dtype="bool")
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    adjusted_labels = np.empty((n_iter, len(data_to_cluster)), dtype=object)
    throw_away_vectors = np.zeros((n_iter, len(data_to_cluster)), dtype="bool")
    euclidian_metric = metrics.pairwise.euclidean_distances(data_to_cluster)
    for i in range(n_iter):
        throw_away_vectors[i,:] = throwAway(throw_away, len(data_to_cluster))
        labels = clust_algo_instance.fit_predict(
            data_to_cluster[throw_away_vectors[i,:]])
        labels = np.insert(labels, throw_to_index_vec(
            throw_away_vectors[i,:]), -1) ## sprawdzic czy mozna insertowac stringi 
        text_labeling = labels
        adjusted_labels[i,:] = text_labeling
    obs_in_every_iter_vec = (np.sum(throw_away_vectors, axis=0) == n_iter) &\
        selection_vec
    obs_in_every_iter_labels = adjusted_labels[..., obs_in_every_iter_vec] ## tutaj dodac selection vector
    neighbour_coeff_list = []
    silhouettes_list = []
    for i in xrange(n_iter):
        silhouettes_list.append(metrics.silhouette_score(euclidian_metric, adjusted_labels[i], 
                metric="precomputed", sample = 3000))
        for j in xrange(n_iter):
            if i < j:
                coeff = adjustedNeighbourCeoff(obs_in_every_iter_labels[i,...],
                                       obs_in_every_iter_labels[j,...])
                neighbour_coeff_list.append(coeff)
    return silhouettes_list, neighbour_coeff_list
    
#sil, neigh = checkCrossStability(3, shapes, k_means, 1)

def multiCrossStability(list_n_clusters, data_to_cluster, clust_algo,
                        path, n_iter = 10,  original_data = original,
                        throw_away = 20):
    """tutaj accordance zmionione jest na silhouettes"""
    data_file = open(path + "data.txt", "w")
    data_file.write("algorithm\t{0}\n"
    "list_of_cluster_n\t{1}\n"
    "n_iter\t{2}\n"
    "throw_away\t{3}".format(str(clust_algo), list_n_clusters, n_iter, throw_away))    
    accordance_indices = []    
    neighbour_coeffs = []
    for i, n_cluster in enumerate(list_n_clusters):
        print path, i, n_cluster
        accordance, neighbour = checkCrossStability(n_cluster, data_to_cluster, clust_algo,
            n_iter, original_data, throw_away)
        neighbour_coeffs.append(neighbour)
        accordance_indices.append(accordance)
    accordance_indices = np.array(accordance_indices)
    neighbour_coeffs = np.array(neighbour_coeffs)
    np.save(path + "silhouettes.npy", accordance_indices)
    np.save(path + "neigh.npy", neighbour_coeffs)
    print path, "finished!"
    return np.array(accordance_indices), np.array(neighbour_coeffs)

#sils, neighs = multiCrossStability([3, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100], shapes, k_means,
#    "results/unlabeled_sil_stab/k_means/", 20)
#    
#sils1, neighs2 = multiCrossStability([3, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100], shapes, ward,
#    "results/unlabeled_sil_stab/ward/", 20)

def plotFromPath(path, npy_file, y_lim = [0.0, 1]):
    trans_dict = dict([("silhouettes.npy", "Silhouette Index"),
                       ("neigh.npy", u"Korygowany Współczynnik Przekrywania")])
    data_file = open(pathJoin(path, "data.txt"))
    matrix = np.load(pathJoin(path, npy_file))
    means = np.mean(matrix, axis = 1)
    errors = np.std(matrix, axis = 1)
    data_dict = {}
    for line in data_file:
        parameter, value = line.rstrip("\n").split("\t")
        data_dict[parameter] = value
    plt.errorbar(np.arange(len(matrix)), means, yerr=errors)
    plt.ylim(y_lim)
    plt.xlim(-1, len(matrix) + 1)
    plt.ylabel(trans_dict[npy_file], fontsize=16)
    plt.xlabel("Liczba Skupisk", fontsize=16)
    plt.plt.tick_params(labelsize=14)
    xticks(np.arange(len(matrix)), eval(data_dict["list_of_cluster_n"]))
    
plotFromPath("results/unlabeled_sil_stab/k_means/", "silhouettes.npy",[0, 0.4])
plotFromPath("results/unlabeled_sil_stab/k_means/", "neigh.npy", [0.3, 1])