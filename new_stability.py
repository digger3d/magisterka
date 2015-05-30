from labeled_stability import *
from collections import Counter
import matplotlib.pylab as plt

#original = np.load("datasets/spines2.npz")
#
###at least one classifier chose label for a spine
#both_label_vector = np.logical_and(np.not_equal(original["kl1"], None),
#                                   np.not_equal(original["kl2"], None))
###both classyfiers agreed on spine
#both_match_vector = np.logical_and(both_label_vector,
#                                   original["kl1"] == original["kl2"])
###both stubby, long & thin or mushroom                                   
#longs = original["kl1"] == "Long & thin"
#mushrooms = original["kl1"] == "Mushroom"
#stubbies = original["kl1"] == "Stubby"
###roznice w populacjach poszczegolnych typow
##print sum(longs), sum(mushrooms), sum(stubbies)
#both_match_long_stubby_mush = lsm = both_match_vector & (longs | mushrooms | stubbies)



def clusterVote(label, labels, original_data = original,
    selection_vec = both_match_long_stubby_mush):
    """returns tuple (old_label(int), new_label(string in ["Stubby", "mush", "long"]"""
    trunc_original_labels = original["kl1"][selection_vec]
    trunc_labels = labels[selection_vec]
    originals_in_cluster = trunc_original_labels[trunc_labels == label]
    freqs = Counter(originals_in_cluster)
    try:
        new_label = freqs.most_common(1)[0][0]
        return label, new_label
    except IndexError:
        return label, str(label)
    

def createTransDict(labels, n_cluster, original_data = original,
        selection_vec = both_match_long_stubby_mush):
    transTable = []
    for label in xrange(n_cluster):
        transTable.append(clusterVote(label, labels, original_data, selection_vec))
    return dict(transTable)

def translateLabeling(labels, n_cluster, original_data = original,
        selection_vec = both_match_long_stubby_mush):
    new_labels = np.empty(len(labels), dtype=object)
    trans_dict = createTransDict(labels, n_cluster, original_data, selection_vec)
    for i, label in enumerate(labels):
        try:
            new_labels[i] = trans_dict[label]
        except KeyError:
            new_labels[i] = label
    return new_labels

#org_labs = original["kl1"]
#a = np.ones(len(org_labs), dtype="uint")
#b = translateLabeling(a,3)

        
def checkCrossStability(n_clusters, data_to_cluster, clust_algo,
                       n_iter = 5,  original_data = original,
                         throw_away = 20, selection_vec = lsm):
    """selection vector is applied after clustering, and throw away
    vector before"""
    ##TODO - selection vector version
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    adjusted_labels = np.empty((n_iter, len(data_to_cluster)), dtype=object)
    throw_away_vectors = np.zeros((n_iter, len(data_to_cluster)), dtype="bool")
    for i in range(n_iter):
        throw_away_vectors[i,:] = throwAway(throw_away, len(data_to_cluster))
        labels = clust_algo_instance.fit_predict(
            data_to_cluster[throw_away_vectors[i,:]])
        adjusted_labeling = np.insert(labels, throw_to_index_vec(
            throw_away_vectors[i,:]), -1) ## sprawdzic czy mozna insertowac stringi 
        text_labeling = translateLabeling(adjusted_labeling, n_clusters,
            original_data, selection_vec)
        adjusted_labels[i,:] = text_labeling
    obs_in_every_iter_vec = (np.sum(throw_away_vectors, axis=0) == n_iter) &\
        selection_vec
    obs_in_every_iter_labels = adjusted_labels[..., obs_in_every_iter_vec] ## tutaj dodac selection vector
    neighbour_coeff_list = []
    accordance_list = []
    for i in xrange(n_iter):
        accordance = np.sum(obs_in_every_iter_labels[i,...] ==\
            original_data["kl1"][obs_in_every_iter_vec]) / float(
            len(original_data["kl1"][obs_in_every_iter_vec]))
        accordance_list.append(accordance)
        for j in xrange(n_iter):
            if i < j:
                coeff = adjustedNeighbourCeoff(obs_in_every_iter_labels[i,...],
                                       obs_in_every_iter_labels[j,...])
                neighbour_coeff_list.append(coeff)
    return accordance_list, neighbour_coeff_list

ward = myAgglomerativeClustering
k_means = cluster.MiniBatchKMeans    
shapes = np.mean(original["shapes_n"], axis=2)    
#a, b = checkCrossStability(3, shapes, k_means)
#print a

def multiCrossStability(list_n_clusters, data_to_cluster, clust_algo,
                        path, n_iter = 10,  original_data = original,
                        throw_away = 20):
    data_file = open(path + "data.txt", "w")
    data_file.write("algorithm\t{0}\n"
    "list_of_cluster_n\t{1}\n"
    "n_iter\t{2}\n"
    "throw_away\t{3}".format(str(clust_algo), list_n_clusters, n_iter, throw_away))    
    accordance_indices = []    
    neighbour_coeffs = []
    for i, n_cluster in enumerate(list_n_clusters):
        print i, n_cluster
        accordance, neighbour = checkCrossStability(n_cluster, data_to_cluster, clust_algo,
            n_iter, original_data, throw_away)
        neighbour_coeffs.append(neighbour)
        accordance_indices.append(accordance)
    accordance_indices = np.array(accordance_indices)
    neighbour_coeffs = np.array(neighbour_coeffs)
    np.save(path + "accord.npy", accordance_indices)
    np.save(path + "neigh.npy", neighbour_coeffs)
    return np.array(accordance_indices), np.array(neighbour_coeffs)

#dolozyc klastrowan
a, b = multiCrossStability([3, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100], shapes,
                           k_means, "results/multi_stab/kmeans3/", n_iter = 50)
    
def plotMultiStab(data_file, matrix_file):
    data_dict = {}
    stab_mat = np.load(matrix_file)
    for line in open(data_file, "r"):
        key, val = line.rstrip("\n").split("\t")
        data_dict[key] = val
    accord_err = np.std(stab_mat, axis = 1)
    accord_mean = np.mean(stab_mat, axis = 1)
    plt.errorbar(np.arange(len(stab_mat)), accord_mean, yerr=accord_err)
    plt.show()

#plotMultiStab("results/multi_stab/kmeans/data.txt",
#              "results/multi_stab/kmeans/neigh.npy")

def plotSmth(array, title):
    plt.plot(np.mean(array, axis=1))
    plt.show()
    
#plotSmth(a, "dupa")