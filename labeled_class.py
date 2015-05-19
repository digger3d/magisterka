##TODO : prediction method for ward clustering, 
## absolute and stability measures for labeled data

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from my_ward import myAgglomerativeClustering

original = np.load("datasets/spines2.npz")
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

#test_adj = np.array([1,2,1,1,3])
#test_clust = np.array([4,4,1,2])
#print change_labels_to_text(test_adj, test_clust)
both_match_long_stubby_mush = both_match_vector & (longs | mushrooms | stubbies)

def change_labels_to_text(cluster_translation, label_vector):
    """
    params
    -------
    cluster_translation : cluster with indication to what group cluster was assigned\
    (see tranlate dictionary)
    
    label_vector : vector with cluster labels obtained from clustering alg.
    
    returns
    -------
    labels : np.array of ints
    vector of text labels ready to compare with reference class
    
    """
    text_labels = [None] * len(label_vector)
    translate_dict = {1 : "Long & thin",
                      2 : "Mushroom",
                      3 : "Stubby"}
    for clust_no, clust_trans_key in enumerate(cluster_translation):
        for i, cluster_label in enumerate(label_vector):
            if cluster_label == clust_no:
                text_labels[i] = translate_dict[clust_trans_key]
    return np.array(text_labels)

def plotClusters(clust_labels, n_clusters, original_data):
    plt.figure()
    for cluster_label in range(n_clusters):
        plt.subplot(7,7,cluster_label + 1) ## lol watch out for +1 !!!!
        plt.title(cluster_label + 1)
        plt.gca().invert_yaxis()
        one_cluster = original_data["shapes_n"][clust_labels == cluster_label, ...]
        mean_image = np.mean(one_cluster, axis=0)
        plt.pcolor(mean_image)
    plt.show(block = False)

def getLabels(N_CLUSTERS):
    """collects integer tuple as described belowe"""        
    adjusted_labels = input("1 long, 2 mushroom, 3 stubby: ")
    if len(adjusted_labels) < N_CLUSTERS or max(adjusted_labels) > 3:
        print "wrong input!!!"
    return adjusted_labels

def randomAccordance(n_cases, n_cluster, N_iter):
    accordance_vec = np.zeros(N_iter)
    for i in range(N_iter):
        a = np.random.randint(n_cluster, size=n_cases)
        b = np.random.randint(n_cluster, size=n_cases)
        accordance_vec[i] = np.sum(a != b) / float(n_cases)
    return np.mean(accordance_vec)
    

def accordance(adjusted_labels, clustering_vec, classification,
               selection_vector=None, N_iter = 50):
    """
    Parameters
    ----------
    adjusted_labels : labels indicating to which group cluster was assigned \
    (eg 3 is for "Stubby" etc.)
    clustering_vec : vector returned from clustering algo
    classification : reference classification (here made by human)
    selection_vector : logic vector indicating whether all observation should \
    be taken under consideration
    N_iter : number of iterations used to determine randomAccordance
    
    Retruns
    -------
    Accordance : adjusted by randomAccordance for that many clusters
    """
    
    adjusted_clustering = change_labels_to_text(adjusted_labels, clustering_vec)
    if selection_vector != None:
        adjusted_clustering = adjusted_clustering[selection_vector]
        classification = classification[selection_vector]
    accordance = np.sum(adjusted_clustering != classification)\
        / float(len(classification))
    return accordance# / randomAccordance(len(clustering_vec),
        #len(adjusted_labels), N_iter)

def accordanceMultiCluster(listOfClusterN, clust_algo,
         result_file, original_data = original, n_iter = 50):
    accordances = np.zeros(len(listOfClusterN))
    print 1
    data_to_cluster = np.mean(original_data["shapes_n"], axis=2)
    print 2    
    for i, n_clusters in enumerate(listOfClusterN):
        print 3
        clust_algo_instance = clust_algo(n_clusters = n_clusters)
        print 4        
        labels = clust_algo_instance.fit_predict(data_to_cluster)
        print 5
        plotClusters(labels, n_clusters, original_data)
        print 6
        plt.pause(0.001)
        adjusted_labels = getLabels(n_clusters)
        accordances[i] = accordance(adjusted_labels, labels,
            original_data["kl1"], both_match_long_stubby_mush, n_iter)
    result_txt = open(result_file, "w")
    for i, accord in enumerate(accordances):
        result_txt.write("{}\t{}\n".format(listOfClusterN[i],accord))
    return accordances
        ## odpalic accordance 
#print accordance(adjusted_labels, clust_labels, original["kl1"],
#                 both_match_long_stubby_mush)
k_means = cluster.MiniBatchKMeans

#transforming original normalised data to line shape
#shapes = np.sum(original["shapes_n"], axis=2)
my_ward = myAgglomerativeClustering
a = accordanceMultiCluster([10, 15, 20, 30, 40], my_ward, "results/ward_accord.txt", original, 50)
#    
    
    
    
    