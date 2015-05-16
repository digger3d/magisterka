##TODO : prediction method for ward clustering, 
## absolute and stability measures for labeled data
## prepare data

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster

original = np.load("spines2.npz")
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
print sum(longs), sum(mushrooms), sum(stubbies)

#test_adj = np.array([1,2,1,1,3])
#test_clust = np.array([4,4,1,2])
#print change_labels_to_text(test_adj, test_clust)
both_match_long_stubby_mush = both_match_vector & (longs | mushrooms | stubbies)

def change_labels_to_text(cluster_translation, label_vector):
    """
    params:
    cluster_translation - cluster with indication to what group cluster 
        was assigned (see tranlate dictionary)
    label_vector - vector with cluster labels obtained from clustering alg.
    
    returns:
        vector of text labels ready to 
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
        
N_CLUSTERS = 9
kmeans = cluster.MiniBatchKMeans(n_clusters=N_CLUSTERS)

#transforming original normalised data to line shape
shapes = np.sum(original["shapes_n"], axis=2)

clust_labels = kmeans.fit_predict(shapes)

plt.figure()
for cluster_label in range(N_CLUSTERS):
    plt.subplot(3,3,cluster_label + 1) ## lol watch out!!!!
    plt.gca().invert_yaxis()
    one_cluster = original["shapes_n"][clust_labels == cluster_label, ...]
    mean_image = np.mean(one_cluster, axis=0)
    plt.pcolor(mean_image)

plt.show()
        
adjusted_labels = input("1 long, 2 mushroom, 3 stubby: ")
#adjusted_labels = map(int, adjusted_labels.split(","))
if len(adjusted_labels) < N_CLUSTERS or max(adjusted_labels) > N_CLUSTERS:
    print "wrong input!!!"

def random_accordance(n_cases, n_cluster, N_iter):
    accordance_vec = np.zeros(N_iter)
    for i in range(N_iter):
        a = np.random.randint(n_cluster, size=n_cases)
        b = np.random.randint(n_cluster, size=n_cases)
        accordance_vec[i] = np.sum(a != b) / float(n_cases)
    return np.mean(accordance_vec)
    

def accordance(adjusted_labels, clustering_vec, classification,
               selection_vector=None, N_iter = 100):
    adjusted_clustering = change_labels_to_text(adjusted_labels, clustering_vec)
    if selection_vector != None:
        adjusted_clustering = adjusted_clustering[selection_vector]
        classification = classification[selection_vector]
    accordance = np.sum(adjusted_clustering != classification)\
        / float(len(classification))
    return accordance / random_accordance(len(clustering_vec),
        len(adjusted_labels), N_iter)

    
print accordance(adjusted_labels, clust_labels, original["kl1"],
                 both_match_long_stubby_mush)
    
    
    
    
    
    