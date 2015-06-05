from toy_problems import Data
import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading


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

def getIndicesOfTrue(vector):
    return np.array([i for i, x in enumerate(vector) if x])
    
def getLearningAndThrowAway(len_learn, len_throw_away, learnable_obs = lsm):
    learnable_indices = set(getIndicesOfTrue(learnable_obs))
    all_indices = set(range(len(learnable_obs)))
    learn_indices = np.random.choice(learn_indices, size = len_learn)
#    learnable_indices = learnable_indices - set(learn_indices)
    all_indices = all_indices - set(learn_indices)
    throw_away_indices = np.random.choice(all_indices, size = len_throw_away)
    return learn_indices, throw_away_indices
    
####TODO zszyc to z semi-labeled data 
    

def multiCrossStabilitySemi(list_n_clusters, data_to_cluster, clust_algo,
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
    print "finished!"
    return np.array(accordance_indices), np.array(neighbour_coeffs)
    
    
    
X = Data.gaussy2
y = np.concatenate((np.zeros(500,dtype="int"), np.ones(500,dtype="int")))
y_semi = y
y_semi[np.random.randint(0, 2)] = -1
print y_semi
