import numpy as np
from my_ward import myAgglomerativeClustering
#from toy_problems import Data
from sklearn import cluster

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
both_match_long_stubby_mush = lsm = both_match_vector & (longs | mushrooms | stubbies)

def createNeighbourMatrix(labels_vec):
    """NeighMatrix[i,j] == 1 iff labels_vec[i] == labels_vec[j]
    Skipping the lower part of matrix because it is symetric
    Maximum 256 clusters"""
    n = len(labels_vec)
    neighMatrix = np.zeros((n, n), dtype="uint8")
    for i in xrange(n):
        neighMatrix[i,i:] = labels_vec[i:] == labels_vec[i]
    return neighMatrix
    
def neighbourCoeff(labels1, labels2):
    """label1 and 2 are same length vectors"""
    neighMat1 = createNeighbourMatrix(labels1)
    neighMat2 = createNeighbourMatrix(labels2)
    return np.sum(neighMat1 * neighMat2)# / float(n)

def adjustedNeighbourCeoff(labels1, labels2):
    """smaller than 1, by schwartz inequality"""
    return neighbourCoeff(labels1,labels2) /\
        np.sqrt(neighbourCoeff(labels1,labels1) * neighbourCoeff(labels2,labels2))
    
def adjustedNeighbourCoeffRatio(reference_labels, labels2):
    """returns ratio of edges present in both labellings to edges in ref
    neighbouringMatrix, tutaj mozliwe jest porownywanie jedynie klastrowan
    z taka sama ilosci klastrow"""
    return neighbourCoeff(reference_labels,labels2) /\
        np.sum(createNeighbourMatrix(reference_labels), dtype="float")

def meanRandomNeighCoeff(n, n_clusters, n_iter):
    coeffs = np.zeros(n_iter)
    for i in xrange(n_iter):
        a = np.random.randint(n_clusters, size=n)
        b = np.random.randint(n_clusters, size=n)
        coeffs[i] = neighbourCoeff(a, b)
    return np.mean(coeffs)

def neighbourCoeffOfClustering(n_clusters, clust_algo,
        original_data=original, selection_vector=both_match_long_stubby_mush):
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    data_to_cluster = np.mean(original_data["shapes_n"], axis=2)
    labels = clust_algo_instance.fit_predict(data_to_cluster)
    ref_labels = original_data["kl1"]    
    if selection_vector != None:
        ref_labels = ref_labels[selection_vector]
        labels = labels[selection_vector]
    return adjustedNeighbourCoeffRatio(ref_labels, labels)
    
def throwAway(throw, N):
    vec = np.array([False] * throw + [True] * (N - throw))
    np.random.shuffle(vec)
    return vec

## respective to reference przechowywac labele
## respective to each other
def checkReferenceStability(n_clusters, data_to_cluster, clust_algo,
        reference_labels, n_iter = 10, selection_vector = None,
        throw_away = 20):
    """selection vector is applied after clustering, and throw awaya
    vector before"""
    ## TODO - no selection vector is used 
    if selection_vector == None:
        selection_vector = np.array([True] * len(reference_labels))
    neigh_coeffs = np.zeros(n_iter)
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    print "begin"
    ### dorobic losowanie przypadkow!
    for i in range(n_iter):
        print i
        print neigh_coeffs
        throw_away_vector = throwAway(throw_away, len(reference_labels))
        new_reference_labels = reference_labels[throw_away_vector]
        labels = clust_algo_instance.fit_predict(data_to_cluster[throw_away_vector])
        neigh_coeffs[i] = adjustedNeighbourCoeffRatio(new_reference_labels,
                                                      labels)
    return neigh_coeffs

def checkCrossStability(n_clusters, data_to_cluster, clust_algo,
                        file_name, n_iter = 10, throw_away = 20):
    """selection vector is applied after clustering, and throw away
    vector before"""
    ##TODO - selection vector version
    clust_algo_instance = clust_algo(n_clusters = n_clusters)
    adjusted_labels = np.zeros((n_iter, len(data_to_cluster)))
    throw_away_vectors = np.zeros((n_iter, len(data_to_cluster)), dtype="bool")
    for i in range(n_iter):
        throw_away_vectors[i,:] = throwAway(throw_away, len(data_to_cluster))
        labels = clust_algo_instance.fit_predict(
            data_to_cluster[throw_away_vectors[i,:]])
        adjusted_labels[i,:] = np.insert(labels, throw_to_index_vec(
            throw_away_vectors[i,:]), -1)
    obs_in_every_iter_vec = np.sum(throw_away_vectors, axis=0) == n_iter
    obs_in_every_iter_labels = adjusted_labels[..., obs_in_every_iter_vec]
    n_every_iter = len(obs_in_every_iter_labels)
    neighbour_coeff_mat = np.zeros((n_every_iter, n_every_iter))   
    for i in xrange(n_every_iter):
        for j in xrange(n_every_iter):
            if i <= j:
                coeff = adjustedNeighbourCeoff(obs_in_every_iter_labels[i,...],
                                       obs_in_every_iter_labels[j,...])
                neighbour_coeff_mat[i,j] = coeff
    np.save(file_name, neighbour_coeff_mat)
    return neighbour_coeff_mat
    ### dorobic losowanie przypadkow!
    ##przechowywac labele z kolejnych klastrowan
def throw_to_index_vec(throw_away_vec):
    indices = np.where(np.logical_not(throw_away_vec))
    return indices[0] - np.arange(len(indices[0]))


if __name__ == "__main__":
    
   
    
    #throw_to_index_vec(np.array([True, False, False, True]))
    data_to_cluster = np. array([[1,1],
                                [0,0],
                                [2,2],
                                [10,10],
                                [9,9],
                                [11,10]])
    
    ward = myAgglomerativeClustering
    k_means = cluster.MiniBatchKMeans
    shapes = np.mean(original["shapes_n"], axis=2)
    #a = checkCrossStability(3, shapes, ward, "results/ward_3_nighbbour_mesh",
    #                        n_iter = 10, throw_away=20)
    
    a = checkCrossStability(20, shapes, ward, "results/ward_20_neighbour_mesh",
                            n_iter = 10, throw_away=20)
    b = checkCrossStability(20, shapes, k_means, "results/k_means_20_neighbour_mesh",
                            n_iter = 10, throw_away=20)
    
    print a
    print b
    
    #labels = k_means(n_clusters = 2).fit_predict(Data.gaussy2)
    ref_labels = np.concatenate((np.ones(500), np.zeros(500)))
    #a = np.random.randint(2, size = 1000)
    #print neighbourCoeffOfClustering(3, ward)
    #print adjustedNeighbourCeoffRatio(ref_labels, a)
    #print neighbourCoeffOfClustering(2, )
    
    #print checkStability(3, Data.gaussy2, ward, ref_labels)
    
    #import time
    #start = time.time()
    #a = checkReferenceStability(10, shapes, ward, original["kl1"], 10, lsm)
    #print time.time() - start
    #a = np.random.randint(8, size=5000)
    #b = np.random.randint(8, size=5000)
    #print neighbourCoeff(a, b)
    
    
    #print meanRandomNeighCoeff(5000, 8, 50)