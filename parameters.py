#coding: utf8
import numpy as np
#from my_ward import myAgglomerativeClustering
#from sklearn import cluster
import matplotlib.pylab as plt

##data
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
#ward = myAgglomerativeClustering
#k_means = cluster.MiniBatchKMeans
def splitLogData(log_data, non_log_cols=[2, 10]):
    return np.delete(log_data, non_log_cols, 1)
    
def delogData(log_data, non_log_cols=[2, 10]):
    deloged = np.exp(splitLogData(log_data, non_log_cols))
    deloged = np.insert(deloged, 2, log_data[:,2], 1)
    deloged = np.insert(deloged, 10, log_data[:,10], 1)
    return deloged

def identity(x):
    return x

def getScaleFactors(deloged_data):
    max_val = max(deloged_data[:,0])
    return max_val / deloged_data[:,0]

def adjust_scale(scale, adjust_vec=[lambda x: x, np.sqrt, np.sqrt, np.sqrt,
                                    np.sqrt, np.sqrt, np.sqrt, lambda x: 1,
                                    np.sqrt, lambda x : 1./np.sqrt(x),
                                    lambda x: 1]):
    adjusted_scales = np.zeros_like(adjust_vec)
    for i, func in enumerate(adjust_vec):
        adjusted_scales[i] = func(scale)
    return adjusted_scales

def adjusted_scales_to_array(adj_scales, shp=(9278, 11)):
    arr = np.zeros(shp)
    for i, col in enumerate(adj_scales):
        arr[:,i] = col
    return arr

def normalise_data(deloged_data):
    adjusted_scales = adjust_scale(getScaleFactors(deloged_data))
    adjusted_scales_arr = adjusted_scales_to_array(adjusted_scales)
    return deloged_data * adjusted_scales_arr
        
#def normaliseRow(scale, row):
#    normed_row = np.zeros_like(row)
#    for i in xrange(len(row)):
#        

#def normaliseData(deloged_data, scale_vec, coeffs=np.array([1]) )
    
##loading parametrised data
par_file = np.load("datasets/morphological_parameters.npz")
par_data_log = par_file["data"]
par_data = delogData(par_data_log)
par_data_normed = normalise_data(par_data)
print par_data_normed
