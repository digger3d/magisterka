# coding: utf-8
#from toy_problems import Data

import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from new_stability import *
GAMMA = 3e-6

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
    
def getLearningAndThrowAwayInd(len_learn, len_throw_away, learnable_obs = lsm):
    learnable_indices = getIndicesOfTrue(learnable_obs)
    all_indices = range(len(learnable_obs))
    learn_indices = np.random.choice(learnable_indices, size=len_learn)
#    learnable_indices = learnable_indices - set(learn_indices)
    all_indices = np.delete(all_indices, learn_indices)
    throw_away_indices = np.random.choice(all_indices, size=len_throw_away)
    return learn_indices, throw_away_indices

LABEL_DICT = {u"Long & thin" : 0,
              u"Mushroom" : 1,
              u"Stubby": 2}
TEXT_DICT = {0 : u"Long & thin",
             1 : u"Mushroom",
             2 :  u"Stubby"}
              
N_OBS = len(original["kl1"])
def translateTextLabels(labels, dictionary=LABEL_DICT):
    new_labels = np.zeros(len(labels), dtype="int")
    for i, label in enumerate(labels):
        try:
            new_labels[i] = dictionary[label]
        except KeyError:
            new_labels[i] = label
    return new_labels

def translateNumericLabels(labels, dictionary):
    new_labels = np.zeros(len(labels), dtype=object)
    for i, label in enumerate(labels):
        try:            
            new_labels[i] = dictionary[label]
        except KeyError:
            new_labels[i] = label
    return new_labels

def prepareLabels(labels, len_learn, len_throw_away):
    learn_vec, throw_away_vec = getLearningAndThrowAwayInd(len_learn,
        len_throw_away, learnable_obs = lsm)
    mask = np.ones(len(labels), dtype="bool")
    mask[learn_vec] = False
    new_labels = labels[:]
    new_labels[mask] = -1
    return translateTextLabels(new_labels)[np.logical_not(fromIdx2Mask(throw_away_vec))],\
        np.logical_not(fromIdx2Mask(throw_away_vec))
    
def fromIdx2Mask(indices, len_of_mask=N_OBS):
    mask = np.zeros(len_of_mask, dtype='bool')
    mask[indices] = True
    return mask
#a = prepareLabels(original["kl2"], 3000, 20)
#print set(a)
#print len(a), N_OBS

from collections import Counter
def checkCrossStabilitySemi(data_to_cluster, clust_algo, knn = 15, kernel="-",
                            learn_cases=100, n_iter = 5,  original_data = original,
                            gamma = GAMMA, throw_away = 20, selection_vec = lsm):
    """selection vector is applied after clustering, and throw away
    vector before"""
    clust_algo_instance = clust_algo(kernel=kernel, gamma=gamma, n_neighbors=knn)
    adjusted_labels = np.empty((n_iter, len(data_to_cluster)), dtype=object)
    throw_away_vectors = np.zeros((n_iter, len(data_to_cluster)), dtype="bool")
    for i in range(n_iter):
        prepared_labels, throw_away_vec = prepareLabels(original_data["kl1"],
            learn_cases, throw_away)
        throw_away_vectors[i,:] = throw_away_vec
        ###
        clust_algo_instance.fit(data_to_cluster[throw_away_vectors[i,:]],
                                prepared_labels)
        labels = clust_algo_instance.predict(data_to_cluster[throw_away_vectors[i,:]])        
        adjusted_labeling = np.insert(labels, throw_to_index_vec(
            throw_away_vectors[i,:]), -1) ## sprawdzic czy mozna insertowac stringi 
        text_labeling = translateNumericLabels(adjusted_labeling, TEXT_DICT)
        adjusted_labels[i,:] = text_labeling
    obs_in_every_iter_vec = (np.sum(throw_away_vectors, axis=0) == n_iter) &\
        selection_vec
    obs_in_every_iter_labels = adjusted_labels[..., obs_in_every_iter_vec]
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

def inspectKNeighbour(data_to_cluster, algo, list_of_knn, path, kernel, n_iter, learn_cases=100):
    data_file = open(path + "data.txt", "w")
    data_file.write("algorithm\t{0}\n"
    "list_of_knns\t{1}\n"
    "n_iter\t{2}\n"
    "learn_cases\t{3}".format(str(algo), list_of_knn, n_iter, learn_cases))    
    accordance_indices = []    
    neighbour_coeffs = []
    for i, knn in enumerate(list_of_knn):
        print path, i, knn
        accordance, neighbour = checkCrossStabilitySemi(data_to_cluster, algo, kernel=kernel,
            knn=knn, n_iter=n_iter)
        neighbour_coeffs.append(neighbour)
        accordance_indices.append(accordance)
    accordance_indices = np.array(accordance_indices)
    neighbour_coeffs = np.array(neighbour_coeffs)
    np.save(path + "accord.npy", accordance_indices)
    np.save(path + "neigh.npy", neighbour_coeffs)
    print path, "finished!"
    return np.array(accordance_indices), np.array(neighbour_coeffs)

def inspectGamma(data_to_cluster, algo, list_of_n_learn, path, kernel, n_iter):
    data_file = open(path + "data.txt", "w")
    data_file.write("algorithm\t{0}\n"
    "list_of_learn_cases\t{1}\n"
    "n_iter\t{2}\n".format(str(algo), list_of_n_learn, n_iter))    
    accordance_indices = []    
    neighbour_coeffs = []
    for i, n_learn_cases in enumerate(list_of_n_learn):
        print path, i, n_learn_cases
        accordance, neighbour = checkCrossStabilitySemi(data_to_cluster, algo, kernel=kernel,
            gamma=n_learn_cases, n_iter=n_iter)
        neighbour_coeffs.append(neighbour)
        accordance_indices.append(accordance)
    accordance_indices = np.array(accordance_indices)
    neighbour_coeffs = np.array(neighbour_coeffs)
    np.save(path + "accord.npy", accordance_indices)
    np.save(path + "neigh.npy", neighbour_coeffs)
    print path, "finished!"
    return np.array(accordance_indices), np.array(neighbour_coeffs)
    

def inspectLearnCases(data_to_cluster, algo, list_of_n_learn, path, kernel, n_iter):
    data_file = open(path + "data.txt", "w")
    data_file.write("algorithm\t{0}\n"
    "list_of_gammas\t{1}\n"
    "n_iter\t{2}\n"
    "kernel\t{3}".format(str(algo), list_of_n_learn, n_iter, kernel))    
    accordance_indices = []    
    neighbour_coeffs = []
    for i, n_learn_cases in enumerate(list_of_n_learn):
        print path, i, n_learn_cases
        accordance, neighbour = checkCrossStabilitySemi(data_to_cluster, algo, kernel=kernel,
            learn_cases=n_learn_cases, n_iter=n_iter)
        neighbour_coeffs.append(neighbour)
        accordance_indices.append(accordance)
    accordance_indices = np.array(accordance_indices)
    neighbour_coeffs = np.array(neighbour_coeffs)
    np.save(path + "accord.npy", accordance_indices)
    np.save(path + "neigh.npy", neighbour_coeffs)
    print path, "finished!"
    return np.array(accordance_indices), np.array(neighbour_coeffs)

def plotFromPath(path, npy_file, title, ylabel, xlabel, ylim=[0,1]):
#    trans_dict = dict([("accord.npy", "Ratio of properly classified spines"),
#                       ("neigh.npy", "Nieghbourhood Index")])
    data_file = open(pathJoin(path, "data.txt"))
    matrix = np.load(pathJoin(path, npy_file))
    means = np.mean(matrix, axis = 1)
    errors = np.std(matrix, axis = 1)
    data_dict = {}
    for i, line in enumerate(data_file):
        if i == 1:
            _, ticks = line.rstrip("\n").split("\t")
        parameter, value = line.rstrip("\n").split("\t")
        data_dict[parameter] = value
    plt.errorbar(np.arange(len(matrix)), means, yerr=errors)
    plt.title(title)
    plt.xlim(-1, len(matrix) + 1)
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(np.arange(len(matrix)), eval(ticks))

##### Plotting
#
#plt.subplot(221)
#plotFromPath("results/semi_labeled/propagation/rbf/gamma/", "accord.npy",\
#"Algorytm Label Propagation", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Wartość parametru gammma")
#plt.subplot(222)
#plotFromPath("results/semi_labeled/propagation/rbf/gamma/", "neigh.npy",\
#"Algorytm Label Propagation", u"Współczynnik Przystawania",\
#u"Wartość parametru gammma")
#plt.subplot(223)
#plotFromPath("results/semi_labeled/spreading/rbf/gamma/", "accord.npy",\
#"Algorytm Label Spreading", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Wartość parametru gammma")
#plt.subplot(224)
#plotFromPath("results/semi_labeled/spreading/rbf/gamma/", "neigh.npy",\
#"Algorytm Label Spreading", u"Współczynnik Przystawania",\
#u"Wartość parametru gammma")
#
#plt.subplot(221)
#plotFromPath("results/semi_labeled/propagation/knn/knns/", "accord.npy",\
#"Algorytm Label Propagation", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Ilość najbliższych sąsiadów")
#plt.subplot(222)
#plotFromPath("results/semi_labeled/propagation/knn/knns/", "neigh.npy",\
#"Algorytm Label Propagation", u"Współczynnik Przystawania",\
#u"Ilość najbliższych sąsiadów")
#plt.subplot(223)
#plotFromPath("results/semi_labeled/spreading/knn/knns/", "accord.npy",\
#"Algorytm Label Spreading", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Ilość najbliższych sąsiadów")
#plt.subplot(224)
#plotFromPath("results/semi_labeled/spreading/knn/knns/", "neigh.npy",\
#"Algorytm Label Spreading", u"Współczynnik Przystawania",\
#u"Ilość najbliższych sąsiadów")

#plt.subplot(221)
#plotFromPath("results/semi_labeled/propagation/rbf/learn_cases/", "accord.npy",\
#"Algorytm Label Propagation", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Wielkość zbioru uczącego")
#plt.subplot(222)
#plotFromPath("results/semi_labeled/propagation/rbf/learn_cases/", "neigh.npy",\
#"Algorytm Label Propagation", u"Współczynnik Przystawania",\
#u"Wielkość zbioru uczącego")
#plt.subplot(223)
#plotFromPath("results/semi_labeled/spreading/rbf/learn_cases/", "accord.npy",\
#"Algorytm Label Spreading", u"Ułamek dobrze zaklasyfikowanych przypadków",\
#u"Wielkość zbioru uczącego")
#plt.subplot(224)
#plotFromPath("results/semi_labeled/spreading/rbf/learn_cases/", "neigh.npy",\
#"Algorytm Label Spreading", u"Współczynnik Przystawania",\
#u"Wielkość zbioru uczącego")

#plotFromPath("results/semi_labeled/spreading/knn/learn_cases/", "accord.npy","Gamma Value", "knn")

### rbf kernel


##TODO zmienic gamme w checStability!!!!
#
#gammas = list(np.arange(1,11) * 1e-6)
#i, j = inspectGamma(shapes, LabelSpreading, gammas,
#    "results/semi_labeled/spreading/rbf/gamma/", "rbf",20)
##
#k, l = inspectGamma(shapes, LabelPropagation, gammas,
#    "results/semi_labeled/propagation/rbf/gamma/", "rbf", 20)

##### knn kernel

#a, b = inspectKNeighbour(shapes, LabelPropagation, [1, 2, 3, 5, 8, 12, 15, 20, 50],
#    "results/semi_labeled/propagation/knn/knns/", "knn", 20)
#c, d = inspectKNeighbour(shapes, LabelSpreading, [1, 2, 3, 5, 8, 12, 15, 20, 50],
#    "results/semi_labeled/spreading/knn/knns/", "knn", 20)
#    
#e, f = inspectLearnCases(shapes, LabelPropagation, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/propagation/knn/learn_cases/", "knn", 20)
#
#g, h = inspectLearnCases(shapes, LabelSpreading, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/spreading/knn/learn_cases/", "knn",  20)  
    
##TODO
#e, f = inspectLearnCases(shapes, LabelPropagation, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/propagation/rbf/learn_cases/", "rbf", 25)
#
#g, h = inspectLearnCases(shapes, LabelSpreading, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/spreading/rbf/learn_cases/", "rbf",  25)      
#    
#m, n = inspectLearnCases(shapes, LabelPropagation, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/propagation/rbf/learn_cases/", "rbf", 20)
#
#o, u = inspectLearnCases(shapes, LabelSpreading, [10, 20, 30, 50, 80, 100, 200, 500, 1000],
#    "results/semi_labeled/spreading/rbf/learn_cases/", "rbf",  20)  

#i, j = inspectGamma(shapes, LabelSpreading, gammas,
#    "results/semi_labeled/spreading/gamma/", 5)
    
#print "\n\n\n"
#k, l = inspectGamma(shapes, LabelPropagation, gammas,
#    "results/semi_labeled/propagation/gamma/", 5)

#g = 1e-6
#import time
#start = time.time()
#a, b = checkCrossStabilitySemi(shapes, LabelSpreading, 100, 5)
#
#print time.time() - start

#a, x1 = prepareLabels(original["kl1"], 100, 0)
#b, x2 = prepareLabels(original["kl1"], 100, 0)
#alg = LabelPropagation(kernel="rbf")
#alg.fit(shapes, a)
#lab1 = alg.predict(shapes)
#alg.fit(shapes, b)
#lab2 = alg.predict(shapes)
#
#print sum(lab1 != lab2)
#from collections import Counter
#
#print Counter(lab1).most_common()

#
#print sum(lab1 != lab2)
#
    
    
#    
#X = Data.gaussy2
##y = ["Kotek"] * 500 + ["Mamrotek"] * 500
#y = np.concatenate((np.zeros(500,dtype="int"), np.ones(500,dtype="int")))
#y_semi = np.array(y[:])
#a, b = getLearningAndThrowAway(100, 20, y)
#y_semi[a] = -1
#a = LabelPropagation()
#a.fit(X,y_semi)
