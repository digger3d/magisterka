# -*- coding: utf-8 -*-
import numpy as np
from munkres import Munkres
from toy_problems import *

def splitData(data):
    if len(data) % 2:
        data = data[1:]
    idxs = np.zeros(data.shape[0])
    idxs[:len(data)/2] = 1
    np.random.shuffle(idxs)
    return data[idxs == 0,:], data[idxs == 1, :]

def createConfusionMatrix(labels1, labels2, n_clusters):
    conf_matrix = np.zeros((n_clusters, n_clusters))    
    for i in xrange(n_clusters):
        for j in xrange(n_clusters):
            set1 = set(np.where(labels1 == i)[0])
            set2 = set(np.where(labels2 == j)[0])
            conf_matrix[i, j] = len((set1 | set2) - (set1 & set2)) # choice
            #of cost function
    return conf_matrix

def findLabelMapping(labels1, labels2, n_clusters, munkres_obj):
    """label 2 is the "original" labelling
    returns list of tuples (a, b) a change to b to match the original"""
    conf_matrix = createConfusionMatrix(labels1, labels2, n_clusters)
    return munkres_obj.compute(conf_matrix)
    
def adjust_labels(mapping, labels):
    new_labels = np.zeros_like(labels)
    for i, j in mapping:
        new_labels[np.where(labels == i)] = j
    return new_labels

def accordLabels(primary_labels, secondary_labels, n_clusters, munkres_obj):
    mapping = findLabelMapping(secondary_labels, primary_labels,
                               n_clusters, munkres_obj)
    return adjust_labels(mapping, secondary_labels)
    
def computeDissimilarity(data, clust_algo, n_clust, munkres_obj):
    data1, data2 = splitData(data)
    clust_algo = clust_algo(n_clusters = n_clust)    
    _ = clust_algo.fit_predict(data1)
    predicted_labels2 = clust_algo.predict(data2)
    cluster_labels2 = clust_algo.fit_predict(data2)
    accorded_labels2 = accordLabels(predicted_labels2, cluster_labels2,
                                    n_clust, munkres_obj)
    return np.sum(predicted_labels2 != accorded_labels2) /\
        float(len(data1))

def computeRandomDissimilarity(len_half_data, n_clust):
    labels1 = np.random.randint(0, n_clust, len_half_data)
    labels2 = np.random.randint(0, n_clust, len_half_data)
    return np.sum(labels1 != labels2) / float(len_half_data)

def computeAverageRandomDissimilarity(data, n_clust, n_iter):
    half_len = len(data) / 2
    dissimilarities = np.zeros(n_iter)
    for i in xrange(n_iter):
        dissimilarities[i] = computeRandomDissimilarity(half_len, n_clust)
    return np.mean(dissimilarities)
    
#print computeAverageRandomDissimilarity(Data.gaussy2, 2, 100)

def computeDissimilarities(data, clust_algo,n_clust, munkres_obj, n_iter):
    dissimilarities = np.zeros(n_iter)
    for i in xrange(n_iter):
        dissimilarities[i] = computeDissimilarity(data, clust_algo,
                                                n_clust, munkres_obj)
    return dissimilarities
    
def computeAverageDissimilarity(data, clust_algo, n_clust, munkres_obj, n_iter):
    return np.mean(computeDissimilarities(data, clust_algo,
                           n_clust, munkres_obj, n_iter))

def adjustedAverageDissimilarity(data, clust_algo, n_clust, 
                                  munkres_obj, n_iter):
    avgDissimilarity = computeAverageDissimilarity(data, clust_algo,
        n_clust, munkres_obj, n_iter)
    avgRandDissimilarity = computeAverageRandomDissimilarity(data, 
        n_clust, n_iter)
    return avgDissimilarity / avgRandDissimilarity

def similarityMeasure(data, clust_algo, list_n_clust, 
                                  munkres_obj, n_iter):
    adj = np.zeros(len(list_n_clust))
    for i, n_clust in enumerate(list_n_clust):
        a = adjustedAverageDissimilarity(data, clust_algo,
            n_clust, munkres_obj, n_iter)
        adj[i] = a
    return adj
    
def plotSimilarityDiag(data, clust_algo, list_n_clust, munkres_obj, n_iter):
    similarity_indeces = similarityMeasure(data, clust_algo, list_n_clust, 
                                  munkres_obj, n_iter)
    clust_ticks = np.arange(len(list_n_clust)) + 1
    plt.plot(clust_ticks, similarity_indeces, "bo", markersize=10)
    plt.xlim([0, len(list_n_clust) + 1])    
    plt.title("Similarity Index (KMeans Algorithm)")   
    plt.xticks(clust_ticks, list_n_clust)
    plt.xlabel("No. of clusters")
    
m = Munkres()


#plotSimilarityDiag(Data.gaussy6, skl.cluster.MiniBatchKMeans,
#                        range(2,11), m, 50)

plotSimilarityDiag(Data.ov_gaussy4, skl.cluster.MiniBatchKMeans,
                        range(2,11), m, 50)


#male_gaussy2 = createGaussianData((5, 5, 10), (10, 10, 10))
#print computeDissimilarity(Data.gaussy2, skl.cluster.MiniBatchKMeans, 3, m)

#print computeDissimilarities(Data.gaussy2, skl.cluster.MiniBatchKMeans,
#                        3, m, 50)
#print computeAverageDissimilarity(Data.gaussy2, skl.cluster.MiniBatchKMeans,
#                        3, m, 50)
#print adjustedAverageDissimilarity(Data.gaussy2, skl.cluster.MiniBatchKMeans,
#                        3, m, 50)

#plotSimilarityMeasure(Data.ov_gaussy4_2, skl.cluster.MiniBatchKMeans,
#                        [2, 3, 4, 5, 6, 7, 8, 9, 10], m, 50)

#print diss
#labels = np.array([1,1,1,1,0,0,0,0])
#matrix = createConfusionMatrix(labels, labels[::-1], 2)
#matrix = createConfusionMatrix(labels, labels, 2)
#indexes = m.compute(matrix)
#print indexes


#mapping = findLabelMapping(labels, labels[::-1], 2, m)
#print labels
#print adjust_labels(mapping, labels)