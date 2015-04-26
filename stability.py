# -*- coding: utf-8 -*-
import numpy as np
from munkres import Munkres

def splitData(data):
    idxs = np.zeros_like(data)
    idxs[:len(data)/2] = 1
    np.random.shuffle(idxs)
    return data[idxs == 0], data[idxs == 1]

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
    returns list of tuples (a, b) a change to be to match the original"""
    conf_matrix = createConfusionMatrix(labels1, labels2, n_clusters)
    return munkres_obj.compute(conf_matrix)
    
def adjust_labels(mapping, labels):
    new_labels = np.zeros_like(labels)
    for i, j in mapping:
        new_labels[np.where(labels == i)] = j
    return new_labels
    
    
labels = np.array([1,1,1,1,0,0,0,0])
#matrix = createConfusionMatrix(labels, labels[::-1], 2)
#matrix = createConfusionMatrix(labels, labels, 2)
#indexes = m.compute(matrix)
#print indexes

#m = Munkres()
#mapping = findLabelMapping(labels, labels[::-1], 2, m)
#print labels
#print adjust_labels(mapping, labels)