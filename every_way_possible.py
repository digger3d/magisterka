from new_stability import *
from semisupervised import *
from os.path import join
from sys import exit
import numpy as np
import matplotlib.pylab as plt


def gatherData3Way(everyPath, algo, index, no):
    path = join(everyPath,algo,index)
    x = np.load(path)
    return x[no]

def gatherData1Way(everyPath, algo, kernel, index):
    path = join(everyPath,algo,kernel,index)
    x = np.load(path)
    return x[0]

def gatherAllData(everyPath, index):
    Data = []
    for i in xrange(3):
        datum = gatherData3Way(everyPath, "k_means", index, i)
        Data.append(datum)
    for i in xrange(3):
        datum = gatherData3Way(everyPath, "ward", index, i)
        Data.append(datum)
    datum_prop_rbf = gatherData1Way(everyPath, "propagation", "rbf", index)
    datum_prop_knn = gatherData1Way(everyPath, "propagation", "knn", index)
    datum_spread_rbf = gatherData1Way(everyPath, "spreading", "rbf", index)
    datum_spread_knn = gatherData1Way(everyPath, "spreading", "knn", index)
    return np.array(Data +\
        [datum_prop_rbf, datum_prop_knn, datum_spread_rbf, datum_spread_knn])

def plotAllData(data, title):
    plt.figure()
    plt.title(title)    
    n = len(data)
    for i, data_vec in enumerate(data):
        plot1Clustering(data_vec, i, "")
        plt.xticks(range(n), ["k-means3", "k-means32", "k-means100",
                              "ward3", "ward32", "ward100",
                              "propagation-rbf","propagation-knn",
                              "spreading-rbf","spreading-knn"], rotation = 70)
    plt.show()

def plotAllDataBox(data, title):
    plt.figure()
    n = len(data)
    plt.boxplot(data.T)
    plt.title(title)
    plt.xticks(range(1, n + 1), ["k-means3", "k-means32", "k-means100",
                          "ward3", "ward32", "ward100",
                          "propagation-rbf","propagation-knn",
                          "spreading-rbf","spreading-knn"], rotation = 70)
    plt.show()
#
#a = gatherAllData("results/every_way_possible/", "accord.npy")
#plotAllDataBox(a, "Accordance Index")
#    
#a = gatherAllData("results/every_way_possible/", "neigh.npy")
#plotAllDataBox(a, "Neighbourhood Index")

a = gatherAllData("results/every_way_possible/", "accord.npy")
plotAllData(a, "Accordance Index")
    
a = gatherAllData("results/every_way_possible/", "neigh.npy")
plotAllData(a, "Neighbourhood Index")

    
def plot1Clustering(val_vec, placement, label):
    x = np.ones(len(val_vec)) * placement
    plt.scatter(x, val_vec)
    plt.xlabel(label)


ward = myAgglomerativeClustering
k_means = cluster.MiniBatchKMeans
N_ITER=25
DATA_TO_CLUSTER = shapes
##DATA_TO_CLUSTER = par_data_normed
#
#g, h = inspectLearnCases(shapes, LabelSpreading, [100],
#    "results/every_way_possible/spreading/knn/", "knn",  N_ITER)
#g, h = inspectLearnCases(shapes, LabelSpreading, [100],
#    "results/every_way_possible/spreading/rbf/", "rbf",  N_ITER)  
#    
##TODO
#e, f = inspectLearnCases(shapes, LabelPropagation, [100],
#    "results/every_way_possible/propagation/rbf/", "rbf", N_ITER)
#e, f = inspectLearnCases(shapes, LabelPropagation, [100],
#    "results/every_way_possible/propagation/knn/", "knn", N_ITER)
#
#d, e = multiCrossStability([3, 32, 100], shapes,
#                           ward, "results/every_way_possible/ward/", n_iter = N_ITER)
#
#d, e = multiCrossStability([3, 32, 100], shapes,
#                           k_means, "results/every_way_possible/k_means/", n_iter = N_ITER)
