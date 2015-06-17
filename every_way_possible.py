from new_stability import *
from semisupervised import *
from os.path import join
from sys import exit
import numpy as np
import matplotlib.pylab as plt


ward = myAgglomerativeClustering
k_means = cluster.MiniBatchKMeans


N_ITER=25
DATA_TO_CLUSTER = shapes

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

def plotAllData(data):
    plt.figure()
    for i, data_vec in enumerate(data):
        plot1Clust(data_vec, i, "")
    plt.show()
    
a = gatherAllData("results/every_way_possible/", "accord.npy")
print a.shape

for el in a:
    print len(el)
exit()
print gatherData1Way("results/every_way_possible/", "spreading", "rbf", "accord.npy") 

def plot1Clustering(val_vec, placement, label):
    x = np.ones(len(val_vec)) * placement
    plt.scatter(x, val_vec)
    plt.xlabel(label)
    
def plotEveryClustering(path, algo, index):
    pass
a = gatherData3Way("results/every_way_possible/", "k_means", "/accord.npy", 2)
plot1Clustering(a, 1, "dupa")
#def plotEveryWay(path)
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

#d, e = multiCrossStability([3, 32, 100], shapes,
#                           k_means, "results/every_way_possible/k_means/", n_iter = N_ITER)
