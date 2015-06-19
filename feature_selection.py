# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import sklearn.metrics
import sklearn.cluster
import stability
#import toy_problems as tp
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



all_data = np.load("./zaczyn/spines.npz")
all_norm = all_data["shapes_n"]
all_norm = all_norm.reshape(all_norm.shape[0], -1)
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


#pca fitting
pca = PCA(n_components = 10)
pca.fit(all_norm)
#np.save("PCAcomps.npy", pca.components_)


def transformPCA(data, components):
    """ components in n_comp x features format"""
    return np.dot(data, components.T)

def SilhouettePCARow(data, components, clustering_algorithm, n_clust):
    sils = []
    for i in xrange(len(components)):
        selected_comp = components[:i+1,:]
        transformed_data = transformPCA(data, selected_comp)
        algo = clustering_algorithm(n_clusters = n_clust)
        labels = algo.fit_predict(transformed_data)
        sils.append(sklearn.metrics.silhouette_score(data, labels, sample_size=2000))
    return sils
    
def silhouettePCAMesh(data, components, clustering_algorithm, n_clust_list):
    """returnS mesh no_of_clusters x no_of_components"""
    sil_indexes = np.zeros((len(n_clust_list), len(components)))
    for row, n_clust in enumerate(n_clust_list):
        print"Computing indices for {}. cluster no.".format(row+1)
        sil_indexes[row,:] = SilhouettePCARow(data, components, 
            clustering_algorithm, n_clust)
    return sil_indexes
    
def stabilityPCAMesh(data, components, clust_algo, list_n_clust, 
                                  munkres_obj, n_iter):
    """returns mesh no_of_clusters x no_of_components"""
    stabilityMesh = np.zeros((len(components), len(list_n_clust)))
    for i in xrange(len(components)):
        print "Computing {} component".format(i)
        selected_comps = components[:i+1,:]
        transformed_data = transformPCA(data, selected_comps) #mozna lekko 
        # przyspieszyc doklejajac po jednej kolumnie do transformed data
        stabilityMesh[i,:] = stability.similarityMeasure(transformed_data,
            clust_algo, list_n_clust, munkres_obj, n_iter)
    return stabilityMesh.T

##GLOBALS
PCA_COMP_10 = np.load("PCAcomps.npy")
M = stability.Munkres()
KMEANS = sklearn.cluster.MiniBatchKMeans
LIST_OF_CLUSTERS = [4, 10, 15, 30, 50]

def PCAReduction(vectorized_obs, pca_comps):
    return transformPCA(vectorized_obs, pca_comps)

def plotPCAScatter(vectorized_obs, pca_comps, longs, stubbies, mushrooms, accorded):
    groups = [longs, stubbies, mushrooms]
    labels = [u"Długie", "Przysadziste", "Grzybkowate"]
    reduced = transformPCA(vectorized_obs, pca_comps)
    plt.figure()    
    colors = "rbg"
#    plt.title(u"Redukcja wymiarowości z użyciem analizy głównych składowych",
#              fontsize=16)
    for i, group in enumerate(groups):
        obs_to_plot = reduced[group & accorded, ...]
        plt.scatter(obs_to_plot[...,0], obs_to_plot[...,1],
                    label=labels[i], color=colors[i])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel(u"Składowa 1.", fontsize=13)
    plt.ylabel(u"Składowa 2.", fontsize=13)
    plt.legend(scatterpoints=1)
    plt.show()
    
def plotPCAScatterAll(vectorized_obs, pca_comps, longs, stubbies, mushrooms, lsm):
    accorded = np.ones(len(longs), dtype="bool")
    uncategorised = accorded & np.logical_not(lsm)
    groups = [longs, stubbies, mushrooms, uncategorised]
    labels = [u"Długie", "Przysadziste", "Grzybkowate", "Nieopisane"]
    reduced = transformPCA(vectorized_obs, pca_comps)
    plt.figure()
    colors = "rbgk"
#    plt.title(u"Redukcja wymiarowości z użyciem analizy głównych składowych",
#              fontsize=16)
    for i, group in enumerate(groups):
        obs_to_plot = reduced[group & accorded, ...]
        plt.scatter(obs_to_plot[...,0], obs_to_plot[...,1],
                    label=labels[i], color=colors[i])
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])
    plt.xlabel(u"Składowa 1.", fontsize=13)
    plt.ylabel(u"Składowa 2.", fontsize=13)
    plt.legend(scatterpoints=1)
    plt.show()
        
#plotPCAScatter(all_norm, PCA_COMP_10[:2,...], longs, stubbies, mushrooms, lsm)
plotPCAScatterAll(all_norm, PCA_COMP_10[:2,...], longs, stubbies, mushrooms, lsm)
##transforming data for plotting 2D
#X = transformPCA(all_norm, PCA_COMP_10[:2,:])
#plt.scatter(X[:,0], X[:,1])
#plt.title("PCA transform for first two components")
#plt.xlabel("Component 1")
#plt.ylabel("component 2")
#plt.xticks([])
#plt.yticks([])
#plt.show()

##3d
#X = transformPCA(all_norm, PCA_COMP_10[:3,:])
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#x = X[:,0]
#y = X[:,1]
#z = X[:,2]
#ax.scatter(x, y, z, c='b', marker='o')
#plt.show()

#meshing
#SILHOUETTE_MESH = silhouettePCAMesh(all_norm, PCA_COMP_10,
#                                 KMEANS, LIST_OF_CLUSTERS)
#np.savez("silhouette_mesh.npz", SILHOUETTE_MESH=SILHOUETTE_MESH,
#         LIST_OF_CLUSTERS=LIST_OF_CLUSTERS)
#STABILITY_MESH = stabilityPCAMesh(all_norm, PCA_COMP_10, KMEANS,
#                                  LIST_OF_CLUSTERS, M, 10)
#np.savez("stability_mesh.npz", STABILITY_MESH=STABILITY_MESH,
#         LIST_OF_CLUSTERS=LIST_OF_CLUSTERS)    
#plotMesh(a, ["dupa", "trupa"], ["a", "b"])

#def averageDissimilarityPCAMesh():
#    passcomputeAverageDissimilarity(data, clust_algo, n_clust, munkres_obj, n_iter):

#a = SilhouettePCAMesh(all_norm, pca.components_[:10,:], KMEANS, [5,10,20,40])
#all_shape_extr = np.sum(all_norm,axis=2)
#all_norm = all_norm.reshape(all_norm.shape[0], -1)