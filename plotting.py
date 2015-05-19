# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plotMesh(meshgrid, title, xlabel, ylabel, xlabels, ylabels):
    plt.imshow(meshgrid, origin="lower", interpolation="nearest")
    plt.legend()
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y in range(meshgrid.shape[0]):
        for x in range(meshgrid.shape[1]):
            plt.text(x, y, "{:.3f}".format(meshgrid[y,x]),
                     horizontalalignment='center',
                     verticalalignment='center', color="white",
                     fontsize=13, weight="bold"
                     )
    plt.show()

def plotMesh2(meshgrid, title, xlabel, ylabel, xlabels, ylabels):
    """ """
    plt.pcolor(meshgrid)
    plt.xticks(np.arange(len(xlabels)) + 0.5, xlabels)
    plt.yticks(np.arange(len(ylabels)) + 0.5, ylabels)
    plt.colorbar()
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for y in range(meshgrid.shape[0]):
        for x in range(meshgrid.shape[1]):
            plt.text(x + 0.5, y + 0.5, "{:.3f}".format(meshgrid[y,x]),
                     horizontalalignment='center',
                     verticalalignment='center', color="white",
                     fontsize=13, weight="bold"
                     )
    plt.show()

NO_OF_PCA_COMP = 10
#PCA_STAB = np.load("stability_mesh.npz")
#plotMesh2(PCA_STAB["STABILITY_MESH"], "Stability index", "First n PCA components",
#         "No. of clusters", range(1,NO_OF_PCA_COMP + 1),
#         PCA_STAB["LIST_OF_CLUSTERS"])

#PCA_SIL = np.load("silhouette_mesh.npz")
#plotMesh2(PCA_SIL["SILHOUETTE_MESH"], "Silhouette index", "First n PCA components",
#         "No. of clusters", range(1,NO_OF_PCA_COMP + 1),
#         PCA_SIL["LIST_OF_CLUSTERS"])