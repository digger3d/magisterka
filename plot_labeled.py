from plotting import plotMesh2
import numpy as  np
import matplotlib.pylab as plt

def plotAccordance(file_name, title):
    f = open(file_name, "r")
    n_clusters = []
    accordances = []
    for line in f:
        n, accord = line.strip("\n").split("\t")
        n_clusters.append(n)
        accordances.append(float(accord))
    plt.plot(accordances, "bo", markersize=10)
    plt.title(title)
    plt.xlabel("No. of clusters")
    plt.ylabel("Accordance Index")
    plt.xticks(np.arange(len(n_clusters)), n_clusters)
    plt.xlim((-1, len(n_clusters) + 1))

k_means_3_mesh = np.load("results/k_means_3_neighbour_mesh.npy")
k_means_10_mesh = np.load("results/k_means_10_neighbour_mesh.npy")
k_means_20_mesh = np.load("results/k_means_20_neighbour_mesh.npy")
ward_3_mesh = np.load("results/ward_3_neighbour_mesh.npy")
ward_10_mesh = np.load("results/ward_10_neighbour_mesh.npy")
ward_20_mesh = np.load("results/ward_20_neighbour_mesh.npy")
#plotMesh2(k_means_3_mesh, "Neighbour Coefficient K-Means Algorithm for 3 clusters",
#          "Cluster 1 no.", "Cluster 2 no. ", range(1,11), range(1,11))
#plotMesh2(k_means_10_mesh, "Neighbour Coefficient K-Means Algorithm for 10 clusters",
#          "Clustering 1 no.", "Clustering 2 no.", range(1,11), range(1,11))
#plotMesh2(k_means_20_mesh, "Neighbour Coefficient K-Means Algorithm for 20 clusters",
#          "Clustering 1 no.", "Clustering 2 no.", range(1,11), range(1,11))
#plotMesh2(ward_3_mesh, "Neighbour Coefficient Ward Algorithm for 3 clusters",
#          "Clustering 1 no.", "Clustering 2 no. ", range(1,11), range(1,11))
#plotMesh2(ward_10_mesh, "Neighbour Coefficient Ward Algorithm for 10 clusters",
#          "Clustering 1 no.", "Clustering 2 no. ", range(1,11), range(1,11))
plotMesh2(ward_20_mesh, "Neighbour Coefficient Ward Algorithm for 20 clusters",
          "Clustering 1 no.", "Clustering 2 no. ", range(1,11), range(1,11))

#plotAccordance("results/kmeans/k_means_accord.txt", "K-Means Algorithm")
#plt.figure()
#plotAccordance("results/ward/ward_accord.txt", "Ward Algorithm")