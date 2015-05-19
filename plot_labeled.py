from plotting import plotMesh2
import numpy as  np

b = np.load("results/k_means_neighbour_mesh.npy")

plotMesh2(b, " ", " ", " ", [0,1], [0,1])