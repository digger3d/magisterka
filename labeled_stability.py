import numpy as np


def createNeighbourMatrix(labels_vec):
    """NeighMatrix[i,j] == 1 iff labels_vec[i] == labels_vec[j]
    Skipping the lower part of matrix because it is symetric
    Maximum 256 clusters"""
    n = len(labels_vec)
    neighMatrix = np.zeros((n, n), dtype="uint8")
    for i in xrange(n):
        neighMatrix[i,i:] = labels_vec[i:] == labels_vec[i]
    return neighMatrix
    
def neighbourCoeff(labels1, labels2):
    """label1 and 2 are same length vectors"""
    n = len(labels1)
    neighMat1 = createNeighbourMatrix(labels1)
    neighMat2 = createNeighbourMatrix(labels2)
    return np.sum(neighMat1 * neighMat2)# / float(n)

def meanRandomNeighCoeff(n, n_clusters, n_iter):
    coeffs = np.zeros(n_iter)
    for i in xrange(n_iter):
        a = np.random.randint(n_clusters, size=n)
        b = np.random.randint(n_clusters, size=n)
        coeffs[i] = neighbourCoeff(a, b)
    return np.mean(coeffs)
    
a = np.random.randint(8, size=5000)
b = np.random.randint(8, size=5000)
print neighbourCoeff(a, b)

#print meanRandomNeighCoeff(5000, 8, 50)