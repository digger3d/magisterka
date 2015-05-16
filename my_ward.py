import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans
from collections import defaultdict
from toy_problems import Data

class myAgglomerativeClustering(AgglomerativeClustering):
    def fit(self, X, y=None):
        super(myAgglomerativeClustering, self).fit(X,y)
        ### determining centroids
        clusters = defaultdict(lambda: [np.zeros_like(X[1]),0])
        for i, observation in enumerate(X):
            clusters[self.labels_[i]][0] += observation
            clusters[self.labels_[i]][1] += 1
        self.centroids_ = np.zeros(tuple((self.n_clusters, ) + X[1].shape))
        for clustLabel, [elementSum, n_elements] in clusters.items():
            self.centroids_[clustLabel,:] = elementSum / float(n_elements)
        
    
    def predict(self, X):
        """Sadfasd saf.
        
        Parameters
        ----------
        X : asf saf.
        
        Returns
        -------
        labels : predicted labels on a new set.
        """
        labels = np.zeros(len(X), dtype="uint8")
        for i, observation in enumerate(X):            
            distances = np.sum((self.centroids_ - observation) ** 2, axis=1)
            labels[i], _ = min(enumerate(distances), key=lambda x: x[1])
        return labels
#            print distances
        # produce labels according to centroids
        
#a = np.array([[1,2], [3,4]])
#b = [[1, [1,2]], [3, [4,5]]]
#for x, y, z in b:
#    print x, y, z

#a = myAgglomerativeClustering(n_clusters=2, linkage="ward")
#a.fit(Data.gaussy2)
#a.predict(np.array([[5,5], [10,10], [6,6], [1,1], [11,11]]))
#print a.predict(Data.gaussy2)
#print a.labels_

#b = MiniBatchKMeans(n_clusters = 3)