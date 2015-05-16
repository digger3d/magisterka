import numpy as np
from sklearn.cluster import AgglomerativeClustering 
from toy_problems import Data

class myAgglomerativeClustering(AgglomerativeClustering):
    def fit(self, X, y=None):
        super(myAgglomerativeClustering, self).fit(X,y)
        print "sialala"

a = myAgglomerativeClustering(n_clusters=2, linkage="ward")

a.fit(Data.gaussy2)
print a.labels_