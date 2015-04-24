# -*- coding: utf-8 -*-

import numpy as np

def makeDataSmaller(source_data, new_data_name, n = 100):
    data = np.load(source_data)
    new_shapes = data["shapes"][0:n]
    new_shapes_n = data["shapes_n"][0:n]
    new_mice = data["mice"][0:n]
#    new_data = dictionary{"shapes" : new_shapes,
#                          "shapes_n" : new_shapes_n,
#                          "mice" : new_mice}
    np.savez(new_data_name + str(n),
             shapes = new_shapes,
             shapes_n = new_shapes_n,
             mice = new_mice)
    
if __name__ == "__main__":
    makeDataSmaller("spines.npz", "smaller_")
    x = np.load("smaller_100.npz")
    print x.keys()
