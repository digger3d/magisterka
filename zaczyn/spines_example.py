import numpy as np
import matplotlib.pyplot as plt

data = np.load('spines.npz')
print data.keys()
print data['shapes'].shape

plt.figure()
plt.subplot(121)
plt.imshow(data['shapes'][0])
plt.subplot(122)
plt.imshow(data['shapes_n'][0])
plt.suptitle(data['mice'][0])

plt.show()