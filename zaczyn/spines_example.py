import numpy as np
import matplotlib.pyplot as plt

data = np.load('spines.npz')
print data.keys()
print data['shapes'].shape

plt.figure()
plt.subplot(131)
plt.imshow(data['shapes'][0])
plt.subplot(132)
plt.imshow(data['shapes_n'][0])
plt.subplot(133)
plt.plot(np.mean(data['shapes_n'][0], axis=1), np.arange(64))
plt.ylim([0,63])
plt.gca().invert_yaxis()
plt.suptitle(data['mice'][0])

plt.show()