import scipy.cluster.hierarchy as sch

# execfile("spines_example.py")

def labels_to_dict(labels):
    return dict([(key, [i for i, x in enumerate(labels) if x == key]) 
                 for key in set(labels)])

s_lines = data['shapes_n'].sum(axis=1).astype('float')
shapes_n = np.array(data['shapes_n'])

# # Ward hierarchical
clust_res = sch.ward(s_lines)

nclus = 36
clust_labels = sch.fcluster(clust_res, nclus, criterion='maxclust')
clust_labels_dict = labels_to_dict(clust_labels)

plt.figure()
plt.set_cmap('hot_r')

ims = np.zeros((nclus, shapes_n.shape[1], shapes_n.shape[2]))
for nn in range(1, nclus + 1):
    plt.subplot(6, 6, nn)
    im = shapes_n[clust_labels_dict[nn], ...].mean(axis=0)
    ims[nn-1, :] = im
    plt.imshow(im, vmin=0, vmax=255)
    plt.yticks([])
    plt.xticks([])
    plt.title(str(nn), va='top')
