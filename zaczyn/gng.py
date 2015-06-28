import mdp
import networkx
from sklearn.decomposition import PCA
import numpy as np
s_lines = np.mean(np.load("../datasets/spines2.npz")["shapes_n"], axis=2)
gng = mdp.nodes.GrowingNeuralGasNode(max_nodes=160)
gng.train(s_lines)
gng.train(s_lines)
gng.stop_training()

gg = gng.graph

G = networkx.Graph()
for ii in range(len(gg.nodes)):
    G.add_node(ii)
for edge in gg.edges:
    G.add_edge(gg.nodes.index(edge.get_ends()[0]),
               gg.nodes.index(edge.get_ends()[1]))
    
pca = PCA(n_components=2)
pca.fit(s_lines)


a = plt.figure()
for idx, node in enumerate(gg.nodes):
    print idx,node.data.pos.shape
    pos = node.data.pos
    plt.text(pca.transform(pos)[:, 0], pca.transform(pos)[:, 1], '%d' %idx)

def get_weight(sl1, sl2):
    return 1/sum((np.array(sl1) - np.array(sl2))**2) / \
        sum((np.array(sl1) + np.array(sl2))**2) #co daje ten czlon?

weights = []
for edge in gg.edges:
    node1 = edge.get_ends()[0]
    node2 = edge.get_ends()[1]
    weights.append(get_weight(node1.data.pos, node2.data.pos))
weights = np.array(weights)

for edge in gg.edges:
    node1 = edge.get_ends()[0]
    node2 = edge.get_ends()[1]
    weight = ((get_weight(node1.data.pos, node2.data.pos)) / max(weights))**2
    pos1 = node1.data.pos
    pos2 = node2.data.pos
    plt.plot([pca.transform(pos1)[:, 0], pca.transform(pos2)[:, 0]],
              [pca.transform(pos1)[:, 1], pca.transform(pos2)[:, 1]],
               'b-', lw=weight)