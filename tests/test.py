# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
from asnu import generate, NetworkXGraph
import time
import pickle
from scipy import stats
# Generate network
pops = 'Data/tab_n_(with oplniv).csv'
# as example we use group interaction data on a work / school layer
links = 'Data/tab_werkschool.csv' 

start = time.perf_counter()
# Your code here
params = {
    'pops_path': pops, 
    'links_path': links, 
    'preferential_attachment':0,
    'scale': 0.01,
    'reciprocity':1,
    'transitivity':0,
    'number_of_communities':25,
    'community_size_distribution':"powerlaw",
    'pa_scope':"local",
    'fill_unfulfilled' : True
          }
graph = generate(
    **params      # Path for the FileBasedGraph's data
)

end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")
G_rx = rx.PyDiGraph()
G_nx = graph.graph
# Create node mapping (NetworkX ID -> rustworkx index)
node_map = {}
for node in G_nx.nodes():
    node_attrs = G_nx.nodes[node]
    idx = G_rx.add_node(node_attrs if node_attrs else node)
    node_map[node] = idx

# Add edges
for u, v, edge_attrs in G_nx.edges(data=True):
    G_rx.add_edge(node_map[u], node_map[v], edge_attrs)

print(f"Graph: {len(G_rx)} nodes, {G_rx.num_edges()} edges")
print(f"Transitivity:{rx.transitivity(G_rx)}")
# print(f"Transitivity (nx):{nx.transitivity(G_nx)}")

# Get degree sequence
degrees = [G_rx.in_degree(node) for node in G_rx.node_indices()]

print(f"Mean degree: {np.mean(degrees):.2f}")
print(f"Std degree: {np.std(degrees):.2f}")
print(f"Max degree: {max(degrees)}")
print(f"Min degree: {min(degrees)}")
print(f"Median degree: {np.median(degrees)}")
print(f"first q degree: {np.quantile(degrees, 0.25)}")
print(f"fourth q degree: {np.quantile(degrees, 0.75)}")
print(f"skew: {stats.skew(degrees)}")


plt.hist(degrees)
plt.show()

# Create filename from params
param_str = '_'.join(f'{k}={v}' for k, v in params.items())
filename = f'a.pkl'
# Result: 'model_lr=0.001_batch_size=32_epochs=100.pkl'

# Save
with open(filename, 'wb') as f:
    pickle.dump(G_nx, f)


