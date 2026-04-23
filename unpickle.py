import pickle

with open('original.pkl', 'rb') as f:
    data = pickle.load(f)


import numpy as np
import igraph as ig
from scipy.stats import skew

# Convert networkx -> igraph
G_ig = ig.Graph.from_networkx(data)
G_undirected = G_ig.as_undirected() if G_ig.is_directed() else G_ig

# Clustering coefficient
cc = G_undirected.transitivity_undirected()
cc_avg = G_undirected.transitivity_avglocal_undirected()

# Modularity via label propagation
partition = G_undirected.community_label_propagation()
modularity = G_undirected.modularity(partition)

# Degree skewness
deg_skew = skew(G_undirected.degree())

print(f"Clustering (global): {cc:.4f}")
print(f"Clustering (avg local): {cc_avg:.4f}")
print(f"Modularity (label prop): {modularity:.4f}")
print(f"Degree skewness: {deg_skew:.4f}")