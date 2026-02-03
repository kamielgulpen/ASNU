# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import rustworkx as rx
import matplotlib.pyplot as plt
from asnu import generate, NetworkXGraph
import time

# Generate network
pops = 'Data/tab_n_(with oplniv).csv'
# as example we use group interaction data on a work / school layer
links = 'Data/tab_huishouden.csv' 

start = time.perf_counter()
# Your code here

graph = generate(
    pops,                             # The group-level population data
    links,                            # The group-level interaction data
    preferential_attachment=0.9,     # Preferential attachment strength
    scale=0.01,                       # Population scaling
    reciprocity=0,                    # Reciprocal edge probability
    transitivity = 0,                 # friend of a friend is my friend probability
    number_of_communities = 500,
    fill_unfulfilled=False,
    pa_scope="global",
    base_path="my_network",
    community_size_distribution="natural"          # Path for the FileBasedGraph's data
)

end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")
G_rx = rx.PyGraph()
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
# print(f"Reciprocity:{nx.reciprocity(G_nx)}")

# Get degree sequence
degrees = [G_rx.degree(node) for node in G_rx.node_indices()]

print(f"Mean degree: {np.mean(degrees):.2f}")
print(f"Std degree: {np.std(degrees):.2f}")
print(f"Max degree: {max(degrees)}")
print(f"Min degree: {min(degrees)}")
print(f"Median degree: {np.median(degrees)}")
print(f"first q degree: {np.quantile(degrees, 0.25)}")
print(f"fourth q degree: {np.quantile(degrees, 0.75)}")

plt.hist(degrees)
plt.show()


