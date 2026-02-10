# Imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
from asnu import generate, create_communities
import time
import pickle
from scipy import stats


import networkx as nx
import igraph as ig

def nx_to_igraph(nx_graph):
    # Get node mapping (igraph uses integer indices)
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build edge list using integer indices
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]
    
    # Create igraph graph
    directed = nx_graph.is_directed()
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=directed)
    
    # Transfer node attributes
    for attr in nx_graph.nodes[nodes[0]].keys() if nodes else []:
        ig_graph.vs[attr] = [nx_graph.nodes[n].get(attr) for n in nodes]
    
    # Store original node names
    ig_graph.vs["name"] = nodes
    
    # Transfer edge attributes
    if nx_graph.edges():
        first_edge = next(iter(nx_graph.edges(data=True)))
        for attr in first_edge[2].keys():
            ig_graph.es[attr] = [
                nx_graph[u][v].get(attr) 
                for u, v in nx_graph.edges()
            ]
    
    return ig_graph

# Generate network
links = 'Data/tab_werkschool.csv'
# as example we use group interaction data on a work / school layer
pops = 'Data/tab_n_(with oplniv).csv' 

start = time.perf_counter()
# Step 1: Create communities separately
create_communities(pops, links,
                   scale=0.1, number_of_communities=50,
                   output_path='my_communities.json')

graph = generate(
    pops,                             # The group-level population data
    links,                            # The group-level interaction data
    preferential_attachment=0.99,        # Preferential attachment strength
    scale=0.1,                          # Population scaling
    reciprocity=1,                    # Reciprocal edge probability
    transitivity =1,
    community_file='my_communities.json',                  # friend of a friend is my friend probability
    base_path="my_network",           # Path for the FileBasedGraph's data
)

end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")
G_nx = graph.graph
end = time.perf_counter()
print(f"Execution time: {end - start:.4f} seconds")
G_ig = nx_to_igraph(G_nx)

num_nodes = G_ig.vcount()
num_edges = G_ig.ecount()
print(f"Calculating metrics for graph with {num_nodes} nodes and {G_ig.ecount()} edges...")

# Reciprocity
print("Calculating reciprocity...")
reciprocity = G_ig.reciprocity()
print(reciprocity)

# Transitivity
print("Calculating transitivity...")
transitivity = G_ig.transitivity_undirected()
print(transitivity)

# Get degree sequence
degrees = degrees = G_ig.degree(mode="in")

print(f"Mean degree: {np.mean(degrees):.2f}")
print(f"Std degree: {np.std(degrees):.2f}")
print(f"Max degree: {max(degrees)}")
print(f"Min degree: {min(degrees)}")
print(f"Median degree: {np.median(degrees)}")
print(f"first q degree: {np.quantile(degrees, 0.25)}")
print(f"fourth q degree: {np.quantile(degrees, 0.75)}")
print(f"skew: {stats.skew(degrees)}")


plt.hist(degrees, bins = 50)
plt.show()

# Create filename from params
param_str = '_'.join(f'{k}={v}' for k, v in params.items())
filename = f'a.pkl'
# Result: 'model_lr=0.001_batch_size=32_epochs=100.pkl'

# Save
with open(filename, 'wb') as f:
    pickle.dump(G_nx, f)


