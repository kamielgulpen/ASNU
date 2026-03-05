# Imports
import numpy as np
import matplotlib.pyplot as plt
import rustworkx as rx
import matplotlib.pyplot as plt
from asnu import generate, create_communities
import time
import pickle
from scipy import stats
# Generate network
<<<<<<< HEAD:tests/test.py
links = 'Data/tab_huishouden.csv'
=======
links = 'Data/tab_werkschool.csv'
>>>>>>> dd3cdf706b2b73eb59a5898efde935a5a0c6787c:network_generation_scripts/generate_single_network.py
# as example we use group interaction data on a work / school layer
pops = 'Data/tab_n_(with oplniv).csv' 

start = time.perf_counter()
# # Step 1: Create communities separately
create_communities(
    pops, 
    links,
<<<<<<< HEAD:tests/test.py
    scale=0.01, 
    number_of_communities=5000,
    output_path='my_communities.json',
    mode='probability'
=======
    scale=0.1, 
    number_of_communities=2000,
    output_path='my_communities.json'
>>>>>>> dd3cdf706b2b73eb59a5898efde935a5a0c6787c:network_generation_scripts/generate_single_network.py
)

graph = generate(
    pops,                             # The group-level population data
    links,                            # The group-level interaction data
<<<<<<< HEAD:tests/test.py
    preferential_attachment=0,     # Preferential attachment strength
    scale=0.01,                        # Population scaling
=======
    preferential_attachment=0.0,     # Preferential attachment strength
    scale=0.1,                        # Population scaling
>>>>>>> dd3cdf706b2b73eb59a5898efde935a5a0c6787c:network_generation_scripts/generate_single_network.py
    reciprocity=1,                    # Reciprocal edge probability
    transitivity =1,                  # Friend of a friend is my friend probability
    community_file='my_communities.json',                  
    base_path="my_network",           # Path for the FileBasedGraph's data
    bridge_probability=0,
    fully_connect_communities = True
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
print(degrees.count(0))
print(degrees.count(1))
print(degrees.count(100))
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


