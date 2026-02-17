import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import entropy
from itertools import combinations

characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

for r in range(1, len(characteristics) + 1):
    for combo in combinations(characteristics, r):
        group_cols = list(combo)
        characteristics_string = '_'.join(group_cols)
        df_edges = pd.read_csv(f"Data/aggregated/tab_werkschool_{characteristics_string}.csv")
        df_nodes = pd.read_csv(f"Data/aggregated/tab_n_{characteristics_string}.csv")



        num_unique_groups = df_nodes.shape[0]
        total_possible_paths = num_unique_groups ** 2
        H_max = np.log2(total_possible_paths)
        print(H_max)

        # Dynamic labeling for Edges (Source)
        df_edges['src_label'] = df_edges[[c + "_src" for c in group_cols]].astype(str).agg('_'.join, axis=1)

        # Dynamic labeling for Edges (Destination)
        df_edges['dst_label'] = df_edges[[c + "_dst" for c in group_cols]].astype(str).agg('_'.join, axis=1)

        # Dynamic labeling for Nodes (matching the population table)
        df_nodes['node_label'] = df_nodes[group_cols].astype(str).agg('_'.join, axis=1)

        # Assuming df_nodes and df_edges are your dataframes...

        # 1. Normalize: n_connections / population_at_source
        df_merged = df_edges.merge(df_nodes, left_on=f'src_label', right_on=f'node_label')
        df_merged['norm_weight'] = df_merged['n_x'] / df_merged['n_y']

        # 2. Entropy of the normalized distribution
        # This measures how 'spread out' the connection probability is across the matrix
        p = df_merged['norm_weight'] / df_merged['norm_weight'].sum()
        print(p)
        p = df_merged['norm_weight'] / df_merged['norm_weight'].sum()
        H_obs = entropy(p, base=2)
        H_norm = H_obs / H_max if H_max > 0 else 0

        # 3. Visualization
        plt.figure(figsize=(12, 8))
        G = nx.from_pandas_edgelist(df_merged, f'src_label', f'dst_label', 
                                    edge_attr='norm_weight', create_using=nx.DiGraph())

        # Layout: Circular makes it easy to see connections between age bins
        pos = nx.circular_layout(G)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='teal', alpha=0.8)
        nx.draw_networkx_labels(G, pos, font_color='white', font_weight='bold')

        # Draw edges - width proportional to the normalized weight
        weights = [G[u][v]['norm_weight'] for u, v in G.edges()]
        # Scale weights so they look good on screen
        scaled_widths = [w * (5 / max(weights)) for w in weights]

        nx.draw_networkx_edges(G, pos, width=scaled_widths, edge_color='gray', 
                            alpha=0.5, arrowsize=20, connectionstyle="arc3,rad=0.1")

        plt.title(f"Normalized Connection Network\nShannon Entropy: {H_norm:.3f} bits")
        plt.axis('off')
        plt.show()

        if len(group_cols) > 2:
            exit()
