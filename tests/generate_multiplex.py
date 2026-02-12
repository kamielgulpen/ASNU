"""
Multiplex Network Generator
============================

Generates a 4-layer multiplex network where every node exists in all layers.
Layers are generated in order with optional hierarchical edge propagation.

Layers:
  1. huishouden  (household)
  2. familie     (family)
  3. buren       (neighbours)
  4. werkschool  (work/school)

Propagation rules (optional):
  - All huishouden edges are copied to the buren layer
  - A fraction of huishouden edges are copied to the familie layer
  - werkschool is independent (no propagation)
"""

import networkx as nx
import numpy as np
import pickle
import time
import os
import random
from itertools import combinations
import igraph as ig
from asnu import generate, create_communities


# ============================================================================
# Layer definitions and per-layer parameters
# ============================================================================

LAYERS = ['huishouden', 'familie', 'buren', 'werkschool']

LAYER_PARAMS = {
    'huishouden': {
        'number_of_communities': 5000,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0,
        'fill_unfulfilled' : False,
        'fully_connect_communities' : True
    },
    'familie': {
        'number_of_communities': 500,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0.1,
        'fill_unfulfilled' : False,
        'fully_connect_communities' : False
    },
    'buren': {
        'number_of_communities': 5,
        'reciprocity': 1,
        'transitivity': 1,
        'preferential_attachment': 0,
        'bridge_probability': 0.2,
        'fill_unfulfilled' : True,
        'fully_connect_communities' : False
    },
    'werkschool': {
        'number_of_communities': 50,
        'reciprocity': 1,
        'transitivity': 0.5,
        'preferential_attachment': 0.1,
        'bridge_probability': 0.3,
        'fill_unfulfilled' : True,
        'fully_connect_communities' : False
    },
}

# Shared parameters
scale = 0.01
characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

# Propagation settings
propagate = True
family_fraction = 0.3  # fraction of household edges that also become family edges


# ============================================================================
# Core functions
# ============================================================================

def generate_layer(layer_name, pops_path, links_path, layer_params, scale):
    """
    Generate a single network layer using ASNU.

    Parameters
    ----------
    layer_name : str
        Name of the layer (for logging and community file naming)
    pops_path : str
        Path to population CSV
    links_path : str
        Path to interaction CSV for this layer
    layer_params : dict
        Per-layer generation parameters
    scale : float
        Population scaling factor

    Returns
    -------
    nx.DiGraph
        The generated network layer
    """
    community_file = f'communities_{layer_name}.json'

    create_communities(
        pops_path, links_path,
        scale=scale,
        number_of_communities=layer_params['number_of_communities'],
        output_path=community_file,
    )

    graph = generate(
        pops_path,
        links_path,
        preferential_attachment=layer_params['preferential_attachment'],
        scale=scale,
        reciprocity=layer_params['reciprocity'],
        transitivity=layer_params['transitivity'],
        fill_unfulfilled = layer_params['fill_unfulfilled'],
        fully_connect_communities= layer_params['fully_connect_communities'],
        community_file=community_file,
        base_path=f'temp_{layer_name}', 
        bridge_probability=layer_params['bridge_probability'],
    )

    print(f"{'='*70}")
    print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
    print(f"  {'-'*60}")

    combined = graph.graph
    edges = list(combined.edges())
    nodes = list(combined.nodes())
    n = len(nodes)
    e = len(edges)
    avg_deg = e / n if n > 0 else 0

    # Reciprocity
    recip_edges = sum(1 for u, v in combined.edges() if combined.has_edge(v, u))
    reciprocity = recip_edges / e if e > 0 else 0

    edges = list(combined.edges())
    nodes = list(combined.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]

    g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=True)

    transitivity_ig = g.transitivity_undirected(mode="nan")

    print(f"  {"multiplex":<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f} ")


    return graph.graph


def propagate_edges_to(source_graph, target_graph, fraction=1.0):
    """
    Copy edges from source_graph into target_graph.

    Parameters
    ----------
    source_graph : nx.DiGraph
        Graph whose edges to propagate
    target_graph : nx.DiGraph
        Graph to add edges into
    fraction : float
        Fraction of source edges to propagate (1.0 = all)

    Returns
    -------
    int
        Number of new edges added
    """
    source_edges = list(source_graph.edges())
    random.shuffle(source_edges)

    n_to_propagate = int(len(source_edges) * fraction)
    propagated = 0

    for u, v in source_edges[:n_to_propagate]:
        if not target_graph.has_edge(u, v):
            target_graph.add_edge(u, v)
            propagated += 1

    return propagated


def save_multiplex(layers, output_dir):
    """
    Save multiplex network: individual pkl per layer + combined multiplex pkl.

    Parameters
    ----------
    layers : dict
        {layer_name: nx.DiGraph}
    output_dir : str
        Directory to save files in
    """
    os.makedirs(output_dir, exist_ok=True)

    # Individual layer files
    for layer_name, graph in layers.items():
        filepath = os.path.join(output_dir, f'{layer_name}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

    # Combined multiplex graph: merge all layers into a single nx.DiGraph
    # Each edge gets a 'layers' attribute (set of layer names it belongs to)
    combined = nx.DiGraph()
    for layer_name, graph in layers.items():
        combined.add_nodes_from(graph.nodes(data=True))
        for u, v, data in graph.edges(data=True):
            if combined.has_edge(u, v):
                combined[u][v]['layers'].add(layer_name)
            else:
                combined.add_edge(u, v, **data, layers={layer_name})

    print(f"\n{'='*70}")
    print("MULTIPLEX NETWORK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
    print(f"  {'-'*60}")


    n = combined.number_of_nodes()
    e = combined.number_of_edges()
    avg_deg = e / n if n > 0 else 0

    # Reciprocity
    recip_edges = sum(1 for u, v in combined.edges() if combined.has_edge(v, u))
    reciprocity = recip_edges / e if e > 0 else 0

    edges = list(combined.edges())
    nodes = list(combined.nodes())
    node_mapping = {node: idx for idx, node in enumerate(nodes)}
    igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]

    g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=True)

    transitivity_ig = g.transitivity_undirected(mode="nan")

    print(f"  {"multiplex":<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f} ")

    multiplex_path = os.path.join(output_dir, 'multiplex.pkl')
    with open(multiplex_path, 'wb') as f:
        pickle.dump(combined, f)


def print_layer_stats(layers):
    """Print summary statistics for all layers."""
    print(f"\n{'='*70}")
    print("MULTIPLEX NETWORK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Layer':<15} {'Nodes':>8} {'Edges':>10} {'Reciprocity':>13} {'Avg Degree':>12} {'Transitivity':>11}")
    print(f"  {'-'*60}")

    for name, G in layers.items():
        n = G.number_of_nodes()
        e = G.number_of_edges()
        avg_deg = e / n if n > 0 else 0

        # Reciprocity
        recip_edges = sum(1 for u, v in G.edges() if G.has_edge(v, u))
        reciprocity = recip_edges / e if e > 0 else 0

        edges = list(G.edges())
        nodes = list(G.nodes())
        node_mapping = {node: idx for idx, node in enumerate(nodes)}
        igraph_edges = [(node_mapping[u], node_mapping[v]) for u, v in edges]

        g = ig.Graph(n=len(nodes), edges=igraph_edges, directed=True)

        transitivity_ig = g.transitivity_undirected(mode="nan")

        print(f"  {name:<15} {n:>8} {e:>10} {reciprocity:>13.3f} {avg_deg:>12.1f} {transitivity_ig:>11.3f} ")

    # Cross-layer overlap
    print(f"\n  Edge Overlap:")
    layer_names = list(layers.keys())
    for i in range(len(layer_names)):
        for j in range(i + 1, len(layer_names)):
            a_name, b_name = layer_names[i], layer_names[j]
            a_edges = set(layers[a_name].edges())
            b_edges = set(layers[b_name].edges())
            overlap = len(a_edges & b_edges)
            if overlap > 0:
                print(f"    {a_name} & {b_name}: {overlap} shared edges")

    print(f"{'='*70}\n")

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MULTIPLEX NETWORK GENERATION")
    print("="*70)
    print(f"\nLayers: {', '.join(LAYERS)}")
    print(f"Scale: {scale}")
    print(f"Propagation: {'enabled' if propagate else 'disabled'}")
    if propagate:
        print(f"  Family fraction: {family_fraction}")
    print()

    params_str = f"multiplex_scale={scale}"

    for r in range(1, len(characteristics) + 1):
        for combo in combinations(characteristics, r):
            group_cols = list(combo)
            characteristics_string = '_'.join(group_cols)

            pops = f'Data/aggregated/tab_n_{characteristics_string}.csv'

            print(f"\n{'='*60}")
            print(f"Generating multiplex: {characteristics_string}")
            print(f"{'='*60}")

            layers = {}
            total_start = time.perf_counter()

            def _generate_and_log(layer_name):
                links = f'Data/aggregated/tab_{layer_name}_{characteristics_string}.csv'
                print(f"\n  --- Layer: {layer_name} ---")
                start = time.perf_counter()
                graph = generate_layer(layer_name, pops, links,
                                       LAYER_PARAMS[layer_name], scale)
                elapsed = time.perf_counter() - start
                print(f"  {layer_name}: {graph.number_of_nodes()} nodes, "
                      f"{graph.number_of_edges()} edges ({elapsed:.2f}s)")
                return graph

            # 1. Generate huishouden first — all other layers build on it
            layers['huishouden'] = _generate_and_log('huishouden')

            # 2. Generate familie, then propagate household ties into it
            layers['familie'] = _generate_and_log('familie')
            if propagate:
                n_fam = propagate_edges_to(layers['huishouden'], layers['familie'],
                                           fraction=family_fraction)
                print(f"    + propagated {n_fam} household edges into familie "
                      f"({family_fraction:.0%} of {layers['huishouden'].number_of_edges()})")

            # 3. Generate buren, then propagate ALL household ties into it
            layers['buren'] = _generate_and_log('buren')
            if propagate:
                n_bur = propagate_edges_to(layers['huishouden'], layers['buren'],
                                           fraction=1.0)
                print(f"    + propagated {n_bur} household edges into buren "
                      f"(all {layers['huishouden'].number_of_edges()})")

            # 4. Generate werkschool independently
            layers['werkschool'] = _generate_and_log('werkschool')

            # Print summary
            print_layer_stats(layers)

            # Save
            output_dir = f'Data/networks/{params_str}/{characteristics_string}'
            save_multiplex(layers, output_dir)

            total_elapsed = time.perf_counter() - total_start
            print(f"Saved to {output_dir}/ ({total_elapsed:.1f}s total)")

            exit()
if __name__ == "__main__":
    main()
