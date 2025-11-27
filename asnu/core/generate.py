"""
Network generation module for PyPleNet NetworkX.

This module provides functions to generate large-scale population networks
using NetworkX in-memory graph storage. It creates nodes from population data and
establishes edges based on interaction patterns, with support for scaling,
reciprocity, and preferential attachment.

This NetworkX-based implementation is significantly faster than file-based
approaches for graphs that fit in memory.

Functions
---------
init_nodes : Initialize nodes in the graph from population data
init_links : Initialize edges in the graph from interaction data
generate : Main function to generate a complete network

Examples
--------
>>> graph = generate('population.csv', 'interactions.xlsx',
...                  fraction=0.4, scale=0.1, reciprocity_p=0.2)
>>> print(f"Generated network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
"""
import os
import math
import shutil
import random

import numpy as np
import networkx as nx

from asnu.core.utils import (find_nodes, read_file, desc_groups)
from asnu.core.grn import establish_links
from asnu.core.graph import NetworkXGraph

def build_group_pair_to_communities_lookup(G, verbose=False):
    """
    Create a lookup dictionary mapping each group pair to their shared communities.

    This precomputes which communities contain which group pairs, making link
    creation much faster by avoiding repeated community membership checks.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community information
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Mapping from (src_id, dst_id) to list of shared community IDs
    """
    if verbose:
        print("Building community lookup for group pairs...")

    group_pair_to_communities = {}

    for community_id in range(G.number_of_communities):
        groups_in_community = set(G.communities_to_groups.get(community_id, []))

        for src_id in groups_in_community:
            for dst_id in groups_in_community:
                pair_key = (src_id, dst_id)

                if pair_key not in group_pair_to_communities:
                    group_pair_to_communities[pair_key] = []

                group_pair_to_communities[pair_key].append(community_id)

    if verbose:
        avg_communities = np.mean([len(v) for v in group_pair_to_communities.values()])
        print(f"  Found {len(group_pair_to_communities)} group pairs")
        print(f"  Average communities per pair: {avg_communities:.1f}")

    return group_pair_to_communities

def populate_communities(G, num_communities, seed_groups):
    """
    Assign nodes to communities based on group affinity patterns.

    Each node is assigned to a community probabilistically, with a bias towards
    placing similar groups in the same community. Seed groups help initialize
    community structure.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Number of communities to create
    seed_groups : list
        Initial group IDs to seed each community
    """
    n_groups = int(np.sqrt(len(G.maximum_num_links)))

    # Initialize community affinity matrix with slight bias
    community_matrix = np.ones((num_communities, n_groups))

    # Handle single community case
    if num_communities == 1:
        seed_groups = [seed_groups[0]]

    # Set up initial community structure with seed groups
    for community_idx, seed_group in enumerate(seed_groups):
        community_matrix[community_idx, seed_group] += 1
        for group_id in range(n_groups):
            G.communities_to_nodes[(community_idx, group_id)] = []
        G.communities_to_groups[community_idx] = []

    # Assign each node to a community based on group probabilities
    for node in G.graph.nodes:
        group = G.nodes_to_group[node]
        probability_vector = G.probability_matrix[group]

        # Calculate community assignment probabilities
        group_probabilities = np.dot(community_matrix, probability_vector)
        group_probabilities = group_probabilities / group_probabilities.sum()

        # Assign node to community
        community = np.random.choice(num_communities, p=group_probabilities)

        # Update community information
        community_matrix[community, group] += 1
        G.communities_to_nodes[(community, group)].append(node)
        G.nodes_to_communities[node] = community
        G.communities_to_groups[community].append(group)
 

def find_separated_groups(G, num_communities, verbose=False):
    """
    Identify groups with minimal inter-connections to seed communities.

    Uses a greedy algorithm to find groups that are least connected to each other,
    which helps create well-separated community structure.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with maximum link counts between groups
    num_communities : int
        Number of communities to create
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    list
        Group IDs selected as community seeds
    """
    n_groups = len(G.group_ids)

    # Create affinity matrix from link counts
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G.maximum_num_links.items():
        affinity[i, j] = count

    # Normalize affinity to get probability matrix
    epsilon = 1e-5
    normalized = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)
    normalized[normalized == 0] = epsilon

    # Store for later use in community assignment
    G.probability_matrix = normalized.copy()
    G.number_of_communities = num_communities
    
    # Start greedy selection with two least connected groups
    selected_indices = []
    remaining = set(range(n_groups))

    # Find the pair of groups with weakest connection
    np.fill_diagonal(normalized, np.inf)
    min_i, min_j = np.unravel_index(np.argmin(normalized), normalized.shape)
    selected_indices.extend([min_i, min_j])
    remaining.discard(min_i)
    remaining.discard(min_j)

    # Iteratively add groups that are minimally connected to already selected groups
    while len(selected_indices) < min(num_communities, n_groups):
        min_total = np.inf
        best_group = None

        for candidate_group in remaining:
            total_connection = normalized[candidate_group, selected_indices].sum()
            if total_connection < min_total:
                min_total = total_connection
                best_group = candidate_group

        selected_indices.append(best_group)
        remaining.discard(best_group)

    # If we need more seeds than available groups, cycle through the selected ones
    if len(selected_indices) < num_communities:
        extended = selected_indices * num_communities
        diff = num_communities - len(selected_indices)
        selected_indices.extend(extended[:diff])

    if verbose:
        print(f"Selected {len(selected_indices)} seed groups for {num_communities} communities")

    # Assign nodes to communities based on selected seed groups
    populate_communities(G, num_communities, selected_indices)

    return selected_indices

def init_nodes(G, pops_path, scale=1, pop_column='n'):
    """
    Initialize nodes from population data using NetworkX directly.

    Parameters
    ----------
    G : NetworkXGraph
        Wrapper with G.graph (nx.DiGraph) and metadata
    pops_path : str
        Path to population file (CSV or Excel)
    scale : float, optional
        Population scaling factor (default 1)
    pop_column : str, optional
        Name of the column containing population counts (default 'n')
    """
    group_desc_dict, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)

    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        G.group_to_attrs[group_id] = attrs
        n_nodes = int(np.ceil(scale * group_info[pop_column]))
        G.group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))

        # Add nodes using NetworkX directly with community attribute
        for _ in range(n_nodes):

            G.graph.add_node(node_id, **attrs)
            G.nodes_to_group[node_id] = group_id
            node_id += 1

    # Create attribute to group mapping
    for group_id, attrs in G.group_to_attrs.items():
        attrs_key = tuple(sorted(attrs.items()))
        G.attrs_to_group[attrs_key] = group_id

    # Initialize link tracking
    group_ids = list(G.group_to_attrs.keys())
    G.group_ids = group_ids
    G.existing_num_links = {(src, dst): 0 for src in group_ids for dst in group_ids}

def init_links(G, links_path, fraction, scale, reciprocity_p, transitivity_p,
               number_of_communities, verbose=True,
               src_suffix='_src', dst_suffix='_dst', link_column='n'):
    """
    Create edges in the graph based on interaction data.

    Reads interaction patterns from a file and creates edges between nodes
    according to group relationships. Supports preferential attachment,
    reciprocity, and community-based link creation.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes already initialized
    links_path : str
        Path to interactions file (CSV or Excel)
    fraction : float
        Preferential attachment parameter (0-1)
    scale : float
        Population scaling factor
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    transitivity_p : float
        Probability of creating transitive edges (0-1)
    number_of_communities : int
        Number of communities to create
    verbose : bool, optional
        Whether to print progress information
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Name of column containing link counts (default 'n')
    """
    warnings = []
    df_n_group_links = read_file(links_path)

    if verbose:
        print("Calculating link requirements...")

    # Initialize maximum link counts for all group pairs
    group_ids = G.group_ids
    G.maximum_num_links = {(i, j): 0 for i in group_ids for j in group_ids}

    # Calculate required links for each group pair
    for idx, row in df_n_group_links.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        G.maximum_num_links[(src_id, dst_id)] = int(math.ceil(row[link_column] * scale))

    if verbose:
        total_links = sum(G.maximum_num_links.values())
        print(f"Total requested links: {total_links}")

    # Create community structure
    find_separated_groups(G, number_of_communities, verbose=verbose)

    # Build lookup for efficient community-based link creation
    group_pair_to_communities = build_group_pair_to_communities_lookup(G, verbose=verbose)

    # Create links for each group pair
    total_rows = len(df_n_group_links)
    for idx, row in df_n_group_links.iterrows():

        if verbose and ((idx + 1) % 500 == 0 or idx == 0 or idx == total_rows - 1):
            print(f"\rProcessing row {idx + 1} of {total_rows}", end="")

        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

        num_requested_links = int(math.ceil(row[link_column] * scale))

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        if not src_nodes or not dst_nodes:
            continue

        # Get valid communities for this group pair
        valid_communities = group_pair_to_communities.get((src_id, dst_id), [])

        # Create links between the groups
        link_success = establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                                      num_requested_links, fraction, reciprocity_p,
                                      transitivity_p, valid_communities)

        if not link_success:
            existing_links = G.existing_num_links[(src_id, dst_id)]
            warnings.append(f"Groups ({src_id})-({dst_id}): {existing_links} exceeds target {num_requested_links}")

    if verbose:
        print()
        if warnings:
            print(f"\nWarnings ({len(warnings)} group pairs):")
            for warning in warnings[:10]:  # Show first 10 warnings
                print(f"  {warning}")
            if len(warnings) > 10:
                print(f"  ... and {len(warnings) - 10} more")

def fill_unfulfilled_group_pairs(G, reciprocity_p, verbose=True):
    """
    Complete any group pairs that didn't reach their target edge count.

    Randomly creates edges between nodes from unfulfilled group pairs until
    targets are met or maximum attempts are reached.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with existing edges
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about the filling process
    """
    if verbose:
        print("\nFilling unfulfilled group pairs...")

    unfulfilled_pairs = []
    stats = {
        'total_pairs': 0,
        'fulfilled_pairs': 0,
        'unfulfilled_pairs': 0,
        'edges_added': 0,
        'reciprocal_edges_added': 0
    }

    # Identify which group pairs need more edges
    for (src_id, dst_id) in G.maximum_num_links.keys():
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        maximum = G.maximum_num_links[(src_id, dst_id)]

        stats['total_pairs'] += 1

        if maximum == 0:
            continue

        # Only try to fill pairs that are genuinely under the target
        if existing < maximum:
            unfulfilled_pairs.append((src_id, dst_id, existing, maximum))
            stats['unfulfilled_pairs'] += 1
        else:
            stats['fulfilled_pairs'] += 1

    if verbose:
        print(f"  Total pairs: {stats['total_pairs']}")
        print(f"  Fulfilled: {stats['fulfilled_pairs']}")
        print(f"  Unfulfilled: {stats['unfulfilled_pairs']}")

    # Add random edges to complete unfulfilled pairs
    partially_filled = 0
    if unfulfilled_pairs:
        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            existing = G.existing_num_links.get((src_id, dst_id), 0)
            maximum = G.maximum_num_links.get((src_id, dst_id), 0)
            needed = maximum - existing
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            attempts = 0
            max_attempts = needed * 20
            edges_added_for_pair = 0

            while edges_added_for_pair < needed and attempts < max_attempts:
                src_node = random.choice(src_nodes)
                dst_node = random.choice(dst_nodes)

                # Add edge if valid (no self-loops, no duplicates)
                if src_node != dst_node and not G.graph.has_edge(src_node, dst_node):
                    G.graph.add_edge(src_node, dst_node)
                    edges_added_for_pair += 1
                    G.existing_num_links[(src_id, dst_id)] += 1
                    stats['edges_added'] += 1

                    # Reciprocity - same pattern as grn.py
                    if random.uniform(0,1) < reciprocity_p:
                        if( G.existing_num_links[(dst_id, src_id)] < G.maximum_num_links[(dst_id, src_id)] and 
                           not G.graph.has_edge(dst_node, src_node)):
                            G.graph.add_edge(dst_node, src_node)
                            G.existing_num_links[(dst_id, src_id)] += 1
                            stats['reciprocal_edges_added'] += 1
                            if (dst_id == src_id):
                                edges_added_for_pair += 1
                                stats['edges_added'] += 1

                attempts += 1

    if verbose:
        print(f"  Edges added: {stats['edges_added']}")
        print(f"  Reciprocal edges added: {stats['reciprocal_edges_added']}")

    return stats

def generate(pops_path, links_path, preferential_attachment, scale, reciprocity,
             transitivity, number_of_communities, base_path="graph_data", verbose=True,
             pop_column='n', src_suffix='_src', dst_suffix='_dst', link_column='n'):
    """
    Generate a population-based network using NetworkX.

    Creates a network by first generating nodes from population data, then
    establishing edges based on interaction patterns. Supports preferential
    attachment, reciprocity, transitivity, and community structure.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    preferential_attachment : float
        Preferential attachment strength (0-1)
    scale : float
        Population scaling factor
    reciprocity : float
        Probability of reciprocal edges (0-1)
    transitivity : float
        Probability of transitive edges (0-1)
    number_of_communities : int
        Number of communities to create
    base_path : str, optional
        Directory for saving graph (default "graph_data")
    verbose : bool, optional
        Whether to print progress information
    pop_column : str, optional
        Column name for population counts in pops_path (default 'n')
    src_suffix : str, optional
        Suffix for source group columns in links_path (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns in links_path (default '_dst')
    link_column : str, optional
        Column name for link counts in links_path (default 'n')

    Returns
    -------
    NetworkXGraph
        Generated network with graph data and metadata
    """
    if verbose:
        print("="*60)
        print("NETWORK GENERATION")
        print("="*60)
        print("\nStep 1: Creating nodes from population data...")

    # Prepare output directory
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path)

    G = NetworkXGraph(base_path)

    # Create nodes
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes")
        print("\nStep 2: Creating edges from interaction patterns...")

    # Invert preferential attachment for internal representation
    preferential_attachment_fraction = 1 - preferential_attachment

    # Create edges
    init_links(G, links_path, preferential_attachment_fraction, scale,
              reciprocity, transitivity, number_of_communities, verbose=verbose,
              src_suffix=src_suffix, dst_suffix=dst_suffix, link_column=link_column)

    if verbose:
        print("\nStep 3: Filling remaining unfulfilled group pairs...")

    # Complete any group pairs that didn't reach their target
    fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    # Save to disk
    G.finalize()

    if verbose:
        # Calculate link fulfillment statistics
        total_requested = sum(G.maximum_num_links.values())
        total_created = sum(G.existing_num_links.values())
        fulfillment_rate = (total_created / total_requested * 100) if total_requested > 0 else 0

        # Count overfulfilled pairs
        overfulfilled = sum(1 for (src, dst) in G.maximum_num_links.keys()
                           if G.existing_num_links.get((src, dst), 0) > G.maximum_num_links[(src, dst)])

        print(f"\n{'='*60}")
        print(f"NETWORK GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Nodes: {G.graph.number_of_nodes()}")
        print(f"Edges: {G.graph.number_of_edges()}")
        print(f"\nLink Fulfillment:")
        print(f"  Requested: {total_requested}")
        print(f"  Created: {total_created}")
        print(f"  Difference: {total_created - total_requested:+d}")
        print(f"  Rate: {fulfillment_rate:.1f}%")
        if overfulfilled > 0:
            print(f"  Overfulfilled pairs: {overfulfilled}")
        print(f"\nSaved to: {base_path}")
        print(f"{'='*60}\n")

    return G