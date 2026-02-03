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
from itertools import product

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

def populate_communities(G, num_communities, seed_groups, community_size_distribution='natural'):
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
    community_size_distribution : str or array-like, optional
        Controls community size distribution:
        - 'natural' (default): Let communities grow naturally based on group affinities
        - 'uniform': Aim for equal-sized communities
        - 'powerlaw': Create power-law distribution (few large, many small communities)
        - array of floats: Custom target proportions (must sum to 1)
    """
    n_groups = int(np.sqrt(len(G.maximum_num_links)))

    # Calculate target community sizes based on distribution type
    total_nodes = len(list(G.graph.nodes))

    if community_size_distribution == 'uniform':
        # Equal-sized communities
        target_sizes = np.ones(num_communities) / num_communities
    elif community_size_distribution == 'powerlaw':
        # Power-law distribution: few large, many small
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / (ranks ** 1.5)  # Adjust exponent for steeper/flatter distribution
        target_sizes = sizes / sizes.sum()
    elif isinstance(community_size_distribution, (list, np.ndarray)):
        # Custom distribution
        target_sizes = np.array(community_size_distribution)
        if not np.isclose(target_sizes.sum(), 1.0):
            raise ValueError("Custom community_size_distribution must sum to 1")
    else:  # 'natural' or default
        target_sizes = None

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

    # OPTIMIZED: Assign nodes in batches with faster random sampling
    all_nodes = list(G.graph.nodes)
    np.random.shuffle(all_nodes)  # Shuffle to mix groups in batches

    # Track community sizes for distribution control
    if target_sizes is not None:
        community_sizes = np.zeros(num_communities, dtype=np.int32)
        target_counts = (target_sizes * total_nodes).astype(np.int32)

    batch_size = 2000  # Process nodes in batches to reduce matrix recalculations
    x = 0
    for batch_start in range(0, len(all_nodes), batch_size):
        batch_end = min(batch_start + batch_size, len(all_nodes))
        batch_nodes = all_nodes[batch_start:batch_end]

        # Pre-calculate affinity matrix once per batch: (num_communities, n_groups)
        affinity_matrix = community_matrix @ G.probability_matrix.T
        # Normalize each group's probabilities (columns sum to 1)
        affinity_matrix = affinity_matrix / affinity_matrix.sum(axis=0, keepdims=True)

        # Apply distribution control if specified
        if target_sizes is not None:
            # Penalize communities that exceed their target size
            size_ratio = community_sizes / (target_counts + 1e-10)
            # Communities above target get their probabilities reduced
            penalty = np.maximum(0.1, 1.0 - size_ratio)  # Min 10% probability
            affinity_matrix = affinity_matrix * penalty[:, np.newaxis]
            # Re-normalize
            affinity_matrix = affinity_matrix / affinity_matrix.sum(axis=0, keepdims=True)

        # Pre-calculate cumulative probabilities for faster sampling
        cumulative_matrix = np.cumsum(affinity_matrix, axis=0)

        # Generate random values for entire batch
        random_values = np.random.random(len(batch_nodes))

        # Track batch updates to community matrix
        batch_updates = np.zeros((num_communities, n_groups), dtype=np.int32)

        # Assign communities for batch
        for idx, node in enumerate(batch_nodes):
            group = G.nodes_to_group[node]

            # Fast sampling using searchsorted (10-20x faster than np.random.choice)
            community = np.searchsorted(cumulative_matrix[:, group], random_values[idx])

            # Track update for batch
            batch_updates[community, group] += 1

            # Update community membership
            G.communities_to_nodes[(community, group)].append(node)
            G.nodes_to_communities[node] = community
            G.communities_to_groups[community].append(group)

        # Apply all batch updates at once to community matrix
        community_matrix += batch_updates

        # Update size tracking
        if target_sizes is not None:
            for comm in range(num_communities):
                community_sizes[comm] += batch_updates[comm, :].sum()

        x += batch_size
        print(x/len(all_nodes))
            
        
 

def find_separated_groups(G, num_communities, verbose=False, community_size_distribution='natural'):
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
    community_size_distribution : str or array-like, optional
        Controls community size distribution (passed to populate_communities)

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
    populate_communities(G, num_communities, selected_indices, community_size_distribution)


    return selected_indices

def init_nodes(G, pops_path, scale=1, pop_column='n', min_group_size=0):
    """
    Initialize nodes from population data using stratified sampling.

    Uses stratified allocation to preserve demographic proportions while scaling.
    This approach ensures that small groups are not over-represented due to
    ceiling effects in simple proportional scaling.

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
    min_group_size : int, optional
        Minimum nodes per group. Groups smaller than this after scaling
        are set to this value to avoid empty groups (default 0)
    """
    group_desc_dict, characteristic_cols = desc_groups(pops_path, pop_column=pop_column)

    # STRATIFIED ALLOCATION: Calculate proportional allocation with remainder distribution
    total_pop = sum(group_info[pop_column] for group_info in group_desc_dict.values())
    target_total = int(scale * total_pop)

    # Allocate nodes proportionally to each group (using floor)
    node_allocations = {}
    allocated_total = 0

    for group_id, group_info in group_desc_dict.items():
        # Proportional allocation (floor to avoid over-allocation)
        base_allocation = int(scale * group_info[pop_column])

        # Apply minimum group size if specified and group is non-empty
        if base_allocation > 0 and base_allocation < min_group_size:
            base_allocation = min_group_size
        elif base_allocation == 0 and group_info[pop_column] > 0 and min_group_size > 0:
            base_allocation = min_group_size

        node_allocations[group_id] = base_allocation
        allocated_total += base_allocation

    # Distribute remainder to maintain exact total (if not using min_group_size)
    remainder = target_total - allocated_total
    if remainder > 0 and min_group_size == 0:
        # Sort groups by original size (largest first) to distribute remainder
        group_sizes = [(gid, group_desc_dict[gid][pop_column])
                       for gid in group_desc_dict.keys()]
        group_sizes.sort(key=lambda x: x[1], reverse=True)

        # Give remainder to largest groups in round-robin fashion
        for i in range(remainder):
            group_id = group_sizes[i % len(group_sizes)][0]
            node_allocations[group_id] += 1

    # Create nodes using stratified allocation
    node_id = 0
    for group_id, group_info in group_desc_dict.items():
        attrs = {col: group_info[col] for col in characteristic_cols}
        G.group_to_attrs[group_id] = attrs
        n_nodes = node_allocations[group_id]
        G.group_to_nodes[group_id] = list(range(node_id, node_id + n_nodes))

        # Add nodes using NetworkX directly
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
               src_suffix='_src', dst_suffix='_dst', link_column='n',
               community_size_distribution='natural', pa_scope='local'):
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

    # STRATIFIED EDGE ALLOCATION: First calculate all link requirements
    link_data = []
    total_original_links = 0

    for idx, row in df_n_group_links.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)

        original_links = row[link_column]
        total_original_links += original_links

        link_data.append({
            'src_id': src_id,
            'dst_id': dst_id,
            'original': original_links
        })

    # Calculate target total links and allocate proportionally
    target_total_links = int(scale * total_original_links)
    allocated_links = 0

    # First pass: allocate floor(scale * links) to each pair
    for item in link_data:
        base_allocation = int(scale * item['original'])
        G.maximum_num_links[(item['src_id'], item['dst_id'])] = base_allocation
        allocated_links += base_allocation

    # Distribute remainder to largest link groups (most statistically stable)
    remainder = target_total_links - allocated_links
    if remainder > 0:
        # Sort by original link count (largest first)
        link_data.sort(key=lambda x: x['original'], reverse=True)

        # Distribute remainder in round-robin to largest groups
        for i in range(remainder):
            item = link_data[i % len(link_data)]
            G.maximum_num_links[(item['src_id'], item['dst_id'])] += 1

    if verbose:
        total_links = sum(G.maximum_num_links.values())
        print(f"Total requested links: {total_links} (target: {target_total_links})")

    # Create community structure
    find_separated_groups(G, number_of_communities, verbose=verbose,
                         community_size_distribution=community_size_distribution)

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
                                      transitivity_p, valid_communities, pa_scope)

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

def analyze_community_distribution(G, verbose=True):
    """
    Analyze the distribution of communities and groups within communities.

    Provides statistics about:
    - Number of nodes per community
    - Number of groups per community
    - Distribution of groups across communities
    - Community size statistics

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community and group assignments
    verbose : bool, optional
        Whether to print detailed statistics

    Returns
    -------
    dict
        Dictionary containing:
        - 'community_sizes': dict mapping community_id to number of nodes
        - 'community_group_counts': dict mapping community_id to number of unique groups
        - 'community_group_distribution': dict mapping community_id to dict of group_id counts
        - 'group_community_membership': dict mapping group_id to list of communities it appears in
        - 'statistics': dict with summary statistics
    """
    analysis = {
        'community_sizes': {},
        'community_group_counts': {},
        'community_group_distribution': {},
        'group_community_membership': {},
        'statistics': {}
    }
    print(analysis)
    # Analyze each community
    for community_id in range(G.number_of_communities):
        # Count nodes in this community
        nodes_in_community = [node for node, comm in G.nodes_to_communities.items()
                             if comm == community_id]
        analysis['community_sizes'][community_id] = len(nodes_in_community)

        # Count groups in this community
        groups_in_community = G.communities_to_groups.get(community_id, [])

        # Get unique groups and their counts
        from collections import Counter
        group_counts = Counter(groups_in_community)
        analysis['community_group_distribution'][community_id] = dict(group_counts)
        analysis['community_group_counts'][community_id] = len(group_counts)

    # Analyze group membership across communities
    for group_id in G.group_ids:
        communities_with_group = []
        for community_id in range(G.number_of_communities):
            groups_in_comm = G.communities_to_groups.get(community_id, [])
            if group_id in groups_in_comm:
                communities_with_group.append(community_id)
        analysis['group_community_membership'][group_id] = communities_with_group

    # Calculate summary statistics
    community_sizes = list(analysis['community_sizes'].values())
    group_counts = list(analysis['community_group_counts'].values())

    analysis['statistics'] = {
        'total_communities': G.number_of_communities,
        'total_nodes': G.graph.number_of_nodes(),
        'total_groups': len(G.group_ids),
        'avg_nodes_per_community': np.mean(community_sizes) if community_sizes else 0,
        'std_nodes_per_community': np.std(community_sizes) if community_sizes else 0,
        'min_nodes_per_community': min(community_sizes) if community_sizes else 0,
        'max_nodes_per_community': max(community_sizes) if community_sizes else 0,
        'avg_groups_per_community': np.mean(group_counts) if group_counts else 0,
        'std_groups_per_community': np.std(group_counts) if group_counts else 0,
        'min_groups_per_community': min(group_counts) if group_counts else 0,
        'max_groups_per_community': max(group_counts) if group_counts else 0,
    }

    # Calculate how many communities each group appears in
    group_community_counts = [len(comms) for comms in analysis['group_community_membership'].values()]
    analysis['statistics']['avg_communities_per_group'] = np.mean(group_community_counts) if group_community_counts else 0
    analysis['statistics']['std_communities_per_group'] = np.std(group_community_counts) if group_community_counts else 0
    analysis['statistics']['min_communities_per_group'] = min(group_community_counts) if group_community_counts else 0
    analysis['statistics']['max_communities_per_group'] = max(group_community_counts) if group_community_counts else 0
    print(analysis)
    if verbose:
        print("\n" + "="*60)
        print("COMMUNITY DISTRIBUTION ANALYSIS")
        print("="*60)

        stats = analysis['statistics']
        print(f"\nOverall Statistics:")
        print(f"  Total communities: {stats['total_communities']}")
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Total groups: {stats['total_groups']}")

        print(f"\nNodes per Community:")
        print(f"  Mean: {stats['avg_nodes_per_community']:.2f} ± {stats['std_nodes_per_community']:.2f}")
        print(f"  Range: [{stats['min_nodes_per_community']}, {stats['max_nodes_per_community']}]")

        print(f"\nUnique Groups per Community:")
        print(f"  Mean: {stats['avg_groups_per_community']:.2f} ± {stats['std_groups_per_community']:.2f}")
        print(f"  Range: [{stats['min_groups_per_community']}, {stats['max_groups_per_community']}]")

        print(f"\nCommunities per Group:")
        print(f"  Mean: {stats['avg_communities_per_group']:.2f} ± {stats['std_communities_per_group']:.2f}")
        print(f"  Range: [{stats['min_communities_per_group']}, {stats['max_communities_per_group']}]")

        # Show detailed breakdown for first few communities if not too many
        if G.number_of_communities <= 10:
            print(f"\nDetailed Community Breakdown:")
            for community_id in range(G.number_of_communities):
                n_nodes = analysis['community_sizes'][community_id]
                n_groups = analysis['community_group_counts'][community_id]
                group_dist = analysis['community_group_distribution'][community_id]
                print(f"  Community {community_id}: {n_nodes} nodes, {n_groups} unique groups")
                print(f"    Group distribution: {dict(sorted(group_dist.items()))}")
        else:
            print(f"\n(Detailed breakdown omitted for {G.number_of_communities} communities)")
            print(f"Sample - Community 0: {analysis['community_sizes'][0]} nodes, "
                  f"{analysis['community_group_counts'][0]} unique groups")

        print("="*60 + "\n")

    return analysis

def connect_all_within_communities(G, verbose=True):
    """
    Connect all nodes within each community to each other.

    Creates a fully connected graph within each community using vectorized
    operations for maximum efficiency.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with community assignments
    verbose : bool, optional
        Whether to print progress information

    Returns
    -------
    dict
        Statistics about edges created
    """
    verbose = False
    if verbose:
        print("\nConnecting all nodes within communities...")

    stats = {
        'total_edges': 0,
        'edges_per_community': {}
    }

    # OPTIMIZED: Build community membership lookup once
    communities_nodes = [[] for _ in range(G.number_of_communities)]
    for node, comm in G.nodes_to_communities.items():
        communities_nodes[comm].append(node)

    # For each community, connect all nodes within it
    for community_id in range(G.number_of_communities):
        community_nodes = communities_nodes[community_id]

        if len(community_nodes) == 0:
            continue

        # Use itertools.product to generate all pairs efficiently
        # Filter out self-loops inline
        edges_to_add = [(src, dst) for src, dst in product(community_nodes, repeat=2)
                       if src != dst]

        # Batch add edges (much faster than individual add_edge calls)
        G.graph.add_edges_from(edges_to_add)

        edges_added = len(edges_to_add)
        stats['edges_per_community'][community_id] = edges_added
        stats['total_edges'] += edges_added

        # Progress reporting for large numbers of communities
        if (community_id + 1) % 5000 == 0 or community_id == 0:
            print(f"  Connected {community_id + 1}/{G.number_of_communities} communities ({(community_id + 1) / G.number_of_communities * 100:.1f}%)")

        if verbose:
            print(f"  Community {community_id}: {len(community_nodes)} nodes, {edges_added} edges")

    if verbose:
        print(f"  Total edges created: {stats['total_edges']}")

    return stats

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
             pop_column='n', src_suffix='_src', dst_suffix='_dst', link_column='n',
             fill_unfulfilled=True, fully_connect_communities=False,
             community_size_distribution='natural', min_group_size=0, pa_scope='local'):
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
    fill_unfulfilled : bool, optional
        Whether to fill unfulfilled group pairs after initial link creation (default True)
    fully_connect_communities : bool, optional
        Whether to fully connect all nodes within each community, bypassing normal
        link formation process (default False). When True, after community assignment,
        all nodes in the same community are connected to each other.
    community_size_distribution : str or array-like, optional
        Controls community size distribution (default 'natural'):
        - 'natural': Let communities grow naturally based on group affinities
        - 'uniform': Aim for equal-sized communities
        - 'powerlaw': Create power-law distribution (few large, many small communities)
        - array of floats: Custom target proportions (must sum to 1)
    min_group_size : int, optional
        Minimum nodes per group after scaling. Groups smaller than this
        are set to this value to avoid empty groups (default 0)
    pa_scope : str, optional
        Scope of preferential attachment popularity (default 'local'):
        - 'local': popularity stays within the community (intra-community)
        - 'global': popularity spreads across all communities (inter-community)

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

    # Create nodes with stratified allocation
    init_nodes(G, pops_path, scale, pop_column=pop_column, min_group_size=min_group_size)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes")
        print("\nStep 2: Creating edges from interaction patterns...")

    if fully_connect_communities:
        # Skip normal link creation and just fully connect within communities
        if verbose:
            print("\nStep 2a: Setting up communities...")

        # Still need to initialize link tracking and communities
        df_n_group_links = read_file(links_path)
        group_ids = G.group_ids
        G.maximum_num_links = {(i, j): 0 for i in group_ids for j in group_ids}

        # STRATIFIED EDGE ALLOCATION (same as in init_links)
        link_data = []
        total_original_links = 0

        for idx, row in df_n_group_links.iterrows():
            src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
            dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}

            src_nodes, src_id = find_nodes(G, **src_attrs)
            dst_nodes, dst_id = find_nodes(G, **dst_attrs)

            original_links = row[link_column]
            total_original_links += original_links

            link_data.append({
                'src_id': src_id,
                'dst_id': dst_id,
                'original': original_links
            })

        # Calculate target total links and allocate proportionally
        target_total_links = int(scale * total_original_links)
        allocated_links = 0

        # First pass: allocate floor(scale * links) to each pair
        for item in link_data:
            base_allocation = int(scale * item['original'])
            G.maximum_num_links[(item['src_id'], item['dst_id'])] = base_allocation
            allocated_links += base_allocation

        # Distribute remainder to largest link groups
        remainder = target_total_links - allocated_links
        if remainder > 0:
            link_data.sort(key=lambda x: x['original'], reverse=True)
            for i in range(remainder):
                item = link_data[i % len(link_data)]
                G.maximum_num_links[(item['src_id'], item['dst_id'])] += 1

        # Create community structure
        find_separated_groups(G, number_of_communities, verbose=verbose,
                             community_size_distribution=community_size_distribution)

        if verbose:
            print("\nStep 2b: Fully connecting nodes within communities...")

        # Fully connect all nodes within each community
        connect_all_within_communities(G, verbose=verbose)

    else:
        # Normal link creation process
        # Invert preferential attachment for internal representation
        preferential_attachment_fraction = 1 - preferential_attachment

        # Create edges
        init_links(G, links_path, preferential_attachment_fraction, scale,
                  reciprocity, transitivity, number_of_communities, verbose=verbose,
                  src_suffix=src_suffix, dst_suffix=dst_suffix, link_column=link_column,
                  community_size_distribution=community_size_distribution, pa_scope=pa_scope)

        if fill_unfulfilled:
            if verbose:
                print("\nStep 3: Filling remaining unfulfilled group pairs...")

            # Complete any group pairs that didn't reach their target
            fill_unfulfilled_group_pairs(G, reciprocity, verbose=verbose)

    # Save to disk
    G.finalize()

    if verbose:
        # Calculate link fulfillment statistics
        total_requested = sum(G.maximum_num_links.values())
        total_created = G.graph.number_of_edges()
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