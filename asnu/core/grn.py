import numpy as np
import math
import random

def establish_links(G, src_nodes, dst_nodes, src_id, dst_id,
                  target_link_count, fraction, reciprocity_p, transitivity_p, valid_communities=None):
    """
    Create edges between source and destination nodes with preferential attachment.

    Connects nodes from source and destination groups using a bounded preferential
    attachment model. Supports reciprocity and transitivity for realistic network
    structure.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with network data and metadata
    src_nodes : list
        Source node IDs
    dst_nodes : list
        Destination node IDs
    src_id : int
        Source group ID
    dst_id : int
        Destination group ID
    target_link_count : int
        Target number of edges to create
    fraction : float
        Preferential attachment parameter (0-1)
    reciprocity_p : float
        Probability of creating reciprocal edges (0-1)
    transitivity_p : float
        Probability of creating transitive edges (0-1)
    valid_communities : list, optional
        Communities shared by both groups (precomputed for efficiency)

    Returns
    -------
    bool
        True if target was met, False if exceeded
    """
    link_n_check = True
    attempts = 0
    max_attempts = target_link_count * 10

    # Get current link count for this group pair
    num_links = G.existing_num_links.get((src_id, dst_id), 0)

    # Check if already over target
    if num_links > target_link_count:
        link_n_check = False

    # Storage for destination nodes with preferential attachment
    d_nodes_bins = {}

    # Use precomputed communities for this group pair
    possible_communities = valid_communities

    if not possible_communities:
        return link_n_check

    # Cache for source node lists by community (created as needed)
    src_node_lists = {}

    # Preselect communities in batches for efficiency
    batch_size = 10000
    community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
    batch_idx = 0

    # Create edges until we reach the target
    while num_links < target_link_count and attempts < max_attempts:

        # Get next community from the batch
        community_id = community_batch[batch_idx]
        batch_idx += 1

        # Refill batch when exhausted
        if batch_idx >= batch_size:
            community_batch = np.random.choice(possible_communities, size=batch_size, replace=True)
            batch_idx = 0

        # Initialize node lists for this community on first use
        if community_id not in src_node_lists:
            # Get source nodes in this community
            src_node_lists[community_id] = G.communities_to_nodes[(community_id, src_id)]

            # Create initial pool of destination nodes for preferential attachment
            dst_community_nodes = G.communities_to_nodes[(community_id, dst_id)]
            if dst_community_nodes:
                sample_size = math.ceil(len(dst_community_nodes) * fraction)
                d_nodes_bins[community_id] = list(np.random.choice(dst_community_nodes,
                                                                   size=sample_size,
                                                                   replace=False))

        # Select random source and destination nodes from this community
        s = random.choice(src_node_lists[community_id])
        d_from_db = random.choice(d_nodes_bins[community_id])

        # Add edge if valid (no self-loops, no duplicates)
        if s != d_from_db and not G.graph.has_edge(s, d_from_db):
            G.graph.add_edge(s, d_from_db)
            num_links += 1
            G.existing_num_links[(src_id, dst_id)] = num_links

            # Reciprocity
            if random.uniform(0,1) < reciprocity_p:
                if G.existing_num_links[(dst_id, src_id)] < G.maximum_num_links[(dst_id, src_id)] and not G.graph.has_edge(d_from_db, s):
                    G.graph.add_edge(d_from_db, s)
                    G.existing_num_links[(dst_id, src_id)] += 1
                    if (dst_id == src_id):
                        num_links += 1
                        G.existing_num_links[(src_id, dst_id)] = num_links

            # Preferential attachment: add popular nodes back to the pool
            if random.uniform(0,1) > fraction and fraction != 1:
                d_nodes_bins[community_id].append(d_from_db)

            # Add edges to neighbors (clustering effect)
            if transitivity_p < random.uniform(0,1):
                continue
            for n in G.graph.neighbors(d_from_db):
                if s == n:
                    continue
                n_id = G.nodes_to_group[n]
                if (src_id, n_id) in G.maximum_num_links:
                    if G.existing_num_links[(src_id, n_id)] < G.maximum_num_links[(src_id, n_id)]:
                        if not G.graph.has_edge(s, n):
                            G.graph.add_edge(s, n)
                            G.existing_num_links[(src_id, n_id)] += 1
                            # Also count toward main target if same destination group
                            if n_id == dst_id:
                                num_links += 1
                                G.existing_num_links[(src_id, dst_id)] = num_links
                            # Reciprocity
                            if random.uniform(0,1) < reciprocity_p:
                                if not G.graph.has_edge(n, s) and G.existing_num_links[(n_id, src_id)] < G.maximum_num_links[(n_id, src_id)]:
                                    G.graph.add_edge(n, s)
                                    G.existing_num_links[(n_id, src_id)] += 1
                                    if (n_id == src_id) & (src_id == dst_id):
                                        num_links += 1
                                        G.existing_num_links[(src_id, dst_id)] = num_links

        attempts += 1

    return link_n_check
