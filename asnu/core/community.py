"""
Community detection and management module for ASNU.

This module provides functions for creating, populating, and analyzing
community structures within population networks. It implements agent-based
decision making for community assignment and provides various community
size distributions.

Functions
---------
build_group_pair_to_communities_lookup : Create lookup for group pairs to communities
populate_communities : Assign nodes to communities using group-aligned decision making
find_separated_groups : Identify groups with minimal inter-connections to seed communities
analyze_community_distribution : Analyze distribution of communities and groups
connect_all_within_communities : Create fully connected subgraphs within communities
fill_unfulfilled_group_pairs : Complete group pairs that didn't reach target edge count
export_community_edge_distribution : Export edge distribution to CSV file
export_community_node_distribution : Export node distribution to CSV file
"""
import csv
import random
from collections import Counter
from itertools import product
from tqdm import tqdm
from asnu_rust import process_nodes_capacity

import numpy as np


def build_group_pair_to_communities_lookup(G, verbose=False):
    """
    Create a lookup dictionary mapping each group pair to their shared communities.
    The algorithm will also resamble the structure when some groups are more representative
    in communities than others.

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
        groups_in_community = G.communities_to_groups.get(community_id, [])
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


def populate_communities(G, num_communities, community_size_distribution='natural', new_comm_penalty=100.0):
    """Thin wrapper around populate_communities_capacity with a high new_comm_penalty."""
    populate_communities_capacity(
        G, num_communities,
        community_size_distribution=community_size_distribution,
        new_comm_penalty=new_comm_penalty,
    )

def _process_nodes_capacity_python(G, all_nodes, node_groups, num_communities,
                                   target, total_nodes, target_counts, new_comm_penalty):
    """Pure-Python fallback for capacity-based node assignment using matrix ops.

    Optimized: only evaluates non-empty communities, avoids matrix copies,
    pre-allocates with room to grow.

    Uses soft penalty for budget overshoot (instead of hard feasibility rejection)
    and always includes a new empty community as a candidate, so communities are
    only created when that genuinely minimises the remaining edge-budget distance.
    """
    OVERSHOOT_PENALTY = 10.0

    n_groups = target.shape[0]

    # Pre-allocate with extra room for dynamic growth
    capacity = num_communities + num_communities // 20
    comp = np.zeros((capacity, n_groups), dtype=np.float64)
    community_sizes = np.zeros(capacity, dtype=np.int64)
    num_active = 0  # number of communities actually in use

    # Global accumulated edge reservations: (n_groups, n_groups)
    accumulated = np.zeros((n_groups, n_groups), dtype=np.float64)

    # Pre-compute target rows/cols for each group (avoids repeated indexing)
    target_rows = [target[g, :].copy() for g in range(n_groups)]
    target_cols = [target[:, g].copy() for g in range(n_groups)]

    random.shuffle(all_nodes)

    for node_idx in range(len(all_nodes)):
        node = all_nodes[node_idx]
        g = node_groups[node_idx]

        tgt_row = target_rows[g]
        tgt_col = target_cols[g]
        acc_row_g = accumulated[g, :]
        acc_col_g = accumulated[:, g]

        # Baseline distance: zero-count contribution for all groups h
        # rem_row[h] = tgt_row[h] - acc_row[g,h],  rem_col[h] = tgt_col[h] - acc_col[h,g]
        # base = sum_h(rem_row[h]^2) + sum_{h!=g}(rem_col[h]^2)
        rem_row_sq = (tgt_row - acc_row_g) ** 2      # (n_groups,)
        rem_col_sq = (tgt_col - acc_col_g) ** 2      # (n_groups,)
        base = float(rem_row_sq.sum() + rem_col_sq.sum() - rem_col_sq[g])

        # New empty community has zero composition → distance is exactly sqrt(base),
        # penalized to discourage creating new communities and encourage larger ones.
        new_comm_dist = new_comm_penalty * (base ** 0.5)

        eval_count = num_active  # index of the "new community" candidate

        if num_active > 0:
            active = comp[:num_active]   # (num_active, n_groups) view, no copy

            # Fully vectorized — no Python loop over groups
            # Step 1: soft-penalty costs assuming hyp = acc + count (h != g treatment)
            rem_row = tgt_row - (acc_row_g + active)   # (num_active, n_groups)
            rem_col = tgt_col - (acc_col_g + active)   # (num_active, n_groups)
            eff_row = np.where(rem_row < 0, OVERSHOOT_PENALTY * rem_row ** 2, rem_row ** 2)
            eff_col = np.where(rem_col < 0, OVERSHOOT_PENALTY * rem_col ** 2, rem_col ** 2)

            # Step 2: where count == 0, revert to zero-count baseline (no penalty)
            has_active = active > 0   # (num_active, n_groups)
            eff_row = np.where(has_active, eff_row, rem_row_sq)
            eff_col = np.where(has_active, eff_col, rem_col_sq)

            # Step 3: sum all h, drop col term for h == g (no col term in Rust for self-group)
            dist_sq = eff_row.sum(axis=1) + eff_col.sum(axis=1) - eff_col[:, g]

            # Step 4: fix row term for h == g — Rust uses 2*count, not 1*count
            hyp_gg = acc_row_g[g] + 2.0 * active[:, g]
            rem_gg = tgt_row[g] - hyp_gg
            gg_cost = np.where(rem_gg < 0, OVERSHOOT_PENALTY * rem_gg ** 2, rem_gg ** 2)
            dist_sq += np.where(has_active[:, g], gg_cost - eff_row[:, g], 0.0)

            distances = np.sqrt(np.maximum(0.0, dist_sq))

            # Hard size limit (if specified) — still respected
            if target_counts is not None:
                tc_len = min(num_active, len(target_counts))
                distances[:tc_len][community_sizes[:tc_len] >= target_counts[:tc_len]] = np.inf

            all_distances = np.append(distances, new_comm_dist)
        else:
            all_distances = np.array([new_comm_dist])

        # Temperature-based SA selection (matches Rust schedule)
        temperature = 1.0 - (node_idx / total_nodes)
        if temperature > 0.05:
            valid_mask = np.isfinite(all_distances)
            n_valid = valid_mask.sum()
            if n_valid > 1:
                d = all_distances[valid_mask]
                scaled = -d / (temperature + 1e-10)
                scaled -= scaled.max()
                probs = np.exp(scaled)
                probs /= probs.sum()
                valid_indices = np.where(valid_mask)[0]
                choice = int(np.random.choice(valid_indices, p=probs))
            elif n_valid == 1:
                choice = int(np.where(valid_mask)[0][0])
            else:
                choice = eval_count  # fallback: new community (matches Rust unwrap_or)
        else:
            finite_mask = np.isfinite(all_distances)
            if finite_mask.any():
                choice = int(np.argmin(np.where(finite_mask, all_distances, np.inf)))
            else:
                choice = eval_count  # fallback: new community

        # If the chosen index is beyond current active communities, create a new one
        if choice >= num_active:
            if num_active >= comp.shape[0]:
                extra = max(500, comp.shape[0] // 2)
                comp = np.vstack([comp, np.zeros((extra, n_groups), dtype=np.float64)])
                community_sizes = np.append(community_sizes, np.zeros(extra, dtype=np.int64))
            best_community = num_active
            num_active += 1
        else:
            best_community = choice

        # Update accumulated (only non-zero groups in this community)
        bc_comp = comp[best_community]
        nz_groups = np.nonzero(bc_comp)[0]
        for h in nz_groups:
            count_h = bc_comp[h]
            if h != g:
                accumulated[g, h] += count_h
                accumulated[h, g] += count_h
            else:
                accumulated[g, g] += 2 * count_h

        # Update community composition
        comp[best_community, g] += 1
        community_sizes[best_community] += 1

        # Assign node
        key = (best_community, g)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)
        G.nodes_to_communities[node] = best_community
        if best_community not in G.communities_to_groups:
            G.communities_to_groups[best_community] = []
        G.communities_to_groups[best_community].append(g)

        if (node_idx + 1) % 500 == 0:
            print(f"Capacity assignment: {node_idx + 1}/{total_nodes} nodes "
                  f"({100*(node_idx+1)/total_nodes:.1f}%), {num_active} communities")

    G.number_of_communities = num_active
    # refine_node_assignments(G, target)

def refine_node_assignments(G, target, max_evals=5000):
    """
    Refines node assignments by moving nodes between communities to 
    minimize the global Mean Squared Error against the target.
    """

    n_groups = target.shape[0]
    # 1. Build initial community composition matrix
    max_comm = max(G.nodes_to_communities.values()) + 1
    comp = np.zeros((max_comm, n_groups))
    for node, comm in G.nodes_to_communities.items():
        g = G.nodes_to_group[node] # Assuming G stores the node's group
        comp[comm, g] += 1

    def get_global_error(composition):
        # Calculates global group-to-group counts
        # Error = sum( (Target - sum_over_communities(C_g * C_h)) ^ 2 )
        current_acc = np.zeros((n_groups, n_groups))
        for c in range(len(composition)):
            c_vec = composition[c].reshape(-1, 1)
            # Outer product represents all pairs in the community
            comm_edges = np.dot(c_vec, c_vec.T)
            # Diagonal adjustment: internal edges are counted differently in the original code
            np.fill_diagonal(comm_edges, composition[c] * (composition[c] - 1))
            current_acc += comm_edges
        return np.sum((target - current_acc) ** 2)

    current_error = get_global_error(comp)
    nodes = list(G.nodes_to_communities.keys())
    u_s = random.choices(nodes, k=max_evals)
    new_comms = random.choices([i for i in range(max_comm)], k=max_evals)
    for i in range(max_evals):
        # Pick a random node and a potential new community
        u = u_s[i]
        old_comm = G.nodes_to_communities[u]
        new_comm = new_comms[i]
        
        if old_comm == new_comm: continue
        
        g = G.nodes_to_group[u]
        
        # Simulate Move
        comp[old_comm, g] -= 1
        comp[new_comm, g] += 1
        
        new_error = get_global_error(comp)
        
        if new_error < current_error:
            # Keep the move
            current_error = new_error
            G.nodes_to_communities[u] = new_comm
        else:
            # Revert the move
            comp[old_comm, g] += 1
            comp[new_comm, g] -= 1

        if i % 100 == 0:
            print(f"Refinement Iteration {i}: Global Error = {current_error:.2f}")

    return G

def populate_communities_capacity(G, num_communities, community_size_distribution='natural', new_comm_penalty=3.0):
    """
    Assign nodes to communities by matching absolute edge counts against
    maximum_num_links budget, with feasibility constraints ensuring
    communities can be fully connected without exceeding the budget.

    Uses same SA temperature schedule as populate_communities() but with
    capacity-based distance (absolute edge counts, not probabilities).

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Initial number of communities (may grow if needed)
    community_size_distribution : str or array-like, optional
        Controls community size distribution
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    # Build target matrix from maximum_num_links
    target = np.zeros((n_groups, n_groups), dtype=np.float64)
    for (i, j), count in G.maximum_num_links.items():
        target[i, j] = count

    # Compute probability matrix (needed for JSON serialization / load_communities)
    epsilon = 1e-5
    row_sums = target.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = epsilon
    G.probability_matrix = target / row_sums
    G.number_of_communities = num_communities

    # Target community sizes
    if isinstance(community_size_distribution, (list, np.ndarray)):
        target_sizes = np.array(community_size_distribution)
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'powerlaw':
        ranks = np.arange(1, num_communities + 1)
        sizes = 1.0 / (ranks ** 1)
        target_sizes = sizes / sizes.sum()
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'uniform':
        target_sizes = np.ones(num_communities) / num_communities
        target_counts = (target_sizes * total_nodes).astype(np.int32)
    elif community_size_distribution == 'normal':
        mean_size = total_nodes / num_communities
        std_size = mean_size * 0.3   # 30% std → ~normal spread around the mean
        raw = np.random.normal(mean_size, std_size, num_communities)
        raw = np.maximum(raw, 1.0)   # no zero-size communities
        # Scale so caps sum to total_nodes (communities collectively cover all nodes)
        target_counts = np.maximum(
            np.round(raw * (total_nodes / raw.sum())).astype(np.int32), 1
        )
    else:
        target_counts = None

    # Initialize community structures
    for community_idx in range(num_communities):
        for group_id in range(n_groups):
            G.communities_to_nodes[(community_idx, group_id)] = []
        G.communities_to_groups[community_idx] = []

    # Shuffle nodes
    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    # Try Rust backend, fall back to Python
    try:  
        print("Using Rust backend for capacity-based node processing...")

        budget = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}

        assignments = process_nodes_capacity(
            all_nodes.astype(np.int64),
            node_groups.astype(np.int64),
            budget,
            n_groups,
            num_communities,
            target_counts if target_counts is not None else None,
            total_nodes,
            new_comm_penalty,
        )

        # Populate G structures from assignments
        for i in range(len(all_nodes)):
            node = int(all_nodes[i])
            comm = int(assignments[i])
            group = int(node_groups[i])
            key = (comm, group)
            if key not in G.communities_to_nodes:
                G.communities_to_nodes[key] = []
            G.communities_to_nodes[key].append(node)
            G.nodes_to_communities[node] = comm
            if comm not in G.communities_to_groups:
                G.communities_to_groups[comm] = []
            G.communities_to_groups[comm].append(group)

        G.number_of_communities = int(assignments.max()) + 1 if len(assignments) > 0 else 0
    except ImportError:
        print("Using Python fallback for capacity-based node processing...")

        _process_nodes_capacity_python(
            G, all_nodes, node_groups, num_communities,
            target, total_nodes, target_counts, new_comm_penalty,
        )

    print(f"\nCapacity-based community assignment complete: "
          f"{total_nodes} nodes → {G.number_of_communities} communities")


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

    def _fill_from_pool(src_pool, dst_pool, src_id, dst_id, needed):
        """Add up to `needed` edges by batch-sampling from src_pool/dst_pool."""
        if not src_pool or not dst_pool or needed <= 0:
            return
        src_arr = np.array(src_pool)
        dst_arr = np.array(dst_pool)
        batch = max(needed * 4, 512)   # oversample to tolerate rejections

        for _ in range(10):            # at most 10 numpy draws per pool
            if G.existing_num_links[(src_id, dst_id)] >= maximum:
                return
            srcs = np.random.choice(src_arr, size=batch).tolist()
            dsts = np.random.choice(dst_arr, size=batch).tolist()
            added_this_round = 0
            for s, d in zip(srcs, dsts):
                if G.existing_num_links[(src_id, dst_id)] >= maximum:
                    return
                if s == d or G.graph.has_edge(s, d):
                    continue
                G.graph.add_edge(s, d)
                G.existing_num_links[(src_id, dst_id)] += 1
                stats['edges_added'] += 1
                added_this_round += 1
                if reciprocity_p > 0 and random.random() < reciprocity_p:
                    if (G.existing_num_links[(dst_id, src_id)] < G.maximum_num_links[(dst_id, src_id)]
                            and not G.graph.has_edge(d, s)):
                        G.graph.add_edge(d, s)
                        G.existing_num_links[(dst_id, src_id)] += 1
                        stats['reciprocal_edges_added'] += 1
                        if dst_id == src_id:
                            stats['edges_added'] += 1
            if added_this_round == 0:
                return  # pool is saturated, stop early

    # Add edges to complete unfulfilled pairs
    total_needed = sum(max - ex for _, _, ex, max in unfulfilled_pairs)
    if unfulfilled_pairs:
        pbar = (tqdm(total=total_needed, unit='edge', desc='Filling pairs', dynamic_ncols=True)
                if tqdm and verbose else None)
        edges_before = stats['edges_added']

        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            existing = G.existing_num_links.get((src_id, dst_id), 0)
            maximum = G.maximum_num_links.get((src_id, dst_id), 0)
            needed = maximum - existing
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            edges_before_pair = stats['edges_added']

            # --- Phase 1: intra-community pool ---
            src_comm = {}
            for node in src_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    src_comm.setdefault(comm, []).append(node)
            dst_comm = {}
            for node in dst_nodes:
                comm = G.nodes_to_communities.get(node)
                if comm is not None:
                    dst_comm.setdefault(comm, []).append(node)

            shared = list(set(src_comm) & set(dst_comm))
            if shared:
                # Build numpy arrays per community for fast sampling
                src_arrs = [np.array(src_comm[c]) for c in shared]
                dst_arrs = [np.array(dst_comm[c]) for c in shared]
                # Weight community selection by pool size product
                weights = np.array([len(s) * len(d) for s, d in zip(src_arrs, dst_arrs)], dtype=float)
                weights /= weights.sum()
                n_comm = len(shared)
                batch = max((maximum - G.existing_num_links[(src_id, dst_id)]) * 4, 512)
                for _ in range(10):
                    if G.existing_num_links[(src_id, dst_id)] >= maximum:
                        break
                    # Sample community indices, then sample one node from each side
                    comm_indices = np.random.choice(n_comm, size=batch, p=weights)
                    added_this_round = 0
                    for ci in comm_indices:
                        if G.existing_num_links[(src_id, dst_id)] >= maximum:
                            break
                        s = src_arrs[ci][np.random.randint(len(src_arrs[ci]))]
                        d = dst_arrs[ci][np.random.randint(len(dst_arrs[ci]))]
                        if s == d or G.graph.has_edge(s, d):
                            continue
                        G.graph.add_edge(s, d)
                        G.existing_num_links[(src_id, dst_id)] += 1
                        stats['edges_added'] += 1
                        added_this_round += 1
                        if reciprocity_p > 0 and random.random() < reciprocity_p:
                            if (G.existing_num_links[(dst_id, src_id)] < G.maximum_num_links[(dst_id, src_id)]
                                    and not G.graph.has_edge(d, s)):
                                G.graph.add_edge(d, s)
                                G.existing_num_links[(dst_id, src_id)] += 1
                                stats['reciprocal_edges_added'] += 1
                                if dst_id == src_id:
                                    stats['edges_added'] += 1
                    if added_this_round == 0:
                        break  # pools saturated

            # # --- Phase 2: cross-community fallback ---
            _fill_from_pool(src_nodes, dst_nodes, src_id, dst_id,
                            needed=maximum - G.existing_num_links[(src_id, dst_id)])

            if pbar is not None:
                pbar.update(stats['edges_added'] - edges_before_pair)

        if pbar is not None:
            pbar.close()

    if verbose:
        print(f"  Edges added: {stats['edges_added']}")
        print(f"  Reciprocal edges added: {stats['reciprocal_edges_added']}")

    return stats


def create_communities(pops_path, links_path, scale, number_of_communities=None,
                       output_path='communities.json', community_size_distribution='natural',
                       pop_column='n', src_suffix='_src', dst_suffix='_dst',
                       link_column='n', min_group_size=0, verbose=True,
                       new_comm_penalty=3.0):
    """
    Create community assignments and save them to a JSON file.

    This is a standalone step that can be run independently from network
    generation. The output file can later be passed to generate() via
    the community_file parameter.

    Parameters
    ----------
    pops_path : str
        Path to population data (CSV or Excel)
    links_path : str
        Path to interaction data (CSV or Excel)
    scale : float
        Population scaling factor
    number_of_communities : int or None
        Number of communities to create. Required for mode='probability'.
        For mode='capacity', this is the initial count (may grow dynamically).
    output_path : str
        Path for the output JSON file
    community_size_distribution : str or array-like, optional
        Controls community size distribution (default 'natural')
    pop_column : str, optional
        Column name for population counts (default 'n')
    src_suffix : str, optional
        Suffix for source group columns (default '_src')
    dst_suffix : str, optional
        Suffix for destination group columns (default '_dst')
    link_column : str, optional
        Column name for link counts (default 'n')
    min_group_size : int, optional
        Minimum nodes per group after scaling (default 0)
    verbose : bool, optional
        Whether to print progress information
    mode : str, optional
        'probability' (default): match probability distributions
        'capacity': match absolute edge counts with feasibility constraints

    Returns
    -------
    str
        Path to the saved JSON file
    """
    import json
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    if verbose:
        print("="*60)
        print(f"COMMUNITY CREATION (penalty={new_comm_penalty})")
        print("="*60)

    # Create a temporary graph and initialize nodes
    G = NetworkXGraph()
    init_nodes(G, pops_path, scale, pop_column=pop_column)

    if verbose:
        print(f"  Created {G.graph.number_of_nodes()} nodes in {len(G.group_ids)} groups")

    # Compute maximum link counts (needed for affinity matrix)
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                                dst_suffix=dst_suffix, link_column=link_column,
                                verbose=verbose)

    if verbose:
        print(f"\nAssigning nodes (penalty={new_comm_penalty}, initial communities={number_of_communities})...")
    populate_communities_capacity(G, number_of_communities,
                                  community_size_distribution=community_size_distribution,
                                  new_comm_penalty=new_comm_penalty)

    # Serialize to JSON (convert numpy types to native Python types)
    data = {
        'number_of_communities': int(G.number_of_communities),
        'probability_matrix': G.probability_matrix.tolist(),
        'nodes_to_communities': {
            str(k): int(v) for k, v in G.nodes_to_communities.items()
        },
        'communities_to_nodes': {
            str(k): [int(n) for n in v] for k, v in G.communities_to_nodes.items()
        },
        'communities_to_groups': {
            str(k): [int(g) for g in v] for k, v in G.communities_to_groups.items()
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"\nCommunity assignments saved to {output_path}")
        print("="*60 + "\n")

    return output_path


def create_hierarchical_community_file(
    household_community_file,
    pops_path,
    links_path,
    scale,
    target_num_communities,
    output_path,
    pop_column='n',
    src_suffix='_src',
    dst_suffix='_dst',
    link_column='n',
    verbose=True,
):
    """
    Create a community file where household communities are grouped into
    super-communities (e.g. 5000 household communities → 5 buren communities).

    Nodes in the same household community are guaranteed to be in the same
    super-community. Uses block assignment for locality preservation.

    Parameters
    ----------
    household_community_file : str
        Path to the household community JSON (from create_communities())
    pops_path : str
        Path to population CSV (for computing probability matrix)
    links_path : str
        Path to interaction CSV for the target layer
    scale : float
        Population scaling factor
    target_num_communities : int
        Number of super-communities to create
    output_path : str
        Path to write the output JSON
    """
    import json
    import math
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    # Load household community assignments
    with open(household_community_file, 'r', encoding='utf-8') as f:
        hh_data = json.load(f)

    hh_num_communities = hh_data['number_of_communities']
    hh_nodes_to_communities = {int(k): v for k, v in hh_data['nodes_to_communities'].items()}

    # Block assignment: group household communities into super-communities
    block_size = math.ceil(hh_num_communities / target_num_communities)

    super_nodes_to_communities = {}
    for node, hh_comm in hh_nodes_to_communities.items():
        super_comm = min(hh_comm // block_size, target_num_communities - 1)
        super_nodes_to_communities[node] = super_comm

    # Compute probability matrix from this layer's own link data
    G_temp = NetworkXGraph('_temp_hierarchical')
    init_nodes(G_temp, pops_path, scale, pop_column=pop_column)
    _compute_maximum_num_links(G_temp, links_path, scale,
                               src_suffix=src_suffix, dst_suffix=dst_suffix,
                               link_column=link_column, verbose=False)

    n_groups = len(G_temp.group_ids)
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G_temp.maximum_num_links.items():
        affinity[i, j] = count
    row_sums = affinity.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-10
    probability_matrix = affinity / row_sums

    # Clean up temp directory
    import shutil, os
    if os.path.exists('_temp_hierarchical'):
        shutil.rmtree('_temp_hierarchical')

    # Write community JSON
    data = {
        'number_of_communities': target_num_communities,
        'probability_matrix': probability_matrix.tolist(),
        'nodes_to_communities': {str(k): int(v) for k, v in super_nodes_to_communities.items()},
        'communities_to_nodes': {},
        'communities_to_groups': {},
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    if verbose:
        print(f"  Hierarchical communities: {hh_num_communities} household → "
              f"{target_num_communities} super-communities (block_size={block_size})")
        print(f"  Saved to {output_path}")

    return output_path


def load_communities(G, community_file_path):
    """
    Load community assignments from a JSON file into a NetworkXGraph object.

    Node-to-community assignments are loaded from the file, but
    communities_to_nodes and communities_to_groups are recalculated from
    the actual graph nodes. This ensures correctness when the graph has
    fewer groups than when the community file was created.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes already initialized
    community_file_path : str
        Path to the JSON file created by create_communities()
    """
    import json

    with open(community_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G.number_of_communities = data['number_of_communities']
    G.probability_matrix = np.array(data['probability_matrix'])

    # Load node-to-community assignments, keeping only nodes present in graph
    graph_nodes = set(G.graph.nodes)

    G.nodes_to_communities = {}
    for k, v in data['nodes_to_communities'].items():
        node = int(k)
        if node in graph_nodes:
            G.nodes_to_communities[node] = v

    unassigned = graph_nodes - set(G.nodes_to_communities.keys())
    if unassigned:
        print(f"Warning: {len(unassigned)} graph nodes have no community assignment")

    # Recalculate communities_to_nodes and communities_to_groups
    # from the actual graph nodes, so they reflect current groups
    communities_groups = {}

    for node, community_id in G.nodes_to_communities.items():
        group_id = G.nodes_to_group[node]
        key = (community_id, group_id)
        if key not in G.communities_to_nodes:
            G.communities_to_nodes[key] = []
        G.communities_to_nodes[key].append(node)

        if community_id not in communities_groups:
            communities_groups[community_id] = set()
        communities_groups[community_id].add(group_id)

    G.communities_to_groups = {
        comm: list(groups) for comm, groups in communities_groups.items()
    }
