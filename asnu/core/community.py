"""
Community detection and management module for ASNU.

Functions
---------
build_group_pair_to_communities_lookup : Create lookup for group pairs to communities
find_separated_groups : Identify groups with minimal inter-connections to seed communities
connect_all_within_communities : Create fully connected subgraphs within communities
fill_unfulfilled_group_pairs : Complete group pairs that didn't reach target edge count
populate_communities_capacity : Assign nodes to communities (capacity_fast mode)
populate_communities_segregation : Assign nodes to communities (segregation mode)
load_communities : Load community assignments from JSON file
"""
import random
from itertools import product
from tqdm import tqdm

import numpy as np

def create_communities(pops_path, links_path, scale, number_of_communities=None,
                       output_path='communities.json',
                       community_size_distribution='natural',
                       pop_column='n', src_suffix='_src', dst_suffix='_dst',
                       link_column='n', verbose=True,
                       new_comm_penalty=3.0,
                       isolation_threshold=0.05, refine_swaps=300_000, mode='capacity_fast'):
    import json
    from asnu.core.graph import NetworkXGraph
    from asnu.core.generate import init_nodes, _compute_maximum_num_links

    G = NetworkXGraph()
    init_nodes(G, pops_path, scale, pop_column=pop_column)
    _compute_maximum_num_links(G, links_path, scale, src_suffix=src_suffix,
                               dst_suffix=dst_suffix, link_column=link_column,
                               verbose=verbose)

    if mode == 'segregation':
        populate_communities_segregation(G, number_of_communities, refine_swaps,
                                         isolation_threshold=isolation_threshold)
    else:  # capacity_fast
        populate_communities_capacity(G, number_of_communities,
                                      community_size_distribution=community_size_distribution,
                                      new_comm_penalty=new_comm_penalty,
                                      refine_swaps=refine_swaps,
                                      fast=True)

    # ── Serialize to JSON ─────────────────────────────────────────────────
    _pm = G.probability_matrix
    _n = _pm.shape[0] if hasattr(_pm, 'shape') else 0

    if _n == 0:
        _pm_serial = {'sparse': True, 'shape': [0, 0], 'rows': [], 'cols': [], 'vals': []}
    elif _n > 500:
        import scipy.sparse as _sp
        if _sp.issparse(_pm):
            _coo = _pm.tocoo()
            _pm_serial = {
                'sparse': True,
                'shape': [int(_n), int(_n)],
                'rows': _coo.row.tolist(),
                'cols': _coo.col.tolist(),
                'vals': _coo.data.tolist(),
            }
        else:
            _nz = np.argwhere(_pm > 0)
            _pm_serial = {
                'sparse': True,
                'shape': [int(_n), int(_n)],
                'rows': _nz[:, 0].tolist(),
                'cols': _nz[:, 1].tolist(),
                'vals': _pm[_nz[:, 0], _nz[:, 1]].tolist(),
            }
    else:
        _pm_serial = _pm.tolist()

    data = {
        'number_of_communities': int(G.number_of_communities),
        'probability_matrix': _pm_serial,
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
    if hasattr(G, 'node_coordinates') and G.node_coordinates:
        data['node_coordinates'] = {str(k): float(v) for k, v in G.node_coordinates.items()}

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print(f"\nCommunity assignments saved to {output_path}")
    return output_path

def build_group_pair_to_communities_lookup(G, verbose=False):
    if verbose:
        print("Building community lookup for group pairs...")

    if G.number_of_communities == 1:
        from collections import defaultdict as _dd

        class _SingleCommLookup(_dd):
            def __missing__(self, key):
                return [0]

        if verbose:
            print("  Single community: using lazy lookup (skipping O(n^2) build)")
        return _SingleCommLookup(list)

    from collections import defaultdict as _dd
    group_to_comms = _dd(list)
    for community_id in range(G.number_of_communities):
        for g in G.communities_to_groups.get(community_id, []):
            group_to_comms[g].append(community_id)

    group_pair_to_communities = {}
    for (src_id, dst_id) in G.maximum_num_links:
        sc = group_to_comms.get(src_id)
        dc = group_to_comms.get(dst_id)
        if not sc or not dc:
            continue
        shared = list(set(sc) & set(dc))
        if shared:
            group_pair_to_communities[(src_id, dst_id)] = shared

    if verbose:
        avg_communities = np.mean([len(v) for v in group_pair_to_communities.values()]) if group_pair_to_communities else 0
        print(f"  Found {len(group_pair_to_communities)} group pairs")
        print(f"  Average communities per pair: {avg_communities:.1f}")

    return group_pair_to_communities


def find_separated_groups(G, num_communities):
    """
    Select groups with minimal mutual interaction for community seeding.

    Uses greedy farthest-point selection: starts with the group that has the
    lowest total interaction, then repeatedly picks the group with the least
    accumulated interaction toward already-selected groups.
    """
    import heapq
    from collections import defaultdict

    groups_with_nodes = [g for g in G.group_ids if G.group_to_nodes.get(g)]
    if not groups_with_nodes:
        return []
    groups_set = set(groups_with_nodes)
    group_index = {g: i for i, g in enumerate(groups_with_nodes)}

    neighbors = defaultdict(dict)
    group_totals = defaultdict(int)
    for (a, b), cnt in G.maximum_num_links.items():
        a_in = a in groups_set
        b_in = b in groups_set
        if a_in:
            group_totals[a] += cnt
        if b_in:
            group_totals[b] += cnt
        if a_in and b_in:
            nb = neighbors[a]; nb[b] = nb.get(b, 0) + cnt
            nb = neighbors[b]; nb[a] = nb.get(a, 0) + cnt

    interaction_sum = dict.fromkeys(groups_with_nodes, 0)
    selected_groups = set()
    used_nodes = set()
    selected = []
    n_target = min(num_communities, len(groups_with_nodes))

    heap = None

    for _ in range(n_target):
        if not selected:
            best_group = min(groups_with_nodes, key=lambda g: group_totals.get(g, 0))
        else:
            if heap is None:
                heap = [(interaction_sum[g], group_index[g], g)
                        for g in groups_with_nodes if g not in selected_groups]
                heapq.heapify(heap)
            best_group = None
            while heap:
                val, _, g = heapq.heappop(heap)
                if g in selected_groups:
                    continue
                if val != interaction_sum[g]:
                    continue
                best_group = g
                break
            if best_group is None:
                break

        candidates = [n for n in G.group_to_nodes[best_group] if n not in used_nodes]
        if not candidates:
            selected_groups.add(best_group)
            continue
        node = random.choice(candidates)
        selected.append((best_group, node))
        selected_groups.add(best_group)
        used_nodes.add(node)

        nb = neighbors.get(best_group)
        if nb:
            for g, w in nb.items():
                if g in selected_groups:
                    continue
                interaction_sum[g] += w
                if heap is not None:
                    heapq.heappush(heap, (interaction_sum[g], group_index[g], g))

    if len(selected) < num_communities:
        extra_candidates = sorted(
            [(g, n) for g in groups_with_nodes for n in G.group_to_nodes[g]],
            key=lambda gn: group_totals.get(gn[0], 0),
        )
        for best_group, node in extra_candidates:
            if len(selected) >= num_communities:
                break
            if node not in used_nodes:
                selected.append((best_group, node))
                used_nodes.add(node)

    return selected


def populate_communities_capacity(G, num_communities, community_size_distribution='natural',
                                   new_comm_penalty=3.0, allow_new_communities=True, fast=False,
                                   refine_swaps=300_000, refine_overshoot_penalty=10.0,
                                   refine_seed=42):
    """
    Assign nodes to communities by matching absolute edge counts against
    maximum_num_links budget. Uses capacity_fast (Rust) backend.

    Parameters
    ----------
    G : NetworkXGraph
        Graph object with nodes and group assignments
    num_communities : int
        Initial number of communities
    community_size_distribution : str or array-like, optional
        Controls community size distribution
    refine_swaps : int, optional
        Number of swap-refinement iterations after assignment. 0 disables.
    refine_overshoot_penalty : float, optional
        Multiplier for over-budget cells in the swap loss (default 10.0).
    refine_seed : int, optional
        RNG seed for reproducible refinement.
    """
    total_nodes = len(list(G.graph.nodes))
    n_groups = int(len(G.group_ids))

    G.number_of_communities = num_communities

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
        std_size = mean_size * 0.3
        raw = np.random.normal(mean_size, std_size, num_communities)
        raw = np.maximum(raw, 1.0)
        target_counts = np.maximum(
            np.round(raw * (total_nodes / raw.sum())).astype(np.int32), 1
        )
    else:
        target_counts = None

    if num_communities == 1 and target_counts is None:
        all_nodes_fp = np.array(list(G.graph.nodes))
        for node in all_nodes_fp:
            group = G.nodes_to_group[node]
            G.communities_to_nodes.setdefault((0, group), []).append(int(node))
            G.nodes_to_communities[int(node)] = 0
            if 0 not in G.communities_to_groups:
                G.communities_to_groups[0] = []
            G.communities_to_groups[0].append(group)
        G.number_of_communities = 1
        G.probability_matrix = np.zeros((0, 0))
        print(f"\nCapacity-based community assignment complete: "
              f"{total_nodes} nodes -> 1 community (fast path)")
        return

    from scipy.sparse import csr_matrix as _csr_mat
    if G.maximum_num_links:
        _ri, _ci, _vi = zip(*[(i, j, v) for (i, j), v in G.maximum_num_links.items() if v > 0]) or ([], [], [])
    else:
        _ri, _ci, _vi = [], [], []
    target_sp = _csr_mat((list(_vi), (list(_ri), list(_ci))), shape=(n_groups, n_groups), dtype=np.float64) if _vi else _csr_mat((n_groups, n_groups), dtype=np.float64)

    epsilon = 1e-5
    row_sums = np.asarray(target_sp.sum(axis=1)).flatten() + epsilon
    target = np.asarray(target_sp.multiply(1.0 / row_sums[:, np.newaxis]).todense())
    G.probability_matrix = target

    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    group_counts: dict = {}
    within_pos = np.empty(len(all_nodes), dtype=np.int64)
    for i, g in enumerate(node_groups):
        within_pos[i] = group_counts.get(g, 0)
        group_counts[g] = within_pos[i] + 1

    order = np.argsort(within_pos, kind='stable')
    all_nodes = all_nodes[order]
    node_groups = node_groups[order]

    seeds = find_separated_groups(G, num_communities)
    seed_node_set = {node for _, node in seeds}

    initial_comp = []
    for comm_id, (group, node) in enumerate(seeds):
        G.communities_to_nodes.setdefault((comm_id, group), []).append(node)
        G.nodes_to_communities[node] = comm_id
        G.communities_to_groups.setdefault(comm_id, []).append(group)
        initial_comp.append({group: 1})

    mask = np.array([int(n) not in seed_node_set for n in all_nodes])
    sa_nodes = all_nodes[mask]
    sa_groups = node_groups[mask]
    sa_total_nodes = len(sa_nodes)
    print(f"  Pre-seeded {len(seeds)} communities; {sa_total_nodes} nodes remaining")

    from asnu_rust import process_nodes_capacity_fast as _rust_fn
    print("Using Rust fast assignment...")

    budget = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}
    rust_initial_comp = {comm_id: {int(g): int(c) for g, c in d.items()}
                         for comm_id, d in enumerate(initial_comp)}

    assignments = _rust_fn(
        sa_nodes.astype(np.int64),
        sa_groups.astype(np.int64),
        budget,
        n_groups,
        num_communities,
        target_counts if target_counts is not None else None,
        sa_total_nodes,
        float('inf') if not allow_new_communities else new_comm_penalty,
        rust_initial_comp if initial_comp else None,
    )

    for i in range(len(sa_nodes)):
        node = int(sa_nodes[i])
        comm = int(assignments[i])
        group = int(sa_groups[i])
        G.communities_to_nodes.setdefault((comm, group), []).append(node)
        G.nodes_to_communities[node] = comm
        G.communities_to_groups.setdefault(comm, []).append(group)

    all_assigned_comms = list(assignments) + list(range(len(seeds)))
    G.number_of_communities = max(all_assigned_comms) + 1 if all_assigned_comms else 0

    print(f"\nCapacity-based community assignment complete: "
          f"{total_nodes} nodes -> {G.number_of_communities} communities")

    if refine_swaps > 0 and G.number_of_communities > 1:
        try:
            from asnu_rust import refine_communities_move
            print(f"\nRefining communities with {refine_swaps} swap iterations...")

            refine_nodes = np.array(list(G.graph.nodes), dtype=np.int64)
            refine_node_groups = np.array(
                [G.nodes_to_group[n] for n in refine_nodes], dtype=np.int64
            )
            refine_assignments = np.array(
                [G.nodes_to_communities[int(n)] for n in refine_nodes], dtype=np.int64
            )
            refine_budget = {(int(k[0]), int(k[1])): int(v)
                             for k, v in G.maximum_num_links.items() if v > 0}

            new_assignments = refine_communities_move(
                refine_assignments,
                refine_node_groups,
                refine_budget,
                n_groups,
                G.number_of_communities,
                refine_swaps,
                10,
                refine_seed,
            )

            G.communities_to_nodes = {}
            G.communities_to_groups = {}
            G.nodes_to_communities = {}
            for i in range(len(refine_nodes)):
                node = int(refine_nodes[i])
                comm = int(new_assignments[i])
                group = int(refine_node_groups[i])
                G.nodes_to_communities[node] = comm
                G.communities_to_nodes.setdefault((comm, group), []).append(node)
                G.communities_to_groups.setdefault(comm, []).append(group)

            G.number_of_communities = int(new_assignments.max()) + 1

            _rng = np.random.default_rng(42)
            comm_coords = _rng.uniform(0.0, 1.0, size=G.number_of_communities)
            node_coordinates = {}
            for node_int, comm in G.nodes_to_communities.items():
                theta_c = float(comm_coords[int(comm)])
                node_coordinates[int(node_int)] = (theta_c) % 1.0
            G.node_coordinates = node_coordinates

            print("Refinement complete.")
        except ImportError:
            print("  refine_communities_move not available; skipping refinement.")


def populate_communities_segregation(G, num_communities, refine_swaps,
                                      isolation_threshold=0.05, seed=42):
    """
    Segregation-driven hierarchical community assignment.

    Measures isolation per characteristic from the edge budget, then partitions
    communities hierarchically: the most segregated characteristic anchors the
    primary split, the second most segregated subdivides within that.
    """
    n_groups = len(G.group_ids)
    rng = np.random.default_rng(seed)
    sorted_groups = sorted(int(g) for g in G.group_ids)
    group_nodes_map = {g: list(G.group_to_nodes.get(g, [])) for g in sorted_groups}
    N = G.graph.number_of_nodes()

    all_chars = set()
    for gid in sorted_groups:
        all_chars.update(G.group_to_attrs.get(gid, {}).keys())
    all_chars = sorted(all_chars)

    char_isolation = {}
    for char in all_chars:
        val_to_groups = {}
        for gid in sorted_groups:
            v = G.group_to_attrs.get(gid, {}).get(char)
            if v is not None:
                val_to_groups.setdefault(v, []).append(gid)

        if len(val_to_groups) <= 1:
            char_isolation[char] = 0.0
            continue

        isolations = []
        for v, vgroups in val_to_groups.items():
            vset = set(vgroups)
            n_v = sum(len(group_nodes_map[g]) for g in vset)
            if n_v == 0 or N == 0:
                continue
            intra = sum(G.maximum_num_links.get((g, h), 0) for g in vset for h in vset)
            total_out = sum(G.maximum_num_links.get((g, h), 0) for g in vset for h in sorted_groups)
            I_v = (intra / total_out - n_v / N) if total_out > 0 else 0.0
            isolations.append(I_v)
        char_isolation[char] = float(np.mean(isolations)) if isolations else 0.0

    meaningful = sorted(
        [(c, iso) for c, iso in char_isolation.items() if iso >= isolation_threshold],
        key=lambda x: x[1], reverse=True
    )
    print(f"  Isolation scores: { {c: f'{iso:.3f}' for c, iso in char_isolation.items()} }")
    print(f"  Meaningful characteristics (>={isolation_threshold}): {[(c, f'{iso:.3f}') for c, iso in meaningful]}")

    K = num_communities

    def _proportional_alloc(weights, total):
        w = np.array(weights, dtype=np.float64)
        w_sum = w.sum()
        if w_sum == 0:
            base = total // len(w)
            sizes = [base] * len(w)
            for i in range(total % len(w)):
                sizes[i] += 1
            return sizes
        floats = w / w_sum * total
        sizes = np.maximum(1, np.floor(floats).astype(int)).tolist()
        remainder = total - sum(sizes)
        fracs = (floats - np.floor(floats)).tolist()
        for i in sorted(range(len(fracs)), key=lambda x: -fracs[x]):
            if remainder <= 0:
                break
            sizes[i] += 1
            remainder -= 1
        return sizes

    if not meaningful:
        group_comms = {gid: list(range(K)) for gid in sorted_groups}

    elif len(meaningful) == 1:
        char1, _ = meaningful[0]
        val_to_groups = {}
        for gid in sorted_groups:
            v = G.group_to_attrs.get(gid, {}).get(char1)
            val_to_groups.setdefault(v, []).append(gid)
        vals1 = sorted(val_to_groups.keys(), key=str)

        pops = [sum(len(group_nodes_map[g]) for g in val_to_groups[v]) for v in vals1]
        sizes = _proportional_alloc(pops, K)

        val1_comms = {}
        start = 0
        for v, size in zip(vals1, sizes):
            val1_comms[v] = list(range(start, start + size))
            start += size

        group_comms = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            group_comms[gid] = val1_comms.get(v1, [])

    else:
        char1, _ = meaningful[0]
        char2, _ = meaningful[1]

        pair_pop = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            v2 = G.group_to_attrs.get(gid, {}).get(char2)
            pair_pop[(v1, v2)] = pair_pop.get((v1, v2), 0) + len(group_nodes_map[gid])

        pairs = sorted(pair_pop.keys(), key=lambda p: (str(p[0]), str(p[1])))
        pops = [pair_pop[p] for p in pairs]
        sizes = _proportional_alloc(pops, K)

        block_comms = {}
        start = 0
        for pair, size in zip(pairs, sizes):
            block_comms[pair] = list(range(start, start + size))
            start += size

        group_comms = {}
        for gid in sorted_groups:
            v1 = G.group_to_attrs.get(gid, {}).get(char1)
            v2 = G.group_to_attrs.get(gid, {}).get(char2)
            group_comms[gid] = block_comms.get((v1, v2), [])

    G.number_of_communities = K
    target = max(1, N // K)
    community_count = np.zeros(K, dtype=np.int64)

    for gid in sorted_groups:
        nodes = np.array(group_nodes_map[gid], dtype=np.int64)
        if len(nodes) == 0:
            continue
        rng.shuffle(nodes)
        chosen = group_comms.get(gid, list(range(K))) or list(range(K))

        headroom = np.array([max(0, target - community_count[c]) for c in chosen], dtype=np.float64)
        if headroom.sum() == 0:
            headroom = np.ones(len(chosen), dtype=np.float64)
        alloc = _proportional_alloc(headroom.tolist(), len(nodes))

        idx = 0
        for comm, count in zip(chosen, alloc):
            community_count[comm] += count
            for node in nodes[idx:idx + count]:
                node_int = int(node)
                G.communities_to_nodes.setdefault((comm, gid), []).append(node_int)
                G.nodes_to_communities[node_int] = comm
                G.communities_to_groups.setdefault(comm, []).append(gid)
            idx += count

    # Refinement
    try:
        from asnu_rust import refine_communities_move
        print(f"\nRefining communities with {refine_swaps} swap iterations...")

        refine_nodes = np.array(list(G.graph.nodes), dtype=np.int64)
        refine_node_groups = np.array(
            [G.nodes_to_group[n] for n in refine_nodes], dtype=np.int64
        )
        refine_assignments = np.array(
            [G.nodes_to_communities[int(n)] for n in refine_nodes], dtype=np.int64
        )
        refine_budget = {(int(k[0]), int(k[1])): int(v)
                         for k, v in G.maximum_num_links.items() if v > 0}

        new_assignments = refine_communities_move(
            refine_assignments,
            refine_node_groups,
            refine_budget,
            n_groups,
            G.number_of_communities,
            refine_swaps,
            1,
            42,
        )

        G.communities_to_nodes = {}
        G.communities_to_groups = {}
        G.nodes_to_communities = {}
        for i in range(len(refine_nodes)):
            node = int(refine_nodes[i])
            comm = int(new_assignments[i])
            group = int(refine_node_groups[i])
            G.nodes_to_communities[node] = comm
            G.communities_to_nodes.setdefault((comm, group), []).append(node)
            G.communities_to_groups.setdefault(comm, []).append(group)

        print("Refinement complete.")

    except ImportError:
        print("  refine_communities_move not available; skipping refinement.")
        new_assignments = refine_assignments  # fall back to pre-refinement assignments

    K_new = int(new_assignments.max()) + 1
    G.number_of_communities = K_new
    coord_pos = rng.permutation(K_new)
    node_coordinates = {}
    for i in range(len(refine_nodes)):
        node_int = int(refine_nodes[i])
        comm = int(new_assignments[i])
        theta_c = float(coord_pos[comm]) / K_new
        node_coordinates[node_int] = (theta_c) % 1.0

    G.node_coordinates = node_coordinates
    print(f"\nSegregation-based assignment complete: {N} nodes -> {K_new} communities")


def connect_all_within_communities(G, verbose=False):
    """
    Connect all nodes within each community to each other.
    """
    stats = {'total_edges': 0, 'edges_per_community': {}}

    communities_nodes = [[] for _ in range(G.number_of_communities)]
    for node, comm in G.nodes_to_communities.items():
        communities_nodes[comm].append(node)

    for community_id in range(G.number_of_communities):
        community_nodes = communities_nodes[community_id]
        if len(community_nodes) == 0:
            continue

        edges_to_add = [(src, dst) for src, dst in product(community_nodes, repeat=2)
                        if src != dst]
        G.graph.add_edges_from(edges_to_add)

        edges_added = len(edges_to_add)
        stats['edges_per_community'][community_id] = edges_added
        stats['total_edges'] += edges_added

        if (community_id + 1) % 5000 == 0 or community_id == 0:
            print(f"  Connected {community_id + 1}/{G.number_of_communities} communities "
                  f"({(community_id + 1) / G.number_of_communities * 100:.1f}%)")

    return stats


def fill_unfulfilled_group_pairs(G, reciprocity_p, verbose=True):
    """
    Complete any group pairs that didn't reach their target edge count.
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

    for (src_id, dst_id) in G.maximum_num_links.keys():
        existing = G.existing_num_links.get((src_id, dst_id), 0)
        maximum = G.maximum_num_links[(src_id, dst_id)]
        stats['total_pairs'] += 1
        if maximum == 0:
            continue
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
        if not src_pool or not dst_pool or needed <= 0:
            return
        src_arr = np.array(src_pool)
        dst_arr = np.array(dst_pool)
        batch = max(needed * 4, 512)

        for _ in range(10):
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
                    if (G.existing_num_links.get((dst_id, src_id), 0) < G.maximum_num_links.get((dst_id, src_id), 0)
                            and not G.graph.has_edge(d, s)):
                        G.graph.add_edge(d, s)
                        G.existing_num_links[(dst_id, src_id)] += 1
                        stats['reciprocal_edges_added'] += 1
                        if dst_id == src_id:
                            stats['edges_added'] += 1
            if added_this_round == 0:
                return

    total_needed = sum(max - ex for _, _, ex, max in unfulfilled_pairs)
    if unfulfilled_pairs:
        pbar = (tqdm(total=total_needed, unit='edge', desc='Filling pairs', dynamic_ncols=True)
                if tqdm and verbose else None)

        for src_id, dst_id, existing, maximum in unfulfilled_pairs:
            existing = G.existing_num_links.get((src_id, dst_id), 0)
            maximum = G.maximum_num_links.get((src_id, dst_id), 0)
            needed = maximum - existing
            src_nodes = G.group_to_nodes.get(src_id, [])
            dst_nodes = G.group_to_nodes.get(dst_id, [])

            if not src_nodes or not dst_nodes:
                continue

            edges_before_pair = stats['edges_added']

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
                src_arrs = [np.array(src_comm[c]) for c in shared]
                dst_arrs = [np.array(dst_comm[c]) for c in shared]
                weights = np.array([len(s) * len(d) for s, d in zip(src_arrs, dst_arrs)], dtype=float)
                weights /= weights.sum()
                n_comm = len(shared)
                batch = max((maximum - G.existing_num_links[(src_id, dst_id)]) * 4, 512)

                src_lens    = np.array([len(a) for a in src_arrs], dtype=np.int64)
                dst_lens    = np.array([len(a) for a in dst_arrs], dtype=np.int64)
                src_offsets = np.zeros(n_comm + 1, dtype=np.int64)
                dst_offsets = np.zeros(n_comm + 1, dtype=np.int64)
                src_offsets[1:] = src_lens.cumsum()
                dst_offsets[1:] = dst_lens.cumsum()
                src_flat = np.concatenate(src_arrs)
                dst_flat = np.concatenate(dst_arrs)

                for _ in range(10):
                    if G.existing_num_links[(src_id, dst_id)] >= maximum:
                        break
                    comm_indices = np.random.choice(n_comm, size=batch, p=weights)
                    s_batch = src_flat[src_offsets[comm_indices] +
                                       (np.random.random(batch) * src_lens[comm_indices]).astype(np.int64)]
                    d_batch = dst_flat[dst_offsets[comm_indices] +
                                       (np.random.random(batch) * dst_lens[comm_indices]).astype(np.int64)]
                    added_this_round = 0
                    for idx in range(batch):
                        if G.existing_num_links[(src_id, dst_id)] >= maximum:
                            break
                        s, d = int(s_batch[idx]), int(d_batch[idx])
                        if s == d or G.graph.has_edge(s, d):
                            continue
                        G.graph.add_edge(s, d)
                        G.existing_num_links[(src_id, dst_id)] += 1
                        stats['edges_added'] += 1
                        added_this_round += 1
                        if reciprocity_p > 0 and random.random() < reciprocity_p:
                            if (G.existing_num_links.get((dst_id, src_id), 0) < G.maximum_num_links.get((dst_id, src_id), 0)
                                    and not G.graph.has_edge(d, s)):
                                G.graph.add_edge(d, s)
                                G.existing_num_links[(dst_id, src_id)] += 1
                                stats['reciprocal_edges_added'] += 1
                                if dst_id == src_id:
                                    stats['edges_added'] += 1
                    if added_this_round == 0:
                        break

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


def load_communities(G, community_file_path):
    """
    Load community assignments from a JSON file into a NetworkXGraph object.
    """
    import json

    with open(community_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    G.number_of_communities = data['number_of_communities']
    _pm_data = data['probability_matrix']
    if isinstance(_pm_data, dict) and _pm_data.get('sparse'):
        _n = _pm_data['shape'][0]
        if _n == 0:
            G.probability_matrix = np.zeros((0, 0), dtype=np.float64)
        elif _n > 5000:
            from scipy.sparse import csr_matrix as _csr
            G.probability_matrix = _csr(
                (_pm_data['vals'], (_pm_data['rows'], _pm_data['cols'])),
                shape=(_n, _n), dtype=np.float64
            )
        else:
            _pm_arr = np.zeros((_n, _n), dtype=np.float64)
            for r, c, v in zip(_pm_data['rows'], _pm_data['cols'], _pm_data['vals']):
                _pm_arr[r, c] = v
            G.probability_matrix = _pm_arr
    else:
        G.probability_matrix = np.array(_pm_data)

    graph_nodes = set(G.graph.nodes)
    G.nodes_to_communities = {}
    for k, v in data['nodes_to_communities'].items():
        node = int(k)
        if node in graph_nodes:
            G.nodes_to_communities[node] = v

    unassigned = graph_nodes - set(G.nodes_to_communities.keys())
    if unassigned:
        print(f"Warning: {len(unassigned)} graph nodes have no community assignment")

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

    if 'node_coordinates' in data:
        G.node_coordinates = {int(k): float(v) for k, v in data['node_coordinates'].items()}