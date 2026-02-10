"""
Compare Rust and Python implementations for both
community assignment (process_nodes) and edge creation (run_edge_creation).

Since RNGs differ, we compare statistical properties rather than exact outputs.
"""
import copy
import time

import numpy as np

from asnu.core.graph import NetworkXGraph
from asnu.core.generate import init_nodes, _compute_maximum_num_links
from asnu.core.community import (
    _process_nodes_python,
    populate_communities,
    build_group_pair_to_communities_lookup,
)
from asnu.core.grn import establish_links
from asnu.core.utils import find_nodes, read_file

# Try importing Rust functions
from asnu_rust import process_nodes as rust_process_nodes
from asnu_rust import run_edge_creation as rust_edge_creation

# ── Data paths ──────────────────────────────────────────────────────
LINKS = 'Data/tab_buren.csv'
POPS  = 'Data/tab_n_(with oplniv).csv'
SCALE = 0.1
NUM_COMMUNITIES = 500


# ════════════════════════════════════════════════════════════════════
#  PART 1 — Community assignment: Rust vs Python
# ════════════════════════════════════════════════════════════════════

def _setup_community_inputs(seed=42):
    """Create identical initial state for both backends."""
    G = NetworkXGraph()
    init_nodes(G, POPS, SCALE)
    _compute_maximum_num_links(G, LINKS, SCALE, verbose=False)

    n_groups = len(G.group_ids)
    num_communities = NUM_COMMUNITIES
    total_nodes = G.graph.number_of_nodes()

    # Affinity → probability matrix
    affinity = np.zeros((n_groups, n_groups))
    for (i, j), count in G.maximum_num_links.items():
        affinity[i, j] = count
    epsilon = 1e-5
    normalized = affinity / (affinity.sum(axis=1, keepdims=True) + epsilon)
    normalized[normalized == 0] = epsilon
    ideal = normalized.copy()

    # Target sizes (natural → None)
    target_sizes = None
    target_counts = None

    # Shared shuffle
    np.random.seed(seed)
    all_nodes = np.array(list(G.graph.nodes))
    np.random.shuffle(all_nodes)
    node_groups = np.array([G.nodes_to_group[n] for n in all_nodes])

    # Tracking arrays (fresh copies for each run)
    def make_arrays():
        return (
            np.zeros((num_communities, n_groups), dtype=np.float64),
            np.zeros(num_communities, dtype=np.int32),
            np.zeros((n_groups, n_groups), dtype=np.float64),
        )

    return G, all_nodes, node_groups, ideal, target_counts, total_nodes, num_communities, n_groups, make_arrays


def run_python_communities():
    G, all_nodes, node_groups, ideal, target_counts, total_nodes, num_communities, n_groups, make_arrays = _setup_community_inputs()
    cc, cs, ge = make_arrays()

    # Init community structures on G
    for c in range(num_communities):
        for g in range(n_groups):
            G.communities_to_nodes[(c, g)] = []
        G.communities_to_groups[c] = []

    t0 = time.perf_counter()
    _process_nodes_python(G, all_nodes, node_groups, cc, cs, ge, ideal, target_counts, total_nodes)
    dt = time.perf_counter() - t0

    return G, cs, dt


def run_rust_communities():
    G, all_nodes, node_groups, ideal, target_counts, total_nodes, num_communities, n_groups, make_arrays = _setup_community_inputs()
    cc, cs, ge = make_arrays()

    # Init community structures on G
    for c in range(num_communities):
        for g in range(n_groups):
            G.communities_to_nodes[(c, g)] = []
        G.communities_to_groups[c] = []

    t0 = time.perf_counter()
    assignments = rust_process_nodes(
        all_nodes.astype(np.int64), node_groups.astype(np.int64),
        cc, cs, ge, ideal, target_counts, total_nodes,
    )
    for i in range(len(all_nodes)):
        node = int(all_nodes[i])
        comm = int(assignments[i])
        group = int(node_groups[i])
        G.communities_to_nodes[(comm, group)].append(node)
        G.nodes_to_communities[node] = comm
        G.communities_to_groups[comm].append(group)
    dt = time.perf_counter() - t0

    return G, cs, dt


# ════════════════════════════════════════════════════════════════════
#  PART 2 — Edge creation: Rust vs Python
# ════════════════════════════════════════════════════════════════════

def _setup_edge_inputs():
    """Create a graph with communities loaded, ready for edge creation."""
    G = NetworkXGraph()
    init_nodes(G, POPS, SCALE)
    _compute_maximum_num_links(G, LINKS, SCALE, verbose=False)

    # Use populate_communities to get a community structure
    # (this uses whichever backend is active, but both tests
    #  will share the same community assignments)
    G.number_of_communities = NUM_COMMUNITIES
    n_groups = len(G.group_ids)
    for c in range(NUM_COMMUNITIES):
        for g in range(n_groups):
            G.communities_to_nodes[(c, g)] = []
        G.communities_to_groups[c] = []

    populate_communities(G, NUM_COMMUNITIES)
    return G


def run_python_edges(G_template):
    """Run Python edge creation on a deep copy of the template graph."""
    G = _deep_copy_graph(G_template)

    src_suffix = '_src'
    dst_suffix = '_dst'
    fraction = 0.5  # 1 - preferential_attachment
    reciprocity_p = 1.0
    transitivity_p = 1.0
    pa_scope = 'local'

    df = read_file(LINKS)
    gp2c = build_group_pair_to_communities_lookup(G, verbose=False)

    t0 = time.perf_counter()
    for _, row in df.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}
        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)
        if not src_nodes or not dst_nodes:
            continue
        target = G.maximum_num_links[(src_id, dst_id)]
        valid = gp2c.get((src_id, dst_id), [])
        establish_links(G, src_id, dst_id, target, fraction, reciprocity_p,
                        transitivity_p, valid, pa_scope)
    dt = time.perf_counter() - t0
    return G, dt


def run_rust_edges(G_template):
    """Run Rust edge creation on a deep copy of the template graph."""
    G = _deep_copy_graph(G_template)

    src_suffix = '_src'
    dst_suffix = '_dst'
    fraction = 0.5
    reciprocity_p = 1.0
    transitivity_p = 1.0
    pa_scope = 'local'

    df = read_file(LINKS)
    gp2c = build_group_pair_to_communities_lookup(G, verbose=False)

    # Build group_pairs
    group_pairs = []
    for _, row in df.iterrows():
        src_attrs = {k.replace(src_suffix, ''): row[k] for k in row.index if k.endswith(src_suffix)}
        dst_attrs = {k.replace(dst_suffix, ''): row[k] for k in row.index if k.endswith(dst_suffix)}
        src_nodes, src_id = find_nodes(G, **src_attrs)
        dst_nodes, dst_id = find_nodes(G, **dst_attrs)
        if not src_nodes or not dst_nodes:
            continue
        target = G.maximum_num_links[(src_id, dst_id)]
        group_pairs.append((src_id, dst_id, target))

    ctn = {(int(k[0]), int(k[1])): [int(n) for n in v]
           for k, v in G.communities_to_nodes.items()}
    ntg = {int(k): int(v) for k, v in G.nodes_to_group.items()}
    mnl = {(int(k[0]), int(k[1])): int(v) for k, v in G.maximum_num_links.items()}
    vcm = {(int(k[0]), int(k[1])): [int(c) for c in v]
           for k, v in gp2c.items()}

    t0 = time.perf_counter()
    new_edges, link_counts = rust_edge_creation(
        group_pairs, vcm, mnl, ctn, ntg,
        fraction, reciprocity_p, transitivity_p,
        pa_scope, G.number_of_communities,
    )
    dt = time.perf_counter() - t0

    G.graph.add_edges_from(new_edges)
    for src, dst, count in link_counts:
        G.existing_num_links[(src, dst)] = count

    return G, dt


def _deep_copy_graph(G_src):
    """Deep copy a NetworkXGraph, preserving all metadata."""
    G = NetworkXGraph()
    G.graph = G_src.graph.copy()
    G.attrs_to_group = copy.deepcopy(G_src.attrs_to_group)
    G.group_to_attrs = copy.deepcopy(G_src.group_to_attrs)
    G.group_to_nodes = copy.deepcopy(G_src.group_to_nodes)
    G.nodes_to_group = copy.deepcopy(G_src.nodes_to_group)
    G.communities_to_nodes = copy.deepcopy(G_src.communities_to_nodes)
    G.nodes_to_communities = copy.deepcopy(G_src.nodes_to_communities)
    G.communities_to_groups = copy.deepcopy(G_src.communities_to_groups)
    G.existing_num_links = copy.deepcopy(G_src.existing_num_links)
    G.maximum_num_links = copy.deepcopy(G_src.maximum_num_links)
    G.probability_matrix = G_src.probability_matrix.copy()
    G.number_of_communities = G_src.number_of_communities
    G.popularity_pool = {}
    G.group_ids = list(G_src.group_ids)
    return G


# ════════════════════════════════════════════════════════════════════
#  REPORTING
# ════════════════════════════════════════════════════════════════════

def compare_communities(G_py, cs_py, dt_py, G_rs, cs_rs, dt_rs):
    print("=" * 60)
    print("COMMUNITY ASSIGNMENT: Rust vs Python")
    print("=" * 60)
    print(f"  Python time: {dt_py:.3f}s")
    print(f"  Rust time:   {dt_rs:.3f}s")
    print(f"  Speedup:     {dt_py / dt_rs:.1f}x")

    # Total nodes assigned
    py_assigned = len(G_py.nodes_to_communities)
    rs_assigned = len(G_rs.nodes_to_communities)
    print(f"\n  Nodes assigned:  Python={py_assigned}  Rust={rs_assigned}")

    # Community sizes
    py_sizes = cs_py[cs_py > 0]
    rs_sizes = cs_rs[cs_rs > 0]
    print(f"  Non-empty communities: Python={len(py_sizes)}  Rust={len(rs_sizes)}")
    print(f"  Mean size:     Python={np.mean(py_sizes):.1f}  Rust={np.mean(rs_sizes):.1f}")
    print(f"  Std size:      Python={np.std(py_sizes):.1f}  Rust={np.std(rs_sizes):.1f}")
    print(f"  Min/Max size:  Python={np.min(py_sizes)}/{np.max(py_sizes)}  "
          f"Rust={np.min(rs_sizes)}/{np.max(rs_sizes)}")


def compare_edges(G_py, dt_py, G_rs, dt_rs):
    print("\n" + "=" * 60)
    print("EDGE CREATION: Rust vs Python")
    print("=" * 60)
    print(f"  Python time: {dt_py:.3f}s")
    print(f"  Rust time:   {dt_rs:.3f}s")
    print(f"  Speedup:     {dt_py / dt_rs:.1f}x")

    py_edges = G_py.graph.number_of_edges()
    rs_edges = G_rs.graph.number_of_edges()
    print(f"\n  Total edges:   Python={py_edges}  Rust={rs_edges}")

    # Degree distributions
    py_deg = [G_py.graph.degree(n) for n in G_py.graph.nodes()]
    rs_deg = [G_rs.graph.degree(n) for n in G_rs.graph.nodes()]
    print(f"  Mean degree:   Python={np.mean(py_deg):.2f}  Rust={np.mean(rs_deg):.2f}")
    print(f"  Std degree:    Python={np.std(py_deg):.2f}  Rust={np.std(rs_deg):.2f}")
    print(f"  Max degree:    Python={np.max(py_deg)}  Rust={np.max(rs_deg)}")

    # Link fulfillment
    total_requested = sum(G_py.maximum_num_links.values())
    py_fulfilled = sum(G_py.existing_num_links.values())
    rs_fulfilled = sum(G_rs.existing_num_links.values())
    print(f"\n  Requested links: {total_requested}")
    print(f"  Fulfilled:     Python={py_fulfilled}  Rust={rs_fulfilled}")
    py_rate = py_fulfilled / total_requested * 100 if total_requested else 0
    rs_rate = rs_fulfilled / total_requested * 100 if total_requested else 0
    print(f"  Fulfillment:   Python={py_rate:.1f}%  Rust={rs_rate:.1f}%")

    # Per-group-pair comparison
    diffs = []
    for key in G_py.maximum_num_links:
        py_count = G_py.existing_num_links.get(key, 0)
        rs_count = G_rs.existing_num_links.get(key, 0)
        if G_py.maximum_num_links[key] > 0:
            diffs.append(abs(py_count - rs_count))
    if diffs:
        print(f"\n  Per-pair |Python - Rust| link count:")
        print(f"    Mean: {np.mean(diffs):.1f}  Max: {np.max(diffs)}  "
              f"Zero-diff: {diffs.count(0)}/{len(diffs)}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Setting up comparison test...")
    print(f"  Scale: {SCALE}  Communities: {NUM_COMMUNITIES}\n")

    # ── Part 1: Communities ──
    print("Running Python community assignment...")
    G_py_comm, cs_py, dt_py_comm = run_python_communities()

    print("\nRunning Rust community assignment...")
    G_rs_comm, cs_rs, dt_rs_comm = run_rust_communities()

    compare_communities(G_py_comm, cs_py, dt_py_comm, G_rs_comm, cs_rs, dt_rs_comm)

    # ── Part 2: Edges ──
    print("\n\nPreparing shared community structure for edge comparison...")
    G_template = _setup_edge_inputs()

    print("Running Python edge creation...")
    G_py_edge, dt_py_edge = run_python_edges(G_template)

    print("Running Rust edge creation...")
    G_rs_edge, dt_rs_edge = run_rust_edges(G_template)

    compare_edges(G_py_edge, dt_py_edge, G_rs_edge, dt_rs_edge)
