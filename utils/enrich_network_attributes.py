"""
enrich_network_attributes_optimized.py
=======================================
Memory-optimized version for very large graphs.

Key optimizations:
1. Pre-compute node degrees once (avoid repeated O(n) operations)
2. Stream CSV writing to avoid loading full DataFrames into memory
3. Reduce redundant dictionary lookups
4. Add progress indicators for long operations
5. Use chunked processing where possible
"""

import argparse
import pickle
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import networkx as nx


import argparse
import pickle
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import networkx as nx

def load_network(file_path):
    """Load network from compressed .npz file"""
    file_path = Path(file_path)
    
    if not str(file_path).endswith('.npz'):
        file_path = Path(f'{file_path}.npz')
    
    if not file_path.exists():
        raise FileNotFoundError(f"Network file not found: {file_path}")
    
    print(f"  Loading from {file_path}...")
    
    # Load compressed archive
    data = np.load(file_path, allow_pickle=True)
    
    # Create graph
    directed = bool(data['directed'])
    G = nx.DiGraph() if directed else nx.Graph()
    
    # CRITICAL: Add ALL nodes first (including isolates)
    if 'nodes' in data:
        G.add_nodes_from(data['nodes'])
    else:
        # Fallback for old format: extract nodes from node_attrs
        node_attrs = data['node_attrs'].item()
        G.add_nodes_from(node_attrs.keys())
    
    # Then add edges
    edges = data['edges']
    G.add_edges_from(edges)
    
    # Restore node attributes
    node_attrs = data['node_attrs'].item()
    nx.set_node_attributes(G, node_attrs)
    
    # Verify node count
    expected = int(data['num_nodes'])
    actual = G.number_of_nodes()
    if expected != actual:
        print(f"  WARNING: Expected {expected} nodes but loaded {actual}!", file=sys.stderr)
    
    return G

# ─────────────────────────────────────────────────────────────────
# CONSTANTS (same as original)
# ─────────────────────────────────────────────────────────────────

AGE_BRACKET_RANGES = {
    '[0,20)':   (0,  19),
    '[20,30)':  (20, 29),
    '[30,40)':  (30, 39),
    '[40,50)':  (40, 49),
    '[50,60)':  (50, 59),
    '[60,70)':  (60, 69),
    '[70,80)':  (70, 79),
    '[80,120]': (80, 99),
}

AGE_BRACKET_MIN = {k: v[0] for k, v in AGE_BRACKET_RANGES.items()}

AGE_5YR_BRACKETS = [
    '[0,5)',   '[5,10)',  '[10,15)', '[15,20)', '[20,25)', '[25,30)',
    '[30,35)', '[35,40)', '[40,45)', '[45,50)', '[50,55)', '[55,60)',
    '[60,65)', '[65,70)', '[70,75)', '[75,80)', '[80,85)', '[85,90)',
    '[90,95)', '[95,100)', '[100,120]'
]

def _get_5yr_bracket(age: int) -> str:
    if age < 100:
        bracket_idx = age // 5
        return AGE_5YR_BRACKETS[bracket_idx]
    else:
        return '[100,120]'

INCOME_BANDS = ['Laag (<20k)', 'Modaal (20-35k)', 'Midden (35-55k)', 'Hoog (>55k)']
INCOME_BAND_ORDER = {b: i for i, b in enumerate(INCOME_BANDS)}

INCOME_BAND_RANGES = {
    'Laag (<20k)':     (1_000,  19_999),
    'Modaal (20-35k)': (20_000, 34_999),
    'Midden (35-55k)': (35_000, 54_999),
    'Hoog (>55k)':     (55_000, 150_000),
}

INKOMEN_PROBS: dict[int, dict[str, float]] = {
    1: {'Laag (<20k)': 0.40, 'Modaal (20-35k)': 0.42, 'Midden (35-55k)': 0.15, 'Hoog (>55k)': 0.03},
    2: {'Laag (<20k)': 0.15, 'Modaal (20-35k)': 0.42, 'Midden (35-55k)': 0.33, 'Hoog (>55k)': 0.10},
    3: {'Laag (<20k)': 0.05, 'Modaal (20-35k)': 0.20, 'Midden (35-55k)': 0.38, 'Hoog (>55k)': 0.37},
}

AGE_INCOME_MODIFIERS: list[tuple[int, dict[str, float]]] = [
    (25,  {'Laag (<20k)': 2.20, 'Modaal (20-35k)': 1.30, 'Midden (35-55k)': 0.35, 'Hoog (>55k)': 0.05}),
    (35,  {'Laag (<20k)': 1.40, 'Modaal (20-35k)': 1.40, 'Midden (35-55k)': 0.75, 'Hoog (>55k)': 0.40}),
    (50,  {'Laag (<20k)': 0.75, 'Modaal (20-35k)': 0.95, 'Midden (35-55k)': 1.25, 'Hoog (>55k)': 1.60}),
    (65,  {'Laag (<20k)': 0.85, 'Modaal (20-35k)': 1.00, 'Midden (35-55k)': 1.15, 'Hoog (>55k)': 1.25}),
    (999, {'Laag (<20k)': 1.60, 'Modaal (20-35k)': 1.35, 'Midden (35-55k)': 0.65, 'Hoog (>55k)': 0.20}),
]

HOMOPHILY_DECAY   = {0: 1.00, 1: 0.50, 2: 0.15, 3: 0.04}
HOMOPHILY_STRENGTH = 0.35

ARBEID_BASE_PROBS: dict[str, dict[str, float]] = {
    '[0,20)':   {'Student/kind': 1.0},
    '[20,30)':  {'Werkend (voltijd)': 0.45, 'Werkend (deeltijd)': 0.15,
                 'Student': 0.25, 'Werkloos': 0.10, 'ZZP': 0.05},
    '[30,40)':  {'Werkend (voltijd)': 0.58, 'Werkend (deeltijd)': 0.18,
                 'ZZP': 0.10, 'Werkloos': 0.08, 'Thuiszorgend': 0.06},
    '[40,50)':  {'Werkend (voltijd)': 0.60, 'Werkend (deeltijd)': 0.16,
                 'ZZP': 0.10, 'Werkloos': 0.07, 'Arbeidsongeschikt': 0.07},
    '[50,60)':  {'Werkend (voltijd)': 0.52, 'Werkend (deeltijd)': 0.15,
                 'ZZP': 0.08, 'Werkloos': 0.10, 'Arbeidsongeschikt': 0.08,
                 'Gepensioneerd': 0.07},
    '[60,70)':  {'Gepensioneerd': 0.55, 'Werkend (deeltijd)': 0.14,
                 'Werkend (voltijd)': 0.09, 'ZZP': 0.07,
                 'Arbeidsongeschikt': 0.08, 'Werkloos': 0.07},
    '[70,80)':  {'Gepensioneerd': 0.90, 'Werkend (deeltijd)': 0.05,
                 'Arbeidsongeschikt': 0.05},
    '[80,120]': {'Gepensioneerd': 0.95, 'Arbeidsongeschikt': 0.05},
}

ARBEID_OPL_MODIFIERS: dict[int, dict[str, float]] = {
    1: {'Werkend (voltijd)': 0.80, 'ZZP': 0.70, 'Werkloos': 1.60,
        'Arbeidsongeschikt': 1.60, 'Thuiszorgend': 1.30,
        'Student': 0.60, 'Werkend (deeltijd)': 1.10},
    2: {'Werkend (voltijd)': 1.00, 'ZZP': 1.00, 'Werkloos': 1.00,
        'Arbeidsongeschikt': 1.00, 'Thuiszorgend': 1.00,
        'Student': 1.00, 'Werkend (deeltijd)': 1.00},
    3: {'Werkend (voltijd)': 1.25, 'ZZP': 1.40, 'Werkloos': 0.55,
        'Arbeidsongeschikt': 0.50, 'Thuiszorgend': 0.60,
        'Student': 1.60, 'Werkend (deeltijd)': 0.85},
}

ARBEID_INKOMEN_MODIFIERS: dict[str, dict[str, float]] = {
    'Laag (<20k)':     {'Werkend (voltijd)': 0.50, 'ZZP': 0.60,
                        'Werkloos': 2.20, 'Arbeidsongeschikt': 2.00,
                        'Thuiszorgend': 1.80, 'Werkend (deeltijd)': 1.40},
    'Modaal (20-35k)': {'Werkend (voltijd)': 1.00, 'ZZP': 0.90,
                        'Werkloos': 0.80, 'Arbeidsongeschikt': 0.80,
                        'Thuiszorgend': 0.90, 'Werkend (deeltijd)': 1.10},
    'Midden (35-55k)': {'Werkend (voltijd)': 1.30, 'ZZP': 1.20,
                        'Werkloos': 0.40, 'Arbeidsongeschikt': 0.40,
                        'Thuiszorgend': 0.50, 'Werkend (deeltijd)': 0.80},
    'Hoog (>55k)':     {'Werkend (voltijd)': 1.50, 'ZZP': 1.80,
                        'Werkloos': 0.15, 'Arbeidsongeschikt': 0.15,
                        'Thuiszorgend': 0.20, 'Werkend (deeltijd)': 0.55},
    'Niet van toepassing': {},
}

UITKERING_PROBS: dict[str, dict[str, float] | None] = {
    'Werkloos':          {'WW': 0.55, 'Bijstand': 0.45},
    'Arbeidsongeschikt': {'WAO/WIA': 1.0},
    'Thuiszorgend':      {'Bijstand': 0.65, 'WW': 0.35},
    'Gepensioneerd':     {'AOW': 0.85, 'AIO': 0.15},
    'Werkend (voltijd)': None,
    'Werkend (deeltijd)':None,
    'ZZP':               None,
    'Student':           None,
    'Student/kind':      None,
}

BSTAT_CATEGORIES = [
    'Ongehuwd', 'Samenwonend', 'Gehuwd', 'Gescheiden', 'Weduwe/weduwnaar'
]
BSTAT_ORDER = {b: i for i, b in enumerate(BSTAT_CATEGORIES)}

BSTAT_BASE_PROBS: dict[str, dict[str, float]] = {
    '[0,20)':   {'Ongehuwd': 1.0},
    '[20,30)':  {'Ongehuwd': 0.65, 'Gehuwd': 0.10,
                 'Samenwonend': 0.22, 'Gescheiden': 0.03},
    '[30,40)':  {'Ongehuwd': 0.28, 'Gehuwd': 0.42,
                 'Samenwonend': 0.20, 'Gescheiden': 0.10},
    '[40,50)':  {'Ongehuwd': 0.18, 'Gehuwd': 0.48,
                 'Samenwonend': 0.14, 'Gescheiden': 0.17,
                 'Weduwe/weduwnaar': 0.03},
    '[50,60)':  {'Ongehuwd': 0.13, 'Gehuwd': 0.50,
                 'Samenwonend': 0.10, 'Gescheiden': 0.20,
                 'Weduwe/weduwnaar': 0.07},
    '[60,70)':  {'Ongehuwd': 0.09, 'Gehuwd': 0.52,
                 'Samenwonend': 0.08, 'Gescheiden': 0.16,
                 'Weduwe/weduwnaar': 0.15},
    '[70,80)':  {'Ongehuwd': 0.07, 'Gehuwd': 0.47,
                 'Gescheiden': 0.10, 'Weduwe/weduwnaar': 0.36},
    '[80,120]': {'Ongehuwd': 0.05, 'Gehuwd': 0.33,
                 'Gescheiden': 0.07, 'Weduwe/weduwnaar': 0.55},
}

BSTAT_HOMOPHILY_DECAY    = {0: 1.00, 1: 0.70, 2: 0.45, 3: 0.25, 4: 0.15}
BSTAT_HOMOPHILY_STRENGTH = 0.20

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)


def _normalise(arr: np.ndarray) -> np.ndarray:
    s = arr.sum()
    return arr / s if s > 0 else np.ones(len(arr)) / len(arr)


def _sample_age(lft: str) -> int:
    lo, hi = AGE_BRACKET_RANGES[lft]
    return int(RNG.integers(lo, hi + 1))


def _age_income_modifier(age: int) -> np.ndarray:
    for max_age, mods in AGE_INCOME_MODIFIERS:
        if age < max_age:
            return np.array([mods[b] for b in INCOME_BANDS])
    return np.ones(len(INCOME_BANDS))


def _blended_income_probs(
    oplniv: int,
    age: int,
    neighbour_bands: list[str],
) -> np.ndarray:
    base = np.array([INKOMEN_PROBS[oplniv][b] for b in INCOME_BANDS])
    base = _normalise(base * _age_income_modifier(age))

    if not neighbour_bands:
        return base

    nb_counts = Counter(neighbour_bands)
    hom = np.zeros(len(INCOME_BANDS))
    for i, band in enumerate(INCOME_BANDS):
        for nb_band, cnt in nb_counts.items():
            dist = abs(INCOME_BAND_ORDER[band] - INCOME_BAND_ORDER[nb_band])
            hom[i] += (cnt / len(neighbour_bands)) * HOMOPHILY_DECAY[dist]
    hom = _normalise(hom)

    return _normalise((1 - HOMOPHILY_STRENGTH) * base + HOMOPHILY_STRENGTH * hom)


def _arbeid_probs(
    lft: str,
    oplniv: int,
    inkomensniveau: str,
) -> tuple[list[str], np.ndarray]:
    base_dict  = ARBEID_BASE_PROBS[lft]
    opl_mods   = ARBEID_OPL_MODIFIERS.get(oplniv, {})
    inc_mods   = ARBEID_INKOMEN_MODIFIERS.get(inkomensniveau, {})

    cats   = list(base_dict.keys())
    probs  = np.array([base_dict[c] for c in cats], dtype=float)

    for j, cat in enumerate(cats):
        probs[j] *= opl_mods.get(cat, 1.0)
        probs[j] *= inc_mods.get(cat, 1.0)

    return cats, _normalise(probs)


def _blended_bstat_probs(
    lft: str,
    neighbour_bstats: list[str],
) -> np.ndarray:
    base_dict = BSTAT_BASE_PROBS[lft]
    base = np.array([base_dict.get(b, 0.0) for b in BSTAT_CATEGORIES])
    base = _normalise(base)

    if not neighbour_bstats:
        return base

    nb_counts = Counter(neighbour_bstats)
    hom = np.zeros(len(BSTAT_CATEGORIES))
    for i, cat in enumerate(BSTAT_CATEGORIES):
        for nb_cat, cnt in nb_counts.items():
            dist = abs(BSTAT_ORDER[cat] - BSTAT_ORDER.get(nb_cat, 0))
            hom[i] += (cnt / len(neighbour_bstats)) * BSTAT_HOMOPHILY_DECAY.get(dist, 0.10)
    hom = _normalise(hom)

    return _normalise(
        (1 - BSTAT_HOMOPHILY_STRENGTH) * base + BSTAT_HOMOPHILY_STRENGTH * hom
    )


def _progress(i: int, total: int, step: int = 100000) -> None:
    """Print progress every `step` nodes."""
    if i > 0 and i % step == 0:
        print(f"  ... {i:,} / {total:,} ({100*i/total:.1f}%)", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────
# OPTIMIZED ASSIGNMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def assign_ages(G) -> None:
    """Assign integer `age` to every node from its `lft` bracket."""
    nodes_data = G.nodes(data=True)
    total = G.number_of_nodes()
    
    for i, (node, attrs) in enumerate(nodes_data):
        G.nodes[node]['age'] = _sample_age(attrs['lft'])
        _progress(i + 1, total)


def assign_5yr_age_brackets(G) -> None:
    """Assign `age_bracket_5yr` to every node based on integer age."""
    nodes_data = G.nodes(data=True)
    total = G.number_of_nodes()
    
    for i, (node, attrs) in enumerate(nodes_data):
        age = attrs.get('age')
        if age is not None:
            G.nodes[node]['age_bracket_5yr'] = _get_5yr_bracket(age)
        _progress(i + 1, total)


def assign_income(G) -> None:
    """
    OPTIMIZED: Pre-compute degrees once, cache neighbor lookups.
    """
    print("  Pre-computing node degrees...", file=sys.stderr)
    # Pre-compute degrees once instead of during sort
    degrees = dict(G.degree())
    
    print("  Sorting nodes by degree...", file=sys.stderr)
    sorted_nodes = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)
    
    total = len(sorted_nodes)
    nodes_data = G.nodes  # Access node view once
    
    for i, node in enumerate(sorted_nodes):
        attrs = nodes_data[node]
        lft = attrs['lft']

        if AGE_BRACKET_MIN[lft] < 20:
            nodes_data[node]['inkomensniveau'] = 'Niet van toepassing'
            nodes_data[node]['inkomen'] = 0
            _progress(i + 1, total)
            continue

        # Cache neighbor lookup - combine predecessors and successors
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        neighbour_bands = [
            nodes_data[nbr].get('inkomensniveau')
            for nbr in neighbors
            if nodes_data[nbr].get('inkomensniveau') in INCOME_BAND_ORDER
        ]

        probs = _blended_income_probs(int(attrs['oplniv']), int(attrs['age']), neighbour_bands)
        chosen_band = str(RNG.choice(INCOME_BANDS, p=probs))
        lo, hi = INCOME_BAND_RANGES[chosen_band]

        nodes_data[node]['inkomensniveau'] = chosen_band
        nodes_data[node]['inkomen'] = int(RNG.integers(lo, hi + 1))
        
        _progress(i + 1, total)


def assign_arbeidsstatus(G) -> None:
    """Assign `arbeidsstatus` - no homophily, so straightforward iteration."""
    nodes_data = G.nodes(data=True)
    total = G.number_of_nodes()
    
    for i, (node, attrs) in enumerate(nodes_data):
        lft = attrs['lft']
        oplniv = int(attrs['oplniv'])
        inkomensniveau = attrs.get('inkomensniveau', 'Niet van toepassing')

        cats, probs = _arbeid_probs(lft, oplniv, inkomensniveau)
        G.nodes[node]['arbeidsstatus'] = str(RNG.choice(cats, p=probs))
        
        _progress(i + 1, total)


def assign_uitkeringstype(G) -> None:
    """Assign `uitkeringstype` from arbeidsstatus."""
    nodes_data = G.nodes(data=True)
    total = G.number_of_nodes()
    
    for i, (node, attrs) in enumerate(nodes_data):
        arbeid = attrs.get('arbeidsstatus', '')
        probs_dict = UITKERING_PROBS.get(arbeid)

        if probs_dict is None:
            G.nodes[node]['uitkeringstype'] = None
            _progress(i + 1, total)
            continue

        if arbeid == 'Gepensioneerd':
            age = int(attrs.get('age', 70))
            if age < 67:
                probs_dict = {'AOW': 0.40, 'AIO': 0.60}
            else:
                probs_dict = {'AOW': 0.85, 'AIO': 0.15}

        cats = list(probs_dict.keys())
        probs = _normalise(np.array(list(probs_dict.values())))
        G.nodes[node]['uitkeringstype'] = str(RNG.choice(cats, p=probs))
        
        _progress(i + 1, total)


def assign_burgerlijke_staat(G) -> None:
    """
    OPTIMIZED: Pre-compute degrees once, cache neighbor lookups.
    """
    print("  Pre-computing node degrees...", file=sys.stderr)
    degrees = dict(G.degree())
    
    print("  Sorting nodes by degree...", file=sys.stderr)
    sorted_nodes = sorted(G.nodes(), key=lambda n: degrees[n], reverse=True)
    
    total = len(sorted_nodes)
    nodes_data = G.nodes
    
    for i, node in enumerate(sorted_nodes):
        attrs = nodes_data[node]
        lft = attrs['lft']

        # Cache neighbor lookup
        neighbors = set(G.predecessors(node)) | set(G.successors(node))
        neighbour_bstats = [
            nodes_data[nbr].get('burgerlijke_staat')
            for nbr in neighbors
            if nodes_data[nbr].get('burgerlijke_staat') in BSTAT_ORDER
        ]

        probs = _blended_bstat_probs(lft, neighbour_bstats)
        valid_cats = [b for b in BSTAT_CATEGORIES if BSTAT_BASE_PROBS[lft].get(b, 0) > 0]
        valid_probs = _normalise(np.array([probs[BSTAT_ORDER[b]] for b in valid_cats]))

        nodes_data[node]['burgerlijke_staat'] = str(RNG.choice(valid_cats, p=valid_probs))
        
        _progress(i + 1, total)


def add_edge_income_distance(G) -> None:
    """Add `income_distance` to each edge."""
    total = G.number_of_edges()
    nodes_data = G.nodes
    
    for i, (u, v) in enumerate(G.edges()):
        band_u = nodes_data[u].get('inkomensniveau')
        band_v = nodes_data[v].get('inkomensniveau')
        if band_u in INCOME_BAND_ORDER and band_v in INCOME_BAND_ORDER:
            dist = abs(INCOME_BAND_ORDER[band_u] - INCOME_BAND_ORDER[band_v])
        else:
            dist = None
        G.edges[u, v]['income_distance'] = dist
        
        _progress(i + 1, total, step=500000)


# ─────────────────────────────────────────────────────────────────
# ENCODING
# ─────────────────────────────────────────────────────────────────

def build_encodings(G) -> dict:
    """Build integer encodings efficiently using sets."""
    attributes_to_encode = [
        'etngrp', 'geslacht', 'lft', 'age_bracket_5yr', 'oplniv',
        'inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat'
    ]
    
    encodings = {}
    nodes_data = G.nodes(data=True)
    
    for attr in attributes_to_encode:
        values = set()
        for node, attrs in nodes_data:
            val = attrs.get(attr)
            if val is not None:
                values.add(val)
        
        sorted_values = sorted(values, key=lambda x: (x is None, x))
        encodings[attr] = {val: idx for idx, val in enumerate(sorted_values)}
    
    print(f"  Created encodings for {len(encodings)} attributes")
    return encodings


def encode_node_attributes(G, encodings: dict) -> None:
    """Encode attributes efficiently."""
    total = G.number_of_nodes()
    nodes_data = G.nodes
    
    for i, node in enumerate(G.nodes()):
        for attr, mapping in encodings.items():
            if attr in nodes_data[node]:
                val = nodes_data[node][attr]
                nodes_data[node][attr] = mapping.get(val, -1)
        
        _progress(i + 1, total)


def export_encoding_mappings(encodings: dict, output_dir: Path) -> None:
    """Export encodings as CSV files."""
    import csv
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for attr, mapping in encodings.items():
        csv_path = output_dir / f"encoding_{attr}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['code', 'value'])
            
            # Sort by code
            for value, code in sorted(mapping.items(), key=lambda x: x[1]):
                writer.writerow([code, value])
    
    print(f"  Exported {len(encodings)} encoding files to {output_dir}")


# ─────────────────────────────────────────────────────────────────
# OPTIMIZED CSV EXPORT - STREAMING
# ─────────────────────────────────────────────────────────────────

GROUP_KEYS = [
    'etngrp', 'geslacht', 'lft', 'age_bracket_5yr', 'oplniv',
    'inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat',
]


def export_edge_csv_streaming(G, csv_path: Path) -> None:
    """
    OPTIMIZED: Stream edge data without building full DataFrame in memory.
    Aggregate groups incrementally.
    """
    import csv
    
    print("  Aggregating edge groups...", file=sys.stderr)
    
    # Use defaultdict to aggregate counts
    edge_groups = defaultdict(int)
    nodes_data = G.nodes
    total = G.number_of_edges()
    
    for i, (u, v) in enumerate(G.edges()):
        # Build group key as tuple
        key_parts = []
        for k in GROUP_KEYS:
            key_parts.append(nodes_data[u].get(k))
        for k in GROUP_KEYS:
            key_parts.append(nodes_data[v].get(k))
        
        key = tuple(key_parts)
        edge_groups[key] += 1
        
        _progress(i + 1, total, step=500000)
    
    print(f"  Writing {len(edge_groups):,} group combinations...", file=sys.stderr)
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write directly to CSV
    with open(csv_path, 'w', newline='') as f:
        cols_src = [f"{k}_src" for k in GROUP_KEYS]
        cols_dst = [f"{k}_dst" for k in GROUP_KEYS]
        
        writer = csv.writer(f)
        writer.writerow(cols_src + cols_dst + ['n'])
        
        # Sort keys and write
        for key in sorted(edge_groups.keys()):
            writer.writerow(list(key) + [edge_groups[key]])
    
    total_edges = sum(edge_groups.values())
    print(f"  {len(edge_groups):,} group-pair rows  |  {total_edges:,} edges  →  {csv_path}")


def export_population_csv_streaming(G, csv_path: Path) -> None:
    """
    OPTIMIZED: Stream population data without building full DataFrame.
    """
    import csv
    
    print("  Aggregating population groups...", file=sys.stderr)
    
    pop_groups = defaultdict(int)
    nodes_data = G.nodes
    total = G.number_of_nodes()
    
    for i, node in enumerate(G.nodes()):
        key_parts = tuple(nodes_data[node].get(k) for k in GROUP_KEYS)
        pop_groups[key_parts] += 1
        
        _progress(i + 1, total)
    
    print(f"  Writing {len(pop_groups):,} group combinations...", file=sys.stderr)
    
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(GROUP_KEYS + ['n'])
        
        for key in sorted(pop_groups.keys()):
            writer.writerow(list(key) + [pop_groups[key]])
    
    total_nodes = sum(pop_groups.values())
    print(f"  {len(pop_groups):,} population groups  |  {total_nodes:,} nodes  →  {csv_path}")


# ─────────────────────────────────────────────────────────────────
# STATS (same as original)
# ─────────────────────────────────────────────────────────────────

def print_stats(G, encodings: dict = None) -> None:
    """Print statistics."""
    n_nodes = G.number_of_nodes()
    
    sample_node = list(G.nodes())[0]
    is_encoded = isinstance(G.nodes[sample_node].get('etngrp'), (int, np.integer))
    
    def _decode(attr, code):
        if encodings and is_encoded:
            for val, c in encodings[attr].items():
                if c == code:
                    return val
        return code
    
    def _dist(attr, categories):
        counts = Counter()
        for node in G.nodes():
            code = G.nodes[node].get(attr)
            val = _decode(attr, code)
            counts[val] += 1
        
        print(f"\n  {attr}:")
        for cat in categories:
            c = counts.get(cat, 0)
            print(f"    {str(cat):<30} {c:>7,}  ({100*c/n_nodes:.1f}%)")
        other = {k: v for k, v in counts.items() if k not in categories}
        if other:
            print(f"    {'(other/None)':<30} {sum(other.values()):>7,}")

    print("\n── 5-year age brackets ──────────────────────────────────")
    _dist('age_bracket_5yr', AGE_5YR_BRACKETS)

    print("\n── Income bands ─────────────────────────────────────────")
    _dist('inkomensniveau', INCOME_BANDS + ['Niet van toepassing'])

    print("\n── Arbeidsstatus ────────────────────────────────────────")
    if encodings and is_encoded:
        all_arbeid = sorted(x for x in encodings['arbeidsstatus'].keys() if x is not None)
    else:
        all_arbeid = sorted(x for x in {G.nodes[n].get('arbeidsstatus') for n in G.nodes()} if x is not None)
    _dist('arbeidsstatus', all_arbeid)

    print("\n── Uitkeringstype ───────────────────────────────────────")
    if encodings and is_encoded:
        all_uitk = sorted(x for x in encodings['uitkeringstype'].keys() if x is not None)
    else:
        all_uitk = sorted(x for x in {G.nodes[n].get('uitkeringstype') for n in G.nodes()} if x is not None)
    _dist('uitkeringstype', all_uitk)

    print("\n── Burgerlijke staat ────────────────────────────────────")
    _dist('burgerlijke_staat', BSTAT_CATEGORIES)

    ages = [G.nodes[n]['age'] for n in G.nodes() if 'age' in G.nodes[n]]
    if ages:
        print(f"\n── Age  min={min(ages)}  max={max(ages)}  "
              f"mean={np.mean(ages):.1f}  median={np.median(ages):.0f}")

    same = cross = na = 0
    for u, v in G.edges():
        d = G.edges[u, v].get('income_distance')
        if d is None:     na    += 1
        elif d == 0:      same  += 1
        else:             cross += 1
    total = same + cross
    if total:
        print(f"\n── Edge income homophily (adults only) ──────────────")
        print(f"  Same band  (distance=0): {same:>7,}  ({100*same/total:.1f}%)")
        print(f"  Cross band (distance>0): {cross:>7,}  ({100*cross/total:.1f}%)")
        print(f"  Edges involving minor:   {na:>7,}")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich network (OPTIMIZED for large graphs)"
    )
    parser.add_argument("--input",   default="my_network/graph.npz")
    parser.add_argument("--output",  default="a_enriched.gpickle")
    parser.add_argument("--csv",     default="Data/enriched/enriched_interactions.csv")
    parser.add_argument("--pop-csv", default="Data/enriched/enriched_pop.csv")
    parser.add_argument("--encoding-dir", default="Data/enriched/encodings")
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    csv_path     = Path(args.csv)
    pop_csv_path = Path(args.pop_csv)
    encoding_dir = Path(args.encoding_dir)

    print(f"Loading {input_path} ...")
    
    # TRY NumPy format first, fall back to pickle

    G = load_network(input_path)
    print(f"  Loaded from NumPy format")
    # except (FileNotFoundError, KeyError):
    #     print(f"  NumPy format not found, trying pickle...")
    #     with open(input_path, "rb") as f:
    #         G = pickle.load(f)
    #     print(f"  Loaded from pickle format")
    
    # print(f"  {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")

    print("Assigning integer ages ...")
    assign_ages(G)

    print("Assigning 5-year age brackets ...")
    assign_5yr_age_brackets(G)

    print("Assigning income (homophily, high-degree first) ...")
    assign_income(G)

    print("Assigning arbeidsstatus ...")
    assign_arbeidsstatus(G)

    print("Assigning uitkeringstype ...")
    assign_uitkeringstype(G)

    print("Assigning burgerlijke_staat (mild homophily) ...")
    assign_burgerlijke_staat(G)

    print("Adding income_distance edge attribute ...")
    add_edge_income_distance(G)

    print_stats(G)

    print(f"\nBuilding encodings ...")
    encodings = build_encodings(G)

    print(f"Exporting encoding mappings ...")
    export_encoding_mappings(encodings, encoding_dir)

    print(f"Encoding node attributes with integers ...")
    encode_node_attributes(G, encodings)

    print(f"Exporting population CSV (with integer encodings) ...")
    export_population_csv_streaming(G, pop_csv_path)

    print(f"Exporting edge CSV (with integer encodings) ...")
    export_edge_csv_streaming(G, csv_path)

if __name__ == "__main__":
    main()