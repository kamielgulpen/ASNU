"""
enrich_network_attributes.py
============================
Reads a .pkl (a NetworkX DiGraph) and enriches each node with:

  age               — integer sampled uniformly within the lft age bracket
  inkomensniveau    — income band, derived from oplniv + homophily with neighbours
  inkomen           — integer annual income (€) sampled within the chosen band
  arbeidsstatus     — employment status, derived from age + oplniv + inkomensniveau
  uitkeringstype    — benefit type, derived from arbeidsstatus (None if not applicable)
  burgerlijke_staat — civil status, derived from age + homophily with neighbours

Dependencies
------------
  age          ← lft
  inkomen      ← oplniv + age + neighbour income (homophily)
  arbeidsstatus← age + oplniv + inkomensniveau
  uitkeringstype← arbeidsstatus
  burgerlijke_staat ← age + neighbour burgerlijke_staat (homophily)

Processing order for homophily attributes: descending degree, so hubs
are assigned first and propagate their signal to neighbours.

Income homophily
----------------
Gaussian-style decay on band distance:
  distance 0 → 1.00,  1 → 0.50,  2 → 0.15,  3 → 0.04
HOMOPHILY_STRENGTH (default 0.35) blends base education probs with
neighbour-driven probs.

Burgerlijke staat homophily
---------------------------
Weaker homophily (BSTAT_HOMOPHILY_STRENGTH = 0.20): social networks
show mild clustering by life stage (married people know married people)
but it is far less pronounced than income segregation.

Usage
-----
  python enrich_network_attributes.py
  python enrich_network_attributes.py --input a.pkl --output a_enriched.pkl
"""

import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
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

# ── Income ────────────────────────────────────────────────────────

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

# ── Arbeidsstatus ─────────────────────────────────────────────────
# Base probs by age bracket; refined by oplniv and inkomensniveau below.

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

# Multipliers on arbeidsstatus given oplniv (education level).
# Higher education → less unemployment/disability, more ZZP/voltijd.
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

# Multipliers on arbeidsstatus given inkomensniveau.
# Low income correlates with unemployment/part-time; high income with voltijd/ZZP.
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
    'Niet van toepassing': {},  # minors — no modifier
}

# ── Uitkeringstype ────────────────────────────────────────────────
# Only assigned when arbeidsstatus implies a benefit entitlement.
# WW  = Werkloosheidswet (unemployment insurance, recent work history required)
# Bijstand = social assistance (last resort)
# WAO/WIA  = disability benefit
# AOW      = state pension (65+)
# AIO      = supplementary income for pensioners below poverty line

UITKERING_PROBS: dict[str, dict[str, float] | None] = {
    'Werkloos':          {'WW': 0.55, 'Bijstand': 0.45},
    'Arbeidsongeschikt': {'WAO/WIA': 1.0},
    'Thuiszorgend':      {'Bijstand': 0.65, 'WW': 0.35},
    'Gepensioneerd':     {'AOW': 0.85, 'AIO': 0.15},
    # All other statuses → no benefit
    'Werkend (voltijd)': None,
    'Werkend (deeltijd)':None,
    'ZZP':               None,
    'Student':           None,
    'Student/kind':      None,
}

# ── Burgerlijke staat ─────────────────────────────────────────────
# Categories ordered by a loose "life-stage rank" used for homophily distance.

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

# Mild homophily: married people tend to connect to married people,
# but effect is weaker than income segregation.
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
    """
    Return (categories, probability_vector) for arbeidsstatus.
    Base probs are multiplied by oplniv and inkomensniveau modifiers,
    then renormalised.
    """
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
    """
    Return probability vector over BSTAT_CATEGORIES.
    Blends age-based prior with mild neighbour homophily.
    """
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


# ─────────────────────────────────────────────────────────────────
# ASSIGNMENT FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def assign_ages(G) -> None:
    """Assign integer `age` to every node from its `lft` bracket."""
    for node, attrs in G.nodes(data=True):
        G.nodes[node]['age'] = _sample_age(attrs['lft'])


def assign_income(G) -> None:
    """
    Assign `inkomensniveau` (band) and `inkomen` (integer €).
    High-degree nodes processed first to maximise homophily propagation.
    Under-20 nodes receive inkomensniveau='Niet van toepassing', inkomen=0.
    """
    for node in sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True):
        attrs = G.nodes[node]
        lft   = attrs['lft']

        if AGE_BRACKET_MIN[lft] < 20:
            G.nodes[node]['inkomensniveau'] = 'Niet van toepassing'
            G.nodes[node]['inkomen'] = 0
            continue

        neighbour_bands = [
            G.nodes[nbr].get('inkomensniveau')
            for nbr in set(list(G.predecessors(node)) + list(G.successors(node)))
            if G.nodes[nbr].get('inkomensniveau') in INCOME_BAND_ORDER
        ]

        probs       = _blended_income_probs(int(attrs['oplniv']), int(attrs['age']), neighbour_bands)
        chosen_band = str(RNG.choice(INCOME_BANDS, p=probs))
        lo, hi      = INCOME_BAND_RANGES[chosen_band]

        G.nodes[node]['inkomensniveau'] = chosen_band
        G.nodes[node]['inkomen']        = int(RNG.integers(lo, hi + 1))


def assign_arbeidsstatus(G) -> None:
    """
    Assign `arbeidsstatus` derived from age + oplniv + inkomensniveau.
    No homophily: employment status is driven by individual attributes,
    not social clustering in the same way income is.
    """
    for node, attrs in G.nodes(data=True):
        lft           = attrs['lft']
        oplniv        = int(attrs['oplniv'])
        inkomensniveau = attrs.get('inkomensniveau', 'Niet van toepassing')

        cats, probs = _arbeid_probs(lft, oplniv, inkomensniveau)
        G.nodes[node]['arbeidsstatus'] = str(RNG.choice(cats, p=probs))


def assign_uitkeringstype(G) -> None:
    """
    Assign `uitkeringstype` derived from arbeidsstatus.
    None for statuses that carry no benefit entitlement.
    AOW age check: Gepensioneerd → AOW only if age >= 67, else AIO more likely.
    """
    for node, attrs in G.nodes(data=True):
        arbeid = attrs.get('arbeidsstatus', '')
        probs_dict = UITKERING_PROBS.get(arbeid)

        if probs_dict is None:
            G.nodes[node]['uitkeringstype'] = None
            continue

        # Age-adjusted pension: below 67 → AIO weight goes up
        if arbeid == 'Gepensioneerd':
            age = int(attrs.get('age', 70))
            if age < 67:
                probs_dict = {'AOW': 0.40, 'AIO': 0.60}
            else:
                probs_dict = {'AOW': 0.85, 'AIO': 0.15}

        cats  = list(probs_dict.keys())
        probs = _normalise(np.array(list(probs_dict.values())))
        G.nodes[node]['uitkeringstype'] = str(RNG.choice(cats, p=probs))


def assign_burgerlijke_staat(G) -> None:
    """
    Assign `burgerlijke_staat` from age-based prior + mild neighbour homophily.
    High-degree nodes processed first.
    """
    for node in sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True):
        attrs = G.nodes[node]
        lft   = attrs['lft']

        neighbour_bstats = [
            G.nodes[nbr].get('burgerlijke_staat')
            for nbr in set(list(G.predecessors(node)) + list(G.successors(node)))
            if G.nodes[nbr].get('burgerlijke_staat') in BSTAT_ORDER
        ]

        probs = _blended_bstat_probs(lft, neighbour_bstats)
        chosen = [b for b in BSTAT_CATEGORIES if BSTAT_BASE_PROBS[lft].get(b, 0) > 0 or probs[BSTAT_ORDER[b]] > 0]
        # Filter to only age-plausible categories (e.g. no widows among 20-year-olds)
        valid_cats   = [b for b in BSTAT_CATEGORIES if BSTAT_BASE_PROBS[lft].get(b, 0) > 0]
        valid_probs  = _normalise(np.array([probs[BSTAT_ORDER[b]] for b in valid_cats]))

        G.nodes[node]['burgerlijke_staat'] = str(RNG.choice(valid_cats, p=valid_probs))


# ─────────────────────────────────────────────────────────────────
# EDGE ATTRIBUTES
# ─────────────────────────────────────────────────────────────────

def add_edge_income_distance(G) -> None:
    """Add `income_distance` (0–3) to each edge; None if either endpoint is a minor."""
    for u, v in G.edges():
        band_u = G.nodes[u].get('inkomensniveau')
        band_v = G.nodes[v].get('inkomensniveau')
        if band_u in INCOME_BAND_ORDER and band_v in INCOME_BAND_ORDER:
            dist = abs(INCOME_BAND_ORDER[band_u] - INCOME_BAND_ORDER[band_v])
        else:
            dist = None
        G.edges[u, v]['income_distance'] = dist


# ─────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────

def print_stats(G) -> None:
    n_nodes = G.number_of_nodes()

    def _dist(attr, categories):
        counts = Counter(G.nodes[n].get(attr) for n in G.nodes())
        print(f"\n  {attr}:")
        for cat in categories:
            c = counts.get(cat, 0)
            print(f"    {str(cat):<30} {c:>7,}  ({100*c/n_nodes:.1f}%)")
        other = {k: v for k, v in counts.items() if k not in categories}
        if other:
            print(f"    {'(other/None)':<30} {sum(other.values()):>7,}")

    print("\n── Income bands ─────────────────────────────────────────")
    _dist('inkomensniveau', INCOME_BANDS + ['Niet van toepassing'])

    print("\n── Arbeidsstatus ────────────────────────────────────────")
    all_arbeid = sorted(x for x in {G.nodes[n].get('arbeidsstatus') for n in G.nodes()} if x is not None)
    _dist('arbeidsstatus', all_arbeid)

    print("\n── Uitkeringstype ───────────────────────────────────────")
    all_uitk = sorted(x for x in {G.nodes[n].get('uitkeringstype') for n in G.nodes()} if x is not None)
    _dist('uitkeringstype', all_uitk)

    print("\n── Burgerlijke staat ────────────────────────────────────")
    _dist('burgerlijke_staat', BSTAT_CATEGORIES)

    ages = [G.nodes[n]['age'] for n in G.nodes() if 'age' in G.nodes[n]]
    print(f"\n── Age  min={min(ages)}  max={max(ages)}  "
          f"mean={np.mean(ages):.1f}  median={np.median(ages):.0f}")

    # Edge income homophily
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
# CSV EXPORT
# ─────────────────────────────────────────────────────────────────

GROUP_KEYS = [
    'etngrp', 'geslacht', 'lft', 'oplniv',
    'inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat',
]


def export_edge_csv(G, csv_path: Path) -> None:
    """
    Write a CSV of (src group × dst group → n edges).
    Group keys: all attributes in GROUP_KEYS.
    """
    import pandas as pd

    rows = []
    for u, v in G.edges():
        row = {f"{k}_src": G.nodes[u].get(k) for k in GROUP_KEYS}
        row.update({f"{k}_dst": G.nodes[v].get(k) for k in GROUP_KEYS})
        rows.append(row)

    cols_src = [f"{k}_src" for k in GROUP_KEYS]
    cols_dst = [f"{k}_dst" for k in GROUP_KEYS]

    df = (
        pd.DataFrame(rows)
        .groupby(cols_src + cols_dst, dropna=False)
        .size()
        .reset_index(name='n')
        .sort_values(cols_src + cols_dst)
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  {len(df):,} group-pair rows  |  {df['n'].sum():,} edges  →  {csv_path}")


def export_population_csv(G, csv_path: Path) -> None:
    """
    Write a CSV of (group combination → n nodes).
    Columns: GROUP_KEYS + n.
    """
    import pandas as pd

    rows = [{k: G.nodes[n].get(k) for k in GROUP_KEYS} for n in G.nodes()]
    df = (
        pd.DataFrame(rows)
        .groupby(GROUP_KEYS, dropna=False)
        .size()
        .reset_index(name='n')
        .sort_values(GROUP_KEYS)
    )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"  {len(df):,} population groups  |  {df['n'].sum():,} nodes  →  {csv_path}")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enrich network pkl with age, income, arbeidsstatus, "
                    "uitkeringstype, and burgerlijke_staat."
    )
    parser.add_argument("--input",   default="original.pkl",          help="Input .pkl path")
    parser.add_argument("--output",  default="a_enriched.pkl", help="Output .pkl path")
    parser.add_argument("--csv",     default="Data/outputs/enriched_interactions.csv")
    parser.add_argument("--pop-csv", default="Data/outputs/enriched_pop.csv")
    args = parser.parse_args()

    input_path   = Path(args.input)
    output_path  = Path(args.output)
    csv_path     = Path(args.csv)
    pop_csv_path = Path(args.pop_csv)

    print(f"Loading {input_path} ...")
    with open(input_path, "rb") as f:
        G = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes  |  {G.number_of_edges():,} edges")

    # Assignment order matters: income must precede arbeidsstatus
    print("Assigning integer ages ...")
    assign_ages(G)

    print("Assigning income (homophily, high-degree first) ...")
    assign_income(G)

    print("Assigning arbeidsstatus (age + oplniv + inkomensniveau) ...")
    assign_arbeidsstatus(G)

    print("Assigning uitkeringstype (from arbeidsstatus) ...")
    assign_uitkeringstype(G)

    print("Assigning burgerlijke_staat (age + mild homophily) ...")
    assign_burgerlijke_staat(G)

    print("Adding income_distance edge attribute ...")
    add_edge_income_distance(G)

    print_stats(G)

    print(f"\nExporting population CSV ...")
    export_population_csv(G, pop_csv_path)

    print(f"Exporting edge group CSV ...")
    export_edge_csv(G, csv_path)

    print(f"\nSaving enriched network to {output_path} ...")
    with open(output_path, "wb") as f:
        pickle.dump(G, f)
    print("Done.")


if __name__ == "__main__":
    main()