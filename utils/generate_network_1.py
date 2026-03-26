"""
Synthetic Amsterdam Interaction Network — Aggregated Group Expansion
=====================================================================
Input:  pop_table.csv         — (etngrp, geslacht, lft, oplniv, n)
        interaction_table.csv — (..._src, ..._dst, n)

Output:
  expanded_pop.csv          — full sub-group breakdown with all demographics + pop count
  expanded_interactions.csv — interaction rows split by income band (the homophily dimension)
                              all other new characteristics stay in expanded_pop as a join key

Why this split:
  Full cross-product of all new dimensions across both sides of each interaction row
  would produce ~265 billion pairs. Instead:
    - Income (4 bands) is the only cross dimension in the interaction file -> max 16x expansion
    - All other demographics live in expanded_pop and can be joined on base group keys
"""

import pandas as pd
import numpy as np
import os

RNG = np.random.default_rng(42)

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

ETHNICITIES  = ['Marokkaans', 'Turks', 'Surinaams', 'Overig', 'Autochtoon']
GENDERS      = ['Man', 'Vrouw']
AGE_BRACKETS = ['[0,20)', '[20,30)', '[30,40)', '[40,50)',
                '[50,60)', '[60,70)', '[70,80)', '[80,120]']
OPLNIV       = [1, 2, 3]

# Low:  Basisonderwijs, VMBO, eerste drie jaar HAVO/VWO, MBO-1
# Mid:  HAVO/VWO (diploma), MBO-2, MBO-3, MBO-4
# High: HBO, WO
OPLNIV_LABELS = {
    1: 'Laag (Basisonderwijs/VMBO/MBO-1)',
    2: 'Midden (HAVO/VWO/MBO-2/3/4)',
    3: 'Hoog (HBO/WO)'
}
OPLNIV_RANK = {1: 1, 2: 2, 3: 3}

AGE_BRACKET_MIN = {
    '[0,20)': 0, '[20,30)': 20, '[30,40)': 30, '[40,50)': 40,
    '[50,60)': 50, '[60,70)': 60, '[70,80)': 70, '[80,120]': 80,
}

# Income bands (18+ only) — the cross-dimension in the interaction file
INCOME_BANDS = ['Laag (<20k)', 'Modaal (20-35k)', 'Midden (35-55k)', 'Hoog (>55k)']
INCOME_BAND_ORDER = {b: i for i, b in enumerate(INCOME_BANDS)}

INKOMEN_PROBS = {
    1: {'Laag (<20k)': 0.40, 'Modaal (20-35k)': 0.42, 'Midden (35-55k)': 0.15, 'Hoog (>55k)': 0.03},
    2: {'Laag (<20k)': 0.15, 'Modaal (20-35k)': 0.42, 'Midden (35-55k)': 0.33, 'Hoog (>55k)': 0.10},
    3: {'Laag (<20k)': 0.05, 'Modaal (20-35k)': 0.20, 'Midden (35-55k)': 0.38, 'Hoog (>55k)': 0.37},
}

MARITAL_PROBS = {
    '[0,20)':   {'Ongehuwd': 0.99, 'Gehuwd': 0.01},
    '[20,30)':  {'Ongehuwd': 0.65, 'Gehuwd': 0.20, 'Samenwonend': 0.13, 'Gescheiden': 0.02},
    '[30,40)':  {'Ongehuwd': 0.30, 'Gehuwd': 0.42, 'Samenwonend': 0.18, 'Gescheiden': 0.10},
    '[40,50)':  {'Ongehuwd': 0.20, 'Gehuwd': 0.48, 'Samenwonend': 0.14, 'Gescheiden': 0.15, 'Weduwe/weduwnaar': 0.03},
    '[50,60)':  {'Ongehuwd': 0.15, 'Gehuwd': 0.50, 'Samenwonend': 0.10, 'Gescheiden': 0.18, 'Weduwe/weduwnaar': 0.07},
    '[60,70)':  {'Ongehuwd': 0.10, 'Gehuwd': 0.52, 'Samenwonend': 0.08, 'Gescheiden': 0.15, 'Weduwe/weduwnaar': 0.15},
    '[70,80)':  {'Ongehuwd': 0.08, 'Gehuwd': 0.48, 'Gescheiden': 0.10, 'Weduwe/weduwnaar': 0.34},
    '[80,120]': {'Ongehuwd': 0.05, 'Gehuwd': 0.35, 'Gescheiden': 0.08, 'Weduwe/weduwnaar': 0.52},
}

EMPLOYMENT_PROBS = {
    '[0,20)':   {'Student': 0.87, 'Werkend (deeltijd)': 0.08, 'Werkloos': 0.05},
    '[20,30)':  {'Werkend (voltijd)': 0.50, 'Werkend (deeltijd)': 0.15, 'Student': 0.20, 'Werkloos': 0.10, 'ZZP': 0.05},
    '[30,40)':  {'Werkend (voltijd)': 0.60, 'Werkend (deeltijd)': 0.18, 'ZZP': 0.10, 'Werkloos': 0.07, 'Thuiszorgend': 0.05},
    '[40,50)':  {'Werkend (voltijd)': 0.62, 'Werkend (deeltijd)': 0.16, 'ZZP': 0.10, 'Werkloos': 0.07, 'Arbeidsongeschikt': 0.05},
    '[50,60)':  {'Werkend (voltijd)': 0.55, 'Werkend (deeltijd)': 0.15, 'ZZP': 0.08, 'Werkloos': 0.10, 'Arbeidsongeschikt': 0.07, 'Gepensioneerd': 0.05},
    '[60,70)':  {'Gepensioneerd': 0.55, 'Werkend (deeltijd)': 0.15, 'Werkend (voltijd)': 0.10, 'ZZP': 0.07, 'Arbeidsongeschikt': 0.08, 'Werkloos': 0.05},
    '[70,80)':  {'Gepensioneerd': 0.90, 'Werkend (deeltijd)': 0.05, 'Arbeidsongeschikt': 0.05},
    '[80,120]': {'Gepensioneerd': 0.95, 'Arbeidsongeschikt': 0.05},
}

HEALTH_PROBS = {
    '[0,20)':   {'Uitstekend': 0.68, 'Goed': 0.29, 'Matig': 0.03},
    '[20,30)':  {'Uitstekend': 0.55, 'Goed': 0.35, 'Matig': 0.08, 'Slecht': 0.02},
    '[30,40)':  {'Uitstekend': 0.45, 'Goed': 0.38, 'Matig': 0.12, 'Slecht': 0.05},
    '[40,50)':  {'Uitstekend': 0.35, 'Goed': 0.38, 'Matig': 0.18, 'Slecht': 0.09},
    '[50,60)':  {'Uitstekend': 0.25, 'Goed': 0.38, 'Matig': 0.24, 'Slecht': 0.13},
    '[60,70)':  {'Uitstekend': 0.18, 'Goed': 0.35, 'Matig': 0.28, 'Slecht': 0.19},
    '[70,80)':  {'Uitstekend': 0.12, 'Goed': 0.30, 'Matig': 0.32, 'Slecht': 0.26},
    '[80,120]': {'Uitstekend': 0.08, 'Goed': 0.25, 'Matig': 0.35, 'Slecht': 0.32},
}

IMMIG_PROBS = {
    'Autochtoon': {'3e generatie+': 1.0},
    'Marokkaans': {'1e generatie': 0.35, '2e generatie': 0.30, '2e generatie (NL-geboren)': 0.35},
    'Turks':      {'1e generatie': 0.35, '2e generatie': 0.30, '2e generatie (NL-geboren)': 0.35},
    'Surinaams':  {'1e generatie': 0.30, '2e generatie': 0.35, '2e generatie (NL-geboren)': 0.35},
    'Overig':     {'1e generatie': 0.55, '2e generatie': 0.30, '3e generatie+': 0.15},
}

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def probs_to_splits(prob_dict, n):
    """Split n people across categories; returns list of (category, count)."""
    keys   = list(prob_dict.keys())
    probs  = np.array(list(prob_dict.values()), dtype=float)
    probs /= probs.sum()
    counts = (probs * n).astype(int)
    remainder = n - counts.sum()
    fracs = probs * n - counts
    counts[np.argsort(fracs)[::-1][:remainder]] += 1
    return [(keys[i], int(counts[i])) for i in range(len(keys)) if counts[i] > 0]


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD REAL INPUT DATA
# ─────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'Data')

INTERACTION_LAYERS = {
    'buren':       'tab_buren.csv',
    'familie':     'tab_familie.csv',
    'huishouden':  'tab_huishouden.csv',
    'werkschool':  'tab_werkschool.csv',
}

def load_pop_table():
    path = os.path.join(DATA_DIR, 'tab_n_(with oplniv).csv')
    print(f"Loading pop table from {path}...")
    df = pd.read_csv(path)
    df = df[df['n'] > 0].reset_index(drop=True)
    print(f"  -> {len(df)} groups, {df['n'].sum():,} people")
    return df


def load_interaction_table(layer_name):
    filename = INTERACTION_LAYERS[layer_name]
    path = os.path.join(DATA_DIR, filename)
    print(f"Loading interaction table '{layer_name}' from {path}...")
    df = pd.read_csv(path)
    # Keep only the columns needed; drop fn/N if present
    keep = ['geslacht_src', 'lft_src', 'oplniv_src', 'etngrp_src',
            'geslacht_dst', 'lft_dst', 'oplniv_dst', 'etngrp_dst', 'n']
    df = df[keep]
    df = df[df['n'] > 0].reset_index(drop=True)
    print(f"  -> {len(df):,} rows, {df['n'].sum():,} interactions")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 2 — EXPANDED POPULATION (all new dimensions)
# ─────────────────────────────────────────────────────────────────

def expand_population(pop_df):
    """
    Expand each base group row into sub-groups along all new demographic dimensions.
    Dimensions added:
      burgerlijke_staat | arbeidsstatus | inkomensniveau | gezondheid
      immigratie_gen    | kiesrecht
    """
    print("Expanding population sub-groups...")
    rows = []
    for _, r in pop_df.iterrows():
        eth, gen, lft, opl, n = r['etngrp'], r['geslacht'], r['lft'], int(r['oplniv']), int(r['n'])
        is_adult = AGE_BRACKET_MIN[lft] >= 20

        for bstat, n1 in probs_to_splits(MARITAL_PROBS[lft], n):
          for arb, n2 in probs_to_splits(EMPLOYMENT_PROBS[lft], n1):
            ink_splits = probs_to_splits(INKOMEN_PROBS[opl], n2) if is_adult else [('Niet van toepassing', n2)]
            for ink, n3 in ink_splits:
              for gez, n4 in probs_to_splits(HEALTH_PROBS[lft], n3):
                for immig, n5 in probs_to_splits(IMMIG_PROBS[eth], n4):
                  kies_splits = probs_to_splits({'Ja': 0.78, 'Nee': 0.22}, n5) if is_adult else [('Nee', n5)]
                  for kies, n6 in kies_splits:
                    if n6 > 0:
                        rows.append({
                            'etngrp': eth, 'geslacht': gen, 'lft': lft,
                            'oplniv': opl, 'oplniv_label': OPLNIV_LABELS[opl],
                            'burgerlijke_staat': bstat, 'arbeidsstatus': arb,
                            'inkomensniveau': ink, 'gezondheid': gez,
                            'immigratie_gen': immig, 'kiesrecht': kies,
                            'n': n6
                        })

    df = pd.DataFrame(rows)
    print(f"  -> {len(df):,} sub-groups, {df['n'].sum():,} people")
    return df


# ─────────────────────────────────────────────────────────────────
# STEP 3 — EXPANDED INTERACTIONS (income band as cross-dimension)
# ─────────────────────────────────────────────────────────────────

def expand_interactions(interaction_df, expanded_pop):
    """
    Split each base interaction row by (inkomensniveau_src × inkomensniveau_dst).
    
    Income homophily: pairs with closer income bands are upweighted via
    a Gaussian-style decay on band distance (0 apart=1.0, 1=0.5, 2=0.15, 3=0.04).
    
    Education homophily: cross-oplniv pairs are downweighted at the base row level
    (0 apart=1.0, 1 apart=0.4, 2 apart=0.1) — already baked into the base n values
    from observed data, but applied here for the synthetic dummy input.

    Other new characteristics (burgerlijke_staat, arbeidsstatus, gezondheid,
    immigratie_gen, kiesrecht) are NOT cross-dimensions in the interaction file —
    they are available per-group via expanded_pop.csv joined on base group keys.
    """
    print("Expanding interactions by income band...")

    # OPL_PENALTY is 1.0 for all — education homophily is already baked into real data n values
    OPL_PENALTY   = {0: 1.0, 1: 1.0, 2: 1.0}
    # Gaussian decay by income band distance
    INCOME_DECAY  = {0: 1.0, 1: 0.50, 2: 0.15, 3: 0.04}

    # Build income band -> population share lookup per base group
    BASE_KEYS = ['etngrp', 'geslacht', 'lft', 'oplniv']
    inc_shares = (
        expanded_pop.groupby(BASE_KEYS + ['inkomensniveau'])['n']
        .sum()
        .reset_index()
    )
    inc_shares['share'] = inc_shares.groupby(BASE_KEYS)['n'].transform(lambda x: x / x.sum())
    inc_lookup = {}
    for key, grp in inc_shares.groupby(BASE_KEYS):
        inc_lookup[key] = dict(zip(grp['inkomensniveau'], grp['share']))

    out_rows = []
    n_rows = len(interaction_df)

    for i, row in enumerate(interaction_df.itertuples(index=False)):
        if i % 5000 == 0:
            print(f"  ... {i:>6,}/{n_rows:,}", end='\r')

        sk = (row.etngrp_src, row.geslacht_src, row.lft_src, int(row.oplniv_src))
        dk = (row.etngrp_dst, row.geslacht_dst, row.lft_dst, int(row.oplniv_dst))

        src_inc = inc_lookup.get(sk, {})
        dst_inc = inc_lookup.get(dk, {})
        n_total = int(row.n)

        opl_pen = OPL_PENALTY[abs(OPLNIV_RANK[sk[3]] - OPLNIV_RANK[dk[3]])]

        # Under-18 groups -> no income split, emit single row
        if not src_inc or not dst_inc:
            out_rows.append({
                **{f'{k}_src': v for k,v in zip(['etngrp','geslacht','lft','oplniv'], sk)},
                'oplniv_label_src':   OPLNIV_LABELS[sk[3]],
                'inkomensniveau_src': 'Niet van toepassing',
                **{f'{k}_dst': v for k,v in zip(['etngrp','geslacht','lft','oplniv'], dk)},
                'oplniv_label_dst':   OPLNIV_LABELS[dk[3]],
                'inkomensniveau_dst': 'Niet van toepassing',
                'n': n_total
            })
            continue

        # Build 4×4 weight matrix over income bands
        bands = INCOME_BANDS
        src_w = np.array([src_inc.get(b, 0.0) for b in bands])
        dst_w = np.array([dst_inc.get(b, 0.0) for b in bands])

        W = np.outer(src_w, dst_w) * opl_pen
        for si, sb in enumerate(bands):
            for di, db in enumerate(bands):
                dist = abs(INCOME_BAND_ORDER[sb] - INCOME_BAND_ORDER[db])
                W[si, di] *= INCOME_DECAY[dist]

        W_sum = W.sum()
        if W_sum == 0:
            continue

        counts = (W / W_sum * n_total).astype(int)
        remainder = n_total - counts.sum()
        fracs = W / W_sum * n_total - counts
        flat_top = np.argsort(fracs.flatten())[::-1][:remainder]
        counts.flat[flat_top] += 1

        for si, sb in enumerate(bands):
            for di, db in enumerate(bands):
                c = int(counts[si, di])
                if c > 0:
                    out_rows.append({
                        'etngrp_src':         sk[0], 'geslacht_src':       sk[1],
                        'lft_src':            sk[2], 'oplniv_src':         sk[3],
                        'oplniv_label_src':   OPLNIV_LABELS[sk[3]],
                        'inkomensniveau_src': sb,
                        'etngrp_dst':         dk[0], 'geslacht_dst':       dk[1],
                        'lft_dst':            dk[2], 'oplniv_dst':         dk[3],
                        'oplniv_label_dst':   OPLNIV_LABELS[dk[3]],
                        'inkomensniveau_dst': db,
                        'n': c
                    })

    result = pd.DataFrame(out_rows)
    print(f"\n  -> {len(result):,} expanded interaction rows, {result['n'].sum():,} interactions")
    return result


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    # Step 1 — load real data
    pop_df = load_pop_table()

    # Step 2 — expanded population (shared across all interaction layers)
    expanded_pop = expand_population(pop_df)
    expanded_pop.to_csv(os.path.join(out_dir, 'expanded_pop.csv'), index=False)
    print(f"  OK Saved expanded_pop.csv  ({len(expanded_pop):,} rows)")

    # Step 3 — expanded interactions for each layer
    print("\n====================== SUMMARY ======================")
    print(f"  Base pop groups:         {len(pop_df):>10,}")
    print(f"  Expanded pop sub-groups: {len(expanded_pop):>10,}")
    for layer in INTERACTION_LAYERS:
        interaction_df = load_interaction_table(layer)
        expanded_int   = expand_interactions(interaction_df, expanded_pop)
        out_path = os.path.join(out_dir, f'expanded_interactions_{layer}.csv')
        expanded_int.to_csv(out_path, index=False)
        print(f"  OK {layer}: {len(interaction_df):>8,} base rows -> {len(expanded_int):>10,} expanded rows  ({expanded_int['n'].sum():,} interactions)")

if __name__ == '__main__':
    main()
