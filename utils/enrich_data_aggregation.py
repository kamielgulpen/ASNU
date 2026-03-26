import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

POP_CSV          = Path('Data/enriched/enriched_pop.csv')
INTERACTIONS_CSV = Path('Data/enriched/enriched_interactions.csv')
OUT_DIR          = Path('Data/enriched/aggregated')

BASE_CHARACTERISTICS = ['etngrp', 'geslacht', 'lft', 'oplniv']
ENRICHED_CHARACTERISTICS = ['inkomensniveau', 'arbeidsstatus', 'uitkeringstype', 'burgerlijke_staat']
CHARACTERISTICS = BASE_CHARACTERISTICS + ENRICHED_CHARACTERISTICS


def iter_combos(r: int):
    """For r <= len(BASE), use only base characteristics; otherwise use all."""
    pool = BASE_CHARACTERISTICS if r <= len(BASE_CHARACTERISTICS) else CHARACTERISTICS
    yield from combinations(pool, r)


OUT_DIR.mkdir(parents=True, exist_ok=True)


def hhi(series: pd.Series) -> float:
    """Herfindahl-Hirschman Index: sum of squared shares. Range [1/k, 1]."""
    shares = series / series.sum()
    return float((shares ** 2).sum())


def sample_combos(summary: pd.DataFrame, data_type: str, n_sample: int = 10) -> pd.DataFrame:
    """Sample n_sample combinations evenly across HHI range, half base-only half enriched."""
    subset = summary[summary['type'] == data_type]
    is_base = subset['characteristics'].apply(
        lambda s: all(c.strip() in BASE_CHARACTERISTICS for c in s.split(','))
    )
    base_sub  = subset[is_base].sort_values('hhi').reset_index(drop=True)
    other_sub = subset[~is_base].sort_values('hhi').reset_index(drop=True)

    def _pick(df, n):
        if len(df) == 0:
            return df
        idx = np.linspace(0, len(df) - 1, min(n, len(df)), dtype=int)
        return df.iloc[idx]

    n_base = n_sample // 2
    return pd.concat([_pick(base_sub, n_base), _pick(other_sub, n_sample - n_base)]).sort_values('hhi')


def plot_sample_distributions(raw: pd.DataFrame, sampled: pd.DataFrame,
                              data_type: str, path: Path) -> None:
    """Plot Lorenz curves for the sampled combinations."""
    n = len(sampled)
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.colormaps['RdYlGn_r']
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    for (_, row), color in zip(sampled.iterrows(), colors):
        cols = [c.strip() for c in row['characteristics'].split(',')]
        if data_type == 'interactions':
            cols = [f"{c}_src" for c in cols] + [f"{c}_dst" for c in cols]
        df_agg = raw.groupby(cols, dropna=False)['n'].sum().reset_index()
        shares = df_agg['n'].sort_values().values
        cum_share = shares.cumsum() / shares.sum()
        cum_pop = np.linspace(0, 1, len(shares))
        ax.plot(cum_pop, cum_share, color=color, alpha=0.8,
                label=f"HHI={row['hhi']:.4f} ({row['characteristics']})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfect equality')
    ax.set_xlabel('Cumulative share of groups')
    ax.set_ylabel('Cumulative population share')
    ax.set_title(f'Lorenz curves for sampled {data_type} aggregations')
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")


def plot_hhi(summary: pd.DataFrame, path: Path) -> None:
    bins = np.logspace(
        np.log10(summary['hhi'].min()),
        np.log10(summary['hhi'].max()),
        40,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, color in [('pop', 'steelblue'), ('interactions', 'darkorange')]:
        subset = summary[summary['type'] == label]
        if subset.empty:
            continue
        ax.hist(subset['hhi'], bins=bins, label=label.capitalize(),
                color=color, alpha=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('HHI (log scale)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of HHI across aggregations')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {path}")


pop = pd.read_csv(POP_CSV)

# ── Pass 1: compute HHI for all combinations to find the sample ──────────────
print("Pass 1: computing HHI for all combinations...")
hhi_rows = []
for r in range(1, len(CHARACTERISTICS) + 1):
    for combo in iter_combos(r):
        group_cols = list(combo)
        df_agg = pop.groupby(group_cols, dropna=False, as_index=False)['n'].sum()
        hhi_rows.append({'type': 'pop', 'characteristics': ", ".join(combo),
                         'n_groups': len(df_agg), 'hhi': hhi(df_agg['n'])})

hhi_summary = pd.DataFrame(hhi_rows)
hhi_summary['n_characteristics'] = hhi_summary['characteristics'].str.count(',') + 1
plot_hhi(hhi_summary, OUT_DIR / 'hhi_plot.png')

sampled = sample_combos(hhi_summary, 'pop')
print(f"\nSampled {len(sampled)} combinations:")
for _, row in sampled.iterrows():
    print(f"  HHI={row['hhi']:.4f}  {row['characteristics']}")

# ── Pass 2: create pop and interaction files for the sampled combinations ─────
print("\nPass 2: creating aggregation files for sampled combinations...")
interactions = pd.read_csv(INTERACTIONS_CSV)
summary_rows = []

for _, row in sampled.iterrows():
    combo = [c.strip() for c in row['characteristics'].split(',')]
    filename = "_".join(combo) + ".csv"

    # Pop
    df_pop = pop.groupby(combo, dropna=False, as_index=False)['n'].sum()
    df_pop.to_csv(OUT_DIR / f"pop_{filename}", index=False)
    summary_rows.append({'type': 'pop', 'characteristics': row['characteristics'],
                         'n_groups': len(df_pop), 'hhi': row['hhi']})

    # Interactions
    int_cols = [f"{c}_src" for c in combo] + [f"{c}_dst" for c in combo]
    df_int = interactions.groupby(int_cols, dropna=False, as_index=False)['n'].sum()
    df_int.to_csv(OUT_DIR / f"interactions_{filename}", index=False)
    h_int = hhi(df_int['n'])
    summary_rows.append({'type': 'interactions', 'characteristics': row['characteristics'],
                         'n_groups': len(df_int), 'hhi': h_int})

    print(f"  {filename}  |  pop groups: {len(df_pop):,}  |  int groups: {len(df_int):,}")

summary = pd.DataFrame(summary_rows)
summary['n_characteristics'] = summary['characteristics'].str.count(',') + 1
summary.to_csv(OUT_DIR / 'hhi_summary.csv', index=False)
print(f"\nSaved HHI summary -> {OUT_DIR / 'hhi_summary.csv'}")

plot_sample_distributions(pop, summary[summary['type'] == 'pop'], 'pop',
                          OUT_DIR / 'hhi_sample_distributions_pop.png')
plot_sample_distributions(interactions, summary[summary['type'] == 'interactions'], 'interactions',
                          OUT_DIR / 'hhi_sample_distributions_interactions.png')
plot_hhi(summary, OUT_DIR / 'hhi_plot_sampled.png')
