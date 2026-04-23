"""
Analyze network statistics from JSON files with real network reference.
Groups and labels results by HHI (from group_charcteristic_mapping.csv).
"""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from scipy.stats import skew, spearmanr
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from pathlib import Path


def compute_real_network_stats(pkl_path='original.pkl'):
    """Compute reference statistics from the real network"""
    print(f"Computing real network stats from {pkl_path}...")
    # with open(pkl_path, 'rb') as f:
    #     data = pickle.load(f)

    # G_ig = ig.Graph.from_networkx(data)
    # G_undirected = G_ig.as_undirected() if G_ig.is_directed() else G_ig

    # stats = {
    #     'transitivity': G_undirected.transitivity_undirected(),
    #     'transitivity_avg': G_undirected.transitivity_avglocal_undirected(),
    #     'modularity': G_undirected.modularity(G_undirected.community_label_propagation()),
    #     'degree_skew': float(skew(G_undirected.degree())),
    #     'degree_mean': float(np.mean(G_undirected.degree())),
    #     'reciprocity': G_ig.reciprocity() if G_ig.is_directed() else 1.0,
    #     'nodes': G_undirected.vcount(),
    #     'edges': G_undirected.ecount(),
    # }

    stats = {
        'transitivity':  0.4941,
        'transitivity_avg': 0.0131,
        'modularity':  0.8813,
        'degree_skew':  7.0774,
        'degree_mean': 26.1996,
        'reciprocity': 0.9701,
        'nodes': 860188,
        'edges': 11268300,
    }

    print("Real network stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    return stats


def load_all_json_files(base_dir='enriched'):
    data = []
    base_path = Path(base_dir)
    json_files = list(base_path.rglob('*.json'))
    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                stats = json.load(f)
                stats['folder'] = json_file.parent.name
                stats['filename'] = json_file.stem
                stats['file_path'] = str(json_file)
                data.append(stats)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} network statistics")
    print(f"Unique filenames: {df['filename'].nunique()}")
    return df


def load_hhi_mapping(csv_path='group_charcteristic_mapping.csv'):
    """Load the HHI mapping CSV."""
    mapping = pd.read_csv(csv_path)
    print(f"Loaded HHI mapping with {len(mapping)} rows")
    print(f"  HHI range: [{mapping['hhi'].min():.4f}, {mapping['hhi'].max():.4f}]")
    return mapping


def merge_hhi(df, mapping):
    """Merge HHI values onto the main dataframe.
    Filenames look like 'geslacht_stats', aggregation is 'geslacht' — strip '_stats'.
    """
    df['aggregation_key'] = df['filename'].str.replace(r'_stats$', '', regex=True)

    df = df.merge(
        mapping[['aggregation', 'hhi', 'n_groups', 'effective_n']],
        left_on='aggregation_key',
        right_on='aggregation',
        how='left',
    )
    matched = df['hhi'].notna().sum()
    print(f"HHI merge: {matched}/{len(df)} rows matched")
    if df['hhi'].isna().any():
        unmatched = df.loc[df['hhi'].isna(), 'aggregation_key'].unique()
        print(f"  Unmatched: {list(unmatched)}")
    df = df.dropna(subset=['hhi'])
    df = df.reset_index(drop=True)
    print(f"  Rows after dropping unmatched: {len(df)}")
    df['hhi_label'] = df['hhi'].map(lambda h: f"HHI={h:.4f}")
    return df


def parse_params(df):
    param_patterns = {
        'scale': r'scale=(\d+)', 'comms': r'comms=(\d+)', 'recip': r'recip=(\d+)',
        'trans': r'trans=([\d.]+)', 'pa': r'pa=([\d.]+)', 'bridge': r'bridge=([\d.]+)'
    }
    for param, pattern in param_patterns.items():
        df[param] = df['params'].str.extract(pattern).astype(float)
    return df


def filter_param_combinations(df, tol=0.02):
    """Keep only rows matching the specified parameter grid."""
    # PREF_ATTACHMENT_VALUES    = np.linspace(0, 0.9999, 2)       # [0, 0.9999]
    # N_COMMUNITIES_VALUES      = np.logspace(0, 4.7, 10).astype(int)
    TRANSITIVITY_VALUES       = np.array([0])      # [0, 0.5, 1]
    BRIDGE_PROBABILITY_VALUES = np.array([0.2])

    def _match(series, valid_values, tol=tol):
        return series.apply(lambda x: any(abs(x - v) < tol for v in valid_values))

    mask = (
        # _match(df['pa'],     PREF_ATTACHMENT_VALUES) &
        # _match(df['comms'],  N_COMMUNITIES_VALUES) &
        _match(df['trans'],  TRANSITIVITY_VALUES) 
        # _match(df['bridge'], BRIDGE_PROBABILITY_VALUES)
    )
    filtered = df[mask].copy()
    print(f"Filtered: {len(filtered)}/{len(df)} rows match the parameter grid")
    return filtered


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_ref_line(ax, value, axis='x', label='Real', color='black'):
    if value is None:
        return
    if axis == 'x':
        ax.axvline(value, color=color, linestyle='-', linewidth=2, alpha=0.9,
                   label=f'{label}: {value:.3f}', zorder=10)
    else:
        ax.axhline(value, color=color, linestyle='-', linewidth=2, alpha=0.9,
                   label=f'{label}: {value:.3f}', zorder=10)


def _add_ref_point(ax, x, y, label='Real', color='black'):
    if x is None or y is None:
        return
    ax.scatter([x], [y], marker='*', s=300, color=color, edgecolor='white',
               linewidth=1.5, zorder=10, label=f'{label} ({x:.2f}, {y:.3f})')


def _avg_distance_to_real(group_df, real):
    """Euclidean distance from each (skew, transitivity) point to the real network, averaged."""
    if real is None:
        return np.nan
    dx = group_df['degree_skew'] - real['degree_skew']
    dt = group_df['transitivity'] - real['transitivity']
    return float(np.sqrt(dx**2 + dt**2).mean())


def _hhi_sorted_groups(df):
    """Return (sorted HHI values, matching labels) for consistent ordering."""
    hhi_vals = sorted(df['hhi'].unique())
    labels = [f"{h:.4f}" for h in hhi_vals]
    return hhi_vals, labels


# ---------------------------------------------------------------------------
# Analysis & plots — all grouped by HHI
# ---------------------------------------------------------------------------

def analyze_by_hhi(df, real=None):
    print("\n" + "="*60 + "\nBY HHI\n" + "="*60)
    grouped = df.groupby('hhi').agg({
        'degree_skew': ['mean', 'std', 'min', 'max'],
        'transitivity': ['mean', 'std', 'min', 'max'],
        'reciprocity': ['mean', 'std'],
        'degree_mean': ['mean', 'std'],
        'nodes': ['mean', 'count'],
        'filename': 'first',
        'aggregation': 'first',
    }).round(4)

    # Add average distance to real network in (skew, transitivity) space
    if real:
        dist_dict = {}
        for hhi_val, group in df.groupby('hhi'):
            dist_dict[hhi_val] = round(_avg_distance_to_real(group, real), 4)
        dist_by_hhi = pd.Series(dist_dict, name='dist_to_real')
        print("\nAvg distance to real (skew, transitivity):")
        for hhi_val in dist_by_hhi.sort_values().index:
            agg = df.loc[df['hhi'] == hhi_val, 'aggregation'].iloc[0]
            print(f"  HHI={hhi_val:.4f} ({agg}): {dist_by_hhi[hhi_val]:.4f}")
        dist_by_hhi.to_csv('distance_by_hhi.csv')

    print(grouped.to_string())
    grouped.to_csv('stats_by_hhi.csv')
    print("\nSaved: stats_by_hhi.csv")
    return grouped


def plot_distributions_by_hhi(df, real=None, save_dir='network_plots'):
    Path(save_dir).mkdir(exist_ok=True)
    hhi_vals, hhi_labels = _hhi_sorted_groups(df)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(hhi_vals)))

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # --- Skewness hist ---
    ax = axes[0, 0]
    for idx, hhi in enumerate(hhi_vals):
        ax.hist(df[df['hhi'] == hhi]['degree_skew'], bins=20, alpha=0.5,
                label=f'HHI={hhi_labels[idx]}', color=colors[idx])
    _add_ref_line(ax, real and real.get('degree_skew'), axis='x')
    ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Frequency')
    ax.set_title('Skewness Distribution by HHI')
    ax.legend(fontsize=6, ncol=2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Transitivity hist ---
    ax = axes[0, 1]
    for idx, hhi in enumerate(hhi_vals):
        ax.hist(df[df['hhi'] == hhi]['transitivity'], bins=20, alpha=0.5,
                label=f'HHI={hhi_labels[idx]}', color=colors[idx])
    _add_ref_line(ax, real and real.get('transitivity'), axis='x')
    ax.set_xlabel('Transitivity'); ax.set_ylabel('Frequency')
    ax.set_title('Transitivity Distribution by HHI')
    ax.legend(fontsize=6, ncol=2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Skewness vs Transitivity scatter ---
    ax = axes[0, 2]
    for idx, hhi in enumerate(hhi_vals):
        d = df[df['hhi'] == hhi]
        dist = _avg_distance_to_real(d, real) if real else np.nan
        label = f'HHI={hhi_labels[idx]} (d={dist:.3f})' if real else f'HHI={hhi_labels[idx]}'
        ax.scatter(d['degree_skew'], d['transitivity'], alpha=0.6, s=30,
                   label=label, color=colors[idx])
    if real:
        _add_ref_point(ax, real.get('degree_skew'), real.get('transitivity'))
    ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Transitivity')
    ax.set_title('Skewness vs Transitivity')
    ax.legend(fontsize=6, ncol=2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Skewness boxplot (skip empty groups) ---
    ax = axes[1, 0]
    skew_pairs = [(df[df['hhi'] == h]['degree_skew'].dropna().values, hhi_labels[i], colors[i])
                  for i, h in enumerate(hhi_vals)]
    skew_pairs = [(d, l, c) for d, l, c in skew_pairs if len(d) > 0]
    if skew_pairs:
        bp = ax.boxplot([d for d, _, _ in skew_pairs],
                        labels=[l for _, l, _ in skew_pairs], patch_artist=True)
        for patch, (_, _, c) in zip(bp['boxes'], skew_pairs):
            patch.set_facecolor(c); patch.set_alpha(0.7)
    _add_ref_line(ax, real and real.get('degree_skew'), axis='y')
    ax.set_xlabel('HHI'); ax.set_ylabel('Degree Skewness'); ax.set_title('Skewness by HHI')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Transitivity boxplot (skip empty groups) ---
    ax = axes[1, 1]
    trans_pairs = [(df[df['hhi'] == h]['transitivity'].dropna().values, hhi_labels[i], colors[i])
                   for i, h in enumerate(hhi_vals)]
    trans_pairs = [(d, l, c) for d, l, c in trans_pairs if len(d) > 0]
    if trans_pairs:
        bp = ax.boxplot([d for d, _, _ in trans_pairs],
                        labels=[l for _, l, _ in trans_pairs], patch_artist=True)
        for patch, (_, _, c) in zip(bp['boxes'], trans_pairs):
            patch.set_facecolor(c); patch.set_alpha(0.7)
    _add_ref_line(ax, real and real.get('transitivity'), axis='y')
    ax.set_xlabel('HHI'); ax.set_ylabel('Transitivity'); ax.set_title('Transitivity by HHI')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    # --- Mean degree vs skewness ---
    ax = axes[1, 2]
    for idx, hhi in enumerate(hhi_vals):
        d = df[df['hhi'] == hhi]
        ax.scatter(d['degree_mean'], d['degree_skew'], alpha=0.6, s=30,
                   label=f'HHI={hhi_labels[idx]}', color=colors[idx])
    if real:
        _add_ref_point(ax, real.get('degree_mean'), real.get('degree_skew'))
    ax.set_xlabel('Mean Degree'); ax.set_ylabel('Degree Skewness')
    ax.set_title('Mean Degree vs Skewness')
    ax.legend(fontsize=6, ncol=2); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_by_hhi.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/comparison_by_hhi.png")
    plt.close()


def plot_individual_hhi(df, real=None, save_dir='network_plots'):
    hhi_dir = Path(save_dir) / 'by_hhi'
    hhi_dir.mkdir(exist_ok=True, parents=True)
    unique_hhi = sorted(df['hhi'].unique())
    print(f"Plotting individual HHI plots for {len(unique_hhi)} groups: {[f'{h:.4f}' for h in unique_hhi]}")

    for hhi_val in unique_hhi:
        file_data = df[df['hhi'] == hhi_val]
        agg_name = file_data['aggregation'].iloc[0] if 'aggregation' in file_data.columns else file_data['filename'].iloc[0]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'HHI={hhi_val:.4f}  ({agg_name})', fontsize=12, fontweight='bold')

        ax = axes[0, 0]
        ax.hist(file_data['degree_skew'], bins=15, color='#4CAF50', alpha=0.7, edgecolor='white')
        ax.axvline(file_data['degree_skew'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Sim mean: {file_data["degree_skew"].mean():.3f}')
        _add_ref_line(ax, real and real.get('degree_skew'), axis='x')
        ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Frequency')
        ax.set_title(f'Mean: {file_data["degree_skew"].mean():.3f}')
        ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax = axes[0, 1]
        ax.hist(file_data['transitivity'], bins=15, color='#42A5F5', alpha=0.7, edgecolor='white')
        ax.axvline(file_data['transitivity'].mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Sim mean: {file_data["transitivity"].mean():.4f}')
        _add_ref_line(ax, real and real.get('transitivity'), axis='x')
        ax.set_xlabel('Transitivity'); ax.set_ylabel('Frequency')
        ax.set_title(f'Mean: {file_data["transitivity"].mean():.4f}')
        ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax = axes[1, 0]
        ax.scatter(file_data['degree_skew'], file_data['transitivity'],
                   alpha=0.6, s=50, color='#FF5722')
        if real:
            _add_ref_point(ax, real.get('degree_skew'), real.get('transitivity'))
            dist = _avg_distance_to_real(file_data, real)
            ax.set_title(f'Skewness vs Transitivity (avg dist={dist:.4f})')
        else:
            ax.set_title('Skewness vs Transitivity')
        ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Transitivity')
        ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        ax = axes[1, 1]
        ax.scatter(file_data['reciprocity'], file_data['transitivity'],
                   alpha=0.6, s=50, color='#F4E04D')
        if real:
            _add_ref_line(ax, real.get('transitivity'), axis='y')
        ax.set_xlabel('Reciprocity'); ax.set_ylabel('Transitivity')
        ax.set_title('Reciprocity vs Transitivity')
        ax.legend(fontsize=8); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        plt.tight_layout()
        safe_name = f'hhi_{hhi_val:.4f}'.replace('.', 'p')
        plt.savefig(hhi_dir / f'{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Saved individual plots to: {hhi_dir}/")


def plot_overall_analysis(df, real=None, save_dir='network_plots'):
    Path(save_dir).mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Overall Network Statistics', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.hist(df['degree_skew'], bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.axvline(df['degree_skew'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Sim mean: {df["degree_skew"].mean():.2f}')
    _add_ref_line(ax, real and real.get('degree_skew'), axis='x')
    ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Frequency')
    ax.set_title('Degree Skewness Distribution'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[0, 1]
    ax.hist(df['transitivity'], bins=30, color='#42A5F5', alpha=0.7, edgecolor='white')
    ax.axvline(df['transitivity'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Sim mean: {df["transitivity"].mean():.4f}')
    _add_ref_line(ax, real and real.get('transitivity'), axis='x')
    ax.set_xlabel('Transitivity'); ax.set_ylabel('Frequency')
    ax.set_title('Transitivity Distribution'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[0, 2]
    ax.scatter(df['degree_skew'], df['transitivity'], alpha=0.5, s=30, color='#FF5722')
    if real:
        _add_ref_point(ax, real.get('degree_skew'), real.get('transitivity'))
    ax.set_xlabel('Degree Skewness'); ax.set_ylabel('Transitivity')
    ax.set_title('Skewness vs Transitivity'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1, 0]
    ax.hist(df['reciprocity'], bins=30, color='#F4E04D', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Reciprocity'); ax.set_ylabel('Frequency')
    ax.set_title('Reciprocity Distribution')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1, 1]
    ax.scatter(df['degree_mean'], df['degree_skew'], alpha=0.5, s=30, color='#9C27B0')
    if real:
        _add_ref_point(ax, real.get('degree_mean'), real.get('degree_skew'))
    ax.set_xlabel('Mean Degree'); ax.set_ylabel('Degree Skewness')
    ax.set_title('Mean Degree vs Skewness'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1, 2]
    ax.scatter(df['nodes'], df['edges'], alpha=0.5, s=30, color='#FF9800')
    if real:
        _add_ref_point(ax, real.get('nodes'), real.get('edges'))
    ax.set_xlabel('Nodes'); ax.set_ylabel('Edges')
    ax.set_title('Network Size'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/overall_analysis.png")
    plt.close()


def summary_statistics(df, real=None):
    print("\n" + "="*60 + "\nOVERALL SUMMARY\n" + "="*60)
    print(f"\nNetworks: {len(df)}")
    print(f"Unique HHI values: {df['hhi'].nunique()}")
    print(f"\nSkewness:\n  Sim mean: {df['degree_skew'].mean():.3f} ± {df['degree_skew'].std():.3f}")
    print(f"  Range: [{df['degree_skew'].min():.3f}, {df['degree_skew'].max():.3f}]")
    if real: print(f"  Real: {real['degree_skew']:.3f}")
    print(f"\nTransitivity:\n  Sim mean: {df['transitivity'].mean():.4f} ± {df['transitivity'].std():.4f}")
    print(f"  Range: [{df['transitivity'].min():.4f}, {df['transitivity'].max():.4f}]")
    if real: print(f"  Real: {real['transitivity']:.4f}")
    print(f"\nReciprocity:\n  Sim mean: {df['reciprocity'].mean():.4f} ± {df['reciprocity'].std():.4f}")
    if real:
        print(f"\nReal network modularity (label prop): {real['modularity']:.4f}")
        print(f"Real network clustering (avg local): {real['transitivity_avg']:.4f}")


# =========================================================================
# NEW ANALYSES
# =========================================================================

METRIC_COLS = ['degree_skew', 'transitivity', 'reciprocity', 'degree_mean']
PARAM_COLS = ['scale', 'comms', 'recip', 'trans', 'pa', 'bridge']


def _normalized_multi_distance(df, real, metrics=METRIC_COLS):
    """Z-score each metric across the full df, then compute Euclidean distance
    from each row to the real network in that normalized space."""
    metrics = [m for m in metrics if m in real and m in df.columns]
    vals = df[metrics].values.astype(float)
    means = np.nanmean(vals, axis=0)
    stds  = np.nanstd(vals, axis=0)
    stds[stds == 0] = 1
    real_z = np.array([(real[m] - means[i]) / stds[i] for i, m in enumerate(metrics)])
    sim_z  = (vals - means) / stds
    return np.sqrt(((sim_z - real_z) ** 2).sum(axis=1))


# --- 1. Normalised multi-metric distance table & bar chart ---------------

def analyze_normalized_distance(df, real, save_dir='network_plots'):
    """Compute z-normalised distance across all metrics, print & plot by HHI."""
    if real is None:
        return
    Path(save_dir).mkdir(exist_ok=True)

    df = df.copy()
    df['norm_dist'] = _normalized_multi_distance(df, real)

    summary = (
        df.groupby(['hhi', 'aggregation'])['norm_dist']
        .agg(['mean', 'std', 'median', 'min'])
        .sort_values('mean')
        .round(4)
    )
    print("\n" + "="*60)
    print("NORMALISED MULTI-METRIC DISTANCE TO REAL (z-scored)")
    print("="*60)
    metrics_used = [m for m in METRIC_COLS if m in real and m in df.columns]
    print(f"Metrics used: {metrics_used}")
    print(summary.to_string())
    summary.to_csv('normalised_distance_by_hhi.csv')
    print("Saved: normalised_distance_by_hhi.csv")

    # --- bar chart ---
    fig, ax = plt.subplots(figsize=(10, 5))
    means = summary['mean'].values
    labels = [f"HHI={h:.4f}\n({a})" for (h, a) in summary.index]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(means)))
    bars = ax.bar(range(len(means)), means, color=colors, edgecolor='white')
    if 'std' in summary.columns:
        ax.errorbar(range(len(means)), means, yerr=summary['std'].values,
                    fmt='none', ecolor='grey', capsize=3)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Mean normalised distance to real')
    ax.set_title('Multi-metric distance to real network (z-normalised)')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/normalised_distance_bar.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/normalised_distance_bar.png")
    plt.close()


# --- 2. Parameter sensitivity heatmaps -----------------------------------

def plot_param_sensitivity(df, real, save_dir='network_plots'):
    """Heatmaps of mean distance-to-real across parameter pairs."""
    if real is None:
        return
    Path(save_dir).mkdir(exist_ok=True)

    df = df.copy()
    df['norm_dist'] = _normalized_multi_distance(df, real)

    param_pairs = [('comms', 'pa'), ('comms', 'trans'), ('pa', 'trans')]
    fig, axes = plt.subplots(1, len(param_pairs), figsize=(6 * len(param_pairs), 5))
    if len(param_pairs) == 1:
        axes = [axes]

    for ax, (p1, p2) in zip(axes, param_pairs):
        pivot = df.groupby([p1, p2])['norm_dist'].mean().reset_index()
        pivot_table = pivot.pivot_table(index=p1, columns=p2, values='norm_dist')
        im = ax.imshow(pivot_table.values, aspect='auto', cmap='RdYlGn_r', origin='lower')
        ax.set_xticks(range(len(pivot_table.columns)))
        ax.set_xticklabels([f'{v:.2f}' for v in pivot_table.columns], fontsize=7, rotation=45)
        ax.set_yticks(range(len(pivot_table.index)))
        ax.set_yticklabels([f'{v:.0f}' if v > 1 else f'{v:.2f}' for v in pivot_table.index], fontsize=7)
        ax.set_xlabel(p2); ax.set_ylabel(p1)
        ax.set_title(f'Dist to real: {p1} × {p2}')
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle('Parameter sensitivity — normalised distance to real', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/param_sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/param_sensitivity.png")
    plt.close()


# --- 3. Radar / spider plots per HHI group --------------------------------

def plot_radar_by_hhi(df, real, save_dir='network_plots'):
    """Radar chart comparing each HHI group to the real network on multiple metrics."""
    if real is None:
        return
    Path(save_dir).mkdir(exist_ok=True)
    metrics = [m for m in METRIC_COLS if m in real and m in df.columns]
    n_metrics = len(metrics)

    # Normalise every metric to [0, 1] using global min/max across sim + real
    mins = {m: min(df[m].min(), real[m]) for m in metrics}
    maxs = {m: max(df[m].max(), real[m]) for m in metrics}

    def _norm(val, m):
        r = maxs[m] - mins[m]
        return (val - mins[m]) / r if r > 0 else 0.5

    real_normed = [_norm(real[m], m) for m in metrics]

    hhi_vals = sorted(df['hhi'].unique())
    n_groups = len(hhi_vals)
    cols = min(4, n_groups)
    rows = int(np.ceil(n_groups / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows),
                             subplot_kw=dict(polar=True))
    axes = np.atleast_2d(axes)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon
    real_plot = real_normed + real_normed[:1]

    for idx, hhi_val in enumerate(hhi_vals):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        group = df[df['hhi'] == hhi_val]
        agg = group['aggregation'].iloc[0]
        group_means = [_norm(group[m].mean(), m) for m in metrics]
        group_plot = group_means + group_means[:1]

        ax.plot(angles, real_plot, 'k-', linewidth=2, label='Real')
        ax.fill(angles, real_plot, alpha=0.1, color='black')
        ax.plot(angles, group_plot, '-', linewidth=2, color='#FF5722', label=f'HHI={hhi_val:.3f}')
        ax.fill(angles, group_plot, alpha=0.15, color='#FF5722')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=7)
        ax.set_title(f'{agg}\nHHI={hhi_val:.4f}', fontsize=9, fontweight='bold', pad=15)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc='upper right')

    # hide unused axes
    for idx in range(n_groups, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    plt.suptitle('Radar: simulated (orange) vs real (black) — normalised [0-1]',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/radar_by_hhi.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/radar_by_hhi.png")
    plt.close()


# --- 4. HHI vs distance-to-real correlation -------------------------------

def plot_hhi_vs_distance(df, real, save_dir='network_plots'):
    """Scatter of HHI vs avg normalised distance, with Spearman correlation."""
    if real is None:
        return
    Path(save_dir).mkdir(exist_ok=True)

    df = df.copy()
    df['norm_dist'] = _normalized_multi_distance(df, real)

    group = df.groupby('hhi').agg(
        mean_dist=('norm_dist', 'mean'),
        std_dist=('norm_dist', 'std'),
        aggregation=('aggregation', 'first'),
    ).reset_index()

    rho, pval = spearmanr(group['hhi'], group['mean_dist'])
    print(f"\nHHI vs distance — Spearman rho={rho:.3f}, p={pval:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(group['hhi'], group['mean_dist'], yerr=group['std_dist'],
                fmt='o', capsize=4, color='#1976D2', ecolor='#90CAF9', markersize=8)
    for _, row in group.iterrows():
        ax.annotate(row['aggregation'], (row['hhi'], row['mean_dist']),
                    fontsize=6, ha='left', va='bottom',
                    xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('HHI (group concentration)')
    ax.set_ylabel('Mean normalised distance to real')
    ax.set_title(f'HHI vs distance to real  (Spearman ρ={rho:.3f}, p={pval:.3f})')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/hhi_vs_distance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/hhi_vs_distance.png")
    plt.close()


# --- 5. Best parameters per aggregation level -----------------------------

def find_top_k_params_per_aggregation(df, real, k=5, metrics=METRIC_COLS,
                                      out_csv='top_params_per_aggregation.csv'):
    """For every aggregation level, return the top-k parameter combos with
    the smallest normalised multi-metric distance to the real network.

    Normalisation is computed globally so distances are comparable across
    aggregation levels. If the same (aggregation, params) appears multiple
    times (e.g. across seeds), metrics and distance are averaged so each
    combo yields one row.
    """
    if real is None:
        return None

    df = df.copy()
    df['norm_dist'] = _normalized_multi_distance(df, real, metrics=metrics)

    # Average duplicates (e.g. seeds) per (aggregation, params) combo.
    metric_cols_present = [m for m in metrics if m in df.columns]
    param_cols_present = [p for p in PARAM_COLS if p in df.columns]
    agg_df = (
        df.groupby(['aggregation', 'hhi'] + param_cols_present, dropna=False)[
            metric_cols_present + ['norm_dist']
        ]
        .mean()
        .reset_index()
    )

    # Rank within each aggregation, keep top-k.
    agg_df['rank'] = agg_df.groupby('aggregation')['norm_dist'].rank(method='first')
    top = (
        agg_df[agg_df['rank'] <= k]
        .sort_values(['aggregation', 'rank'])
        .reset_index(drop=True)
    )
    top['rank'] = top['rank'].astype(int)

    # --- pretty printout ---
    print("\n" + "=" * 78)
    print(f"TOP-{k} PARAMETER COMBINATIONS PER AGGREGATION LEVEL")
    print("(ranked by normalised multi-metric distance to the real network)")
    print("=" * 78)
    print("\nReal network reference values:")
    for m in metric_cols_present:
        print(f"  {m:15s} = {real[m]:.4f}")

    display_cols = ['rank'] + param_cols_present + metric_cols_present + ['norm_dist']
    for agg, block in top.groupby('aggregation'):
        hhi = block['hhi'].iloc[0]
        print("\n" + "-" * 78)
        print(f"Aggregation: {agg}   (HHI = {hhi:.4f})")
        print("-" * 78)
        print(block[display_cols].to_string(index=False))

    top.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    return top


def main(base_dir='enriched', pkl_path='original.pkl',
         mapping_csv='group_charcteristic_mapping.csv'):
    print("="*60 + "\nNETWORK STATISTICS ANALYSIS\n" + "="*60)

    real = compute_real_network_stats(pkl_path)

    # Load & filter simulated networks
    df = load_all_json_files(base_dir)
    df = df[df['nodes'] > 100000]
    df = parse_params(df)
    # df = filter_param_combinations(df)

    # Map filenames → HHI
    mapping = load_hhi_mapping(mapping_csv)
    df = merge_hhi(df, mapping)

    summary_statistics(df, real=real)
    analyze_by_hhi(df, real=real)

    plot_overall_analysis(df, real=real)
    plot_distributions_by_hhi(df, real=real)
    plot_individual_hhi(df, real=real)

    # New analyses
    analyze_normalized_distance(df, real)
    plot_param_sensitivity(df, real)
    plot_radar_by_hhi(df, real)
    plot_hhi_vs_distance(df, real)

    # Top-k best parameters per aggregation level
    find_top_k_params_per_aggregation(df, real, k=5)

    df.to_csv('network_statistics_full.csv', index=False)
    print(f"\nFull data saved to: network_statistics_full.csv")
    return df


if __name__ == "__main__":
    df = main('enriched3', 'original.pkl', 'group_charcteristic_mapping.csv')