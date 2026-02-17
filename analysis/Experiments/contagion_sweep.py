"""
Parameter sweeps for contagion experiments (Centola-style).

Experiments:
  1. Uncontested (absolute) threshold sweep — τ = 1, 2, 3, ...
  2. Contested (fractional) threshold sweep — τ = 1/z, 2/z, 3/z, ...
  3. Hybrid contagion — base τ=2, varying % vulnerable nodes (τ=1)
  4. Combined heatmap — threshold × seed fraction

Seeding follows Centola:
  - τ = 1 (simple): single seed node
  - τ ≥ 2 (complex): focal node + all its neighbors
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from scipy import stats as sp_stats
from itertools import combinations
import scipy

from contagion_experiment import (
    ContagionSimulator, load_networks, assign_colors,
    print_network_properties,
)

sns.set_style("whitegrid")
np.random.seed(42)


# ---------------------------------------------------------------------------
# Sweep functions
# ---------------------------------------------------------------------------

def sweep_uncontested(networks, thresholds=None, n_simulations=20, max_steps=50):
    """
    Absolute threshold sweep (uncontested contagion).
    τ=1 uses single seed, τ≥2 uses focal+neighbors.
    """
    if thresholds is None:
        thresholds = np.array([1, 2, 3, 4, 5])
    results = {}

    for name, G in networks.items():
        n = len(G)
        sim = ContagionSimulator(G, name)
        finals = np.zeros((len(thresholds), n_simulations))

        for i, tau in enumerate(thresholds):
            seeding = 'random' if tau <= 1 else 'focal_neighbors'
            initial = 1 if tau <= 1 else 1
            ts_list = sim.complex_contagion(
                threshold=tau, threshold_type='absolute',
                initial_infected=initial, seeding=seeding,
                max_steps=max_steps, n_simulations=n_simulations)
            finals[i] = np.array([ts[-1] / n * 100 for ts in ts_list])

        results[name] = finals
        print(f"  Uncontested sweep done: {name}")

    return results


def sweep_contested(networks, fractions=None, n_simulations=20, max_steps=50):
    """
    Fractional threshold sweep (contested contagion).
    τ = fraction of neighbors required.
    """
    if fractions is None:
        fractions = np.linspace(0.05, 0.50, 15)
    results = {}

    for name, G in networks.items():
        n = len(G)
        sim = ContagionSimulator(G, name)
        finals = np.zeros((len(fractions), n_simulations))

        for i, tau in enumerate(fractions):
            ts_list = sim.complex_contagion(
                threshold=tau, threshold_type='fractional',
                seeding='focal_neighbors',
                max_steps=max_steps, n_simulations=n_simulations)
            finals[i] = np.array([ts[-1] / n * 100 for ts in ts_list])

        results[name] = finals
        print(f"  Contested sweep done: {name}")

    return results


def sweep_hybrid(networks, vulnerable_fractions=None,
                 base_threshold=2, vulnerable_threshold=1,
                 n_simulations=20, max_steps=50):
    """
    Hybrid contagion: base τ=2, varying fraction of vulnerable nodes (τ=1).
    Replicates Centola Figure 8 style.
    """
    if vulnerable_fractions is None:
        vulnerable_fractions = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    results = {}

    for name, G in networks.items():
        n = len(G)
        sim = ContagionSimulator(G, name)
        finals = np.zeros((len(vulnerable_fractions), n_simulations))

        for i, vf in enumerate(vulnerable_fractions):
            ts_list = sim.hybrid_contagion(
                base_threshold=base_threshold,
                vulnerable_threshold=vulnerable_threshold,
                vulnerable_fraction=vf,
                threshold_type='absolute',
                seeding='focal_neighbors',
                max_steps=max_steps, n_simulations=n_simulations)
            finals[i] = np.array([ts[-1] / n * 100 for ts in ts_list])

        results[name] = finals
        print(f"  Hybrid sweep done: {name}")

    return results


def sweep_combined(networks, thresholds=np.linspace(0.05, 0.5, 15),
                   seed_fractions=np.linspace(0.01, 0.20, 15),
                   n_simulations=10, max_steps=50):
    """
    2D sweep: fractional threshold × seed size. One heatmap per network.
    """
    results = {}

    for name, G in networks.items():
        n = len(G)
        sim = ContagionSimulator(G, name)
        grid = np.zeros((len(thresholds), len(seed_fractions)))

        for i, thresh in enumerate(thresholds):
            for j, frac in enumerate(seed_fractions):
                initial = max(1, int(frac * n))
                ts_list = sim.complex_contagion(
                    threshold=thresh, threshold_type='fractional',
                    initial_infected=initial, seeding='random',
                    max_steps=max_steps, n_simulations=n_simulations)
                grid[i, j] = np.mean([ts[-1] / n * 100 for ts in ts_list])

        results[name] = grid
        print(f"  Combined sweep done: {name}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(data), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def plot_uncontested(thresholds, results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = assign_colors(list(results.keys()))

    for name, finals in results.items():
        mean = finals.mean(axis=1)
        # print(finals)
        std = finals.std(axis=1)
        ax.plot(thresholds, mean, 'o-', label=name, color=colors[name], linewidth=2)
        # ax.fill_between(thresholds, mean - std, mean + std,
        #                 alpha=0.15, color=colors[name])

    ax.set_xlabel('Absolute Threshold (τ)', fontsize=12)
    ax.set_ylabel('Final Cascade Size (%)', fontsize=12)
    ax.set_title('Uncontested Contagion: Absolute Threshold Sweep',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(thresholds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_contested(fractions, results):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = assign_colors(list(results.keys()))

    for name, finals in results.items():
        mean = finals.mean(axis=1)
        std = finals.std(axis=1)
        ax.plot(fractions, mean, '-', label=name, color=colors[name], linewidth=2)
        # ax.fill_between(fractions, mean - std, mean + std,
        #                 alpha=0.15, color=colors[name])

    ax.set_xlabel('Fractional Threshold (τ)', fontsize=12)
    ax.set_ylabel('Final Cascade Size (%)', fontsize=12)
    ax.set_title('Contested Contagion: Fractional Threshold Sweep\n'
                 '(seeding: focal node + neighbors)',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_hybrid(vulnerable_fractions, results, base_threshold, vulnerable_threshold):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = assign_colors(list(results.keys()))

    for name, finals in results.items():
        mean = finals.mean(axis=1)
        std = finals.std(axis=1)
        ax.plot(vulnerable_fractions * 100, mean, 'o-', label=name,
                color=colors[name], linewidth=2)
        ax.fill_between(vulnerable_fractions * 100, mean - std, mean + std,
                        alpha=0.15, color=colors[name])

    ax.set_xlabel('Vulnerable Nodes (%)', fontsize=12)
    ax.set_ylabel('Final Cascade Size (%)', fontsize=12)
    ax.set_title(f'Hybrid Contagion: base τ={base_threshold}, '
                 f'vulnerable τ={vulnerable_threshold}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_combined(thresholds, seed_fractions, results):
    n_nets = len(results)
    ncols = min(n_nets, 3)
    nrows = math.ceil(n_nets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    fig.suptitle('Contested Contagion: Threshold × Seed Size',
                 fontsize=16, fontweight='bold')

    axes = np.array(axes).flatten()

    for idx, (name, grid) in enumerate(results.items()):
        ax = axes[idx]
        im = ax.imshow(grid, origin='lower', aspect='auto',
                       extent=[seed_fractions[0]*100, seed_fractions[-1]*100,
                               thresholds[0], thresholds[-1]],
                       vmin=0, vmax=100, cmap='YlOrRd')
        ax.set_xlabel('Initial Infected (%)')
        ax.set_ylabel('Fractional Threshold (τ)')
        ax.set_title(name, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Final Cascade %')

    for idx in range(n_nets, len(axes)):
        axes[idx].axis('off')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def _find_critical_point(param_values, mean_cascade, target=50.0):
    """
    Find the parameter value where the mean cascade crosses `target`%
    via linear interpolation. Returns NaN if no crossing exists.
    """
    above = mean_cascade >= target
    # Check if there's a crossing at all
    if above.all() or (~above).all():
        return np.nan

    # Find first crossing (works for both increasing and decreasing curves)
    # For decreasing: find last index where above is True
    # For increasing: find first index where above is True
    if mean_cascade[0] > mean_cascade[-1]:  # decreasing
        idx = np.where(above)[0][-1]  # last point above target
    else:  # increasing
        idx = np.where(above)[0][0] - 1  # point just before first above
        idx = max(idx, 0)

    if idx >= len(param_values) - 1:
        return param_values[-1]

    # Linear interpolation between idx and idx+1
    x0, x1 = param_values[idx], param_values[idx + 1]
    y0, y1 = mean_cascade[idx], mean_cascade[idx + 1]
    if abs(y1 - y0) < 1e-10:
        return (x0 + x1) / 2
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


def analyze_sweep(param_values, results, param_name='threshold', cascade_target=50.0):
    """
    Compute summary statistics for a parameter sweep.

    For each network returns:
      - critical_point: parameter value where cascade crosses target%
      - auc: area under the cascade-vs-parameter curve
      - steepness: max |d(cascade)/d(param)| — sharpness of the transition
      - cascade_prob: at each param value, fraction of sims with cascade > target%

    Args:
        param_values: 1D array of swept parameter values
        results: {name: array(n_params, n_sims)} from a sweep function
        param_name: label for the parameter (for printing)
        cascade_target: % threshold for "global cascade" (default 50%)

    Returns:
        dict of {name: {stat_name: value}}
    """
    stats = {}

    for name, finals in results.items():
        mean = finals.mean(axis=1)

        # Critical point: where does the mean cascade cross the target?
        tc = _find_critical_point(param_values, mean, target=cascade_target)

        # AUC (trapezoidal integration, normalized to [0,1] range of params)
        auc = np.trapezoid(mean, param_values)

        # Steepness: max absolute gradient of the mean curve
        gradient = np.gradient(mean, param_values)
        steepness = np.max(np.abs(gradient))

        # Cascade probability: fraction of sims exceeding the target at each point
        cascade_prob = (finals > cascade_target).mean(axis=1)

        stats[name] = {
            'critical_point': tc,
            'auc': auc,
            'steepness': steepness,
            'cascade_prob': cascade_prob,
            'mean': mean,
            'std': finals.std(axis=1),
        }

    return stats


def pairwise_tests(param_values, results):
    """
    Run Mann-Whitney U tests between all network pairs at each parameter value.

    Returns:
        List of dicts with keys: param_idx, param_value, net_a, net_b,
        U_statistic, p_value, effect_size (rank-biserial r)
    """
    names = list(results.keys())
    records = []

    for i, val in enumerate(param_values):
        for na, nb in combinations(names, 2):
            a = results[na][i]
            b = results[nb][i]

            # Skip if both distributions are identical (all same value)
            if np.std(a) == 0 and np.std(b) == 0 and np.mean(a) == np.mean(b):
                continue

            U, p = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
            n1, n2 = len(a), len(b)
            # Rank-biserial correlation as effect size
            r = 1 - (2 * U) / (n1 * n2)

            records.append({
                'param_value': val,
                'net_a': na, 'net_b': nb,
                'U': U, 'p': p, 'effect_size': r,
            })

    return records


def print_sweep_stats(stats, param_name='threshold'):
    """Print a formatted summary table of sweep statistics."""
    print(f"\n{'='*75}")
    print(f"SWEEP STATISTICS ({param_name})")
    print(f"{'='*75}")
    print(f"  {'Network':<30} {'τ_c (50%)':>10} {'AUC':>10} {'Steepness':>12}")
    print(f"  {'-'*65}")

    for name, s in stats.items():
        tc = f"{s['critical_point']:.3f}" if not np.isnan(s['critical_point']) else "N/A"
        print(f"  {name:<30} {tc:>10} {s['auc']:>10.1f} {s['steepness']:>12.1f}")

    print(f"{'='*75}")


def print_pairwise_summary(records, alpha=0.05):
    """Print significant pairwise differences."""
    sig = [r for r in records if r['p'] < alpha]
    if not sig:
        print("\n  No significant pairwise differences found.")
        return

    # Group by network pair, count how many param values are significant
    from collections import defaultdict
    pair_counts = defaultdict(list)
    for r in sig:
        pair_counts[(r['net_a'], r['net_b'])].append(r)

    print(f"\n  Significant pairwise differences (p < {alpha}):")
    print(f"  {'Pair':<50} {'# sig. points':>14} {'Mean |effect|':>14}")
    print(f"  {'-'*80}")

    for (na, nb), recs in sorted(pair_counts.items(), key=lambda x: -len(x[1])):
        mean_effect = np.mean([abs(r['effect_size']) for r in recs])
        print(f"  {na} vs {nb:<30} {len(recs):>10} {mean_effect:>14.3f}")


def plot_analysis(param_values, stats, param_name='threshold'):
    """
    Summary figure with 4 panels:
      1. Bar chart of critical thresholds
      2. Bar chart of AUC
      3. Cascade probability curves
      4. Steepness comparison
    """
    names = list(stats.keys())
    colors = assign_colors(names)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Sweep Analysis: {param_name}', fontsize=16, fontweight='bold')

    # 1. Critical thresholds
    ax = axes[0, 0]
    tc_vals = [stats[n]['critical_point'] for n in names]
    bar_colors = [colors[n] for n in names]
    bars = ax.bar(range(len(names)), tc_vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel(f'Critical {param_name} (τ_c)')
    ax.set_title('Critical Point (50% cascade)', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate NaN bars
    for i, v in enumerate(tc_vals):
        if np.isnan(v):
            ax.text(i, 0.01, 'N/A', ha='center', va='bottom', fontweight='bold')

    # 2. AUC
    ax = axes[0, 1]
    auc_vals = [stats[n]['auc'] for n in names]
    ax.bar(range(len(names)), auc_vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title('Area Under Cascade Curve', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Cascade probability curves
    ax = axes[1, 0]
    for n in names:
        ax.plot(param_values, stats[n]['cascade_prob'],
                'o-', label=n, color=colors[n], linewidth=2)
    ax.set_xlabel(param_name)
    ax.set_ylabel('P(cascade > 50%)')
    ax.set_title('Cascade Probability', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # 4. Steepness
    ax = axes[1, 1]
    steep_vals = [stats[n]['steepness'] for n in names]
    ax.bar(range(len(names)), steep_vals, color=bar_colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Max |d(cascade)/d(param)|')
    ax.set_title('Transition Steepness', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_sweep_on_folder(network_folder, n_simulations=20):
    """Run the full parameter sweep on a single folder containing .pkl networks."""
    print(f"\nLoading networks from: {network_folder}\n")

    # Output directory
    out_dir = Path(network_folder) / 'diffusion_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    networks = load_networks(network_folder, add_random=True)

    if not networks:
        print(f"  No networks found in {network_folder}, skipping.")
        return None

    print_network_properties(networks)

    # --- 1. Uncontested (absolute) threshold sweep ---
    print("\n[1/4] Uncontested threshold sweep...")
    abs_thresholds = np.array([1, 2, 3, 4, 5])
    uncontested_results = sweep_uncontested(
        networks, thresholds=abs_thresholds, n_simulations=n_simulations)

    fig1 = plot_uncontested(abs_thresholds, uncontested_results)
    fig1.savefig(out_dir / 'sweep_uncontested.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'sweep_uncontested.png'}")

    unc_stats = analyze_sweep(abs_thresholds, uncontested_results, 'abs. threshold')
    print_sweep_stats(unc_stats, 'abs. threshold')
    unc_pw = pairwise_tests(abs_thresholds, uncontested_results)
    print_pairwise_summary(unc_pw)

    fig1a = plot_analysis(abs_thresholds, unc_stats, 'Absolute Threshold')
    fig1a.savefig(out_dir / 'analysis_uncontested.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'analysis_uncontested.png'}")

    # --- 2. Contested (fractional) threshold sweep ---
    print("\n[2/4] Contested threshold sweep...")
    frac_thresholds = np.linspace(0.05, 0.50, 15)
    contested_results = sweep_contested(
        networks, fractions=frac_thresholds, n_simulations=n_simulations)

    fig2 = plot_contested(frac_thresholds, contested_results)
    fig2.savefig(out_dir / 'sweep_contested.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'sweep_contested.png'}")

    con_stats = analyze_sweep(frac_thresholds, contested_results, 'frac. threshold')
    print_sweep_stats(con_stats, 'frac. threshold')
    con_pw = pairwise_tests(frac_thresholds, contested_results)
    print_pairwise_summary(con_pw)

    fig2a = plot_analysis(frac_thresholds, con_stats, 'Fractional Threshold')
    fig2a.savefig(out_dir / 'analysis_contested.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'analysis_contested.png'}")

    # --- 3. Hybrid contagion sweep ---
    print("\n[3/4] Hybrid contagion sweep...")
    vuln_fracs = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    base_t, vuln_t = 2, 1
    hybrid_results = sweep_hybrid(
        networks, vulnerable_fractions=vuln_fracs,
        base_threshold=base_t, vulnerable_threshold=vuln_t,
        n_simulations=n_simulations)

    fig3 = plot_hybrid(vuln_fracs, hybrid_results, base_t, vuln_t)
    fig3.savefig(out_dir / 'sweep_hybrid.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'sweep_hybrid.png'}")

    hyb_stats = analyze_sweep(vuln_fracs, hybrid_results, 'vulnerable %')
    print_sweep_stats(hyb_stats, 'vulnerable %')
    hyb_pw = pairwise_tests(vuln_fracs, hybrid_results)
    print_pairwise_summary(hyb_pw)

    fig3a = plot_analysis(vuln_fracs, hyb_stats, 'Vulnerable Fraction')
    fig3a.savefig(out_dir / 'analysis_hybrid.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'analysis_hybrid.png'}")

    # --- 4. Combined heatmap ---
    print("\n[4/4] Combined threshold × seed size sweep...")
    combined_thresholds = np.linspace(0.05, 0.50, 15)
    combined_seeds = np.linspace(0.01, 0.20, 15)
    combined_results = sweep_combined(
        networks, thresholds=combined_thresholds,
        seed_fractions=combined_seeds,
        n_simulations=max(n_simulations // 2, 5))

    fig4 = plot_combined(combined_thresholds, combined_seeds, combined_results)
    fig4.savefig(out_dir / 'sweep_combined.png', dpi=300, bbox_inches='tight')
    print(f"  Saved: {out_dir / 'sweep_combined.png'}")

    return {
        'uncontested': (abs_thresholds, uncontested_results, unc_stats),
        'contested': (frac_thresholds, contested_results, con_stats),
        'hybrid': (vuln_fracs, hybrid_results, hyb_stats),
        'combined': (combined_thresholds, combined_seeds, combined_results),
    }


def main(network_folder='Data/networks/scale=0.01_comms=1_recip=1_trans=0_pa=0',
         n_simulations=20):
    """
    Run parameter sweeps on networks in a folder.

    If the folder contains .pkl files directly, runs the sweep once.
    If the folder contains subfolders (multiplex structure), runs the
    sweep separately for each characteristic group.
    """
    print("\n" + "="*70)
    print("CONTAGION PARAMETER SWEEP (Centola-style)")
    print("="*70)

    folder = Path(network_folder)

    # Check if this folder has .pkl files directly or subfolders
    pkl_files = list(folder.glob('*.pkl'))
    subfolders = sorted([d for d in folder.iterdir() if d.is_dir()
                         and d.name not in ('diffusion_analysis', 'node_distribution')])

    if pkl_files:
        # Direct pkl files — single sweep
        return _run_sweep_on_folder(network_folder, n_simulations)
    elif subfolders:
        # Multiplex structure — iterate over characteristic subfolders
        print(f"\nFound {len(subfolders)} characteristic groups in: {network_folder}")
        all_results = {}
        for subfolder in subfolders:
            print(f"\n{'='*70}")
            print(f"CHARACTERISTIC GROUP: {subfolder.name}")
            print(f"{'='*70}")
            result = _run_sweep_on_folder(subfolder, n_simulations)
            if result is not None:
                all_results[subfolder.name] = result
        return all_results
    else:
        print(f"No .pkl files or subfolders found in {network_folder}")
        return None


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    kwargs = {'network_folder': folder} if folder else {}
    all_results = main(**kwargs)
    plt.show()
