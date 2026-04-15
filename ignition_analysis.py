"""
Ignition / cascade-size decomposition for threshold-diffusion simulation outputs.

For each (aggregation, threshold) cell:
  - Parse individual run outcomes from `final_values`.
  - Classify each run as IGNITED vs FAILED using an adaptive cutoff.
  - Report P(ignite), E[size | ignite], E[size | fail], and bimodality diagnostics.
  - Compare against the reported mean to show how misleading it is.

Input : CSV with columns
        [n_communities, pref_attachment, folder, network, threshold_idx,
         threshold_value, mean_final_adoption, variance_final_adoption,
         final_values, ratio]
        `final_values` is a quoted comma-separated string of integers.

Usage :  python ignition_decomposition.py --input results.csv --outdir ./out
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- parsing ----------------------------------------------------------

def parse_final_values(s):
    """Parse the quoted comma-separated `final_values` field into a numpy array."""
    if pd.isna(s) or s == "":
        return np.array([])
    return np.fromstring(str(s).strip().strip('"'), sep=",", dtype=np.int64)


def load(csv_path):
    df = pd.read_csv(csv_path)
    df["runs"] = df["final_values"].apply(parse_final_values)
    df["n_runs"] = df["runs"].apply(len)
    # True network size: max observed adoption across ALL rows of that network
    # (low-threshold rows reveal graph size; high-threshold ones don't).
    per_network_max = (df.groupby("network")["runs"]
                         .apply(lambda s: max((a.max() for a in s if len(a)), default=np.nan)))
    df["network_size_est"] = df["network"].map(per_network_max).astype(float)
    return df


# ---------- ignition cutoff --------------------------------------------------

def ignition_cutoff(runs, network_size, rel=0.01):
    """Run is 'ignited' if adoption >= rel * network_size."""
    if len(runs) == 0:
        return np.nan
    net = network_size if network_size and network_size > 0 else runs.max()
    return rel * net


# ---------- per-cell decomposition ------------------------------------------

def decompose_row(runs, network_size, rel=0.01):
    if len(runs) == 0:
        return dict(n_runs=0)
    n = len(runs)
    net = network_size if network_size and network_size > 0 else runs.max()
    rel_cut = rel * net

    # All-saturated: even the smallest run is above the relative threshold.
    if runs.min() >= rel_cut:
        ignited = np.ones(n, dtype=bool)
        cut = rel_cut
    # All-dead: even the largest run is below the relative threshold.
    elif runs.max() < rel_cut:
        ignited = np.zeros(n, dtype=bool)
        cut = rel_cut
    else:
        cut = ignition_cutoff(runs, net, rel=rel)
        ignited = runs >= cut
    n_ig = int(ignited.sum())
    out = {
        "n_runs": n,
        "cutoff": float(cut),
        "mean_raw": float(runs.mean()),
        "var_raw": float(runs.var(ddof=0)),
        "p_ignite": n_ig / n,
        "mean_ignited": float(runs[ignited].mean()) if n_ig else np.nan,
        "std_ignited": float(runs[ignited].std(ddof=0)) if n_ig > 1 else np.nan,
        "mean_failed": float(runs[~ignited].mean()) if n - n_ig else np.nan,
        "std_failed": float(runs[~ignited].std(ddof=0)) if n - n_ig > 1 else np.nan,
        # bimodality coefficient (Sarle): b in (0,1]; >5/9 ~ bimodal
        "bimodality": _sarle_bimodality(runs),
        "frac_at_seed": float((runs <= np.quantile(runs, 0.25)).mean()),
    }
    # decomposition identity check: mean = p*E[ig] + (1-p)*E[fail]
    reconstructed = out["p_ignite"] * (out["mean_ignited"] if n_ig else 0) + \
                    (1 - out["p_ignite"]) * (out["mean_failed"] if n - n_ig else 0)
    out["mean_reconstruction_err"] = float(abs(reconstructed - out["mean_raw"]))
    return out


def _sarle_bimodality(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or x.std() == 0:
        return np.nan
    m3 = ((x - x.mean()) ** 3).mean()
    m2 = ((x - x.mean()) ** 2).mean()
    m4 = ((x - x.mean()) ** 4).mean()
    skew = m3 / m2 ** 1.5
    kurt = m4 / m2 ** 2 - 3.0
    return (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


# ---------- aggregation ------------------------------------------------------

def decompose_all(df, rel=0.01, pool=True):
    """
    If pool=True: pool all runs sharing (network, threshold_value) across
    folders / pref_attachment / n_communities before decomposing.
    If pool=False: one decomposition per CSV row (replicates stay separate).
    """
    if pool:
        grouped = (df.groupby(["network", "threshold_value"], as_index=False)
                     .agg(runs=("runs", lambda s: np.concatenate(list(s))),
                          network_size_est=("network_size_est", "max"),
                          n_replicates=("runs", "size")))
        rows = []
        for _, r in grouped.iterrows():
            d = decompose_row(r["runs"], r["network_size_est"], rel=rel)
            d.update({
                "network": r["network"],
                "threshold_value": r["threshold_value"],
                "n_replicates": r["n_replicates"],
            })
            rows.append(d)
        return pd.DataFrame(rows)

    rows = []
    for _, r in df.iterrows():
        d = decompose_row(r["runs"], r.get("network_size_est"), rel=rel)
        d.update({
            "folder": r["folder"],
            "network": r["network"],
            "threshold_idx": r["threshold_idx"],
            "threshold_value": r["threshold_value"],
            "n_communities": r.get("n_communities"),
            "pref_attachment": r.get("pref_attachment"),
        })
        rows.append(d)
    return pd.DataFrame(rows)


# ---------- plots ------------------------------------------------------------

def plot_per_network(dec, outdir):
    os.makedirs(outdir, exist_ok=True)
    networks = sorted(dec["network"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    print(dec)
    dec = dec[dec["threshold_value"] > 0.05]
    # 1) P(ignite) vs threshold, coloured by network
    ax = axes[0, 0]
    for net in networks:
        sub = dec[dec["network"] == net].sort_values("threshold_value")
        ax.plot(sub["threshold_value"], sub["p_ignite"], "o-",
                label=net[:100], alpha=0.8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("P(ignite)")
    ax.set_title("Ignition probability vs threshold")
    ax.legend(fontsize=6, loc="best")
    ax.grid(alpha=0.3)

    # 2) E[size | ignite] vs threshold
    ax = axes[0, 1]
    for net in networks:
        sub = dec[dec["network"] == net].sort_values("threshold_value")
        ax.plot(sub["threshold_value"], sub["mean_ignited"], "s-",
                label=net[:24], alpha=0.8)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("E[size | ignited]")
    ax.set_yscale("log")
    ax.set_title("Conditional cascade size (log scale)")
    ax.grid(alpha=0.3, which="both")

    # 3) raw mean vs decomposed mean — shows how misleading the raw mean is
    ax = axes[1, 0]
    ax.scatter(dec["mean_raw"], dec["p_ignite"] * dec["mean_ignited"].fillna(0),
               c=dec["threshold_value"], cmap="viridis", alpha=0.7)
    lim = [dec["mean_raw"].min(), dec["mean_raw"].max()]
    ax.plot(lim, lim, "k--", alpha=0.4)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("Raw mean")
    ax.set_ylabel("P(ignite) · E[size | ignited]")
    ax.set_title("Decomposition sanity check")
    ax.grid(alpha=0.3)

    # 4) bimodality coefficient vs threshold
    ax = axes[1, 1]
    for net in networks:
        sub = dec[dec["network"] == net].sort_values("threshold_value")
        ax.plot(sub["threshold_value"], sub["bimodality"], "^-",
                label=net[:24], alpha=0.8)
    ax.axhline(5 / 9, color="red", ls="--", alpha=0.5, label="bimodal threshold (5/9)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Sarle bimodality coefficient")
    ax.set_title("Bimodality vs threshold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "decomposition_overview.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_run_histograms(df, outdir, max_cells=24, pool=True):
    """Plot raw run-level histograms for the most bimodal cells.
    If pool=True, pool runs across replicates per (network, threshold) so
    the top-N are distinct (network, threshold) pairs, not duplicates.
    """
    os.makedirs(outdir, exist_ok=True)
    if pool:
        df = (df.groupby(["network", "threshold_value"], as_index=False)
                .agg(runs=("runs", lambda s: np.concatenate(list(s)))))
    else:
        df = df.copy()
    df["cv"] = df["runs"].apply(
        lambda a: a.std() / a.mean() if len(a) and a.mean() > 0 else 0
    )
    top = df.sort_values("cv", ascending=False).head(max_cells)

    n = len(top)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes).ravel()
    for i, (_, r) in enumerate(top.iterrows()):
        ax = axes[i]
        ax.hist(r["runs"], bins=40, color="steelblue", alpha=0.7)
        ax.set_yscale("log")
        ax.set_title(f"{r['network'][:100]}\nthr={r['threshold_value']}, CV={r['cv']:.1f}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    path = os.path.join(outdir, "bimodal_histograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- main -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with final_values column")
    ap.add_argument("--outdir", default="./decomposition_out")
    ap.add_argument("--rel", type=float, default=0.01,
                    help="Ignition cutoff as fraction of network size")
    ap.add_argument("--no-pool", action="store_true",
                    help="Keep replicates separate instead of pooling by (network, threshold)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load(args.input)
    df =df[df.network != "etngrp_geslacht_lft_oplniv_burgerlijke_staat"]
    print(f"Loaded {len(df)} cells, "
          f"{df['n_runs'].sum()} total runs, "
          f"{df['network'].nunique()} networks, "
          f"{df['threshold_value'].nunique()} thresholds.")

    pool = not args.no_pool
    dec = decompose_all(df, rel=args.rel, pool=pool)
    dec_path = os.path.join(args.outdir, "decomposition.csv")
    dec.to_csv(dec_path, index=False)
    print(f"Wrote per-cell decomposition -> {dec_path}")

    # summary table: per-network, per-threshold key numbers
    summary = dec[[
        "network", "threshold_value", "n_runs", "p_ignite",
        "mean_ignited", "mean_failed", "mean_raw", "bimodality"
    ]].sort_values(["threshold_value", "network"])
    print("\n=== Decomposition summary ===")
    print(summary.to_string(index=False,
                            float_format=lambda x: f"{x:.3g}" if pd.notna(x) else "nan"))

    p1 = plot_per_network(dec, args.outdir)
    p2 = plot_run_histograms(df, args.outdir, pool=pool)
    print(f"\nPlots: {p1}\n       {p2}")

    # Headline numbers for the writeup
    med = dec.groupby("threshold_value").agg(
        p_ignite_median=("p_ignite", "median"),
        size_ignited_median=("mean_ignited", "median"),
        bimodality_median=("bimodality", "median"),
    )
    print("\n=== Per-threshold medians across networks ===")
    print(med.to_string())


if __name__ == "__main__":
    main()