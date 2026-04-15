"""
Between-replicate variance of ignition metrics vs aggregation level.

For each (network, threshold), treat each CSV row (folder/PA/n_comms combo)
as one "network realization". Compute:
  - var(p_ignite) across realizations
  - var(log10 mean_ignited) across realizations
with bootstrap 95% CIs.

Plot against aggregation level (HHI, effective_n, or n_blocks).

Input: original combined_tasks.csv (has folder/n_communities/pref_attachment
       as separate rows) + group_charcteristic_mapping.csv for HHI/effective_n.

Uses the ignition decomposition per row (not pooled) so each row yields one
P(ignite) / E[size|ignite] estimate.
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_final_values(s):
    if pd.isna(s) or s == "":
        return np.array([])
    return np.fromstring(str(s).strip().strip('"'), sep=",", dtype=np.int64)


def decompose(runs, net_size, rel=0.01):
    if len(runs) == 0:
        return np.nan, np.nan
    cut = rel * net_size
    if runs.min() >= cut:
        return 1.0, float(runs.mean())
    if runs.max() < cut:
        return 0.0, np.nan
    ig = runs >= cut
    return float(ig.mean()), float(runs[ig].mean()) if ig.any() else np.nan


def bootstrap_var(x, n_boot=1000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = x.var(ddof=1)
    vs = np.array([
        rng.choice(x, len(x), replace=True).var(ddof=1) for _ in range(n_boot)
    ])
    return v, float(np.quantile(vs, 0.025)), float(np.quantile(vs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="combined_tasks.csv")
    ap.add_argument("--mapping", required=True,
                    help="group_charcteristic_mapping.csv")
    ap.add_argument("--outdir", default="./ignite_variance_out")
    ap.add_argument("--agg", default="hhi",
                    choices=["hhi", "effective_n", "n_groups"])
    ap.add_argument("--rel", type=float, default=0.005)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input)
    df["runs"] = df["final_values"].apply(parse_final_values)

    # Network size per network = max run across all rows of that network
    df["row_max"] = df["runs"].apply(lambda a: int(a.max()) if len(a) else np.nan)
    df["net_size"] = df.groupby("network")["row_max"].transform("max")

    # Per-row decomposition
    p_size = df.apply(lambda r: decompose(r["runs"], r["net_size"], args.rel),
                      axis=1, result_type="expand")
    df["p_ignite"] = p_size[0]
    df["log_size_ignited"] = np.log10(p_size[1].clip(lower=1))

    # Attach aggregation metric
    mp = pd.read_csv(args.mapping).rename(columns={"aggregation": "network"})
    df = df.merge(mp[["network", args.agg]], on="network", how="left")

    # Variance across replicates (rows) per (network, threshold)
    rows = []
    for (net, thr), g in df.groupby(["network", "threshold_value"]):
        vp, vp_lo, vp_hi = bootstrap_var(g["p_ignite"].values)
        vs, vs_lo, vs_hi = bootstrap_var(g["log_size_ignited"].values)
        rows.append({
            "network": net, "threshold": thr,
            args.agg: g[args.agg].iloc[0],
            "n_replicates": len(g),
            "mean_pig": g["p_ignite"].mean(),
            "var_pig": vp, "var_pig_lo": vp_lo, "var_pig_hi": vp_hi,
            "mean_log_size": g["log_size_ignited"].mean(),
            "var_log_size": vs, "var_log_size_lo": vs_lo, "var_log_size_hi": vs_hi,
        })
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.outdir, "variance_by_agg.csv"), index=False)
    print(out.to_string(index=False))

    # Plot
    thresholds = sorted(out["threshold"].unique())
    fig, axes = plt.subplots(2, len(thresholds), figsize=(5 * len(thresholds), 8),
                              squeeze=False)
    for j, thr in enumerate(thresholds):
        sub = out[out["threshold"] == thr].sort_values(args.agg)

        ax = axes[0, j]
        x = sub[args.agg].values
        y = sub["var_pig"].values
        yerr = np.vstack([y - sub["var_pig_lo"].values,
                          sub["var_pig_hi"].values - y])
        ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, linewidth=2)
        for _, r in sub.iterrows():
            ax.annotate(r["network"][:18], (r[args.agg], r["var_pig"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(args.agg)
        ax.set_ylabel("Var(P(ignite)) across replicates")
        ax.set_title(f"thr={thr}")
        ax.grid(alpha=0.3)
        if args.agg == "hhi":
            ax.set_xscale("log")

        ax = axes[1, j]
        y = sub["var_log_size"].values
        yerr = np.vstack([y - sub["var_log_size_lo"].values,
                          sub["var_log_size_hi"].values - y])
        ax.errorbar(x, y, yerr=yerr, fmt="s-", color="darkorange",
                    capsize=4, linewidth=2)
        for _, r in sub.iterrows():
            ax.annotate(r["network"][:18], (r[args.agg], r["var_log_size"]),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords="offset points")
        ax.set_xlabel(args.agg)
        ax.set_ylabel("Var(log10 E[size|ignite])")
        ax.grid(alpha=0.3)
        if args.agg == "hhi":
            ax.set_xscale("log")

    fig.suptitle(f"Between-replicate variance of ignition metrics vs {args.agg}",
                 y=1.00)
    fig.tight_layout()
    p = os.path.join(args.outdir, f"ignite_variance_{args.agg}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"\nPlot: {p}")

    # Correlations
    print("\n=== Spearman corr(agg, variance) per threshold ===")
    for thr in thresholds:
        s = out[out["threshold"] == thr]
        if len(s) < 4:
            continue
        c1 = s[args.agg].corr(s["var_pig"], method="spearman")
        c2 = s[args.agg].corr(s["var_log_size"], method="spearman")
        print(f"thr={thr}:  var(P(ig)) r={c1:+.2f}   "
              f"var(log size) r={c2:+.2f}")


if __name__ == "__main__":
    main()