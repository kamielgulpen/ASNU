"""
Test: does aggregation hurt less when the partition is homophilous?

Approach:
  x-axis = aggregation level (n_blocks or HHI)
  y-axis = cascade metric (E[size|ignite], P(ignite))
  color  = homophily (modularity_q or frac_within)
  one panel per threshold

Plus: stratify networks into homophily tertiles and fit aggregation slopes
      within each tertile. Hypothesis predicts:
        - high-homophily slope ~ flat (aggregation doesn't hurt)
        - low-homophily slope steep (aggregation destroys structure)

Input: merged.csv from homophily_analysis.py
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_merged(path, hhi_path=None):
    df = pd.read_csv(path)
    if hhi_path:
        hhi = pd.read_csv(hhi_path)[["aggregation", "hhi"]].rename(
            columns={"aggregation": "network"})
        df = df.merge(hhi, on="network", how="left")
    return df


def aggregation_metric(df, kind="log_n_blocks"):
    if kind == "log_n_blocks":
        return np.log10(df["n_blocks"]), "log10(n_blocks)  →  finer aggregation"
    if kind == "hhi":
        return df["hhi"], "HHI  →  coarser aggregation"
    if kind == "neg_log_hhi":
        return -np.log10(df["hhi"]), "-log10(HHI)  →  finer aggregation"
    raise ValueError(kind)


def plot_interaction(df, outdir, agg_kind="log_n_blocks",
                     hom_metric="modularity_q"):
    os.makedirs(outdir, exist_ok=True)
    df = df.dropna(subset=[hom_metric, "n_blocks", "mean_ignited", "p_ignite"]).copy()
    df["agg"], agg_label = aggregation_metric(df, agg_kind)

    thresholds = sorted(df["threshold_value"].unique())
    fig, axes = plt.subplots(2, len(thresholds), figsize=(5 * len(thresholds), 8.5),
                              squeeze=False)

    vmin, vmax = df[hom_metric].min(), df[hom_metric].max()

    for j, thr in enumerate(thresholds):
        sub = df[df["threshold_value"] == thr]

        # E[size|ignite] vs aggregation, colored by homophily
        ax = axes[0, j]
        sc = ax.scatter(sub["agg"], sub["mean_ignited"],
                        c=sub[hom_metric], cmap="RdBu_r",
                        vmin=vmin, vmax=vmax, s=100,
                        edgecolor="k", linewidth=0.6)
        for _, r in sub.iterrows():
            ax.annotate(r["network"][:18], (r["agg"], r["mean_ignited"]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel(agg_label)
        ax.set_ylabel("E[size | ignite]")
        ax.set_title(f"thr={thr}")
        ax.grid(alpha=0.3, which="both")
        plt.colorbar(sc, ax=ax, label=hom_metric)

        # P(ignite) vs aggregation, colored by homophily
        ax = axes[1, j]
        sc = ax.scatter(sub["agg"], sub["p_ignite"],
                        c=sub[hom_metric], cmap="RdBu_r",
                        vmin=vmin, vmax=vmax, s=100,
                        edgecolor="k", linewidth=0.6)
        for _, r in sub.iterrows():
            ax.annotate(r["network"][:18], (r["agg"], r["p_ignite"]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(agg_label)
        ax.set_ylabel("P(ignite)")
        ax.grid(alpha=0.3)
        plt.colorbar(sc, ax=ax, label=hom_metric)

    fig.suptitle(f"Aggregation × homophily interaction "
                 f"(color = {hom_metric})", y=1.00)
    fig.tight_layout()
    p = os.path.join(outdir, f"interaction_{agg_kind}_{hom_metric}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def stratified_slopes(df, agg_kind="log_n_blocks",
                      hom_metric="modularity_q", outdir="."):
    """Split networks into homophily tertiles, fit log(E[size|ig]) ~ aggregation
    within each tertile, report slopes per threshold."""
    df = df.dropna(subset=[hom_metric, "n_blocks", "mean_ignited"]).copy()
    df["agg"], _ = aggregation_metric(df, agg_kind)
    df["log_size"] = np.log10(df["mean_ignited"].clip(lower=1))

    # per-network homophily tertile (homophily is constant within network)
    net_hom = df.groupby("network")[hom_metric].first()
    tertile = pd.qcut(net_hom, 3, labels=["low", "mid", "high"], duplicates="drop")
    df["hom_tertile"] = df["network"].map(tertile)

    print(f"\n=== Hypothesis test: aggregation slope by homophily tertile ===")
    print(f"  agg metric: {agg_kind} ; homophily: {hom_metric}")
    print(f"  hypothesis: |slope| smaller in HIGH-homophily tertile\n")
    rows = []
    for thr in sorted(df["threshold_value"].unique()):
        for tert in ["low", "mid", "high"]:
            sub = df[(df["threshold_value"] == thr) & (df["hom_tertile"] == tert)]
            if len(sub) < 3:
                continue
            slope, intercept = np.polyfit(sub["agg"], sub["log_size"], 1)
            r = sub["agg"].corr(sub["log_size"])
            rows.append({"threshold": thr, "tertile": tert, "n": len(sub),
                          "slope": slope, "r": r})
            print(f"  thr={thr}  hom={tert:>4}  n={len(sub):2d}  "
                  f"slope={slope:+.2f}  r={r:+.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(outdir, "stratified_slopes.csv"), index=False)
    return out


def plot_slopes(slopes, outdir):
    if slopes.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for tert, sub in slopes.groupby("tertile"):
        ax.plot(sub["threshold"], sub["slope"], "o-", label=f"{tert} homophily",
                linewidth=2, markersize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Slope: log E[size|ignite] vs aggregation")
    ax.set_title("Aggregation sensitivity by homophily tertile\n"
                 "(hypothesis: high-homophily slope flatter than low)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "slopes_by_tertile.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True,
                    help="merged.csv from homophily_analysis.py")
    ap.add_argument("--hhi", default=None,
                    help="optional group_charcteristic_mapping.csv to attach HHI")
    ap.add_argument("--outdir", default="./interaction_out")
    ap.add_argument("--hom", default="modularity_q",
                    choices=["modularity_q", "frac_within", "ei_index"])
    ap.add_argument("--agg", default="log_n_blocks",
                    choices=["log_n_blocks", "hhi", "neg_log_hhi"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_merged(args.merged, args.hhi)

    p = plot_interaction(df, args.outdir, agg_kind=args.agg, hom_metric=args.hom)
    print(f"Plot: {p}")
    slopes = stratified_slopes(df, args.agg, args.hom, args.outdir)
    p2 = plot_slopes(slopes, args.outdir)
    if p2:
        print(f"Plot: {p2}")


if __name__ == "__main__":
    main()