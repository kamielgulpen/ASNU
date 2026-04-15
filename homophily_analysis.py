"""
Compute homophily metrics from block-aggregated interaction files and merge
with the ignition/cascade decomposition.

Input:
  --interactions-dir : folder containing files named
                       interactions_<partition>.csv with columns
                       <col>_src, <col>_dst, n
  --decomposition    : decomposition.csv produced by ignition_decomposition.py
                       (must contain `network`, `threshold_value`, `p_ignite`,
                        `mean_ignited`, ...)
  --outdir           : where to write merged CSV and plots

Outputs:
  homophily.csv         : one row per partition with newman_r, ei_index,
                          per-dimension EI, n_blocks, total_edges
  merged.csv            : decomposition joined to homophily on `network`
  homophily_plots.png   : E[size|ignite] and P(ignite) vs homophily
"""

import argparse
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- partition parsing ----------------------------------------------

INTERACTIONS_RE = re.compile(r"interactions_(.+)\.csv$")


def partition_from_filename(path):
    m = INTERACTIONS_RE.search(os.path.basename(path))
    return m.group(1) if m else None


def partition_cols(df):
    """Detect partition dimensions from columns ending in _src."""
    return [c[:-4] for c in df.columns if c.endswith("_src")]


# ---------- homophily metrics ----------------------------------------------

def compute_homophily(path):
    df = pd.read_csv(path)
    cols = partition_cols(df)
    if not cols or "n" not in df.columns:
        return None

    src = list(zip(*[df[f"{c}_src"] for c in cols]))
    dst = list(zip(*[df[f"{c}_dst"] for c in cols]))
    n = df["n"].to_numpy()

    total = n.sum()
    within = np.array([s == d for s, d in zip(src, dst)])
    internal = n[within].sum()
    external = total - internal

    # E-I index on full block: -1 fully homophilous, +1 fully heterophilous
    ei_full = (external - internal) / total

    # Mixing matrix e[i,j] = fraction of edge weight from block i to block j
    blocks = sorted(set(src) | set(dst))
    idx = {b: i for i, b in enumerate(blocks)}
    k = len(blocks)
    e = np.zeros((k, k))
    for s, d, w in zip(src, dst, n):
        e[idx[s], idx[d]] += w
    e /= e.sum()
    a = e.sum(axis=1)
    b = e.sum(axis=0)

    # Modularity Q: excess within-block weight vs degree-preserving null.
    # Same units across partitions of any granularity; directly meaningful as
    # "how much more within-block density than random rewiring would give".
    modularity_q = float(np.trace(e) - (a * b).sum())

    # Newman assortativity (kept for reference; normalized version of Q)
    denom = 1 - (a * b).sum()
    newman_r = (np.trace(e) - (a * b).sum()) / denom if denom > 0 else np.nan

    # Per-dimension E-I
    per_dim = {}
    for col in cols:
        same = (df[f"{col}_src"].to_numpy() == df[f"{col}_dst"].to_numpy())
        intra = n[same].sum()
        per_dim[f"ei_{col}"] = (total - 2 * intra) / total

    out = {
        "n_blocks": k,
        "total_edges": int(total),
        "frac_within": internal / total,
        "ei_index": ei_full,
        "modularity_q": modularity_q,
        "newman_r": newman_r,
        "n_dimensions": len(cols),
    }
    out.update(per_dim)
    return out


def homophily_table(interactions_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(interactions_dir, "interactions_*.csv"))):
        partition = partition_from_filename(path)
        print(f"  {partition} ...", end=" ", flush=True)
        h = compute_homophily(path)
        if h is None:
            print("skipped (no _src/_dst columns or no n)")
            continue
        h["network"] = partition
        rows.append(h)
        print(f"newman_r={h['newman_r']:.3f}  ei={h['ei_index']:+.3f}  "
              f"blocks={h['n_blocks']}")
    return pd.DataFrame(rows)


# ---------- merge + plot ---------------------------------------------------

def merge_with_decomposition(hom_df, dec_path):
    dec = pd.read_csv(dec_path)
    merged = dec.merge(hom_df, on="network", how="left")
    missing = merged[merged["newman_r"].isna()]["network"].unique()
    if len(missing):
        print(f"WARNING: no homophily for: {list(missing)}")
    return merged


def plot_homophily(merged, outdir, x_metric="modularity_q",
                   x_label="Modularity Q (excess within-block density)"):
    os.makedirs(outdir, exist_ok=True)
    thresholds = sorted(merged["threshold_value"].unique())
    fig, axes = plt.subplots(2, len(thresholds), figsize=(5 * len(thresholds), 8),
                              squeeze=False)

    for j, thr in enumerate(thresholds):
        sub = merged[merged["threshold_value"] == thr].dropna(subset=[x_metric])

        ax = axes[0, j]
        ax.scatter(sub[x_metric], sub["mean_ignited"],
                   s=80, c=sub["n_blocks"], cmap="viridis", alpha=0.85,
                   edgecolor="k", linewidth=0.5)
        for _, r in sub.iterrows():
            if pd.notna(r["mean_ignited"]):
                ax.annotate(r["network"][:18], (r[x_metric], r["mean_ignited"]),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("E[size | ignite]")
        ax.set_title(f"Conditional cascade size — thr={thr}")
        ax.grid(alpha=0.3, which="both")

        ax = axes[1, j]
        ax.scatter(sub[x_metric], sub["p_ignite"],
                   s=80, c=sub["n_blocks"], cmap="viridis", alpha=0.85,
                   edgecolor="k", linewidth=0.5)
        for _, r in sub.iterrows():
            ax.annotate(r["network"][:18], (r[x_metric], r["p_ignite"]),
                        fontsize=6, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel(x_label)
        ax.set_ylabel("P(ignite)")
        ax.set_title(f"Ignition probability — thr={thr}")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, f"homophily_plots_{x_metric}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_per_dimension(hom_df, outdir):
    """Heatmap of per-dimension E-I across partitions."""
    ei_cols = [c for c in hom_df.columns if c.startswith("ei_") and c != "ei_index"]
    if not ei_cols:
        return None
    mat = hom_df.set_index("network")[ei_cols]
    mat.columns = [c[3:] for c in mat.columns]

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(mat.columns) + 4),
                                     0.4 * len(mat) + 2))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels([n[:35] for n in mat.index], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="E-I (−1 homophilous, +1 heterophilous)")
    ax.set_title("Per-dimension homophily across partitions")
    fig.tight_layout()
    path = os.path.join(outdir, "per_dimension_ei.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions-dir", required=True)
    ap.add_argument("--decomposition", required=True,
                    help="decomposition.csv from ignition_decomposition.py")
    ap.add_argument("--outdir", default="./homophily_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Computing homophily per partition:")
    hom = homophily_table(args.interactions_dir)
    hom_path = os.path.join(args.outdir, "homophily.csv")
    hom.to_csv(hom_path, index=False)
    print(f"\nWrote {hom_path}")

    merged = merge_with_decomposition(hom, args.decomposition)
    merged_path = os.path.join(args.outdir, "merged.csv")
    merged.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path}")

    p1 = plot_homophily(merged, args.outdir)
    p2 = plot_per_dimension(hom, args.outdir)
    print(f"Plots: {p1}")
    if p2:
        print(f"       {p2}")

    # Quick correlation summary
    print("\n=== Correlations across networks (at each threshold) ===")
    for thr in sorted(merged["threshold_value"].unique()):
        sub = merged[merged["threshold_value"] == thr].dropna(
            subset=["newman_r", "mean_ignited", "p_ignite"])
        if len(sub) < 3:
            continue
        c1 = sub["newman_r"].corr(sub["mean_ignited"])
        c2 = sub["newman_r"].corr(sub["p_ignite"])
        c3 = sub["newman_r"].corr(np.log(sub["mean_ignited"].clip(lower=1)))
        print(f"thr={thr}: corr(newman_r, E[size|ig]) = {c1:+.2f}   "
              f"corr(newman_r, log E[size|ig]) = {c3:+.2f}   "
              f"corr(newman_r, P(ig)) = {c2:+.2f}")


if __name__ == "__main__":
    main()