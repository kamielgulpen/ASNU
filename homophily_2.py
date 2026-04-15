"""
Per-characteristic homophily, decoupled from partition granularity.

Idea:
  1. Pick the finest interactions file as reference (it contains all dimensions).
  2. For each demographic dimension d, compute homophily by collapsing the
     mixing matrix to that single dimension: same_d / total. This is the
     marginal probability that an edge stays within the same value of d,
     and it's a property of the *characteristic*, not the partition.
  3. For each partition P (= subset of dimensions), define
        partition_homophily(P) = aggregate of per-dim scores for d in P
     (mean / min / max — choose what fits the hypothesis).
  4. Now: aggregation level = n_blocks (or HHI) of P; homophily = step 3.
     These two axes are independent — small partitions can be high-homophily
     if built from age+ethnicity, low if built from gender alone.

Outputs:
  per_characteristic.csv      : one row per dimension, with raw within-share
                                and Coleman-style excess over null
  partition_homophily.csv     : one row per partition, with aggregated scores
  merged_decoupled.csv        : decomposition merged with the above
  decoupled_plots.png         : cascade vs aggregation, color = char-homophily
"""

import argparse, os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INTERACTIONS_RE = re.compile(r"interactions_(.+)\.csv$")


# ---------- per-characteristic homophily (computed once) -------------------

def per_characteristic_homophily(reference_path):
    """For each dimension in the reference file, compute:
       - within_share: fraction of edge weight where src==dst on that dim
       - expected_within: same under independence (sum p_i^2 over values)
       - excess: within_share - expected_within  (Coleman-style, partition-free)
       - ratio: within_share / expected_within   (odds-ratio-like)
    """
    df = pd.read_csv(reference_path)
    dims = [c[:-4] for c in df.columns if c.endswith("_src")]
    n = df["n"].to_numpy()
    total = n.sum()

    rows = []
    for d in dims:
        s = df[f"{d}_src"].to_numpy()
        t = df[f"{d}_dst"].to_numpy()
        within = (s == t)
        within_share = n[within].sum() / total

        # marginal distribution of the dimension over edge endpoints
        marg = (pd.Series(np.r_[s, t]).repeat(np.r_[n, n] // 1)
                  .value_counts(normalize=True))
        # cheaper: weight by n on each side
        marg_src = pd.Series(s).groupby(s).apply(lambda x: n[x.index].sum())
        marg_dst = pd.Series(t).groupby(t).apply(lambda x: n[x.index].sum())
        marg = (marg_src.add(marg_dst, fill_value=0)) / (2 * total)
        expected = float((marg ** 2).sum())

        rows.append({
            "characteristic": d,
            "within_share": within_share,
            "expected_within": expected,
            "excess_homophily": within_share - expected,
            "ratio_homophily": within_share / expected if expected > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ---------- partition-level aggregation of per-char scores -----------------

def parse_partition_dims(network_name, known_dims):
    """Greedy match: find which known_dims appear in the partition name."""
    found = []
    rest = network_name
    # match longest dim names first to avoid 'lft' matching inside 'inkomensniveau' etc.
    for d in sorted(known_dims, key=len, reverse=True):
        if d in rest:
            found.append(d)
            rest = rest.replace(d, "")
    return sorted(found)


def n_blocks_per_network(interactions_dir):
    """Count distinct blocks in each interactions file."""
    rows = []
    for path in glob.glob(os.path.join(interactions_dir, "interactions_*.csv")):
        m = INTERACTIONS_RE.search(os.path.basename(path))
        if not m: continue
        df = pd.read_csv(path)
        dims = [c[:-4] for c in df.columns if c.endswith("_src")]
        src = list(zip(*[df[f"{d}_src"] for d in dims]))
        dst = list(zip(*[df[f"{d}_dst"] for d in dims]))
        rows.append({"network": m.group(1), "n_blocks": len(set(src) | set(dst))})
    return pd.DataFrame(rows)


def partition_homophily_table(networks, per_char):
    char_to_score = dict(zip(per_char["characteristic"], per_char["excess_homophily"]))
    known = list(char_to_score.keys())
    rows = []
    for net in networks:
        dims = parse_partition_dims(net, known)
        scores = [char_to_score[d] for d in dims if d in char_to_score]
        rows.append({
            "network": net,
            "dimensions": ",".join(dims),
            "n_dims": len(dims),
            "mean_char_homophily": float(np.mean(scores)) if scores else np.nan,
            "min_char_homophily": float(np.min(scores)) if scores else np.nan,
            "max_char_homophily": float(np.max(scores)) if scores else np.nan,
        })
    return pd.DataFrame(rows)


# ---------- plot -----------------------------------------------------------

def plot_decoupled(merged, outdir, hom_col="mean_char_homophily"):
    os.makedirs(outdir, exist_ok=True)
    df = merged.dropna(subset=[hom_col, "n_blocks", "mean_ignited"]).copy()
    df["log_nb"] = np.log10(df["n_blocks"])
    thresholds = sorted(df["threshold_value"].unique())

    fig, axes = plt.subplots(2, len(thresholds), figsize=(5 * len(thresholds), 8.5),
                              squeeze=False)
    vmin, vmax = df[hom_col].min(), df[hom_col].max()

    for j, thr in enumerate(thresholds):
        sub = df[df["threshold_value"] == thr]
        for row, (ycol, ylab, ylog) in enumerate([
            ("mean_ignited", "E[size | ignite]", True),
            ("p_ignite", "P(ignite)", False),
        ]):
            ax = axes[row, j]
            sc = ax.scatter(sub["log_nb"], sub[ycol],
                            c=sub[hom_col], cmap="RdBu_r",
                            vmin=vmin, vmax=vmax, s=110,
                            edgecolor="k", linewidth=0.6)
            for _, r in sub.iterrows():
                ax.annotate(r["network"][:18], (r["log_nb"], r[ycol]),
                            fontsize=6, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
            if ylog: ax.set_yscale("log")
            ax.set_xlabel("log10(n_blocks)  →  finer aggregation")
            ax.set_ylabel(ylab)
            ax.set_title(f"thr={thr}")
            ax.grid(alpha=0.3, which="both")
            plt.colorbar(sc, ax=ax, label=hom_col)

    fig.suptitle(f"Aggregation × characteristic-homophily (color = {hom_col})",
                 y=1.00)
    fig.tight_layout()
    p = os.path.join(outdir, f"decoupled_{hom_col}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions-dir", required=True)
    ap.add_argument("--reference",
                    help="Specific interactions file to use as reference. "
                         "Default = the one with the most _src columns.")
    ap.add_argument("--decomposition", required=True)
    ap.add_argument("--outdir", default="./decoupled_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # pick reference = file with most dimensions
    files = sorted(glob.glob(os.path.join(args.interactions_dir, "interactions_*.csv")))
    if args.reference:
        ref = args.reference
    else:
        ref = max(files, key=lambda f: sum(1 for c in pd.read_csv(f, nrows=0).columns
                                            if c.endswith("_src")))
    print(f"Reference: {os.path.basename(ref)}")

    per_char = per_characteristic_homophily(ref)
    per_char.to_csv(os.path.join(args.outdir, "per_characteristic.csv"), index=False)
    print("\nPer-characteristic homophily:")
    print(per_char.to_string(index=False))

    dec = pd.read_csv(args.decomposition)
    networks = dec["network"].unique()
    part_hom = partition_homophily_table(networks, per_char)
    nb = n_blocks_per_network(args.interactions_dir)
    part_hom = part_hom.merge(nb, on="network", how="left")
    part_hom.to_csv(os.path.join(args.outdir, "partition_homophily.csv"), index=False)
    print("\nPartition homophily:")
    print(part_hom.to_string(index=False))

    merged = dec.merge(part_hom, on="network", how="left")
    merged.to_csv(os.path.join(args.outdir, "merged_decoupled.csv"), index=False)

    for col in ["mean_char_homophily", "min_char_homophily", "max_char_homophily"]:
        p = plot_decoupled(merged, args.outdir, hom_col=col)
        print(f"Plot: {p}")

    # correlations
    print("\n=== Correlations (across networks, per threshold) ===")
    for thr in sorted(merged["threshold_value"].unique()):
        sub = merged[merged["threshold_value"] == thr].dropna(
            subset=["mean_char_homophily", "n_blocks", "mean_ignited"])
        if len(sub) < 3: continue
        log_size = np.log10(sub["mean_ignited"].clip(lower=1))
        log_nb = np.log10(sub["n_blocks"])
        print(f"thr={thr}:  "
              f"corr(log n_blocks, log size) = {log_nb.corr(log_size):+.2f}   "
              f"corr(char_hom, log size)     = {sub['mean_char_homophily'].corr(log_size):+.2f}   "
              f"partial: hom controlling agg = "
              f"{partial_corr(sub['mean_char_homophily'], log_size, log_nb):+.2f}")


def partial_corr(x, y, z):
    """corr(x,y | z) via residuals."""
    rx = x - np.poly1d(np.polyfit(z, x, 1))(z)
    ry = y - np.poly1d(np.polyfit(z, y, 1))(z)
    return float(np.corrcoef(rx, ry)[0, 1])


if __name__ == "__main__":
    main()