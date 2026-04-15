"""
Test: when you drop dimension d from a partition, does the cascade outcome
change more if d is homophilous than if d is heterophilous?

Find all (P_fine, P_coarse) pairs where P_coarse = P_fine minus exactly one
dimension d. Compute |delta cascade|, plot against homophily(d).
"""

import argparse, os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INTERACTIONS_RE = re.compile(r"interactions_(.+)\.csv$")


def per_characteristic_homophily(reference_path):
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
        marg_src = pd.Series(s).groupby(s).apply(lambda x: n[x.index].sum())
        marg_dst = pd.Series(t).groupby(t).apply(lambda x: n[x.index].sum())
        marg = (marg_src.add(marg_dst, fill_value=0)) / (2 * total)
        expected = float((marg ** 2).sum())
        rows.append({
            "characteristic": d,
            "within_share": within_share,
            "expected_within": expected,
            "excess_homophily": within_share - expected,
        })
    return pd.DataFrame(rows)


def parse_dims(network_name, known_dims):
    found = []
    rest = network_name
    for d in sorted(known_dims, key=len, reverse=True):
        if d in rest:
            found.append(d)
            rest = rest.replace(d, "")
    return frozenset(found)


def find_pairs(networks, known_dims):
    parsed = {n: parse_dims(n, known_dims) for n in networks}
    pairs = []
    for fine, fine_dims in parsed.items():
        for coarse, coarse_dims in parsed.items():
            if fine == coarse:
                continue
            diff = fine_dims - coarse_dims
            if len(diff) == 1 and coarse_dims < fine_dims:
                pairs.append((fine, coarse, next(iter(diff))))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions-dir", required=True)
    ap.add_argument("--decomposition", required=True)
    ap.add_argument("--outdir", default="./pair_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.interactions_dir, "interactions_*.csv")))
    ref = max(files, key=lambda f: sum(1 for c in pd.read_csv(f, nrows=0).columns
                                        if c.endswith("_src")))
    print(f"Reference: {os.path.basename(ref)}")

    per_char = per_characteristic_homophily(ref)
    char_to_h = dict(zip(per_char["characteristic"], per_char["excess_homophily"]))
    print("\nPer-characteristic homophily:")
    print(per_char.sort_values("excess_homophily", ascending=False).to_string(index=False))

    dec = pd.read_csv(args.decomposition)
    networks = dec["network"].unique()
    pairs = find_pairs(networks, list(char_to_h.keys()))
    print(f"\nFound {len(pairs)} fine-to-coarse pairs")

    rows = []
    for fine, coarse, d in pairs:
        h = char_to_h.get(d, np.nan)
        for thr in dec["threshold_value"].unique():
            f = dec[(dec["network"] == fine) & (dec["threshold_value"] == thr)]
            c = dec[(dec["network"] == coarse) & (dec["threshold_value"] == thr)]
            if f.empty or c.empty:
                continue
            f, c = f.iloc[0], c.iloc[0]
            rows.append({
                "fine": fine, "coarse": coarse, "dropped_dim": d,
                "homophily_dropped": h, "threshold": thr,
                "fine_size": f["mean_ignited"], "coarse_size": c["mean_ignited"],
                "fine_pig": f["p_ignite"], "coarse_pig": c["p_ignite"],
                "delta_log_size": np.log10(max(f["mean_ignited"], 1)) -
                                  np.log10(max(c["mean_ignited"], 1)),
                "delta_pig": f["p_ignite"] - c["p_ignite"],
            })
    pairs_df = pd.DataFrame(rows)
    pairs_df.to_csv(os.path.join(args.outdir, "pairs.csv"), index=False)
    print(f"\n{len(pairs_df)} pair-threshold rows")

    thresholds = sorted(pairs_df["threshold"].unique())
    fig, axes = plt.subplots(2, len(thresholds), figsize=(5 * len(thresholds), 8),
                              squeeze=False)
    dims = sorted(pairs_df["dropped_dim"].unique())
    cmap = plt.cm.tab10
    color = {d: cmap(i) for i, d in enumerate(dims)}

    for j, thr in enumerate(thresholds):
        sub = pairs_df[pairs_df["threshold"] == thr]
        for row, (ycol, ylab) in enumerate([
            ("delta_log_size", "Δ log10 E[size|ignite]\n(fine − coarse)"),
            ("delta_pig", "Δ P(ignite)  (fine − coarse)"),
        ]):
            ax = axes[row, j]
            for d in dims:
                s = sub[sub["dropped_dim"] == d]
                if s.empty:
                    continue
                ax.scatter(s["homophily_dropped"], s[ycol], color=color[d],
                           s=80, alpha=0.8, edgecolor="k", linewidth=0.5,
                           label=d if (row == 0 and j == 0) else None)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("Homophily of dropped dimension")
            ax.set_ylabel(ylab)
            ax.set_title(f"thr={thr}")
            ax.grid(alpha=0.3)
        if j == 0:
            axes[0, 0].legend(fontsize=7, loc="best")

    fig.suptitle("Cascade change when dropping one dimension, vs that dimension's homophily",
                 y=1.00)
    fig.tight_layout()
    p = os.path.join(args.outdir, "pairwise_drop.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Plot: {p}")

    print("\n=== Correlations: |delta| vs homophily of dropped dim ===")
    for thr in thresholds:
        sub = pairs_df[pairs_df["threshold"] == thr].dropna(subset=["homophily_dropped"])
        if len(sub) < 3:
            continue
        c1 = sub["homophily_dropped"].corr(sub["delta_log_size"].abs())
        c2 = sub["homophily_dropped"].corr(sub["delta_pig"].abs())
        print(f"thr={thr}:  corr(hom, |Δlog size|) = {c1:+.2f}   "
              f"corr(hom, |Δ P(ig)|) = {c2:+.2f}   n={len(sub)}")


if __name__ == "__main__":
    main()