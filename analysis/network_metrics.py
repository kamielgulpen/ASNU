"""
Compute clustering, degree skewness, and modularity for all werkschool networks
and save results to CSV + scatterplots.
"""
import pickle
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.Experiments.contagion_experiment import load_networks


def nx_to_igraph(nx_graph):
    """Convert a NetworkX graph to an igraph graph."""
    nodes = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in nx_graph.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=nx_graph.is_directed())
    ig_graph.vs["name"] = nodes
    return ig_graph


def compute_metrics(G_nx):
    """Compute clustering, degree skewness, and modularity."""
    G_ig = nx_to_igraph(G_nx)

    # Clustering coefficient (global transitivity)
    clustering = G_ig.transitivity_undirected()

    # Degree skewness
    degrees = G_ig.degree(mode="all")
    skewness = stats.skew(degrees)

    # Modularity via Louvain on undirected version
    G_undirected = G_ig.as_undirected()
    partition = G_undirected.community_multilevel()
    modularity = G_undirected.modularity(partition)

    return {
        "clustering": clustering,
        "skewness": skewness,
        "modularity": modularity,
        "nodes": G_ig.vcount(),
        "edges": G_ig.ecount(),
        "mean_degree": np.mean(degrees),
    }


def process_folder(folder, results):
    """Load all networks from a folder and compute metrics."""
    print(f"\nProcessing: {folder}")
    network_graphs = load_networks(folder, add_random=False)

    for name, obj in network_graphs.items():
        # Handle both plain nx.Graph and wrapper objects with .graph attribute
        if isinstance(obj, nx.Graph):
            G_nx = obj
        else:
            G_nx = obj.graph

        metrics = compute_metrics(G_nx)
        metrics["folder"] = str(folder)
        metrics["network"] = name
        results.append(metrics)
        print(f"  {name}: clustering={metrics['clustering']:.4f}, "
              f"skewness={metrics['skewness']:.2f}, modularity={metrics['modularity']:.4f}")


def main():
    base = Path("Data/networks")
    results = []

    # 1. Single-layer werkschool folders (scale=... parameter sweep folders)
    for subfolder in sorted(base.iterdir()):
        if not subfolder.is_dir():
            continue
        # Check if this folder directly contains pkl files
        pkl_files = list(subfolder.glob("*.pkl"))
        if pkl_files:
            process_folder(subfolder, results)

        # 2. Multiplex folders: load werkschool.pkl specifically
        for char_folder in sorted(subfolder.iterdir()):
            if not char_folder.is_dir():
                continue
            ws_pkl = char_folder / "werkschool.pkl"
            if ws_pkl.exists():
                print(f"\nProcessing multiplex: {char_folder}")
                with open(ws_pkl, "rb") as f:
                    obj = pickle.load(f)
                G_nx = obj.graph if hasattr(obj, "graph") and not isinstance(obj, nx.Graph) else obj
                metrics = compute_metrics(G_nx)
                metrics["folder"] = str(char_folder)
                metrics["network"] = "werkschool"
                results.append(metrics)
                print(f"  werkschool: clustering={metrics['clustering']:.4f}, "
                      f"skewness={metrics['skewness']:.2f}, modularity={metrics['modularity']:.4f}")

    if not results:
        print("No networks found!")
        return

    # Save to CSV
    df = pd.DataFrame(results)
    output_dir = Path("analysis/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "network_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to {csv_path}")
    print(df.to_string())

    # Scatterplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].scatter(df["clustering"], df["skewness"], alpha=0.7)
    axes[0].set_xlabel("Clustering Coefficient")
    axes[0].set_ylabel("Degree Skewness")
    axes[0].set_title("Clustering vs Skewness")

    axes[1].scatter(df["clustering"], df["modularity"], alpha=0.7)
    axes[1].set_xlabel("Clustering Coefficient")
    axes[1].set_ylabel("Modularity")
    axes[1].set_title("Clustering vs Modularity")

    axes[2].scatter(df["modularity"], df["skewness"], alpha=0.7)
    axes[2].set_xlabel("Modularity")
    axes[2].set_ylabel("Degree Skewness")
    axes[2].set_title("Modularity vs Skewness")

    # Label points with network name
    for _, row in df.iterrows():
        axes[0].annotate(row["network"], (row["clustering"], row["skewness"]),
                         fontsize=6, alpha=0.6)
        axes[1].annotate(row["network"], (row["clustering"], row["modularity"]),
                         fontsize=6, alpha=0.6)
        axes[2].annotate(row["network"], (row["modularity"], row["skewness"]),
                         fontsize=6, alpha=0.6)

    plt.tight_layout()
    plot_path = output_dir / "network_metrics_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()
