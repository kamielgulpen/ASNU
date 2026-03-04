"""
Compute clustering, degree skewness, and modularity for all werkschool networks
and save results to CSV + scatterplots.

Memory-efficient version: processes networks one at a time and writes incrementally.
"""
import pickle
import sys
import os
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
# sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from modeling.experiments.contagion_experiment import load_networks


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

    result = {
        "clustering": clustering,
        "skewness": skewness,
        "modularity": modularity,
        "nodes": G_ig.vcount(),
        "edges": G_ig.ecount(),
        "mean_degree": np.mean(degrees),
    }
    
    # Explicitly delete large objects
    del G_ig, G_undirected, partition, degrees
    gc.collect()
    
    return result


def load_and_process_single_network(pkl_path, folder_name, network_name):
    """Load a single network, compute metrics, and immediately release memory."""
    try:
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        
        # Extract NetworkX graph
        if isinstance(obj, nx.Graph):
            G_nx = obj
        else:
            G_nx = obj.graph if hasattr(obj, "graph") else obj
        
        # Compute metrics
        metrics = compute_metrics(G_nx)
        metrics["folder"] = folder_name
        metrics["network"] = network_name
        
        # Clean up
        del obj, G_nx
        gc.collect()
        
        return metrics
    except Exception as e:
        print(f"Error processing {pkl_path}: {e}")
        return None


def process_folder_iteratively(folder, csv_path, write_header):
    """Process networks from a folder one at a time, writing results incrementally."""
    print(f"\nProcessing: {folder}")
    
    # Get list of pickle files without loading them
    pkl_files = list(folder.glob("*.pkl"))
    
    if not pkl_files:
        return 0
    
    count = 0
    for pkl_file in pkl_files:
        network_name = pkl_file.stem
        metrics = load_and_process_single_network(pkl_file, str(folder), network_name)
        
        if metrics:
            # Write to CSV immediately (append mode)
            df_row = pd.DataFrame([metrics])
            df_row.to_csv(csv_path, mode='a', header=write_header, index=False)
            write_header = False  # Only write header once
            
            print(f"  {network_name}: clustering={metrics['clustering']:.4f}, "
                  f"skewness={metrics['skewness']:.2f}, modularity={metrics['modularity']:.4f}")
            count += 1
            
            # Clean up
            del metrics, df_row
            gc.collect()
    
    return count


def process_multiplex_network(char_folder, csv_path, write_header):
    """Process a single multiplex network."""
    ws_pkl = char_folder / "werkschool.pkl"
    if not ws_pkl.exists():
        return False
    
    print(f"\nProcessing multiplex: {char_folder}")
    metrics = load_and_process_single_network(ws_pkl, str(char_folder), "werkschool")
    
    if metrics:
        # Write to CSV immediately
        df_row = pd.DataFrame([metrics])
        df_row.to_csv(csv_path, mode='a', header=write_header, index=False)
        
        print(f"  werkschool: clustering={metrics['clustering']:.4f}, "
              f"skewness={metrics['skewness']:.2f}, modularity={metrics['modularity']:.4f}")
        
        # Clean up
        del metrics, df_row
        gc.collect()
        return True
    
    return False


def create_plots(csv_path, output_dir):
    """Create plots from the saved CSV data."""
    # Load data in chunks if needed, but for plotting we need it all
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("No data to plot!")
        return
    
    print(f"\nLoaded {len(df)} networks for plotting")
    
    # Merge with mapping
    try:
        df_mapping = pd.read_csv("../../utils/group_charcteristic_mapping.csv")
        df = df.merge(df_mapping, left_on="network", right_on="aggregation")
        print(f"After merge: {len(df)} networks")
    except FileNotFoundError:
        print("Warning: group_charcteristic_mapping.csv not found, skipping merge")
        return
    
    # Create discrete HHI bins
    n_bins = 5
    df['hhi_bin'] = pd.cut(df['hhi'], bins=n_bins, labels=False)
    
    # Calculate variances
    variances_skew = []
    variances_mod = []
    variances_clus = []
    bins = []

    for i in sorted(df['hhi_bin'].dropna().unique()):
        df_x = df[df['hhi_bin'] == i]
        bins.append(i)
        variances_skew.append(np.var(df_x["skewness"]))
        variances_mod.append(np.var(df_x["modularity"]))
        variances_clus.append(np.var(df_x["clustering"]))

    # Variance plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(bins, variances_skew, marker='o')
    axes[0].set_xlabel('HHI Bin')
    axes[0].set_ylabel('Variance')
    axes[0].set_title('Skewness Variance vs Bins')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(bins, variances_mod, marker='o', color='orange')
    axes[1].set_xlabel('HHI Bin')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Modularity Variance vs Bins')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(bins, variances_clus, marker='o', color='green')
    axes[2].set_xlabel('HHI Bin')
    axes[2].set_ylabel('Variance')
    axes[2].set_title('Clustering Variance vs Bins')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    variance_plot_path = output_dir / "network_variance_plots.png"
    plt.savefig(variance_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved variance plot to {variance_plot_path}")
    
    # Scatterplots with discrete colormap
    cmap = plt.cm.viridis
    hhi_bins = df['hhi_bin'].unique()
    hhi_bins = np.sort(hhi_bins[~np.isnan(hhi_bins)])
    colors = cmap(np.linspace(0, 1, len(hhi_bins)))
    hhi_colors = dict(zip(hhi_bins, colors))
    df['color'] = df['hhi_bin'].map(hhi_colors)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot with colors
    for _, row in df.iterrows():
        if not np.isnan(row['hhi_bin']):
            axes[0].scatter(row["clustering"], row["skewness"], 
                        c=[row["color"]], s=50, alpha=0.6)
            axes[1].scatter(row["clustering"], row["modularity"], 
                        c=[row["color"]], s=50, alpha=0.6)
            axes[2].scatter(row["modularity"], row["skewness"], 
                        c=[row["color"]], s=50, alpha=0.6)

    axes[0].set_xlabel("Clustering Coefficient")
    axes[0].set_ylabel("Degree Skewness")
    axes[1].set_xlabel("Clustering Coefficient")
    axes[1].set_ylabel("Modularity")
    axes[2].set_xlabel("Modularity")
    axes[2].set_ylabel("Degree Skewness")
    
    # Create legend
    handles = []
    for count, i in enumerate(hhi_bins):
        if count == 0 or count == len(hhi_bins) - 1:
            value = "low" if count == 0 else "high"
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=hhi_colors[i], 
                            markersize=8, 
                            label=f'{value}'))
        else:
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                            markerfacecolor=hhi_colors[i], 
                            markersize=8, 
                            label=f''))

    axes[2].legend(handles=handles, title='Aggregation level', loc='best', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / "network_properties_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {plot_path}")
    
    # Clean up
    del df, fig, axes
    gc.collect()


def main():
    base = Path(__file__).resolve().parents[3] / "Data" / "networks" / "werkschool"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "network_properties.csv"
    
    # # Remove old CSV if it exists
    # if csv_path.exists():
    #     csv_path.unlink()
    
    # write_header = True
    # total_networks = 0

    # # Process single-layer werkschool folders
    # for subfolder in sorted(base.iterdir()):
    #     if not subfolder.is_dir():
    #         continue
        
    #     if "scale=0.1" not in str(subfolder):
    #         continue
        
    #     # Check if this folder directly contains pkl files
    #     pkl_files = list(subfolder.glob("*.pkl"))
        
    #     if pkl_files:
    #         count = process_folder_iteratively(subfolder, csv_path, write_header)
    #         if count > 0:
    #             write_header = False
    #             total_networks += count

    #     # Process multiplex folders
    #     for char_folder in sorted(subfolder.iterdir()):
    #         if not char_folder.is_dir():
    #             continue
            
    #         if process_multiplex_network(char_folder, csv_path, write_header):
    #             write_header = False
    #             total_networks += 1

    # if total_networks == 0:
    #     print("No networks found!")
    #     return

    # print(f"\nProcessed {total_networks} networks total")
    # print(f"Results saved to {csv_path}")
    
    # Now create plots from the saved data
    print("\nGenerating plots...")
    create_plots(csv_path, output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parents[1])
    main()