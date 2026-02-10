"""
PAWN SENSITIVITY ANALYSIS WITH MLFLOW
======================================
Analyzes how network generation parameters affect network properties
Features checkpointing for resuming after interruptions
"""

import numpy as np
import mlflow
import mlflow.sklearn
from SALib.sample import latin
from SALib.analyze import pawn
import matplotlib.pyplot as plt
from asnu import generate
import pickle
import os
from pathlib import Path
import shutil
from matplotlib import rcParams
import seaborn as sns
import igraph as ig
import networkx as nx
import numpy as np
from scipy import stats
import warnings

def networkx_to_igraph(G_nx):
    """
    Convert a NetworkX graph to igraph.
    
    Parameters:
    -----------
    G_nx : networkx.Graph or networkx.DiGraph
        NetworkX graph to convert
        
    Returns:
    --------
    igraph.Graph : Converted igraph graph
    """
    # Check if directed
    directed = G_nx.is_directed()
    
    # Create igraph graph
    G_ig = ig.Graph(directed=directed)
    
    # Add vertices
    nodes = list(G_nx.nodes())
    G_ig.add_vertices(len(nodes))
    
    # Create mapping from node labels to indices
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # Store original node labels as vertex attribute
    G_ig.vs['name'] = nodes
    
    # Copy node attributes
    for attr in list(G_nx.nodes[nodes[0]].keys()) if nodes else []:
        G_ig.vs[attr] = [G_nx.nodes[node].get(attr) for node in nodes]
    
    # Add edges
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in G_nx.edges()]
    G_ig.add_edges(edges)
    
    # Copy edge attributes
    if G_nx.edges():
        edge_attrs = list(G_nx.edges[list(G_nx.edges())[0]].keys())
        for attr in edge_attrs:
            G_ig.es[attr] = [G_nx.edges[edge].get(attr) for edge in G_nx.edges()]
    
    return G_ig


def calculate_network_metrics(G, is_networkx=False):
    """
    Calculate network metrics using igraph.
    Automatically converts from NetworkX if needed.
    
    Parameters:
    -----------
    G : igraph.Graph or networkx.Graph
        The network graph
    is_networkx : bool, optional
        If True, treats G as NetworkX graph and converts it
        If False (default), treats G as igraph graph
        
    Returns:
    --------
    dict : Dictionary of calculated metrics
    """
    # Convert from NetworkX if needed
    if is_networkx or isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        print("Converting NetworkX graph to igraph...")
        G_ig = networkx_to_igraph(G)
    else:
        G_ig = G
    
    num_nodes = G_ig.vcount()
    print(f"Calculating metrics for graph with {num_nodes} nodes and {G_ig.ecount()} edges...")
    
    # Reciprocity
    print("Calculating reciprocity...")
    reciprocity = G_ig.reciprocity()
    
    # Transitivity
    print("Calculating transitivity...")
    transitivity = G_ig.transitivity_undirected()
    
    # Average shortest path length
    print("Calculating average path length...")
    if num_nodes < 1000:
        if G_ig.is_connected(mode="weak"):
            avg_path_length = G_ig.average_path_length(directed=True)
        else:
            components = G_ig.connected_components(mode="weak")
            largest_cc_idx = np.argmax(components.sizes())
            largest_cc = components.subgraph(largest_cc_idx)
            avg_path_length = (largest_cc.average_path_length(directed=True) 
                             if largest_cc.vcount() > 1 else 0)
    else:
        # Sample for large networks
        sample_size = min(500, num_nodes)
        sample_nodes = np.random.choice(range(num_nodes), 
                                       size=sample_size, replace=False)
        path_lengths = []
        
        # Suppress the "Couldn't reach some vertices" warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            for i in range(min(100, sample_size)):
                source = int(sample_nodes[i])
                targets = sample_nodes[i+1:min(i+10, sample_size)]
                
                for target in targets:
                    target = int(target)
                    try:
                        # Get shortest path
                        paths = G_ig.get_shortest_paths(source, target, mode="out")
                        
                        # Check if path exists (non-empty path)
                        if paths and len(paths[0]) > 0:
                            path_lengths.append(len(paths[0]) - 1)
                    except Exception as e:
                        # Skip if any error occurs
                        continue
        
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
    
    # Degree distribution skewness
    print("Calculating degree skewness...")
    degrees = G_ig.degree(mode="in" if G_ig.is_directed() else "all")
    degree_skewness = stats.skew(degrees) if len(degrees) > 1 else 0
    
    # Degree assortativity
    print("Calculating degree assortativity...")
    try:
        degree_assortativity = G_ig.assortativity_degree(directed=G_ig.is_directed())
    except:
        degree_assortativity = 0
    
    # Modularity via community detection
    print("Calculating modularity...")
    try:
        # Try Leiden algorithm first (best quality)
        communities = G_ig.community_leiden(objective_function='modularity')
        modularity = communities.modularity
        n_communities = len(communities)
    except:
        try:
            # Fallback to Louvain
            communities = G_ig.community_multilevel()
            modularity = communities.modularity
            n_communities = len(communities)
        except:
            modularity = 0
            n_communities = 0
    
    print("Done!")
    
    return {
        'reciprocity': reciprocity,
        'transitivity': transitivity,
        'avg_path_length': avg_path_length,
        'degree_skewness': degree_skewness,
        'degree_assortativity': degree_assortativity,
        'modularity': modularity,
        'n_communities': n_communities
    }

def run_model_sample(params, pops_path, links_path, scale=0.05):
    """Run network generation with given parameters and return metrics."""

    # Extract parameters
    preferential_attachment = params[0]
    reciprocity = params[1]
    # transitivity = params[2]
    number_of_communities = int(params[2])

    # Generate network in temporary directory
    temp_path = "temp_network_sa"

    try:
        G = generate(
            pops_path=pops_path,
            links_path=links_path,
            preferential_attachment=preferential_attachment,
            pa_scope="global",
            scale=scale,
            reciprocity=reciprocity,
            transitivity=0,
            number_of_communities=number_of_communities,
            base_path=temp_path
        )

        num_nodes = G.graph.number_of_nodes()
        num_edges = G.graph.number_of_edges()

        if num_nodes > 0 and num_edges > 0:

            metrics = calculate_network_metrics(G.graph)

        # Clean up temporary network
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        return metrics

    except Exception as e:
        print(f"Error in model run: {e}")
        import traceback
        traceback.print_exc()
        # Clean up on error
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        return {
            'reciprocity': 0,
            'transitivity': 0,
            'avg_path_length': 0,
            'degree_skewness': 0,
            'degree_assortativity': 0,
            'modularity': 0
        }


def save_checkpoint(param_values, outputs, completed_idx):
    """Save checkpoint to MLflow."""
    checkpoint = {
        'param_values': param_values,
        'outputs': outputs,
        'completed_idx': completed_idx
    }

    checkpoint_file = 'checkpoint.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    mlflow.log_artifact(checkpoint_file)
    os.remove(checkpoint_file)


def load_checkpoint(run_id):
    """Load checkpoint from MLflow run."""
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, 'checkpoint.pkl')

    with open(local_path, 'rb') as f:
        return pickle.load(f)


def pawn_analysis(pops_path, links_path, scale=0.05, save_interval=50, samples = 1000, resume_run_id=None):
    """Perform PAWN sensitivity analysis with MLflow checkpointing."""

    # Define parameter space (without scale)
    problem = {
        'num_vars': 3,
        'names': [
            'preferential_attachment', 
            'reciprocity', 
            # 'transitivity', 
            'number_of_communities'
            ],
        'bounds': [
            [0.0, 0.99],   # preferential_attachment
            [0.0, 1],      # reciprocity
            # [0.0, 1],      # transitivity
            [1, 1000]       # number_of_communities
        ]
    }

    # Define output metrics
    metric_names = ['reciprocity', 'transitivity', 'avg_path_length', 'degree_skewness', 'degree_assortativity', 'modularity']

    # Load checkpoint if resuming, otherwise generate new samples
    if resume_run_id:
        checkpoint = load_checkpoint(resume_run_id)
        print(checkpoint)
        param_values = checkpoint['param_values']  # Use saved parameter values
        outputs = checkpoint['outputs']
        start_idx = checkpoint['completed_idx']
        total_samples = len(param_values)
        print(f"Resuming from sample {start_idx}/{total_samples}")
    else:
        # Generate samples using Latin Hypercube only for fresh runs
        param_values = latin.sample(problem, samples)
        total_samples = len(param_values)
        outputs = {metric: np.zeros(total_samples) for metric in metric_names}
        start_idx = 0
        print(f"Starting fresh: {total_samples} simulations...")

    # Run analysis - resume existing run or start new one
    with mlflow.start_run(run_id=resume_run_id, run_name="pawn_sensitivity") as run:
        # Print MLflow tracking information
        print("\n" + "="*60)
        print("MLFLOW TRACKING INFO")
        print("="*60)
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Run Name: {run.info.run_name}")
        print(f"\nView experiment at: {mlflow.get_tracking_uri()}")
        print("="*60 + "\n")

        mlflow.log_param("n_samples", total_samples)
        mlflow.log_param("checkpoint_interval", save_interval)
        mlflow.log_param("pops_path", pops_path)
        mlflow.log_param("links_path", links_path)
        mlflow.log_param("scale", scale)
        if resume_run_id:
            mlflow.log_param("resumed_from", resume_run_id)

        for i in range(start_idx, total_samples):
            metrics = run_model_sample(param_values[i], pops_path, links_path, scale)

            # Store all metrics
            for metric_name in metric_names:
                outputs[metric_name][i] = metrics[metric_name]

            # Log current sample metrics to MLflow in real-time
            step = i + 1

            # Log input parameters
            mlflow.log_metric("input_preferential_attachment", param_values[i][0], step=step)
            mlflow.log_metric("input_reciprocity", param_values[i][1], step=step)
            # mlflow.log_metric("input_transitivity", param_values[i][2], step=step)
            mlflow.log_metric("input_number_of_communities", param_values[i][2], step=step)

            # Log output metrics
            for metric_name in metric_names:
                mlflow.log_metric(f"sample_{metric_name}", metrics[metric_name], step=step)

            # Log running averages
            if i > 0:
                for metric_name in metric_names:
                    running_avg = np.mean(outputs[metric_name][:i+1])
                    mlflow.log_metric(f"running_avg_{metric_name}", running_avg, step=step)

            # Save checkpoint periodically
            if (i + 1) % save_interval == 0:
                save_checkpoint(param_values, outputs, i + 1)
                progress = 100 * (i + 1) / total_samples
                print(f"Progress: {i + 1}/{total_samples} ({progress:.1f}%) - Checkpoint saved")

        print(f"Completed: {total_samples}/{total_samples} (100%)")

        # Perform PAWN analysis for each metric
        print("\n=== PAWN SENSITIVITY ANALYSIS RESULTS ===\n")

        results = {}
        for metric_name in metric_names:
            print(f"\n--- {metric_name.upper().replace('_', ' ')} ---")

            try:
                Si = pawn.analyze(problem, param_values, outputs[metric_name], print_to_console=False)
                results[metric_name] = Si

                # Log sensitivity indices
                for i, param_name in enumerate(problem['names']):
                    mlflow.log_metric(f"{metric_name}_{param_name}_median", Si['median'][i])
                    mlflow.log_metric(f"{metric_name}_{param_name}_mean", Si['mean'][i])

                # Print results
                print(f"Parameter Sensitivity (Median KS statistic):")
                for i, param_name in enumerate(problem['names']):
                    print(f"  {param_name:30s}: {Si['median'][i]:.3f}")

            except Exception as e:
                print(f"Error analyzing {metric_name}: {e}")
                continue

        # Create comprehensive visualization
        n_metrics = len(results)
        if n_metrics > 0:

            # Set publication-quality defaults
            rcParams['font.family'] = 'serif'
            rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
            rcParams['font.size'] = 11
            rcParams['axes.labelsize'] = 12
            rcParams['axes.titlesize'] = 14
            rcParams['xtick.labelsize'] = 10
            rcParams['ytick.labelsize'] = 10
            rcParams['legend.fontsize'] = 10
            rcParams['figure.dpi'] = 300
            
            # Create single figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            metric_names = list(results.keys())
            param_names = problem['names']
            n_params = len(param_names)
            n_metrics = len(metric_names)
            
            # Set up grouped bar chart
            bar_width = 0.8 / n_metrics
            x_pos = np.arange(n_params)
            
            # Professional color palette
            colors = sns.color_palette("Set2", n_colors=n_metrics)
            
            # Plot bars for each metric
            for idx, (metric_name, Si) in enumerate(results.items()):
                offset = (idx - n_metrics/2 + 0.5) * bar_width
                bars = ax.bar(x_pos + offset, Si['median'], bar_width, 
                            label=metric_name.replace("_", " ").title(),
                            alpha=0.85, color=colors[idx], 
                            edgecolor='black', linewidth=0.5)
            
            # Styling
            ax.set_xlabel('Parameters', fontweight='bold')
            ax.set_ylabel('PAWN Index (Median KS)', fontweight='bold')
            ax.set_title('Sensitivity Analysis: PAWN Indices Across Metrics', 
                        fontweight='bold', pad=15)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.set_ylim([0, 1.05])
            
            # Grid and spines
            ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            
            # Legend
            ax.legend(loc='upper right', frameon=True, fancybox=False, 
                    shadow=False, framealpha=0.95, edgecolor='black')
            
            # Save in multiple formats
            plt.savefig('user-data/outputs/pawn_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
            plt.savefig('user-data/outputs/pawn_sensitivity_analysis.pdf', bbox_inches='tight')
            

            plt.tight_layout()
            os.makedirs('user-data/outputs', exist_ok=True)
            mlflow.log_artifact('user-data/outputs/pawn_sensitivity_analysis.png')
            mlflow.log_artifact('user-data/outputs/pawn_sensitivity_analysis.pdf')

        print(f"\nResults logged to MLflow")
        print(f"Run ID: {run.info.run_id}")


if __name__ == '__main__':
    mlflow.set_experiment("global_network_sensitivity")
    mlflow.set_tracking_uri("./mlruns")
    print(mlflow.get_tracking_uri())

    # Specify your data paths
    pops_path = 'Data/tab_n_(with oplniv).csv'
    links_path = 'Data/tab_werkschool.csv'
    import subprocess
    import sys

    subprocess.Popen([
        sys.executable, '-m', 'mlflow', 'ui',
        '--backend-store-uri', mlflow.get_tracking_uri(),
        '--port', '5000'
    ])

    # Start fresh
    # pawn_analysis(pops_path, links_path, scale=0.1, save_interval=10, samples = 500)

    # To resume from a previous run:
    pawn_analysis(pops_path, links_path, scale=0.1, save_interval=10, samples =500, resume_run_id="d1615b7df4d643d7b89777bd54e4e65e")

    # Launch MLflow UI

