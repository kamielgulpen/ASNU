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


def run_model_sample(params, pops_path, links_path, scale=0.05):
    """Run network generation with given parameters and return metrics."""

    # Extract parameters
    preferential_attachment = params[0]
    reciprocity = params[1]
    transitivity = params[2]
    number_of_communities = int(params[3])

    # Generate network in temporary directory
    temp_path = "temp_network_sa"

    try:
        G = generate(
            pops_path=pops_path,
            links_path=links_path,
            preferential_attachment=preferential_attachment,
            scale=scale,
            reciprocity=reciprocity,
            transitivity=transitivity,
            number_of_communities=number_of_communities,
            base_path=temp_path
        )

        # Calculate network metrics
        import networkx as nx
        from scipy import stats

        num_nodes = G.graph.number_of_nodes()
        num_edges = G.graph.number_of_edges()

        if num_nodes > 0 and num_edges > 0:

            # Reciprocity
            reciprocity_actual = nx.reciprocity(G.graph)

            # Transitivity (global clustering coefficient)
            transitivity_actual = nx.transitivity(G.graph)

            # Average shortest path length (sample for large networks)
            if num_nodes < 1000:
                # For small networks, try to compute on largest connected component
                if nx.is_weakly_connected(G.graph):
                    avg_path_length = nx.average_shortest_path_length(G.graph)
                else:
                    # Get largest weakly connected component
                    largest_cc = max(nx.weakly_connected_components(G.graph), key=len)
                    subgraph = G.graph.subgraph(largest_cc)
                    if len(largest_cc) > 1:
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                    else:
                        avg_path_length = 0
            else:
                # For large networks, sample pairs of nodes
                sample_size = min(500, num_nodes)
                sample_nodes = np.random.choice(list(G.graph.nodes()), size=sample_size, replace=False)
                path_lengths = []
                for i in range(min(100, sample_size)):
                    source = sample_nodes[i]
                    for j in range(i+1, min(i+10, sample_size)):
                        target = sample_nodes[j]
                        try:
                            length = nx.shortest_path_length(G.graph, source, target)
                            path_lengths.append(length)
                        except nx.NetworkXNoPath:
                            pass
                avg_path_length = np.mean(path_lengths) if path_lengths else 0

            # Degree distribution skewness
            degrees = [d for n, d in G.graph.degree()]
            if len(degrees) > 1:
                degree_skewness = stats.skew(degrees)
            else:
                degree_skewness = 0

        else:
            reciprocity_actual = 0
            transitivity_actual = 0
            avg_path_length = 0
            degree_skewness = 0

        metrics = {
            'reciprocity': reciprocity_actual,
            'transitivity': transitivity_actual,
            'avg_path_length': avg_path_length,
            'degree_skewness': degree_skewness
        }

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
            'degree_skewness': 0
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
        'num_vars': 4,
        'names': ['preferential_attachment', 'reciprocity', 'transitivity', 'number_of_communities'],
        'bounds': [
            [0.0, 0.99],      # preferential_attachment
            [0.0, 1],      # reciprocity
            [0.0, 1],      # transitivity
            [1, 100]          # number_of_communities
        ]
    }

    # Define output metrics
    metric_names = ['reciprocity', 'transitivity', 'avg_path_length', 'degree_skewness']

    # Generate samples using Latin Hypercube
    param_values = latin.sample(problem, samples)
    total_samples = len(param_values)

    # Load checkpoint if resuming
    if resume_run_id:
        checkpoint = load_checkpoint(resume_run_id)
        print(checkpoint)
        outputs = checkpoint['outputs']
        start_idx = checkpoint['completed_idx']
        print(f"Resuming from sample {start_idx}/{total_samples}")
    else:
        outputs = {metric: np.zeros(total_samples) for metric in metric_names}
        start_idx = 0
        print(f"Starting fresh: {total_samples} simulations...")

    # Run analysis
    with mlflow.start_run(run_name="pawn_sensitivity") as run:
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
            mlflow.log_metric("input_transitivity", param_values[i][2], step=step)
            mlflow.log_metric("input_number_of_communities", param_values[i][3], step=step)

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
    

    # Start fresh
    pawn_analysis(pops_path, links_path, scale=0.01, save_interval=10, samples = 500)

    # To resume from a previous run:
    # pawn_analysis(pops_path, links_path, scale=0.1, save_interval=10, samples =50, resume_run_id="6ae3d9cbba9e4dde910f9c47583ce280")

    # Launch MLflow UI
    import subprocess
    import sys

    subprocess.Popen([
        sys.executable, '-m', 'mlflow', 'ui',
        '--backend-store-uri', mlflow.get_tracking_uri(),
        '--port', '5000'
    ])
