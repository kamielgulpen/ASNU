"""
LOCAL SENSITIVITY ANALYSIS (OFAT) WITH MLFLOW
==============================================
One-Factor-At-a-Time (OFAT) analysis to understand local parameter sensitivity
around a baseline operating point. Features checkpointing for resuming after interruptions.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from asnu import generate
import pickle
import os
from pathlib import Path
import shutil
from matplotlib import rcParams
import seaborn as sns
import pandas as pd


def run_model_sample(params, pops_path, links_path, scale=0.05):
    """Run network generation with given parameters and return metrics."""

    # Extract parameters
    preferential_attachment = params[0]
    reciprocity = params[1]
    transitivity = params[2]
    number_of_communities = int(params[3])

    # Generate network in temporary directory
    temp_path = "temp_network_la"

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


def save_checkpoint(results_data, completed_params):
    """Save checkpoint to MLflow."""
    checkpoint = {
        'results_data': results_data,
        'completed_params': completed_params
    }

    checkpoint_file = 'checkpoint_ofat.pkl'
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint, f)

    mlflow.log_artifact(checkpoint_file)
    os.remove(checkpoint_file)


def load_checkpoint(run_id):
    """Load checkpoint from MLflow run."""
    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(run_id, 'checkpoint_ofat.pkl')

    with open(local_path, 'rb') as f:
        return pickle.load(f)


def ofat_analysis(pops_path, links_path, scale=0.05,
                  baseline=None, n_samples_per_param=20,
                  save_interval=5, resume_run_id=None):
    """
    Perform One-Factor-At-a-Time (OFAT) local sensitivity analysis with MLflow.

    Parameters:
    -----------
    pops_path : str
        Path to population data
    links_path : str
        Path to links data
    scale : float
        Scale parameter for network generation
    baseline : dict or None
        Baseline parameter values. If None, uses midpoint of ranges.
    n_samples_per_param : int
        Number of samples to take for each parameter
    save_interval : int
        How often to save checkpoints
    resume_run_id : str or None
        MLflow run ID to resume from
    """

    # Define parameter space
    param_info = {
        'preferential_attachment': {'bounds': [0.0, 0.99], 'index': 0},
        'reciprocity': {'bounds': [0.0, 1.0], 'index': 1},
        'transitivity': {'bounds': [0.0, 1.0], 'index': 2},
        'number_of_communities': {'bounds': [1, 100], 'index': 3, 'integer': True}
    }

    # Set baseline values (midpoint if not specified)
    if baseline is None:
        baseline = {
            'preferential_attachment': 0,
            'reciprocity': 0,
            'transitivity': 0,
            'number_of_communities': 1
        }

    # Define output metrics
    metric_names = ['reciprocity', 'transitivity', 'avg_path_length', 'degree_skewness']

    # Load checkpoint if resuming
    if resume_run_id:
        checkpoint = load_checkpoint(resume_run_id)
        results_data = checkpoint['results_data']
        completed_params = checkpoint['completed_params']
        print(f"Resuming from checkpoint. Completed parameters: {completed_params}")
    else:
        results_data = {}
        completed_params = []
        print(f"Starting fresh OFAT analysis...")

    # Calculate total simulations
    total_simulations = len(param_info) * n_samples_per_param
    completed_simulations = len(completed_params) * n_samples_per_param

    # Run analysis
    with mlflow.start_run(run_name="ofat_local_sensitivity") as run:
        # Print MLflow tracking information
        print("\n" + "="*60)
        print("MLFLOW TRACKING INFO - LOCAL OFAT SENSITIVITY")
        print("="*60)
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Experiment ID: {run.info.experiment_id}")
        print(f"Run ID: {run.info.run_id}")
        print(f"Run Name: {run.info.run_name}")
        print(f"\nView experiment at: {mlflow.get_tracking_uri()}")
        print("="*60 + "\n")

        # Log configuration
        mlflow.log_param("analysis_type", "OFAT")
        mlflow.log_param("n_samples_per_param", n_samples_per_param)
        mlflow.log_param("total_simulations", total_simulations)
        mlflow.log_param("checkpoint_interval", save_interval)
        mlflow.log_param("pops_path", pops_path)
        mlflow.log_param("links_path", links_path)
        mlflow.log_param("scale", scale)
        for param_name, value in baseline.items():
            mlflow.log_param(f"baseline_{param_name}", value)
        if resume_run_id:
            mlflow.log_param("resumed_from", resume_run_id)

        # Perform OFAT analysis
        simulation_count = completed_simulations

        for param_name, info in param_info.items():

            if param_name in completed_params:
                print(f"Skipping {param_name} (already completed)")
                continue

            print(f"\n{'='*60}")
            print(f"Analyzing parameter: {param_name}")
            print(f"{'='*60}")

            # Generate parameter values to test
            bounds = info['bounds']
            if info.get('integer', False):
                param_values = np.linspace(bounds[0], bounds[1], n_samples_per_param, dtype=int)
            else:
                param_values = np.linspace(bounds[0], bounds[1], n_samples_per_param)

            # Initialize storage for this parameter
            results_data[param_name] = {
                'param_values': param_values,
                'metrics': {metric: [] for metric in metric_names}
            }

            # Run simulations for this parameter
            for i, param_value in enumerate(param_values):
                # Create parameter array with baseline values
                params = [
                    baseline['preferential_attachment'],
                    baseline['reciprocity'],
                    baseline['transitivity'],
                    baseline['number_of_communities']
                ]

                # Replace the current parameter with test value
                params[info['index']] = param_value

                # Run model
                metrics = run_model_sample(params, pops_path, links_path, scale)

                # Store results
                for metric_name in metric_names:
                    results_data[param_name]['metrics'][metric_name].append(metrics[metric_name])

                # Log to MLflow
                simulation_count += 1

                # Log all input parameters used in this simulation
                mlflow.log_metric("input_preferential_attachment", params[0], step=simulation_count)
                mlflow.log_metric("input_reciprocity", params[1], step=simulation_count)
                mlflow.log_metric("input_transitivity", params[2], step=simulation_count)
                mlflow.log_metric("input_number_of_communities", params[3], step=simulation_count)

                # Log which parameter is being varied and its value
                mlflow.log_metric(f"varied_param_{param_name}", param_value, step=simulation_count)

                # Log output metrics
                for metric_name in metric_names:
                    mlflow.log_metric(f"output_{metric_name}", metrics[metric_name], step=simulation_count)

                # Save checkpoint periodically
                if (i + 1) % save_interval == 0:
                    # Mark parameter as partially completed
                    save_checkpoint(results_data, completed_params)
                    progress = 100 * simulation_count / total_simulations
                    print(f"  Sample {i + 1}/{n_samples_per_param} - Overall progress: {progress:.1f}%")

            # Mark parameter as completed
            completed_params.append(param_name)
            save_checkpoint(results_data, completed_params)

            param_progress = 100 * len(completed_params) / len(param_info)
            overall_progress = 100 * simulation_count / total_simulations
            print(f"Completed {param_name}: {len(completed_params)}/{len(param_info)} parameters ({param_progress:.1f}%)")
            print(f"Overall progress: {simulation_count}/{total_simulations} simulations ({overall_progress:.1f}%)")

        print(f"\n{'='*60}")
        print(f"All simulations completed!")
        print(f"{'='*60}\n")

        # Calculate sensitivity metrics (normalized derivatives)
        sensitivity_results = {}

        for param_name, data in results_data.items():
            param_values = data['param_values']
            sensitivity_results[param_name] = {}

            for metric_name in metric_names:
                metric_values = np.array(data['metrics'][metric_name])

                # Calculate normalized local sensitivity (approximate derivative)
                # S = (dy/y) / (dx/x) where x is parameter, y is metric
                param_range = param_values.max() - param_values.min()
                metric_range = metric_values.max() - metric_values.min()

                if param_range > 0 and metric_range > 0:
                    # Simple normalized sensitivity: range of output / range of input
                    sensitivity = metric_range / param_range
                else:
                    sensitivity = 0

                sensitivity_results[param_name][metric_name] = sensitivity
                mlflow.log_metric(f"sensitivity_{param_name}_{metric_name}", sensitivity)

        # Create visualizations
        create_ofat_visualizations(results_data, baseline, metric_names, sensitivity_results)

        print(f"\nResults logged to MLflow")
        print(f"Run ID: {run.info.run_id}")

        return results_data, sensitivity_results


def create_ofat_visualizations(results_data, baseline, metric_names, sensitivity_results):
    """Create comprehensive OFAT visualization plots."""

    # Set publication-quality defaults
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 11
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 9
    rcParams['figure.dpi'] = 300

    os.makedirs('user-data/outputs', exist_ok=True)

    # 1. Parameter sweep plots (one subplot per parameter)
    n_params = len(results_data)
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = sns.color_palette("Set2", n_colors=n_metrics)

    for idx, (param_name, data) in enumerate(results_data.items()):
        ax = axes[idx]
        param_values = data['param_values']

        for metric_idx, metric_name in enumerate(metric_names):
            metric_values = data['metrics'][metric_name]
            ax.plot(param_values, metric_values, 'o-',
                   label=metric_name.replace('_', ' ').title(),
                   color=colors[metric_idx], alpha=0.7, linewidth=2, markersize=4)

        # Add baseline line
        baseline_value = baseline[param_name]
        ax.axvline(baseline_value, color='red', linestyle='--',
                  alpha=0.5, linewidth=1.5, label='Baseline')

        ax.set_xlabel(param_name.replace('_', ' ').title(), fontweight='bold')
        ax.set_ylabel('Metric Value', fontweight='bold')
        ax.set_title(f'OFAT: {param_name.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='best', frameon=True, framealpha=0.95, edgecolor='black')

        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('user-data/outputs/ofat_parameter_sweeps.png', dpi=300, bbox_inches='tight')
    plt.savefig('user-data/outputs/ofat_parameter_sweeps.pdf', bbox_inches='tight')
    mlflow.log_artifact('user-data/outputs/ofat_parameter_sweeps.png')
    mlflow.log_artifact('user-data/outputs/ofat_parameter_sweeps.pdf')
    plt.close()

    # 2. Sensitivity heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for heatmap
    param_names = list(results_data.keys())
    sensitivity_matrix = np.zeros((len(metric_names), len(param_names)))

    for i, metric_name in enumerate(metric_names):
        for j, param_name in enumerate(param_names):
            sensitivity_matrix[i, j] = sensitivity_results[param_name][metric_name]

    # Create heatmap
    im = ax.imshow(sensitivity_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels([p.replace('_', ' ').title() for p in param_names], rotation=45, ha='right')
    ax.set_yticklabels([m.replace('_', ' ').title() for m in metric_names])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Sensitivity (Range Ratio)', rotation=270, labelpad=20, fontweight='bold')

    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{sensitivity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Local Sensitivity Matrix (OFAT)', fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('user-data/outputs/ofat_sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('user-data/outputs/ofat_sensitivity_heatmap.pdf', bbox_inches='tight')
    mlflow.log_artifact('user-data/outputs/ofat_sensitivity_heatmap.png')
    mlflow.log_artifact('user-data/outputs/ofat_sensitivity_heatmap.pdf')
    plt.close()

    # 3. Sensitivity bar chart (similar to global analysis)
    fig, ax = plt.subplots(figsize=(12, 6))

    param_names = list(results_data.keys())
    n_params = len(param_names)
    bar_width = 0.8 / n_metrics
    x_pos = np.arange(n_params)
    colors = sns.color_palette("Set2", n_colors=n_metrics)

    for idx, metric_name in enumerate(metric_names):
        sensitivities = [sensitivity_results[param][metric_name] for param in param_names]
        offset = (idx - n_metrics/2 + 0.5) * bar_width
        ax.bar(x_pos + offset, sensitivities, bar_width,
               label=metric_name.replace("_", " ").title(),
               alpha=0.85, color=colors[idx],
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Parameters', fontweight='bold')
    ax.set_ylabel('Local Sensitivity', fontweight='bold')
    ax.set_title('Local Sensitivity Analysis (OFAT): Parameter Impact on Metrics',
                fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in param_names],
                       rotation=45, ha='right')

    # Grid and spines
    ax.grid(axis='y', alpha=0.25, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.legend(loc='upper right', frameon=True, fancybox=False,
             shadow=False, framealpha=0.95, edgecolor='black')

    plt.tight_layout()
    plt.savefig('user-data/outputs/ofat_sensitivity_bars.png', dpi=300, bbox_inches='tight')
    plt.savefig('user-data/outputs/ofat_sensitivity_bars.pdf', bbox_inches='tight')
    mlflow.log_artifact('user-data/outputs/ofat_sensitivity_bars.png')
    mlflow.log_artifact('user-data/outputs/ofat_sensitivity_bars.pdf')
    plt.close()

    print("\nVisualization files created:")
    print("  - ofat_parameter_sweeps.png/pdf")
    print("  - ofat_sensitivity_heatmap.png/pdf")
    print("  - ofat_sensitivity_bars.png/pdf")


if __name__ == '__main__':
    mlflow.set_experiment("local_network_sensitivity")
    mlflow.set_tracking_uri("./mlruns")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    # Specify your data paths
    pops_path = 'Data/tab_n_(with oplniv).csv'
    links_path = 'Data/tab_werkschool.csv'

    # Define baseline operating point
    baseline = {
        'preferential_attachment': 0,
        'reciprocity': 0,
        'transitivity': 0,
        'number_of_communities': 1
    }

    # Start fresh OFAT analysis
    results, sensitivities = ofat_analysis(
        pops_path=pops_path,
        links_path=links_path,
        scale=0.01,
        baseline=baseline,
        n_samples_per_param=20,
        save_interval=5
    )

    # To resume from a previous run:
    # results, sensitivities = ofat_analysis(
    #     pops_path=pops_path,
    #     links_path=links_path,
    #     scale=0.01,
    #     baseline=baseline,
    #     n_samples_per_param=20,
    #     save_interval=5,
    #     resume_run_id="your_run_id_here"
    # )

    # Launch MLflow UI
    import subprocess
    import sys

    subprocess.Popen([
        sys.executable, '-m', 'mlflow', 'ui',
        '--backend-store-uri', mlflow.get_tracking_uri(),
        '--port', '5000'
    ])
