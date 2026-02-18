import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from scipy import stats as sp_stats
from itertools import combinations
from scipy.cluster.hierarchy import dendrogram, linkage

from contagion_experiment import (
    ContagionSimulator, load_networks, assign_colors,
    print_network_properties,
)


def sweep_contested(networks, fractions=None, n_simulations=20, max_steps=50):
    """
    Fractional threshold sweep (contested contagion).
    τ = fraction of neighbors required.
    """

    if fractions is None:
        fractions = np.linspace(0.05, 0.15, 5)

    results = {}
    ratios = {}
    for name, G in networks.items():
        if not name.startswith("Random"):

            # Method 1: Normalize by product of group sizes (expected links under random mixing)
            df = pd.read_csv(f"Data/aggregated/tab_werkschool_{name}.csv")
            df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")

            ratio = df_n.n.max()/df_n.n.sum()
            ratios[name] = ratio
        n = len(G)
        sim = ContagionSimulator(G, name)
        finals = np.zeros((len(fractions), n_simulations))
        finals = {}
        for i, tau in enumerate(fractions):
            ts_list = sim.complex_contagion(
                threshold=tau, threshold_type='fractional',
                seeding='focal_neighbors',
                max_steps=max_steps, 
                n_simulations=n_simulations,
                initial_infected=n/30)
            # finals[i] = np.array([ts[-1] / n * 100 for ts in ts_list])
            finals[i] = np.median([i[-1] for i in ts_list])
        results[name] = finals
        print(f"  Contested sweep done: {name}")
 
    return results, ratios


def _run_sweep_on_folder(network_folder, n_simulations=20):
    """Run the full parameter sweep on a single folder containing .pkl networks."""
    print(f"\nLoading networks from: {network_folder}\n")

    # Output directory
    out_dir = Path(network_folder) / 'diffusion_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    networks = load_networks(network_folder, add_random=True)

    if not networks:
        print(f"  No networks found in {network_folder}, skipping.")
        return None


    # --- 2. Contested (fractional) threshold sweep ---
    print("\n[2/4] Contested threshold sweep...")
    frac_thresholds = np.linspace(0.05, 0.30, 5)
    contested_results, ratios = sweep_contested(
        networks, fractions=frac_thresholds, n_simulations=n_simulations)
    return contested_results, ratios


def main(network_folder='Data/networks/werkschool/scale=0.01_comms=1_recip=1_trans=0_pa=0_bridge=0',
         n_simulations=20):
    """
    Run parameter sweeps on networks in a folder.

    If the folder contains .pkl files directly, runs the sweep once.
    If the folder contains subfolders (multiplex structure), runs the
    sweep separately for each characteristic group.
    """
    print("\n" + "="*70)
    print("CONTAGION PARAMETER SWEEP (Centola-style)")
    print("="*70)
    i = 0
    results = {}
    keys = []
    n_c = np.linspace(1,100, 10)
    p_a = np.linspace(0,0.99,10)

    np.random.shuffle(n_c)
    np.random.shuffle(p_a)

    indexes = [(i,j) for j in range(10) for i in range(10)]
    print(indexes)
    for number_of_communities in n_c:
        for preferential_attachment in p_a:
            print(number_of_communities, preferential_attachment)
            # number_of_communities, preferential_attachment = number_of_communities, preferential_attachment
            i+=1 
            
            keys.append((number_of_communities, preferential_attachment))
            print(number_of_communities, preferential_attachment)
            network_folder = f"Data/networks/werkschool/scale=0.01_comms={number_of_communities}_recip=1_trans=0_pa={preferential_attachment}_bridge=0.2"
            folder = Path(network_folder)

            print(network_folder)

            # Check if this folder has .pkl files directly or subfolders
            pkl_files = list(folder.glob('*.pkl'))
            subfolders = sorted([d for d in folder.iterdir() if d.is_dir()
                                and d.name not in ('diffusion_analysis', 'node_distribution')])
            print(folder)
            if pkl_files:
                # Direct pkl files — single sweep
                results[(number_of_communities, preferential_attachment)], ratios = _run_sweep_on_folder(network_folder, n_simulations)
            elif subfolders:
                # Multiplex structure — iterate over characteristic subfolders
                print(f"\nFound {len(subfolders)} characteristic groups in: {network_folder}")
                all_results = {}
                for subfolder in subfolders:
                    print(f"\n{'='*70}")
                    print(f"CHARACTERISTIC GROUP: {subfolder.name}")
                    print(f"{'='*70}")
                    result, ratios = _run_sweep_on_folder(subfolder, n_simulations)
                    if result is not None:
                        all_results[subfolder.name] = result
                return all_results
            else:
                print(f"No .pkl files or subfolders found in {network_folder}")
                return None
            
            
            if i == 10:
                # Prepare data for plotting with different thresholds
                plot_data = []
                frac_thresholds = np.linspace(0.05, 0.30, 5)
                for name in ratios.keys():
                    ratio_val = ratios[name]
                    
                    # Loop through all threshold values
                    for threshold_idx, threshold_val in enumerate(frac_thresholds):
                        results_vals = []
                        
                        # Collect results across all parameter combinations for this threshold
                        for key in keys:
                            results_vals.append(results[key][name][threshold_idx])
                        
                        # Calculate variance for this threshold
                        variance = np.var(results_vals)
                        
                        # Add to plot data
                        plot_data.append({
                            'ratio': ratio_val,
                            'variance': variance,
                            'threshold': threshold_val,
                            'network': name
                        })
                
                # Convert to DataFrame for seaborn
                plot_df = pd.DataFrame(plot_data)
                
                # Create the plot with lines and markers
                plt.figure(figsize=(10, 6))
    
                # sns.lineplot(data=plot_df, x='ratio', y='variance', hue='threshold', 
                #             palette='viridis', linewidth=2, alpha=0.5, legend=False)
                sns.scatterplot(data=plot_df, x='ratio', y='variance', hue='threshold', 
                                palette='viridis', s=100, alpha=0.8)
                plt.xlabel('Ratio', fontsize=12)
                plt.ylabel('Variance', fontsize=12)
                plt.title('Variance vs Ratio for Different Threshold Values', fontsize=14)
                plt.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig("x.png")
                plt.show()
                exit()

    return results

if __name__ == "__main__":
   results =  main()

