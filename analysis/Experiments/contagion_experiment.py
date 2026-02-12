"""
Experiment: Simple vs Complex Contagion on Different Network Topologies
=========================================================================

Demonstrates that topology matters differently for simple vs complex contagion,
even when networks have similar degree distributions.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import seaborn as sns
import pickle
from scipy import sparse
import math

sns.set_style("whitegrid")
np.random.seed(42)


class ContagionSimulator:
    """Simulates contagion spreading on networks using vectorized sparse operations"""

    def __init__(self, network, name="Network"):
        self.G = network
        self.name = name
        self.n = len(network)
        # Pre-compute sparse adjacency matrix (CSR for fast matrix-vector multiply)
        # Symmetrize directed graphs so contagion uses undirected connections
        adj = nx.to_scipy_sparse_array(network, format='csr', dtype=np.float64)
        # if network.is_directed():
        #     adj = ((adj + adj.T) > 0).astype(np.float64)
        self.adj = adj
        self.degree = np.array(self.adj.sum(axis=1)).flatten()

    def simple_contagion(self, p_transmit=0.2, initial_infected=1, max_steps=30, n_simulations=1):
        """
        Vectorized SI model: runs n_simulations in parallel via sparse matmul.

        Returns:
            List of time_series lists (one per simulation)
        """
        # State matrix: (n_nodes, n_sims) — 1=infected, 0=susceptible
        state = np.zeros((self.n, n_simulations), dtype=np.float64)

        for sim in range(n_simulations):
            initial_nodes = np.random.choice(self.n, initial_infected, replace=False)
            state[initial_nodes, sim] = 1.0

        totals = np.sum(state, axis=0)
        time_series = [totals.copy()]

        for step in range(max_steps):
            # Sparse matmul: count infected neighbors for all nodes × all sims
            infected_counts = self.adj @ state  # (n, n_sims)

            susceptible = (state == 0)

            # p_infection = 1 - (1-p)^k
            p_infection = 1.0 - (1.0 - p_transmit) ** infected_counts

            rolls = np.random.random(state.shape)
            new_infections = susceptible & (rolls < p_infection)
            state[new_infections] = 1.0

            prev_totals = totals
            totals = np.sum(state, axis=0)
            time_series.append(totals.copy())

            if np.all(totals == self.n) or np.all(totals == prev_totals):
                break

        # Convert to list-of-lists format
        return [[int(time_series[t][sim]) for t in range(len(time_series))]
                for sim in range(n_simulations)]

    def _seed_state(self, state, n_simulations, seeding, initial_infected):
        """Initialize infection state based on seeding strategy."""
        if seeding == 'focal_neighbors':
            for sim in range(n_simulations):
                focal = np.random.randint(self.n)
                state[focal, sim] = 1.0
                # Get neighbor indices directly from CSR structure
                neighbors = self.adj.indices[self.adj.indptr[focal]:self.adj.indptr[focal + 1]]
                state[neighbors, sim] = 1.0
        else:  # 'random'
            for sim in range(n_simulations):
                nodes = np.random.choice(self.n, initial_infected, replace=False)
                state[nodes, sim] = 1.0

    def complex_contagion(self, threshold=2, threshold_type='absolute',
                         initial_infected=1, max_steps=30, n_simulations=1,
                         seeding='random'):
        """
        Deterministic threshold model.

        Args:
            threshold: Absolute count or fraction of neighbors needed
            threshold_type: 'absolute' or 'fractional'
            initial_infected: Number of seed nodes (for seeding='random')
            max_steps: Maximum simulation steps
            n_simulations: Number of parallel runs
            seeding: 'random' (N random nodes) or 'focal_neighbors'
                     (a random focal node + all its neighbors)
        """
        state = np.zeros((self.n, n_simulations), dtype=np.float64)
        self._seed_state(state, n_simulations, seeding, initial_infected)

        totals = np.sum(state, axis=0)
        time_series = [totals.copy()]

        for step in range(max_steps):
            infected_counts = self.adj @ state  # (n, n_sims)
            susceptible = (state == 0)

            if threshold_type == 'absolute':
                meets_threshold = infected_counts >= threshold
            else:  # fractional (contested)
                with np.errstate(divide='ignore', invalid='ignore'):
                    fraction = infected_counts / self.degree[:, np.newaxis]
                fraction = np.where(self.degree[:, np.newaxis] == 0, 0, fraction)
                meets_threshold = fraction >= threshold

            new_adopters = susceptible & meets_threshold
            state[new_adopters] = 1.0

            prev_totals = totals
            totals = np.sum(state, axis=0)
            time_series.append(totals.copy())

            if np.all(totals == self.n) or np.all(totals == prev_totals):
                break

        return [[int(time_series[t][sim]) for t in range(len(time_series))]
                for sim in range(n_simulations)]

    def hybrid_contagion(self, base_threshold=2, vulnerable_threshold=1,
                        vulnerable_fraction=0.1, threshold_type='absolute',
                        max_steps=30, n_simulations=1, seeding='focal_neighbors'):
        """
        Hybrid contagion: most nodes have base_threshold, a fraction have
        vulnerable_threshold (lower). Models heterogeneous adoption thresholds.

        Args:
            base_threshold: Threshold for normal nodes
            vulnerable_threshold: Lower threshold for vulnerable nodes
            vulnerable_fraction: Fraction of nodes that are vulnerable
            threshold_type: 'absolute' or 'fractional'
            seeding: 'random' or 'focal_neighbors'
        """
        state = np.zeros((self.n, n_simulations), dtype=np.float64)
        self._seed_state(state, n_simulations, seeding, 1)

        # Per-node thresholds (same across simulations, different vulnerable sets)
        node_thresholds = np.full(self.n, base_threshold, dtype=np.float64)
        n_vulnerable = int(vulnerable_fraction * self.n)
        if n_vulnerable > 0:
            vulnerable_nodes = np.random.choice(self.n, n_vulnerable, replace=False)
            node_thresholds[vulnerable_nodes] = vulnerable_threshold

        totals = np.sum(state, axis=0)
        time_series = [totals.copy()]

        for step in range(max_steps):
            infected_counts = self.adj @ state
            susceptible = (state == 0)

            if threshold_type == 'absolute':
                meets_threshold = infected_counts >= node_thresholds[:, np.newaxis]
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    fraction = infected_counts / self.degree[:, np.newaxis]
                fraction = np.where(self.degree[:, np.newaxis] == 0, 0, fraction)
                meets_threshold = fraction >= node_thresholds[:, np.newaxis]

            new_adopters = susceptible & meets_threshold
            state[new_adopters] = 1.0

            prev_totals = totals
            totals = np.sum(state, axis=0)
            time_series.append(totals.copy())

            if np.all(totals == self.n) or np.all(totals == prev_totals):
                break

        return [[int(time_series[t][sim]) for t in range(len(time_series))]
                for sim in range(n_simulations)]

def get_avg_shortest_path(G):
    is_connected = nx.is_weakly_connected if G.is_directed() else nx.is_connected
    components = nx.weakly_connected_components if G.is_directed() else nx.connected_components

    if is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        largest_cc = max(components(G), key=len)
        return nx.average_shortest_path_length(G.subgraph(largest_cc))


def load_networks(folder, add_random=True, multiplex = False):
    """
    Load all .pkl networks from a folder and optionally add a Random baseline.

    Handles both single-layer folders (each pkl is a graph) and multiplex
    folders (skips multiplex.pkl which is a dict of layers).

    Args:
        folder: Path to folder containing .pkl network files
            (e.g. 'Data/networks/scale=0.1_comms=1_recip=1_trans=0_pa=0')
        add_random: If True, add an Erdős-Rényi graph matched to the first
            network's node count and average degree

    Returns:
        Dictionary of {name: nx.Graph}
    """
    folder = Path(folder)
    networks = {}
    for pkl_file in sorted(folder.glob('*.pkl')):
        name = pkl_file.stem  # filename without extension
        if multiplex and name != "multiplex": continue
        with open(pkl_file, 'rb') as f:
            obj = pickle.load(f)
        # Skip non-graph objects (e.g. multiplex.pkl is a dict of layers)
        if not isinstance(obj, nx.Graph):
            print(f"  Skipped {name} (not a graph)")
            continue
        networks[name] = obj
        print(f"  Loaded {name} ({networks[name].number_of_nodes()} nodes, "
              f"{networks[name].number_of_edges()} edges)")

    if add_random and networks:
        ref = next(iter(networks.values()))
        n = ref.number_of_nodes()
        avg_degree = 2 * ref.number_of_edges() / n
        if ref.is_directed():
            avg_degree /= 2  # directed edges are counted once each
        p_er = avg_degree / (n - 1)
        networks['Random (ER)'] = nx.erdos_renyi_graph(n, p_er)
        print(f"  Added Random (ER) baseline ({n} nodes, avg_degree={avg_degree:.1f})")

    return networks


def assign_colors(names):
    """Auto-assign distinct colors for an arbitrary number of networks."""
    cmap = plt.cm.get_cmap('tab10' if len(names) <= 10 else 'tab20')
    return {name: cmap(i / max(len(names) - 1, 1)) for i, name in enumerate(names)}


def print_network_properties(networks):
    """Print key properties of each network"""
    print("\n" + "="*70)
    print("NETWORK PROPERTIES")
    print("="*70)
    print(f"{'Network':<15} {'Nodes':>8} {'Edges':>8} {'Avg Deg':>10} {'Clustering':>12} {'Avg Path':>10}")
    print("-"*70)
    
    for name, G in networks.items():
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        avg_deg = 2 * n_edges / n_nodes
        clustering = nx.average_clustering(G)

        avg_path = 5

        # Degree distribution diagnostics
        if G.is_directed():
            degrees = [d for _, d in G.out_degree()]
        else:
            degrees = [d for _, d in G.degree()]
        degrees = np.array(degrees)
        low_deg = np.sum(degrees <= 5)


        print(f"{name:<15} {n_nodes:>8} {n_edges:>8} {avg_deg:>10.2f} {clustering:>12.3f} {avg_path:>10.2f}")
        print(f"  degree: min={degrees.min()}, median={int(np.median(degrees))}, max={degrees.max()}, nodes with deg<=5: {low_deg} ({100*low_deg/n_nodes:.1f}%)")

    print("="*70 + "\n")


def run_experiment(networks, n_simulations=50):
    """
    Run contagion simulations on all networks
    
    Args:
        networks: Dictionary of network topologies
        n_simulations: Number of simulation runs per configuration
        
    Returns:
        results: Dictionary with simulation results
    """
    results = {
        'simple': defaultdict(list),
        'complex': defaultdict(list)
    }

    print("Running simulations...")

    for name, G in networks.items():
        print(f"  Processing {name} network...")
        sim = ContagionSimulator(G, name)

        # Simple contagion — all simulations in one batched call
        results['simple'][name] = sim.simple_contagion(
            p_transmit=0.01, initial_infected=800, n_simulations=n_simulations)

        # Complex contagion — all simulations in one batched call
        results['complex'][name] = sim.complex_contagion(
            threshold=0.25, threshold_type='fraction', initial_infected=800,
            n_simulations=n_simulations)

    print("Simulations complete!\n")
    return results


def plot_results(results, networks):
    """
    Visualize simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simple vs Complex Contagion: Effect of Network Topology',
                 fontsize=16, fontweight='bold')

    colors = assign_colors(list(networks.keys()))

    # Get network size
    n = list(networks.values())[0].number_of_nodes()
    
    # 1. Simple Contagion - Time Series
    ax = axes[0, 0]
    for name in results['simple'].keys():
        time_series_list = results['simple'][name]
        
        # Get max length
        max_len = max(len(ts) for ts in time_series_list)
        
        # Pad and average
        padded = []
        for ts in time_series_list:
            padded_ts = ts + [ts[-1]] * (max_len - len(ts))
            padded.append(padded_ts)
        
        mean_ts = np.mean(padded, axis=0)
        std_ts = np.std(padded, axis=0)
        
        ax.plot(mean_ts / n * 100, label=name, color=colors[name], linewidth=2)
        ax.fill_between(range(len(mean_ts)), 
                        (mean_ts - std_ts) / n * 100,
                        (mean_ts + std_ts) / n * 100,
                        alpha=0.2, color=colors[name])
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('% Infected', fontsize=11)
    ax.set_title('Simple Contagion (Single Contact Spreads)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Complex Contagion - Time Series
    ax = axes[0, 1]
    for name in results['complex'].keys():
        time_series_list = results['complex'][name]
        print(time_series_list)
        max_len = max(len(ts) for ts in time_series_list)
        print(max_len)
        padded = []
        for ts in time_series_list:
            padded_ts = ts + [ts[-1]] * (max_len - len(ts))
            padded.append(padded_ts)
        
        mean_ts = np.mean(padded, axis=0)
        std_ts = np.std(padded, axis=0)
        
        ax.plot(mean_ts / n * 100, label=name, color=colors[name], linewidth=2)
        ax.fill_between(range(len(mean_ts)), 
                        (mean_ts - std_ts) / n * 100,
                        (mean_ts + std_ts) / n * 100,
                        alpha=0.2, color=colors[name])
    
    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('% Adopted', fontsize=11)
    ax.set_title('Complex Contagion (Requires ≥2 Contacts)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Final Adoption Rates - Bar Chart
    ax = axes[1, 0]
    
    final_rates = {}
    for contagion_type in ['simple', 'complex']:
        final_rates[contagion_type] = {}
        for name in results[contagion_type].keys():
            finals = [ts[-1] / n * 100 for ts in results[contagion_type][name]]
            final_rates[contagion_type][name] = (np.mean(finals), np.std(finals))
    
    x = np.arange(len(networks))
    width = 0.35
    
    simple_means = [final_rates['simple'][name][0] for name in networks.keys()]
    simple_stds = [final_rates['simple'][name][1] for name in networks.keys()]
    complex_means = [final_rates['complex'][name][0] for name in networks.keys()]
    complex_stds = [final_rates['complex'][name][1] for name in networks.keys()]
    
    ax.bar(x - width/2, simple_means, width, label='Simple', 
           yerr=simple_stds, capsize=5, color='#3498db', alpha=0.8)
    ax.bar(x + width/2, complex_means, width, label='Complex', 
           yerr=complex_stds, capsize=5, color='#e74c3c', alpha=0.8)
    
    ax.set_ylabel('Final % Infected/Adopted', fontsize=11)
    ax.set_title('Final Spread: Topology Matters More for Complex Contagion', 
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(networks.keys(), rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Clustering vs Final Adoption (Scatter)
    ax = axes[1, 1]
    
    clustering_coeffs = {name: nx.average_clustering(G) for name, G in networks.items()}
    
    simple_finals = [final_rates['simple'][name][0] for name in networks.keys()]
    complex_finals = [final_rates['complex'][name][0] for name in networks.keys()]
    clustering_vals = [clustering_coeffs[name] for name in networks.keys()]
    
    for i, name in enumerate(networks.keys()):
        ax.scatter(clustering_vals[i], simple_finals[i], 
                  s=200, marker='o', color=colors[name], 
                  label=f'{name} (Simple)', alpha=0.6, edgecolors='black', linewidth=1.5)
        ax.scatter(clustering_vals[i], complex_finals[i], 
                  s=200, marker='s', color=colors[name], 
                  label=f'{name} (Complex)', alpha=0.9, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('Clustering Coefficient', fontsize=11)
    ax.set_ylabel('Final % Spread', fontsize=11)
    ax.set_title('Clustering Matters for Complex Contagion', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend with markers explanation
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor='gray', markersize=10, label='Simple Contagion'),
                      Line2D([0], [0], marker='s', color='w', 
                             markerfacecolor='gray', markersize=10, label='Complex Contagion')]
    ax.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    return fig

def visualize_networks(networks):
    """
    Visualize the structure of different network topologies
    """
    n_nets = len(networks)
    ncols = min(n_nets, 3)
    nrows = math.ceil(n_nets / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    fig.suptitle('Network Topologies Comparison', fontsize=16, fontweight='bold')

    axes = np.array(axes).flatten()
    colors = assign_colors(list(networks.keys()))

    for idx, (name, G) in enumerate(networks.items()):
        ax = axes[idx]
        
        # Sample nodes for large networks
        if G.number_of_nodes() > 200:
            nodes_to_draw = list(G.nodes())[:200]
            G_sample = G.subgraph(nodes_to_draw)
        else:
            G_sample = G
        
        # Choose layout based on network type
        if name == 'Lattice':
            pos = nx.circular_layout(G_sample)
        elif name == 'Scale-Free':
            pos = nx.spring_layout(G_sample, k=0.3, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(G_sample, k=0.5, iterations=50, seed=42)
        
        # Draw network
        node_size = 30 if G.number_of_nodes() > 200 else 50
        nx.draw_networkx_nodes(G_sample, pos, node_color=colors[name], 
                              node_size=node_size, alpha=0.7, ax=ax)
        nx.draw_networkx_edges(G_sample, pos, alpha=0.3, width=0.5, ax=ax)
        
        ax.set_title(f'{name}\n({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide extra subplots
    for idx in range(len(networks), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig

def plot_degree_distributions(networks, output_dir):
    """Save individual degree distribution plots per network."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, G in networks.items():
        if G.is_directed():
            degrees = [d for _, d in G.out_degree()]
        else:
            degrees = [d for _, d in G.degree()]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(degrees, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Degree Distribution — {name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        filepath = output_dir / f'{safe_name}.png'
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved degree distribution: {filepath}")


def _run_experiment_on_folder(network_folder, multiplex=False):
    """Run the contagion experiment on a single folder containing .pkl networks."""
    print(f"\nLoading networks from: {network_folder}\n")

    # Output directories within this folder
    diffusion_dir = Path(network_folder) / 'diffusion_analysis'
    node_dist_dir = Path(network_folder) / 'node_distribution'
    diffusion_dir.mkdir(parents=True, exist_ok=True)
    node_dist_dir.mkdir(parents=True, exist_ok=True)
    print(network_folder)

    networks = load_networks(network_folder, add_random=True, multiplex = multiplex)

    if not networks:
        print(f"  No networks found in {network_folder}, skipping.")
        return None, None, None

    # Print properties
    print_network_properties(networks)

    # Save degree distributions
    print("Saving degree distributions...")
    plot_degree_distributions(networks, node_dist_dir)

    # Run simulations
    results = run_experiment(networks, n_simulations=20)

    # Analyze results
    print("="*70)
    print("KEY FINDINGS")
    print("="*70)

    n = list(networks.values())[0].number_of_nodes()

    for contagion_type in ['simple', 'complex']:
        print(f"\n{contagion_type.upper()} CONTAGION:")
        for name in results[contagion_type].keys():
            finals = [ts[-1] / n * 100 for ts in results[contagion_type][name]]
            mean_final = np.mean(finals)
            std_final = np.std(finals)
            print(f"  {name:<15}: {mean_final:>6.1f}% ± {std_final:>5.1f}%")

    # Visualize
    fig = plot_results(results, networks)
    fig.savefig(diffusion_dir / 'contagion_results.png', dpi=300, bbox_inches='tight')
    print(f"Results saved to: {diffusion_dir / 'contagion_results.png'}\n")

    # Visualize network structures
    print("Visualizing network topologies...")
    fig_networks = visualize_networks(networks)
    fig_networks.savefig(diffusion_dir / 'network_topologies.png', dpi=300, bbox_inches='tight')
    print(f"Network visualizations saved to: {diffusion_dir / 'network_topologies.png'}")

    return networks, results, fig


def main(network_folder='Data/networks/multiplex_scale=0.01'):
    """
    Run the complete experiment on networks in a folder.

    If the folder contains .pkl files directly, runs the experiment once.
    If the folder contains subfolders (multiplex structure), runs the
    experiment separately for each characteristic group.
    """
    print("\n" + "="*70)
    print("CONTAGION EXPERIMENT: SIMPLE vs COMPLEX")
    print("="*70)

    folder = Path(network_folder)

    # Check if this folder has .pkl files directly or subfolders
    pkl_files = list(folder.glob('*.pkl'))
    subfolders = sorted([d for d in folder.iterdir() if d.is_dir()
                         and d.name not in ('diffusion_analysis', 'node_distribution')])

    if pkl_files:
        # Direct pkl files — single experiment
        return _run_experiment_on_folder(network_folder)
    elif subfolders:
        # Multiplex structure — iterate over characteristic subfolders
        print(f"\nFound {len(subfolders)} characteristic groups in: {network_folder}")
        all_results = {}
        for subfolder in subfolders:
            print(subfolder)
            print(f"\n{'='*70}")
            print(f"CHARACTERISTIC GROUP: {subfolder.name}")
            print(f"{'='*70}")
            networks, results, fig = _run_experiment_on_folder(subfolder, multiplex = True)
            if networks is not None:
                all_results[subfolder.name] = (networks, results, fig)
        return all_results
    else:
        print(f"No .pkl files or subfolders found in {network_folder}")
        return None


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else None
    kwargs = {'network_folder': folder} if folder else {}
    main(**kwargs)
    plt.show()