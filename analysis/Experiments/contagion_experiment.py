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
import seaborn as sns
import pickle

sns.set_style("whitegrid")
np.random.seed(42)


class ContagionSimulator:
    """Simulates contagion spreading on networks"""
    
    def __init__(self, network, name="Network"):
        self.G = network
        self.name = name
        self.n = len(network)
        
    def simple_contagion(self, p_transmit=0.2, initial_infected=1, max_steps=30):
        """
        Simple contagion: SI model where single contact can cause infection
        
        Args:
            p_transmit: Probability of transmission per infected neighbor
            initial_infected: Number of initially infected nodes
            max_steps: Maximum simulation steps
            
        Returns:
            time_series: List of infection counts at each time step
        """
        # Initialize states: 0 = susceptible, 1 = infected
        state = np.zeros(self.n)
        
        # Random initial infections
        initial_nodes = np.random.choice(self.n, initial_infected, replace=False)
        state[initial_nodes] = 1
        
        time_series = [np.sum(state)]
        
        for step in range(max_steps):
            new_infections = []
            
            # Check each susceptible node
            for node in range(self.n):
                if state[node] == 0:  # Susceptible
                    # Count infected neighbors
                    neighbors = list(self.G.neighbors(node))
                    infected_neighbors = sum(state[n] for n in neighbors)
                    
                    if infected_neighbors > 0:
                        # Probability of infection: 1 - (1-p)^k
                        # where k is number of infected neighbors
                        p_infection = 1 - (1 - p_transmit)**infected_neighbors
                        
                        if np.random.random() < p_infection:
                            new_infections.append(node)
            
            # Update states
            for node in new_infections:
                state[node] = 1
            
            time_series.append(np.sum(state))
            
            # Stop if no change or all infected
            if  np.sum(state) == self.n:
                break
        
        return time_series
    
    def complex_contagion(self, threshold=2, threshold_type='relative', 
                         initial_infected=1, max_steps=30):
        """
        Complex contagion: Threshold model requiring multiple exposures
        
        Args:
            threshold: Number or fraction of neighbors needed for adoption
            threshold_type: 'absolute' (number) or 'fractional' (proportion)
            initial_infected: Number of initially infected nodes
            max_steps: Maximum simulation steps
            
        Returns:
            time_series: List of adoption counts at each time step
        """
        # Initialize states: 0 = not adopted, 1 = adopted
        state = np.zeros(self.n)
        
        # Random initial adopters
        initial_nodes = np.random.choice(self.n, initial_infected, replace=False)
        state[initial_nodes] = 1
        
        time_series = [np.sum(state)]
        
        for step in range(max_steps):
            new_adopters = []
            
            # Check each non-adopter
            for node in range(self.n):
                if state[node] == 0:  # Not yet adopted
                    neighbors = list(self.G.neighbors(node))
                    
                    if len(neighbors) == 0:
                        continue
                    
                    # Count adopted neighbors
                    adopted_neighbors = sum(state[n] for n in neighbors)
                    
                    # Check threshold
                    if threshold_type == 'absolute':
                        if adopted_neighbors >= threshold:
                            new_adopters.append(node)
                    else:  # fractional
                        fraction = adopted_neighbors / len(neighbors)
                        if fraction >= threshold:
                            new_adopters.append(node)
            
            # Update states
            for node in new_adopters:
                state[node] = 1
            
            time_series.append(np.sum(state))
            
            # Stop if no change or all adopted
            if np.sum(state) == self.n:
                break
        
        return time_series

def get_avg_shortest_path(G):
    is_connected = nx.is_weakly_connected if G.is_directed() else nx.is_connected
    components = nx.weakly_connected_components if G.is_directed() else nx.connected_components
    
    if is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        largest_cc = max(components(G), key=len)
        return nx.average_shortest_path_length(G.subgraph(largest_cc))
    
def create_networks(n=500, avg_degree=6):
    """
    Create different network topologies with similar average degree
    
    Args:
        n: Number of nodes
        avg_degree: Target average degree
        
    Returns:
        Dictionary of networks
    """
    networks = {}
    
    # 1. Random Network (Erdős-Rényi)
    p_er = avg_degree / (n - 1)
    networks['Random'] = nx.erdos_renyi_graph(n, p_er)
    
    # 2. Small-World Network (Watts-Strogatz)
    # High clustering, short paths
    k_ws = avg_degree  # nearest neighbors
    p_rewire = 0.1  # rewiring probability
    networks['Small-World'] = nx.watts_strogatz_graph(n, k_ws, p_rewire)
    
    # 3. Scale-Free Network (Barabási-Albert)
    # Has hubs
    m_ba = avg_degree // 2  # edges to add per new node
    networks['Scale-Free'] = nx.barabasi_albert_graph(n, m_ba)
    
    # 4. Regular Lattice (for comparison)
    # Very high clustering, long paths
    k_lattice = avg_degree
    networks['Lattice'] = nx.watts_strogatz_graph(n, k_lattice, 0)  # p=0 = regular lattice
    
    # 5. Amsterdam network
    with open('z.pkl', 'rb') as f:
        networks['Amsterdam1']  = pickle.load(f)

    # 6. Amsterdam network
    with open('a.pkl', 'rb') as f:
        networks['Amsterdam2']  = pickle.load(f)

    return networks


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
        
        print(f"{name:<15} {n_nodes:>8} {n_edges:>8} {avg_deg:>10.2f} {clustering:>12.3f} {avg_path:>10.2f}")
    
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
        
        # Simple contagion
        for _ in range(n_simulations):
            ts = sim.simple_contagion(p_transmit=0.01, initial_infected=600)
            results['simple'][name].append(ts)
        
        # Complex contagion (threshold = 2 neighbors)
        for _ in range(n_simulations):
            ts = sim.complex_contagion(threshold=0.2, threshold_type='fraction', initial_infected=600)
            results['complex'][name].append(ts)
    
    print("Simulations complete!\n")
    return results


def plot_results(results, networks):
    """
    Visualize simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Simple vs Complex Contagion: Effect of Network Topology', 
                 fontsize=16, fontweight='bold')
    
    colors = {'Random': '#e74c3c', 'Small-World': '#3498db', 
              'Scale-Free': '#2ecc71', 'Lattice': '#f39c12',
              'Amsterdam1': "#6a2e81", 'Amsterdam2': "#ef4de4"}
        
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
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Network Topologies Comparison', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    colors = {'Random': '#e74c3c', 'Small-World': '#3498db', 
              'Scale-Free': '#2ecc71', 'Lattice': '#f39c12',
              'Amsterdam1': "#6a2e81", 'Amsterdam2': "#ef4de4"}
    
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
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    return fig

def main():
    """Run the complete experiment"""
    print("\n" + "="*70)
    print("CONTAGION EXPERIMENT: SIMPLE vs COMPLEX")
    print("="*70)
    print("\nDemonstrating that network topology matters differently")
    print("for simple vs complex contagion processes.\n")
    
    # Create networks
    print("Creating network topologies...")
    networks = create_networks(n=8601, avg_degree=71)
    
    # Print properties
    print_network_properties(networks)
    
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
    plt.savefig('contagion_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to: contagion_results.png\n")

    # Visualize network structures
    print("Visualizing network topologies...")
    fig_networks = visualize_networks(networks)
    plt.savefig('network_topologies.png', dpi=300, bbox_inches='tight')
    print("Network visualizations saved to: network_topologies.png")
    
    return networks, results, fig


if __name__ == "__main__":
    networks, results, fig = main()
    plt.show()