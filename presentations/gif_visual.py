import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np
from PIL import Image
import io

# Configuration
n = 20
seed = 42

# Color palette - matching original style
color_susceptible = '#E0E0E0'  # Gray for non-activated nodes
color_activated = '#4CAF50'    # Green (same as original)
color_seed = '#2E7D32'         # Dark green for seed nodes
border_color = '#4A1C1C'       # Same border as original

def create_network_for_threshold_demo():
    """
    Create a network similar to the original style
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Create a network with a core-periphery structure
    # Similar to the spring layout networks in the original
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Create a well-connected core (first 8 nodes)
    core_size = 8
    for i in range(core_size):
        for j in range(i + 1, core_size):
            if random.random() < 0.6:  # Dense core
                G.add_edge(i, j)
    
    # Create periphery nodes connected to core with fewer connections
    for i in range(core_size, n):
        # Each peripheral node connects to 2-3 core nodes
        num_connections = random.randint(2, 3)
        core_nodes = random.sample(range(core_size), num_connections)
        for core_node in core_nodes:
            G.add_edge(i, core_node)
    
    # Add some connections between peripheral nodes (sparse)
    for i in range(core_size, n):
        for j in range(i + 1, n):
            if random.random() < 0.1:
                G.add_edge(i, j)
    
    return G

def initialize_contagion(G, num_seeds=3):
    """Initialize the contagion with seed nodes in the core"""
    for node in G.nodes():
        G.nodes[node]['state'] = 'susceptible'
    
    # Select seed nodes from the core
    seed_nodes = list(range(num_seeds))
    
    for node in seed_nodes:
        G.nodes[node]['state'] = 'seed'
    
    return seed_nodes

def complex_contagion_step(G, threshold):
    """
    Perform one step of complex contagion.
    A node becomes activated if fraction of activated neighbors >= threshold.
    Returns True if any node changed state.
    """
    changed = False
    new_activations = []
    
    for node in G.nodes():
        if G.nodes[node]['state'] == 'susceptible':
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            
            # Count activated neighbors (both seed and activated)
            activated_neighbors = sum(
                1 for n in neighbors 
                if G.nodes[n]['state'] in ['seed', 'activated']
            )
            
            # Check if threshold is met
            fraction = activated_neighbors / len(neighbors)
            if fraction >= threshold:
                new_activations.append(node)
                changed = True
    
    # Apply new activations
    for node in new_activations:
        G.nodes[node]['state'] = 'activated'
    
    return changed

def get_node_colors(G):
    """Get node colors based on their state"""
    colors = []
    for node in G.nodes():
        state = G.nodes[node]['state']
        if state == 'seed':
            colors.append(color_seed)
        elif state == 'activated':
            colors.append(color_activated)
        else:
            colors.append(color_susceptible)
    return colors

def plot_network_frame(G, pos, step, threshold, total_activated):
    """Create a single frame matching original network style"""
    # Transparent background like original
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='none')
    ax.set_facecolor('none')
    
    # Get node colors
    node_colors = get_node_colors(G)
    
    # Draw edges with BOLD, VISIBLE black lines (same as original)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        width=4,  # Thick, bold edges like original
        alpha=1.0,  # Fully opaque
        ax=ax
    )
    
    # Calculate node sizes based on degree (same as original)
    degrees = dict(G.degree())
    base_size = 1400
    node_sizes = [base_size + degrees[node] * 100 for node in G.nodes()]
    
    # Draw nodes with thick dark borders (same style as original)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=1.0,  # Fully opaque
        ax=ax,
        edgecolors=border_color,
        linewidths=6  # Thick border like original
    )
    
    # No titles or labels - clean like original
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    
    return img

def simulate_contagion(threshold, filename, num_seeds=3):
    """Run complete contagion simulation and create GIF"""
    print(f"\nSimulating complex contagion with threshold = {threshold}...")
    
    # Create network (same for both simulations)
    random.seed(seed)
    np.random.seed(seed)
    G = create_network_for_threshold_demo()
    
    # Spring layout like original (same parameters)
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=seed)
    
    # Initialize
    seed_nodes = initialize_contagion(G, num_seeds)
    
    # Collect frames
    frames = []
    step = 0
    max_steps = 30
    
    # Initial frame
    total_activated = sum(1 for node in G.nodes() if G.nodes[node]['state'] != 'susceptible')
    frame = plot_network_frame(G, pos, step, threshold, total_activated)
    frames.append(frame)
    frames.append(frame)  # Hold first frame
    
    # Simulation loop
    no_change_count = 0
    while step < max_steps:
        step += 1
        changed = complex_contagion_step(G, threshold)
        
        total_activated = sum(1 for node in G.nodes() if G.nodes[node]['state'] != 'susceptible')
        frame = plot_network_frame(G, pos, step, threshold, total_activated)
        frames.append(frame)
        
        if not changed:
            no_change_count += 1
            if no_change_count >= 2:  # No change for 2 steps
                # Hold final frame longer
                frames.extend([frame] * 5)
                break
        else:
            no_change_count = 0
    
    # Save as GIF
    frames[0].save(
        filename,
        save_all=True,
        append_images=frames[1:],
        duration=500,  # 500ms per frame
        loop=0
    )
    
    print(f"✓ Saved: {filename}")
    print(f"  Final activation: {total_activated}/{n} nodes ({100*total_activated/n:.1f}%)")
    print(f"  Steps: {step}")
    
    return total_activated

# Generate both simulations
print("Creating complex contagion simulations with original network style...\n")

activated_low = simulate_contagion(threshold=0.05, filename='contagion_threshold_0.05.gif', num_seeds=3)
activated_high = simulate_contagion(threshold=0.50, filename='contagion_threshold_0.30.gif', num_seeds=3)

print("\n" + "="*60)
print("✨ Contagion simulations complete!")
print("="*60)
print(f"\nResults:")
print(f"  Low threshold (0.05):  {activated_low}/{n} nodes activated ({100*activated_low/n:.0f}%)")
print(f"  High threshold (0.30): {activated_high}/{n} nodes activated ({100*activated_high/n:.0f}%)")