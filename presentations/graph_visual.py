import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import numpy as np

# Configuration
n = 12
m = 24  # Number of edges to maintain
seed = 42

# Clean, bold color palette matching reference style
colors = [
    '#F4E04D',  # Yellow
    '#4CAF50',  # Green
    '#42A5F5',  # Blue
]

# Dark border color for all nodes (dark brown/maroon like reference)
border_color = '#4A1C1C'  # Dark reddish-brown

def assign_groups(G, num_groups):
    """Assign nodes to groups and return node colors"""
    group_size = n // num_groups
    node_colors = []
    
    for i in G.nodes():
        group_idx = min(i // group_size, num_groups - 1)
        G.nodes[i]['group'] = group_idx
        node_colors.append(colors[group_idx])
    
    return node_colors

def generate_random_network(num_groups):
    """Random network - edges added uniformly at random"""
    G = nx.gnm_random_graph(n, m, seed=seed)
    node_colors = assign_groups(G, num_groups)
    return G, node_colors

def generate_homophily_network(num_groups, homophily_strength=0.8):
    """Homophily network - nodes prefer same-group connections"""
    G = nx.Graph()
    G.add_nodes_from(range(n))
    node_colors = assign_groups(G, num_groups)
    
    group_size = n // num_groups
    
    # Calculate how many internal vs external edges we want
    target_internal = int(m * homophily_strength)
    target_external = m - target_internal
    
    # Generate internal edges within groups
    edges_added = 0
    attempts = 0
    max_attempts = m * 100
    
    while edges_added < target_internal and attempts < max_attempts:
        attempts += 1
        group = random.randint(0, num_groups - 1)
        start_node = group * group_size
        end_node = min(start_node + group_size, n)
        
        u = random.randint(start_node, end_node - 1)
        v = random.randint(start_node, end_node - 1)
        
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)
            edges_added += 1
    
    # Generate external edges between groups
    attempts = 0
    while edges_added < m and attempts < max_attempts:
        attempts += 1
        u, v = random.sample(range(n), 2)
        if not G.has_edge(u, v) and G.nodes[u]['group'] != G.nodes[v]['group']:
            G.add_edge(u, v)
            edges_added += 1
    
    # Fill remaining edges randomly if needed
    while G.number_of_edges() < m and attempts < max_attempts:
        attempts += 1
        u, v = random.sample(range(n), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            
    return G, node_colors

def generate_clustered_network(num_groups):
    """Highly clustered network - mostly intra-group edges"""
    return generate_homophily_network(num_groups, homophily_strength=0.95)

def generate_preferential_attachment(num_groups):
    """Preferential attachment (scale-free) network with groups"""
    # Generate with approximate edge count
    m_ba = max(1, m // n)
    G = nx.barabasi_albert_graph(n, m_ba, seed=seed)
    node_colors = assign_groups(G, num_groups)
    
    # Quickly adjust edge count
    current_edges = G.number_of_edges()
    all_possible_edges = list(nx.non_edges(G))
    
    if current_edges < m and all_possible_edges:
        edges_to_add = min(m - current_edges, len(all_possible_edges))
        new_edges = random.sample(all_possible_edges, edges_to_add)
        G.add_edges_from(new_edges)
    elif current_edges > m:
        edges_to_remove = current_edges - m
        edges_list = list(G.edges())
        remove_edges = random.sample(edges_list, edges_to_remove)
        G.remove_edges_from(remove_edges)
    
    return G, node_colors

def generate_small_world(num_groups):
    """Small-world network (Watts-Strogatz) with groups"""
    k = min(4, n - 1)  # Each node connected to k nearest neighbors
    p = 0.3  # Rewiring probability
    
    G = nx.watts_strogatz_graph(n, k, p, seed=seed)
    node_colors = assign_groups(G, num_groups)
    
    # Adjust edge count efficiently
    current_edges = G.number_of_edges()
    
    if current_edges < m:
        all_possible_edges = list(nx.non_edges(G))
        if all_possible_edges:
            edges_to_add = min(m - current_edges, len(all_possible_edges))
            new_edges = random.sample(all_possible_edges, edges_to_add)
            G.add_edges_from(new_edges)
    elif current_edges > m:
        edges_to_remove = current_edges - m
        edges_list = list(G.edges())
        remove_edges = random.sample(edges_list, edges_to_remove)
        G.remove_edges_from(remove_edges)
    
    return G, node_colors

def plot_five_topologies():
    """Create a single plot with 5 different network topologies"""
    
    num_groups = 1
    
    # Network generators with descriptions
    networks = [
        (generate_random_network, "Random Network"),
        # (generate_homophily_network, "Low Homophily"),
        (lambda ng: generate_homophily_network(ng, 0.9), "High Homophily"),
        # (generate_clustered_network, "Clustered"),
        (generate_small_world, "Small-World"),
    ]
    
    # Create figure with 5 subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 25), facecolor='white')
    
    for idx, (ax, (generator, title)) in enumerate(zip(axes, networks)):
        # Set seed for consistency
        random.seed(seed + idx)
        np.random.seed(seed + idx)
        
        # Generate network
        G, node_colors = generator(num_groups)
        
        # Spring layout
        pos = nx.spring_layout(G, k=0.8, iterations=50, seed=seed + idx)
        
        # Draw edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color='black',
            width=3,
            alpha=1.0,
            ax=ax
        )
        
        # Calculate node sizes
        degrees = dict(G.degree())
        base_size = 800
        node_sizes = [base_size + degrees[node] * 80 for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=1.0,
            ax=ax,
            edgecolors=border_color,
            linewidths=4
        )
        
        # # Add title
        # ax.set_title(
        #     title,
        #     fontsize=16,
        #     fontweight='bold',
        #     pad=15,
        #     color='#2C3E50'
        # )
        
        # Remove axes
        ax.axis('off')
    
    # Add overall title
    # fig.suptitle(
    #     'Five Network Topologies (3 Groups)',
    #     fontsize=20,
    #     fontweight='bold',
    #     y=0.98,
    #     color='#2C3E50'
    # )
    
    # Create shared legend
    legend_elements = []
    group_names = ["Group A", "Group B", "Group C"]
    
    for i in range(num_groups):
        legend_elements.append(
            mpatches.Patch(
                facecolor=colors[i],
                edgecolor=border_color,
                linewidth=3,
                label=group_names[i]
            )
        )
    
    # fig.legend(
    #     handles=legend_elements,
    #     loc='lower center',
    #     fontsize=12,
    #     frameon=True,
    #     ncol=3,
    #     bbox_to_anchor=(0.5, -0.05)
    # )
    
    plt.tight_layout()
    plt.savefig('network_five_topologies.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: network_five_topologies.png")
    plt.close()

def generate_homophily_graph(num_groups):
    G = nx.Graph()
    nodes = list(range(n))
    G.add_nodes_from(nodes)
    
    # Assign groups
    group_size = n // num_groups
    node_colors = []
    
    for i in nodes:
        group_idx = min(i // group_size, num_groups - 1)
        G.nodes[i]['group'] = group_idx
        node_colors.append(colors[group_idx])
    
    # Add edges with homophily logic - OPTIMIZED
    edges_added = 0
    attempts = 0
    max_attempts = m * 100
    
    while edges_added < m and attempts < max_attempts:
        attempts += 1
        u, v = random.sample(nodes, 2)
        if G.has_edge(u, v):
            continue
            
        # Homophily factor: 80% chance to keep edge if same group, 10% if different
        same_group = G.nodes[u]['group'] == G.nodes[v]['group']
        chance = random.random()
        
        if (same_group and chance < 0.8) or (not same_group and chance < 0.1):
            G.add_edge(u, v)
            G[u][v]['same_group'] = same_group
            edges_added += 1
            
    return G, node_colors

def plot_bubbly_network(num_groups, title, filename):
    """Create a beautiful network visualization with bold edges and clean style"""
    
    # Set random seed for consistency
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate graph
    G, node_colors = generate_homophily_graph(num_groups)
    
    # Create figure with transparent background
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='none')
    ax.set_facecolor('none')
    
    # Spring layout for organic clustering
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=seed)
    
    # Draw edges with BOLD, VISIBLE black lines
    nx.draw_networkx_edges(
        G, pos,
        edge_color='black',
        width=4,  # Thick, bold edges
        alpha=1.0,  # Fully opaque
        ax=ax
    )
    
    # Calculate node sizes based on degree
    degrees = dict(G.degree())
    base_size = 1400
    node_sizes = [base_size + degrees[node] * 100 for node in G.nodes()]
    
    # Draw nodes with thick dark borders (single layer, clean style)
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=1.0,  # Fully opaque
        ax=ax,
        edgecolors=border_color,
        linewidths=6  # Thick border like reference image
    )
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filename}', dpi=300, bbox_inches='tight', transparent=True)
    print(f"✓ Saved: {filename}")
    plt.close()

# Generate all three networks as separate plots
print("Creating clean, bold network visualizations...\n")

plot_bubbly_network(1, "Uniform Network\n(1 Group)", "network_1_uniform.png")
plot_bubbly_network(2, "Homophily Network\n(2 Groups)", "network_2_homophily.png")
plot_bubbly_network(3, "Strong Homophily Network\n(3 Groups)", "network_3_homophily.png")

# Generate the five topology comparison
print("\nCreating five topology comparison...\n")
plot_five_topologies()

print("\n✨ All networks created successfully!")
print("Each network features bold, visible edges and clean styling.")