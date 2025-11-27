"""
Basic Example: Generating a Synthetic Network with ASNU
========================================================

This example demonstrates how to generate a basic synthetic network
using population and interaction data.
"""

from asnu import generate
import pandas as pd

# Create sample population data
population_data = pd.DataFrame({
    'age_group': ['18-25', '18-25', '26-35', '26-35', '36-50', '36-50'],
    'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
    'n': [1000, 1050, 800, 850, 1200, 1250]
})

# Create sample interaction data
interaction_data = pd.DataFrame({
    'age_group_src': ['18-25', '18-25', '26-35', '26-35', '18-25'],
    'gender_src': ['M', 'F', 'M', 'F', 'M'],
    'age_group_dst': ['18-25', '26-35', '26-35', '18-25', '18-25'],
    'gender_dst': ['F', 'M', 'F', 'F', 'M'],
    'n': [500, 300, 400, 250, 200]
})

# Save to CSV files
population_data.to_csv('population.csv', index=False)
interaction_data.to_csv('interactions.csv', index=False)

# Generate the network
print("Generating network...")
G = generate(
    pops_path='population.csv',
    links_path='interactions.csv',
    preferential_attachment=0.5,
    scale=0.1,
    reciprocity=0.2,
    transitivity=0.3,
    number_of_communities=5,
    base_path='output_network',
    verbose=True
)

# Print network statistics
print(f"\nNetwork Statistics:")
print(f"  Nodes: {G.graph.number_of_nodes()}")
print(f"  Edges: {G.graph.number_of_edges()}")
print(f"  Number of groups: {len(G.group_to_nodes)}")

# Analyze degree distribution
import networkx as nx
degrees = [d for n, d in G.graph.degree()]
print(f"\nDegree Statistics:")
print(f"  Average degree: {sum(degrees) / len(degrees):.2f}")
print(f"  Max degree: {max(degrees)}")
print(f"  Min degree: {min(degrees)}")

print("\nNetwork saved to 'output_network/' directory")
