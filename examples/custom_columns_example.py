"""
Custom Column Names Example
============================

This example shows how to use ASNU with input files that have
different column naming conventions.
"""

from asnu import generate
import pandas as pd

# Create population data with custom column names
population_data = pd.DataFrame({
    'age': ['young', 'young', 'old', 'old'],
    'sex': ['male', 'female', 'male', 'female'],
    'population': [1000, 1050, 800, 850]  # Custom column name
})

# Create interaction data with custom suffixes
interaction_data = pd.DataFrame({
    'age_source': ['young', 'young', 'old'],
    'sex_source': ['male', 'female', 'male'],
    'age_target': ['young', 'old', 'old'],
    'sex_target': ['female', 'male', 'female'],
    'num_interactions': [500, 300, 400]  # Custom column name
})

# Save to files
population_data.to_csv('custom_population.csv', index=False)
interaction_data.to_csv('custom_interactions.csv', index=False)

# Generate network with custom column specifications
print("Generating network with custom column names...")
G = generate(
    pops_path='custom_population.csv',
    links_path='custom_interactions.csv',
    pop_column='population',           # Instead of default 'n'
    src_suffix='_source',               # Instead of default '_src'
    dst_suffix='_target',               # Instead of default '_dst'
    link_column='num_interactions',     # Instead of default 'n'
    preferential_attachment=0.5,
    scale=0.15,
    reciprocity=0.25,
    transitivity=0.35,
    number_of_communities=4,
    base_path='custom_output',
    verbose=True
)

print(f"\nGenerated network with {G.graph.number_of_nodes()} nodes and {G.graph.number_of_edges()} edges")
print("Network saved to 'custom_output/' directory")
