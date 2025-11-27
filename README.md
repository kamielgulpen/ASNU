# ASNU - Aggregated Social Network Unfolder

**Generate large-scale population-based networks with realistic structure**

ASNU is a Python package for generating synthetic networks from aggregated population and interaction data. It creates realistic network structures by incorporating:

- **Community structure** - Nodes organized into overlapping communities
- **Preferential attachment** - Popular nodes attract more connections
- **Reciprocity** - Mutual connections between nodes
- **Transitivity** - Friend-of-friend connections (clustering)

## Installation

```bash
pip install asnu
```

### From source

```bash
git clone https://github.com/kamielgulpen/asnu.git
cd asnu
pip install -e .
```

## Quick Start

```python
from asnu import generate

# Generate a network
G = generate(
    pops_path="population.csv",           # Population data
    links_path="interactions.csv",       # Interaction data
    preferential_attachment=0.5,          # Strength of preferential attachment
    scale=0.1,                            # Network size scaling
    reciprocity=0.2,                      # Probability of reciprocal edges
    transitivity=0.3,                     # Probability of transitive edges
    number_of_communities=10,             # Number of communities
    base_path="my_network"                # Output directory
)

# Access the NetworkX graph
print(f"Nodes: {G.graph.number_of_nodes()}")
print(f"Edges: {G.graph.number_of_edges()}")
```

## Input Data Format

### Population File (CSV/Excel)

Describes population groups with their characteristics:

```csv
age_group,gender,region,n
18-25,M,North,1000
18-25,F,North,1050
26-35,M,North,800
...
```

- Characteristic columns: Any columns except the population count column
- Population column: `n` by default (configurable with `pop_column`)

### Interactions File (CSV/Excel)

Describes interaction patterns between groups:

```csv
age_group_src,gender_src,age_group_dst,gender_dst,n
18-25,M,18-25,F,500
18-25,M,26-35,M,300
...
```

- Source group columns: End with `_src` by default (configurable with `src_suffix`)
- Destination group columns: End with `_dst` by default (configurable with `dst_suffix`)
- Interaction count column: `n` by default (configurable with `link_column`)

## Custom Column Names

ASNU supports flexible column naming:

```python
G = generate(
    pops_path="population.csv",
    links_path="interactions.xlsx",
    pop_column='population',              # Instead of 'n'
    src_suffix='_source',                 # Instead of '_src'
    dst_suffix='_destination',            # Instead of '_dst'
    link_column='num_interactions',       # Instead of 'n'
    ...
)
```

## Parameters

### Network Generation

- `preferential_attachment` (float, 0-1): Strength of preferential attachment (higher = stronger)
- `scale` (float): Population scaling factor (smaller = smaller network)
- `reciprocity` (float, 0-1): Probability of creating reciprocal edges
- `transitivity` (float, 0-1): Probability of creating transitive edges
- `number_of_communities` (int): Number of communities to create

### Input/Output

- `pops_path` (str): Path to population data file
- `links_path` (str): Path to interactions data file
- `base_path` (str): Directory for saving the network
- `verbose` (bool): Whether to print progress information

### Column Names (Optional)

- `pop_column` (str): Population count column name (default: `'n'`)
- `src_suffix` (str): Source group column suffix (default: `'_src'`)
- `dst_suffix` (str): Suffix for destination group columns (default: `'_dst'`)
- `link_column` (str): Interaction count column name (default: `'n'`)

## Output

ASNU generates a `NetworkXGraph` object with:

- `G.graph`: NetworkX DiGraph with the generated network
- `G.group_to_nodes`: Mapping from group IDs to node lists
- `G.nodes_to_group`: Mapping from node IDs to group IDs
- `G.existing_num_links`: Link counts between group pairs
- `G.maximum_num_links`: Target link counts between group pairs

The network is automatically saved to disk at `base_path/`:
- `graph.gpickle`: NetworkX graph file
- `metadata.json`: Network metadata

## Examples

See the `examples/` directory for complete examples:
- Basic network generation
- Sensitivity analysis with MLflow
- Network analysis and visualization

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- pandas >= 1.3.0
- networkx >= 2.6.0
- scipy >= 1.7.0
- openpyxl >= 3.0.0

## License

MIT License

## Citation

If you use ASNU in your research, please cite:

```bibtex
@software{asnu2024,
  title={ASNU: Aggregated Social Network Unfolder},
  author={Your Name},
  year={2024},
  url={https://github.com/kamielgulpen/asnu}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions and support, please open an issue on [GitHub](https://github.com/kamielgulpen/asnu/issues).
