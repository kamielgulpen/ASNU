# ASNU Quick Start Guide

Get started with ASNU in 5 minutes!

## Installation

```bash
pip install asnu
```

## Your First Network

### 1. Prepare Your Data

Create two CSV files:

**population.csv** - describes your population groups:
```csv
age_group,gender,n
18-25,M,1000
18-25,F,1050
26-35,M,800
26-35,F,850
```

**interactions.csv** - describes how groups interact:
```csv
age_group_src,gender_src,age_group_dst,gender_dst,n
18-25,M,18-25,F,500
18-25,F,26-35,M,300
26-35,M,26-35,F,400
```

### 2. Generate Your Network

```python
from asnu import generate

# Generate the network
G = generate(
    pops_path="population.csv",
    links_path="interactions.csv",
    preferential_attachment=0.5,
    scale=0.1,
    reciprocity=0.2,
    transitivity=0.3,
    number_of_communities=10,
    base_path="my_network"
)

# Check the results
print(f"Nodes: {G.graph.number_of_nodes()}")
print(f"Edges: {G.graph.number_of_edges()}")
```

### 3. Analyze Your Network

```python
import networkx as nx

# The graph is a standard NetworkX DiGraph
G_nx = G.graph

# Calculate basic metrics
print(f"Density: {nx.density(G_nx):.4f}")
print(f"Average clustering: {nx.average_clustering(G_nx.to_undirected()):.4f}")

# Find most connected nodes
degrees = dict(G_nx.degree())
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"Top 5 nodes by degree: {top_nodes}")
```

## Key Parameters

- **preferential_attachment** (0-1): Higher = popular nodes get more connections
- **scale** (float): Controls network size (0.1 = 10% of population)
- **reciprocity** (0-1): Probability of mutual connections
- **transitivity** (0-1): Probability of friend-of-friend connections
- **number_of_communities**: How many communities to create

## Tips

1. **Start Small**: Use `scale=0.05` for quick testing
2. **Verbose Mode**: Add `verbose=True` to see progress
3. **Custom Columns**: Use `pop_column`, `src_suffix`, `dst_suffix`, `link_column` for different file formats
4. **Save & Load**: Networks auto-save to `base_path/graph.gpickle`

## Next Steps

- See `examples/` for more advanced usage
- Read the full README for parameter details
- Check `tests/` for usage patterns

## Common Issues

**"File not found"**: Use absolute paths or check your working directory
**"Out of memory"**: Reduce `scale` parameter
**"Not enough edges"**: Increase interaction counts in your data

## Getting Help

- GitHub Issues: https://github.com/yourusername/asnu/issues
- Documentation: See README.md
- Examples: Check the `examples/` directory
