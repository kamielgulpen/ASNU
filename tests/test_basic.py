"""
Basic tests for ASNU package
"""

import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from asnu import generate, NetworkXGraph


@pytest.fixture
def sample_data():
    """Create sample population and interaction data"""
    population_data = pd.DataFrame({
        'age_group': ['18-25', '18-25', '26-35', '26-35'],
        'gender': ['M', 'F', 'M', 'F'],
        'n': [100, 105, 80, 85]
    })

    interaction_data = pd.DataFrame({
        'age_group_src': ['18-25', '18-25', '26-35'],
        'gender_src': ['M', 'F', 'M'],
        'age_group_dst': ['18-25', '26-35', '26-35'],
        'gender_dst': ['F', 'M', 'F'],
        'n': [50, 30, 40]
    })

    return population_data, interaction_data


def test_import():
    """Test that package can be imported"""
    import asnu
    assert hasattr(asnu, 'generate')
    assert hasattr(asnu, 'NetworkXGraph')
    assert asnu.__version__ == '0.1.0'


def test_generate_basic(sample_data, tmp_path):
    """Test basic network generation"""
    population_data, interaction_data = sample_data

    # Save to temporary files
    pop_file = tmp_path / "population.csv"
    links_file = tmp_path / "interactions.csv"
    output_dir = tmp_path / "output"

    population_data.to_csv(pop_file, index=False)
    interaction_data.to_csv(links_file, index=False)

    # Generate network
    G = generate(
        pops_path=str(pop_file),
        links_path=str(links_file),
        preferential_attachment=0.5,
        scale=0.5,
        reciprocity=0.2,
        transitivity=0.3,
        number_of_communities=2,
        base_path=str(output_dir),
        verbose=False
    )

    # Verify network was created
    assert isinstance(G, NetworkXGraph)
    assert G.graph.number_of_nodes() > 0
    assert G.graph.number_of_edges() >= 0

    # Verify output files were created
    assert (output_dir / "graph.gpickle").exists()
    assert (output_dir / "metadata.json").exists()


def test_custom_column_names(tmp_path):
    """Test generation with custom column names"""
    population_data = pd.DataFrame({
        'age': ['young', 'old'],
        'population': [100, 80]
    })

    interaction_data = pd.DataFrame({
        'age_source': ['young'],
        'age_target': ['old'],
        'count': [50]
    })

    pop_file = tmp_path / "pop.csv"
    links_file = tmp_path / "links.csv"
    output_dir = tmp_path / "output"

    population_data.to_csv(pop_file, index=False)
    interaction_data.to_csv(links_file, index=False)

    # Generate with custom column names
    G = generate(
        pops_path=str(pop_file),
        links_path=str(links_file),
        pop_column='population',
        src_suffix='_source',
        dst_suffix='_target',
        link_column='count',
        preferential_attachment=0.5,
        scale=0.5,
        reciprocity=0.2,
        transitivity=0.3,
        number_of_communities=2,
        base_path=str(output_dir),
        verbose=False
    )

    assert G.graph.number_of_nodes() > 0


def test_network_properties(sample_data, tmp_path):
    """Test that generated network has expected properties"""
    population_data, interaction_data = sample_data

    pop_file = tmp_path / "population.csv"
    links_file = tmp_path / "interactions.csv"
    output_dir = tmp_path / "output"

    population_data.to_csv(pop_file, index=False)
    interaction_data.to_csv(links_file, index=False)

    G = generate(
        pops_path=str(pop_file),
        links_path=str(links_file),
        preferential_attachment=0.5,
        scale=0.5,
        reciprocity=0.0,  # No reciprocity for simpler testing
        transitivity=0.0,  # No transitivity for simpler testing
        number_of_communities=2,
        base_path=str(output_dir),
        verbose=False
    )

    # Check that groups were created
    assert len(G.group_to_nodes) > 0

    # Check that nodes are assigned to groups
    assert len(G.nodes_to_group) == G.graph.number_of_nodes()

    # Check that network is directed
    assert G.graph.is_directed()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
