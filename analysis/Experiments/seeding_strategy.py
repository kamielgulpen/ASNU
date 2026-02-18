import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from contagion_experiment import ContagionSimulator, load_networks

def get_strategy_groups(characteristics, network_graphs):
    strategies = {}

    for r in range(1, len(characteristics) + 1):
        for combo in combinations(characteristics, r):
            group_cols = list(combo)
            characteristics_string = '_'.join(group_cols)

            df_edges = pd.read_csv(f"Data/aggregated/tab_werkschool_{characteristics_string}.csv")
            df_nodes = pd.read_csv(f"Data/aggregated/tab_n_{characteristics_string}.csv")

            # Build group_id lookup from network
            df_group_id = pd.DataFrame([
                {col: val for col, val in key} | {'group_id': idx}
                for key, idx in network_graphs[characteristics_string].attrs_to_group.items()
            ])

            # Labels
            df_edges['src_label'] = df_edges[[c + "_src" for c in group_cols]].astype(str).agg(','.join, axis=1)
            df_edges['dst_label'] = df_edges[[c + "_dst" for c in group_cols]].astype(str).agg(','.join, axis=1)
            df_nodes['node_label'] = df_nodes[group_cols].astype(str).agg(','.join, axis=1)
            df_group_id['node_label'] = df_group_id[group_cols].astype(str).agg(','.join, axis=1)

            df_nodes = df_nodes.merge(df_group_id[['node_label', 'group_id']], on='node_label')
            df_merged = df_edges.merge(df_nodes, left_on='src_label', right_on='node_label')

            # Seeding metrics
            seeding_metrics = []
            for _, node_row in df_nodes.iterrows():
                group    = node_row['node_label']
                n        = node_row['n']
                group_id = node_row['group_id']

                internal_edges = df_merged[
                    (df_merged['src_label'] == group) & (df_merged['dst_label'] == group)
                ]['n_x'].sum()

                max_internal = (n * (n - 1)) / 2
                density = internal_edges / max_internal if max_internal > 0 else 0

                external_edges = df_merged[
                    (df_merged['src_label'] == group) & (df_merged['dst_label'] != group)
                ]['n_x'].sum()

                ext_ratio = external_edges / internal_edges if internal_edges > 0 else 0

                seeding_metrics.append({
                    'group':                    group,
                    'size':                     n,
                    'group_id':                 group_id,
                    'internal_density':         density,
                    'external_exposure_ratio':  ext_ratio,
                })

            df_seeding = pd.DataFrame(seeding_metrics)

            # Normalize + combined score
            for col in ['internal_density', 'external_exposure_ratio']:
                max_val = df_seeding[col].max()
                df_seeding[f'{col}_norm'] = df_seeding[col] / max_val if max_val > 0 else 0

            df_seeding['combined_score'] = (
                0.9 * df_seeding['internal_density_norm'] +
                0.1 * df_seeding['external_exposure_ratio_norm']
            )

            strategies[characteristics_string] = df_seeding.sort_values('combined_score', ascending=False)

    return strategies


def choose_strategy(possible_strategies, strategy, network_graphs):
    groups = {}
    threshold = (0.01 * 8601) * 100
    
    for key, df in possible_strategies.items():
        if strategy != "random":
            # Sort by strategy value in descending order
            df_sorted = df.sort_values(by=strategy, ascending=False).reset_index(drop=True)
            
            combined_nodes = []
            total_size = 0
            
            # Keep adding groups until size threshold is met
            for idx in range(len(df_sorted)):
                group_id = df_sorted.loc[idx, 'group_id']
                group_size = df_sorted.loc[idx, 'size']
                strategy_value = df_sorted.loc[idx, strategy]
                
                # Add this group's nodes
                combined_nodes.extend(network_graphs[key].group_to_nodes[group_id])
                total_size += group_size
                
                print(f"{key}: Adding group {group_id}, size={group_size}, {strategy}={strategy_value}, total_size={total_size}")
                
                # Stop when threshold is exceeded
                if total_size > threshold:
                    break
            
            groups[key] = np.array(combined_nodes)
        else:
            groups[key] = np.array([i for i in range(86001)])
            np.random.shuffle(groups[key])
    
    return groups


def sweep_contested(networks, groups, fractions=None, n_simulations=20, max_steps=50):
    if fractions is None:
        fractions = np.linspace(0.05, 0.20, 8)

    results = {}
    for name, G in networks.items():
        seedings = np.array(groups[name])
        sim = ContagionSimulator(G, name)
        print(len(seedings), int(sim.n * 0.01))
        finals = {}
        if len(seedings) < int(sim.n * 0.01):
            seedings = "random"
            print(len(seedings))
            print(name)
        for i, tau in enumerate(fractions):
            ts_list = sim.complex_contagion(
                threshold=tau,
                threshold_type='fractional',
                seeding=seedings,
                max_steps=max_steps,
                n_simulations=n_simulations,
                initial_infected=int(sim.n * 0.01)
            )
            finals[i] = np.median([ts[-1] for ts in ts_list])
        results[name] = finals
        print(f"  Contested sweep done: {name}")

    return results


def plot_results(results, fractions):
    df = pd.DataFrame(results).T
    df.columns = fractions
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Value'})
    plt.title('Comparison of Values Across Grouping Categories')
    plt.xlabel('Threshold index')
    plt.ylabel('Grouping Category')
    plt.tight_layout()
    plt.savefig('heatmap_plot.png')
    plt.show()

    plt.figure(figsize=(12, 7))
    for index, row in df.iterrows():
        plt.plot(df.columns, row.values, marker='o', label=index, alpha=0.7)
    plt.title('Trend of Values per Category')
    plt.xlabel('Threshold index')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('line_plot.png')
    plt.show()


def main():
    characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

    network_folder = "Data/networks/werkschool/scale=0.01_comms=1.0_recip=1_trans=0_pa=0.33_bridge=0.2"

    network_graphs = load_networks(network_folder, add_random=False)
    networks = {key: network_graphs[key].graph for key in network_graphs}

    possible_strategies = get_strategy_groups(characteristics, network_graphs)
    groups = choose_strategy(possible_strategies, 'combined_score', network_graphs)
    fractions = np.round(np.linspace(0.05, 0.35, 6), 3)
    results = sweep_contested(networks=networks, groups=groups, fractions = fractions)
    plot_results(results, fractions = fractions)


if __name__ == "__main__":
    main()