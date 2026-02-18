"""
Contagion Analysis - Parameter sweeps and strategic seeding for network contagion.
Combines variance analysis and seeding strategy comparison into a unified framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

from contagion_experiment import ContagionSimulator, load_networks


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    n_simulations: int = 20
    max_steps: int = 50
    threshold_type: str = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.35
    n_thresholds: int = 8
    
    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


@dataclass
class NetworkConfig:
    """Network parameters."""
    base_folder: str = "Data/networks/werkschool"
    scale: float = 0.01
    reciprocity: int = 1
    transitivity: int = 0
    bridge: float = 0.2
    n_communities_range: Tuple[float, float, int] = (1, 100, 10)
    preferential_attachment_range: Tuple[float, float, int] = (0, 0.99, 10)
    
    def folder_path(self, n_comms: float, pref_att: float) -> str:
        return (f"{self.base_folder}/scale={self.scale}_comms={n_comms}_"
                f"recip={self.reciprocity}_trans={self.transitivity}_"
                f"pa={pref_att}_bridge={self.bridge}")
    
    @property
    def community_values(self) -> np.ndarray:
        return np.linspace(*self.n_communities_range)
    
    @property
    def attachment_values(self) -> np.ndarray:
        return np.linspace(*self.preferential_attachment_range)


class StrategicSeeding:
    """Calculate and select strategic seeding groups based on network structure."""
    
    def __init__(self, characteristics: List[str], network_graphs: Dict):
        self.characteristics = sorted(characteristics)
        self.network_graphs = network_graphs
        self.strategies = {}
        
    def calculate_group_metrics(self) -> Dict[str, pd.DataFrame]:
        """Calculate seeding metrics for all characteristic combinations."""
        print("Calculating strategic seeding metrics...")
        
        for combo in tqdm(list(self._get_combinations()), desc="Processing combinations"):
            group_cols = list(combo)
            char_str = '_'.join(group_cols)
            
            try:
                df_seeding = self._calc_combo_metrics(group_cols, char_str)
                self.strategies[char_str] = df_seeding
            except FileNotFoundError:
                continue
        
        return self.strategies
    
    def _get_combinations(self):
        for r in range(1, len(self.characteristics) + 1):
            for combo in combinations(self.characteristics, r):
                yield combo
    
    def _calc_combo_metrics(self, group_cols: List[str], char_str: str) -> pd.DataFrame:
        # Load data
        df_edges = pd.read_csv(f"Data/aggregated/tab_werkschool_{char_str}.csv")
        df_nodes = pd.read_csv(f"Data/aggregated/tab_n_{char_str}.csv")
        
        # Build group_id lookup
        df_group_id = pd.DataFrame([
            {col: val for col, val in key} | {'group_id': idx}
            for key, idx in self.network_graphs[char_str].attrs_to_group.items()
        ])
        
        # Add labels
        for suffix in [('src', 'dst')]:
            df_edges['src_label'] = df_edges[[f"{c}_src" for c in group_cols]].astype(str).agg(','.join, axis=1)
            df_edges['dst_label'] = df_edges[[f"{c}_dst" for c in group_cols]].astype(str).agg(','.join, axis=1)
        
        df_nodes['node_label'] = df_nodes[group_cols].astype(str).agg(','.join, axis=1)
        df_group_id['node_label'] = df_group_id[group_cols].astype(str).agg(','.join, axis=1)
        
        # Merge
        df_nodes = df_nodes.merge(df_group_id[['node_label', 'group_id']], on='node_label')
        df_merged = df_edges.merge(df_nodes, left_on='src_label', right_on='node_label')
        
        # Calculate metrics
        metrics = []
        for _, row in df_nodes.iterrows():
            group, n, gid = row['node_label'], row['n'], row['group_id']
            
            internal = df_merged[(df_merged['src_label'] == group) & 
                               (df_merged['dst_label'] == group)]['n_x'].sum()
            external = df_merged[(df_merged['src_label'] == group) & 
                               (df_merged['dst_label'] != group)]['n_x'].sum()
            
            max_internal = (n * (n - 1)) / 2
            density = internal / max_internal if max_internal > 0 else 0
            ext_ratio = external / internal if internal > 0 else 0
            
            metrics.append({
                'group': group, 'size': n, 'group_id': gid,
                'internal_density': density, 'external_exposure_ratio': ext_ratio
            })
        
        # Normalize and score
        df = pd.DataFrame(metrics)
        for col in ['internal_density', 'external_exposure_ratio']:
            max_val = df[col].max()
            df[f'{col}_norm'] = df[col] / max_val if max_val > 0 else 0
        
        df['combined_score'] = 0.9 * df['internal_density_norm'] + 0.1 * df['external_exposure_ratio_norm']
        return df.sort_values('combined_score', ascending=False)
    
    def select_seeding_groups(self, strategy: str = 'combined_score',
                             threshold_fraction: float = 0.01,
                             total_nodes: int = 8601) -> Dict[str, np.ndarray]:
        """Select nodes for seeding based on strategy metric."""
        if not self.strategies:
            raise ValueError("Call calculate_group_metrics() first")
        
        selected = {}
        threshold = threshold_fraction * total_nodes
        
        for key, df in self.strategies.items():
            if strategy == "random":
                nodes = np.arange(total_nodes)
                np.random.shuffle(nodes)
                selected[key] = nodes
            else:
                df_sorted = df.sort_values(strategy, ascending=False).reset_index(drop=True)
                nodes, total = [], 0
                
                for idx in range(len(df_sorted)):
                    gid = df_sorted.loc[idx, 'group_id']
                    nodes.extend(self.network_graphs[key].group_to_nodes[gid])
                    total += df_sorted.loc[idx, 'size']
                    if total >= threshold:
                        break
                
                selected[key] = np.array(nodes)
        
        return selected


class ContagionAnalyzer:
    """Run contagion analysis across network configurations and seeding strategies."""
    
    def __init__(self, sim_config: SimulationConfig = None, net_config: NetworkConfig = None):
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()
        
    def run_parameter_sweep(self, n_iterations: int = 10, shuffle: bool = True) -> pd.DataFrame:
        """Run parameter sweep over network configurations."""
        print("\n" + "="*70)
        print("CONTAGION PARAMETER SWEEP")
        print("="*70)
        
        comms = self.net.community_values
        atts = self.net.attachment_values
        combinations = list(product(comms, atts))
        if shuffle:
            np.random.shuffle(combinations)

        params = combinations[:n_iterations]

        results = []
        for n_comms, pref_att in tqdm(params, desc="Parameter sweep"):
            result = self._run_config(n_comms, pref_att)
            if result:
                results.extend(result)
        
        return pd.DataFrame(results)
    
    def _run_config(self, n_comms: float, pref_att: float) -> Optional[List[Dict]]:
        """Run simulation for single parameter configuration."""
        folder = Path(self.net.folder_path(n_comms, pref_att))
        
        if not folder.exists():
            return None
        
        try:
            network_graphs = load_networks(folder, add_random=False)
            networks = {key: network_graphs[key].graph for key in network_graphs}
        
        except:
            return None
        
        if not networks:
            return None
        
        contested_results, ratios = self._sweep_contested(networks)
        
        results = []
        for net_name, thresh_results in contested_results.items():
            if net_name in ratios:
                for thresh_idx, median_val in thresh_results.items():
                    results.append({
                        'n_communities': n_comms,
                        'pref_attachment': pref_att,
                        'network': net_name,
                        'threshold_idx': thresh_idx,
                        'threshold_value': self.sim.thresholds[thresh_idx],
                        'median_final_adoption': median_val,
                        'ratio': ratios[net_name]
                    })
        
        return results
    
    def _sweep_contested(self, networks: Dict) -> Tuple[Dict, Dict]:
        """Run contested contagion sweep across thresholds."""
        results, ratios = {}, {}
        
        for name, G in networks.items():
            if not name.startswith("Random"):
                try:
                    df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                    ratios[name] = df_n.n.max() / df_n.n.sum()
                except FileNotFoundError:
                    continue
            print(G)
            sim = ContagionSimulator(G, name)
            n, initial = len(G), int(len(G) * self.sim.initial_infected_fraction)
            
            finals = {}
            for i, tau in enumerate(self.sim.thresholds):
                ts_list = sim.complex_contagion(
                    threshold=tau, threshold_type=self.sim.threshold_type,
                    seeding='focal_neighbors', max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations, initial_infected=initial
                )
                finals[i] = np.median([ts[-1] for ts in ts_list])
            
            results[name] = finals
        
        return results, ratios
    
    def run_strategic_analysis(self, network_folder: str, characteristics: List[str]) -> pd.DataFrame:
        """Run analysis with strategic seeding approaches."""
        print("\n" + "="*70)
        print("STRATEGIC SEEDING ANALYSIS")
        print("="*70)
        
        network_graphs = load_networks(network_folder, add_random=False)
        networks = {key: network_graphs[key].graph for key in network_graphs}
        
        strategic = StrategicSeeding(characteristics, network_graphs)
        strategic.calculate_group_metrics()
        
        strategies = ['combined_score', 'internal_density', 'external_exposure_ratio', 'random', 'size']
        
        all_results = []
        for strategy in tqdm(strategies, desc="Testing strategies"):
            seeding_groups = strategic.select_seeding_groups(strategy=strategy)
            results = self._run_strategic_sweep(networks, seeding_groups)
            
            for net_name, thresh_results in results.items():
                for thresh_idx, median_val in thresh_results.items():
                    all_results.append({
                        'strategy': strategy,
                        'network': net_name,
                        'threshold_idx': thresh_idx,
                        'threshold_value': self.sim.thresholds[thresh_idx],
                        'median_final_adoption': median_val
                    })
        
        return pd.DataFrame(all_results)
    
    def _run_strategic_sweep(self, networks: Dict, seeding_groups: Dict) -> Dict:
        """Run sweep with strategic seeding."""
        results = {}
        
        for name, G in networks.items():
            seedings = seeding_groups[name]
            sim = ContagionSimulator(G, name)
            initial = int(sim.n * self.sim.initial_infected_fraction)
            
            if len(seedings) < initial:
                seedings = "random"
            
            finals = {}
            for i, tau in enumerate(self.sim.thresholds):
                ts_list = sim.complex_contagion(
                    threshold=tau, threshold_type=self.sim.threshold_type,
                    seeding=seedings, max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations, initial_infected=initial
                )
                finals[i] = np.median([ts[-1] for ts in ts_list])
            
            results[name] = finals
        
        return results


class Visualizer:
    """Create visualizations for contagion analysis results."""
    
    @staticmethod
    def plot_variance_vs_ratio(df: pd.DataFrame, out: str = "variance_analysis.png"):
        """Plot variance vs ratio for different thresholds."""
        plot_data = []
        
        for net in df['network'].unique():
            for thresh in df['threshold_value'].unique():
                subset = df[(df['network'] == net) & (df['threshold_value'] == thresh)]
                if len(subset) > 1:
                    plot_data.append({
                        'network': net,
                        'threshold': thresh,
                        'variance': subset['median_final_adoption'].var(),
                        'ratio': subset['ratio'].iloc[0] if 'ratio' in subset.columns else 0
                    })
        
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 7))
        
        sns.scatterplot(data=plot_df, x='ratio', y='variance', hue='threshold',
                       palette='viridis', s=150, alpha=0.7, ax=ax)
        
        ax.set_xlabel('Group Size Ratio (max/total)', fontsize=13)
        ax.set_ylabel('Adoption Variance', fontsize=13)
        ax.set_title('Contagion Variance vs Network Homogeneity', fontsize=15, fontweight='bold')
        ax.legend(title='Threshold', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()
    
    @staticmethod
    def plot_strategy_comparison(df: pd.DataFrame, out: str = "strategy_comparison.png"):
        """Heatmap and line plots comparing seeding strategies."""
        pivot = df.pivot_table(index='network', columns='threshold_value',
                              values='median_final_adoption', aggfunc='mean')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                   cbar_kws={'label': 'Final Adoption (%)'}, ax=ax1)
        ax1.set_title('Adoption by Network and Threshold', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Threshold Value')
        ax1.set_ylabel('Network Category')
        
        # Line plot
        for strategy in df['strategy'].unique():
            subset = df[df['strategy'] == strategy]
            grouped = subset.groupby('threshold_value')['median_final_adoption'].mean()
            ax2.plot(grouped.index, grouped.values, marker='o', 
                    label=strategy, linewidth=2, markersize=6, alpha=0.8)
        
        ax2.set_title('Strategy Performance', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Threshold Value')
        ax2.set_ylabel('Mean Final Adoption (%)')
        ax2.legend(title='Strategy', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()
    
    @staticmethod
    def plot_threshold_curves(df: pd.DataFrame, out: str = "threshold_curves.png"):
        """Plot adoption curves across thresholds."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for net in df['network'].unique():
            subset = df[df['network'] == net]
            grouped = subset.groupby('threshold_value')['median_final_adoption'].mean()
            ax.plot(grouped.index, grouped.values, marker='o',
                   label=net, linewidth=2, markersize=5, alpha=0.7)
        
        ax.set_xlabel('Adoption Threshold', fontsize=13)
        ax.set_ylabel('Final Adoption (%)', fontsize=13)
        ax.set_title('Threshold Sensitivity Analysis', fontsize=15, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_parameter_sweep(n_iterations: int = 10, output_dir: str = "results/parameter_sweep"):
    """Run and visualize parameter sweep analysis."""
    analyzer = ContagionAnalyzer()
    results = analyzer.run_parameter_sweep(n_iterations=n_iterations)
    
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path / "sweep_results.csv", index=False)
    print(f"\nSaved results to {out_path / 'sweep_results.csv'}")
    
    # Visualize
    viz = Visualizer()
    viz.plot_variance_vs_ratio(results, str(out_path / "variance_analysis.png"))
    viz.plot_threshold_curves(results, str(out_path / "threshold_curves.png"))
    
    return results


def run_strategic_seeding(characteristics: List[str] = None, 
                          network_folder: str = None,
                          output_dir: str = "results/strategic_seeding"):
    """Run and visualize strategic seeding analysis."""
    if characteristics is None:
        characteristics = ["geslacht", "lft", "etngrp", "oplniv"]
    if network_folder is None:
        network_folder = "Data/networks/werkschool/scale=0.01_comms=1.0_recip=1_trans=0_pa=0.0_bridge=0"
    
    analyzer = ContagionAnalyzer()
    results = analyzer.run_strategic_analysis(network_folder, characteristics)
    
    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path / "seeding_results.csv", index=False)
    print(f"\nSaved results to {out_path / 'seeding_results.csv'}")
    
    # Visualize
    viz = Visualizer()
    viz.plot_strategy_comparison(results, str(out_path / "strategy_comparison.png"))
    viz.plot_threshold_curves(results, str(out_path / "threshold_curves.png"))
    
    return results


if __name__ == "__main__":
    # Run parameter sweep
    print("="*70)
    print("PARAMETER SWEEP")
    print("="*70)
    sweep_results = run_parameter_sweep(n_iterations=3)
    print("\nSummary:")
    print(sweep_results.groupby('threshold_value')['median_final_adoption'].describe())
    
    # # Run strategic seeding
    # print("\n" + "="*70)
    # print("STRATEGIC SEEDING")
    # print("="*70)
    # seeding_results = run_strategic_seeding()
    # print("\nStrategy Performance:")
    # print(seeding_results.groupby('strategy')['median_final_adoption'].mean())