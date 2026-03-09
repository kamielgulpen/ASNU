"""
Memory-Optimized Contagion Analysis - Parameter sweeps and strategic seeding.
Implements incremental processing, garbage collection, and checkpointing to prevent memory overloads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
import gc
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from contagion_experiment import ContagionSimulator, load_networks


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    n_simulations: int = 50
    max_steps: int = 50
    threshold_type: str = 'fractional'
    initial_infected_fraction: float = 0.01
    min_threshold: float = 0.05
    max_threshold: float = 0.30
    n_thresholds: int = 4
    
    @property
    def thresholds(self) -> np.ndarray:
        return np.linspace(self.min_threshold, self.max_threshold, self.n_thresholds)


@dataclass
class NetworkConfig:
    """Network parameters."""
    base_folder: str = "Data/networks/werkschool"
    scale: float = 0.1
    reciprocity: int = 1
    transitivity: int = 0
    bridge: float = 0.2
    n_communities_range: Tuple[float, float, int] = (1, 1000, 10)
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


class MemoryManager:
    """Helper class for memory management."""
    
    @staticmethod
    def clear_memory():
        """Force garbage collection."""
        gc.collect()
    
    @staticmethod
    def save_checkpoint(data: pd.DataFrame, filepath: Path, append: bool = True):
        """Save checkpoint to CSV."""
        if append and filepath.exists():
            data.to_csv(filepath, mode='a', header=False, index=False)
        else:
            data.to_csv(filepath, index=False)
    
    @staticmethod
    def load_checkpoint(filepath: Path) -> pd.DataFrame:
        """Load checkpoint from CSV."""
        if filepath.exists():
            return pd.read_csv(filepath)
        return pd.DataFrame()
    
    @staticmethod
    def get_completed_configs(filepath: Path) -> set:
        """Get set of completed configurations from checkpoint."""
        if not filepath.exists():
            return set()
        df = pd.read_csv(filepath)
        if 'n_communities' in df.columns and 'pref_attachment' in df.columns:
            return set(zip(df['n_communities'], df['pref_attachment']))
        return set()


class StrategicSeeding:
    """Calculate and select strategic seeding groups based on network structure."""
    
    def __init__(self, characteristics: List[str], network_graphs: Dict):
        self.characteristics = sorted(characteristics)
        self.network_graphs = network_graphs
        self.strategies = {}
    
    def calculate_group_metrics(self, max_combinations: int = None) -> Dict[str, pd.DataFrame]:
        """Calculate seeding metrics for characteristic combinations.
        
        Args:
            max_combinations: Maximum number of combinations to process (None = all)
        """
        print("Calculating strategic seeding metrics...")
        
        combinations_list = list(self._get_combinations())
        if max_combinations:
            combinations_list = combinations_list[:max_combinations]
        
        for combo in tqdm(combinations_list, desc="Processing combinations"):
            group_cols = list(combo)
            char_str = '_'.join(group_cols)
            
            # Skip if network doesn't exist
            if char_str not in self.network_graphs:
                continue
            
            try:
                df_seeding = self._calc_combo_metrics(group_cols, char_str)
                self.strategies[char_str] = df_seeding
                
                # Clear memory after each combination
                gc.collect()
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Error processing {char_str}: {e}")
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
        
        # Clear temporary dataframes
        del df_edges, df_nodes, df_merged, df_group_id
        gc.collect()
        
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
    """Run contagion analysis with memory-efficient processing."""
    
    def __init__(self, sim_config: SimulationConfig = None, net_config: NetworkConfig = None):
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()
        self.memory_manager = MemoryManager()
    
    def run_parameter_sweep(self, n_iterations: int = 10, shuffle: bool = True,
                           batch_size: int = 5, output_dir: str = "results/parameter_sweep",
                           resume: bool = True) -> pd.DataFrame:
        """Run parameter sweep with batching and checkpointing.
        
        Args:
            n_iterations: Number of parameter combinations to test
            shuffle: Randomly shuffle parameter combinations
            batch_size: Save results after every batch_size iterations
            output_dir: Output directory for results and checkpoints
            resume: Resume from checkpoint if available
        """
        print("\n" + "="*70)
        print("MEMORY-OPTIMIZED CONTAGION PARAMETER SWEEP")
        print("="*70)
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_path / "checkpoint_sweep.csv"
        
        # Get parameter combinations
        comms = self.net.community_values
        atts = self.net.attachment_values
        combinations_list = list(product(comms, atts))
        
        if shuffle:
            np.random.shuffle(combinations_list)
        
        params = combinations_list[:n_iterations]
        
        # Get completed configurations if resuming
        completed = self.memory_manager.get_completed_configs(checkpoint_file) if resume else set()
        print(f"Found {len(completed)} completed configurations")
        
        # Filter out completed
        params = [(nc, pa) for nc, pa in params if (nc, pa) not in completed]
        print(f"Processing {len(params)} remaining configurations")
        
        # Process in batches
        batch_results = []
        for idx, (n_comms, pref_att) in enumerate(tqdm(params, desc="Parameter sweep")):
            result = self._run_config(n_comms, pref_att)
            if result:
                batch_results.extend(result)
            
            # Save checkpoint every batch_size iterations
            if (idx + 1) % batch_size == 0 and batch_results:
                df_batch = pd.DataFrame(batch_results)
                self.memory_manager.save_checkpoint(df_batch, checkpoint_file, append=True)
                print(f"\nCheckpoint saved: {len(batch_results)} results")
                batch_results = []  # Clear batch
                self.memory_manager.clear_memory()
        
        # Save remaining results
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            self.memory_manager.save_checkpoint(df_batch, checkpoint_file, append=True)
        
        # Load all results
        results = self.memory_manager.load_checkpoint(checkpoint_file)
        return results
    
    def run_parameter_strategy_sweep(self, n_iterations: int = 10, shuffle: bool = True,
                                    characteristics: List[str] = None,
                                    strategies: List[str] = None,
                                    batch_size: int = 3,
                                    output_dir: str = "results/parameter_strategy_sweep",
                                    resume: bool = True) -> pd.DataFrame:
        """Run parameter + strategy sweep with memory optimization.
        
        Args:
            n_iterations: Number of parameter combinations
            shuffle: Randomly shuffle combinations
            characteristics: Node characteristics for strategic seeding
            strategies: Seeding strategies to test
            batch_size: Save after every batch_size iterations
            output_dir: Output directory
            resume: Resume from checkpoint
        """
        print("\n" + "="*70)
        print("MEMORY-OPTIMIZED PARAMETER + STRATEGY SWEEP")
        print("="*70)
        
        if characteristics is None:
            characteristics = ["geslacht", "lft", "etngrp", "oplniv"]
        if strategies is None:
            strategies = ['combined_score', 'internal_density', 'external_exposure_ratio', 
                         'random', 'size']
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_path / "checkpoint_param_strategy.csv"
        
        # Get parameter combinations
        comms = self.net.community_values
        atts = self.net.attachment_values
        combinations_list = list(product(comms, atts))
        
        if shuffle:
            np.random.shuffle(combinations_list)
        
        params = combinations_list[:n_iterations]
        
        # Get completed configurations
        completed = self.memory_manager.get_completed_configs(checkpoint_file) if resume else set()
        params = [(nc, pa) for nc, pa in params if (nc, pa) not in completed]
        print(f"Processing {len(params)} configurations")
        
        # Process in batches
        batch_results = []
        for idx, (n_comms, pref_att) in enumerate(tqdm(params, desc="Parameter+Strategy sweep")):
            result = self._run_config_with_strategies(n_comms, pref_att, characteristics, strategies)
            if result:
                batch_results.extend(result)
            
            # Save checkpoint and clear memory
            if (idx + 1) % batch_size == 0 and batch_results:
                df_batch = pd.DataFrame(batch_results)
                self.memory_manager.save_checkpoint(df_batch, checkpoint_file, append=True)
                print(f"\nCheckpoint: {len(batch_results)} results saved")
                batch_results = []
                self.memory_manager.clear_memory()
        
        # Save remaining
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            self.memory_manager.save_checkpoint(df_batch, checkpoint_file, append=True)
        
        results = self.memory_manager.load_checkpoint(checkpoint_file)
        return results
    
    def _run_config(self, n_comms: float, pref_att: float) -> Optional[List[Dict]]:
        """Run simulation for single parameter configuration."""
        folder = Path(self.net.folder_path(n_comms, pref_att))
        
        if not folder.exists():
            return None
        
        try:
            # Load only needed networks
            network_graphs = load_networks(str(folder), add_random=False)

            
            characteristic_groups = ["geslacht_lft", "etngrp", "lft", "etngrp_geslacht_lft_oplniv", "etngrp_geslacht_lft", "geslacht"]
            
            # Extract only the graphs we need
            networks = {key: network_graphs[key].graph 
                       for key in network_graphs 
                       if key in characteristic_groups}
            
            # networks = {key: network_graphs[key].graph 
            #            for key in network_graphs}
            
            # Run simulations
            contested_results, ratios = self._sweep_contested(networks)
            
            # Build results
            results = []
            for net_name, thresh_results in contested_results.items():
                if net_name in ratios:
                    for thresh_idx, stat_val in thresh_results.items():
                        results.append({
                            'n_communities': n_comms,
                            'pref_attachment': pref_att,
                            'network': net_name,
                            'threshold_idx': thresh_idx,
                            'threshold_value': self.sim.thresholds[thresh_idx],
                            'median_final_adoption': stat_val["mean"],
                            'internal_variance_adoption': stat_val["variance"],
                            'ratio': ratios[net_name]
                        })
            
            # Explicitly clear network data
            del network_graphs, networks
            self.memory_manager.clear_memory()
            
            return results
        except Exception as e:
            print(f"Error with config ({n_comms}, {pref_att}): {e}")
            return None
    
    def _run_config_with_strategies(self, n_comms: float, pref_att: float,
                                    characteristics: List[str],
                                    strategies: List[str]) -> Optional[List[Dict]]:
        """Run simulation with strategies, optimized for memory."""
        folder = Path(self.net.folder_path(n_comms, pref_att))
        
        if not folder.exists():
            return None
        
        try:
            # Load networks
            network_graphs = load_networks(str(folder), add_random=False)
            characteristic_groups = ["geslacht", "etngrp_geslacht_lft_oplniv", 
                                   "geslacht_oplniv", "lft"]
            
            networks = {key: network_graphs[key].graph 
                       for key in network_graphs 
                       if key in characteristic_groups}
            
            if not networks:
                return None
            
            # Calculate strategic seeding (limit combinations to reduce memory)
            strategic = StrategicSeeding(characteristics, network_graphs)
            strategic.calculate_group_metrics(max_combinations=10)  # Limit combinations
            
            results = []
            
            # Process each strategy
            for strategy in strategies:
                try:
                    seeding_groups = strategic.select_seeding_groups(strategy=strategy)
                    strategy_results = self._run_strategic_sweep(networks, seeding_groups)
                    
                    for net_name, thresh_results in strategy_results.items():
                        for thresh_idx, median_val in thresh_results.items():
                            results.append({
                                'n_communities': n_comms,
                                'pref_attachment': pref_att,
                                'strategy': strategy,
                                'network': net_name,
                                'threshold_idx': thresh_idx,
                                'threshold_value': self.sim.thresholds[thresh_idx],
                                'median_final_adoption': median_val
                            })
                    
                    # Clear after each strategy
                    del seeding_groups, strategy_results
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error with strategy {strategy}: {e}")
                    continue
            
            # Clear all network data
            del network_graphs, networks, strategic
            self.memory_manager.clear_memory()
            
            return results
            
        except Exception as e:
            print(f"Error with config ({n_comms}, {pref_att}): {e}")
            return None
    
    def _sweep_contested(self, networks: Dict) -> Tuple[Dict, Dict]:
        """Run contested contagion sweep across thresholds."""
        results, ratios = {}, {}
        
        for name, G in networks.items():
            if not name.startswith("Random"):
                try:
                    df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                    ratios[name] = df_n.n.max() / df_n.n.sum()
                    del df_n  # Clear immediately
                except FileNotFoundError:
                    continue
            
            sim = ContagionSimulator(G, name)
            n, initial = len(G), int(len(G) * self.sim.initial_infected_fraction)
            
            finals = {}
            for i, tau in enumerate(self.sim.thresholds):
                ts_list = sim.complex_contagion(
                    threshold=tau, threshold_type=self.sim.threshold_type,
                    seeding='focal_neighbors', max_steps=self.sim.max_steps,
                    n_simulations=self.sim.n_simulations, initial_infected=initial
                )
                finals[i] = {
                    "mean": np.mean([ts[-1] for ts in ts_list]),
                    "initial_mean": np.mean([ts[0] for ts in ts_list]),
                    "variance": np.var([ts[-1] for ts in ts_list])
                }
                del ts_list  # Clear after processing

            results[name] = finals            
            del sim  # Clear simulator
            gc.collect()
        
        return results, ratios
    
    def _run_strategic_sweep(self, networks: Dict, seeding_groups: Dict) -> Dict:
        """Run sweep with strategic seeding."""
        results = {}
        
        for name, G in networks.items():
            if name not in seeding_groups:
                continue
                
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
                del ts_list
            
            results[name] = finals
            del sim
            gc.collect()
        
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
        
        if not plot_data:
            print("No data for variance plot")
            return
        
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
        
        # Clear figure from memory
        del fig, ax
        gc.collect()
    
    @staticmethod
    def plot_strategy_comparison(df: pd.DataFrame, out: str = "strategy_comparison.png"):
        """Heatmap and line plots comparing seeding strategies."""
        if 'strategy' not in df.columns:
            print("No strategy column for comparison plot")
            return
        
        pivot = df.pivot_table(index='network', columns='threshold_value',
                              values='median_final_adoption', aggfunc='mean')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu",
                   cbar_kws={'label': 'Final Adoption'}, ax=ax1)
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
        ax2.set_ylabel('Mean Final Adoption')
        ax2.legend(title='Strategy', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()
        
        del fig, ax1, ax2
        gc.collect()
    
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
        ax.set_ylabel('Final Adoption', fontsize=13)
        ax.set_title('Threshold Sensitivity Analysis', fontsize=15, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.4)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()
        
        del fig, ax
        gc.collect()
    
    @staticmethod
    def plot_strategy_performance_grid(df: pd.DataFrame, out: str = "strategy_grid.png"):
        """Create a comprehensive grid showing strategy performance."""
        if 'strategy' not in df.columns:
            print("No strategy column found in data")
            return
        
        strategies = df['strategy'].unique()
        n_strategies = len(strategies)
        
        fig, axes = plt.subplots(2, int(np.ceil(n_strategies/2)), figsize=(16, 10))
        axes = axes.flatten()
        
        for idx, strategy in enumerate(strategies):
            subset = df[df['strategy'] == strategy]
            
            if 'n_communities' in subset.columns and 'pref_attachment' in subset.columns:
                pivot = subset.pivot_table(
                    index='n_communities',
                    columns='pref_attachment', 
                    values='median_final_adoption',
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot, ax=axes[idx], cmap='RdYlGn', 
                           cbar_kws={'label': 'Adoption %'},
                           annot=False, fmt='.1f')
                axes[idx].set_title(f'{strategy}', fontweight='bold')
                axes[idx].set_xlabel('Preferential Attachment')
                axes[idx].set_ylabel('N Communities')
            else:
                grouped = subset.groupby('threshold_value')['median_final_adoption'].mean()
                axes[idx].plot(grouped.index, grouped.values, marker='o', linewidth=2)
                axes[idx].set_title(f'{strategy}', fontweight='bold')
                axes[idx].set_xlabel('Threshold')
                axes[idx].set_ylabel('Adoption %')
                axes[idx].grid(True, alpha=0.3)
        
        for idx in range(n_strategies, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved to {out}")
        plt.close()
        
        del fig, axes
        gc.collect()


# ============================================================================
# Main Execution Functions with Memory Optimization
# ============================================================================

def run_parameter_sweep(n_iterations: int = 10, 
                       batch_size: int = 5,
                       output_dir: str = "results/parameter_sweep",
                       resume: bool = True):
    """Run and visualize parameter sweep with memory optimization."""
    analyzer = ContagionAnalyzer()
    
    results = analyzer.run_parameter_sweep(
        n_iterations=n_iterations,
        batch_size=batch_size,
        output_dir=output_dir,
        resume=resume
    )
    
    # Save final results
    out_path = Path(output_dir)
    results.to_csv(out_path / f"sweep_results_final{analyzer.net.scale}.csv", index=False)
    print(f"\nFinal results saved to {out_path / 'sweep_results_final.csv'}")
    
    # Visualize (use smaller dataset if too large)
    viz = Visualizer()
    if len(results) > 10000:
        print("Large dataset - sampling for visualization")
        viz_data = results.sample(min(10000, len(results)))
    else:
        viz_data = results
    
    viz.plot_variance_vs_ratio(viz_data, str(out_path / "variance_analysis.png"))
    viz.plot_threshold_curves(viz_data, str(out_path / "threshold_curves.png"))
    
    return results


def run_parameter_strategy_sweep(n_iterations: int = 10,
                                 batch_size: int = 3,
                                 characteristics: List[str] = None,
                                 strategies: List[str] = None,
                                 output_dir: str = "results/parameter_strategy_sweep",
                                 resume: bool = True):
    """Run combined parameter and strategy sweep with memory optimization."""
    if characteristics is None:
        characteristics = ["geslacht", "lft", "etngrp", "oplniv"]
    if strategies is None:
        strategies = ['combined_score', 'internal_density', 'external_exposure_ratio', 
                     'random', 'size']
    
    analyzer = ContagionAnalyzer()
    results = analyzer.run_parameter_strategy_sweep(
        n_iterations=n_iterations,
        batch_size=batch_size,
        characteristics=characteristics,
        strategies=strategies,
        output_dir=output_dir,
        resume=resume
    )
    
    # Save final results
    out_path = Path(output_dir)
    results.to_csv(out_path / "param_strategy_results_final.csv", index=False)
    print(f"\nFinal results saved to {out_path / 'param_strategy_results_final.csv'}")
    
    # Visualize
    viz = Visualizer()
    viz.plot_strategy_comparison(results, str(out_path / "strategy_comparison.png"))
    viz.plot_strategy_performance_grid(results, str(out_path / "strategy_grid.png"))
    viz.plot_threshold_curves(results, str(out_path / "threshold_curves.png"))
    
    if 'ratio' in results.columns:
        viz.plot_variance_vs_ratio(results, str(out_path / "variance_analysis.png"))
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    print("="*70)
    print("MEMORY-OPTIMIZED PARAMETER SWEEP")
    print("="*70)
    sweep_results = run_parameter_sweep(
        n_iterations=100,
        batch_size=1,  # Save every 5 iterations
        resume=True     # Resume from checkpoint if available
    )
    print("\nSummary:")
    print(sweep_results.groupby('threshold_value')['median_final_adoption'].describe())
    
    # Uncomment to run strategy sweep
    # print("\n" + "="*70)
    # print("PARAMETER + STRATEGY SWEEP")
    # print("="*70)
    # param_strategy_results = run_parameter_strategy_sweep(
    #     n_iterations=20,
    #     batch_size=3,
    #     resume=True
    # )
    # print("\nCombined Results Summary:")
    # print(param_strategy_results.groupby(['strategy', 'threshold_value'])['median_final_adoption'].mean())