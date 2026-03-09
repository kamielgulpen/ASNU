"""
Parallelized Contagion Analysis - Memory-efficient parallel parameter sweeps.
Supports multiprocessing with safe checkpointing and distributed workload management.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations, product
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc
import json
from datetime import datetime
from tqdm import tqdm
import warnings
from joblib import Parallel, delayed
from filelock import FileLock
import multiprocessing as mp
import os

warnings.filterwarnings('ignore')

from contagion_experiment import ContagionSimulator, load_networks


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    n_simulations: int = 20
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


class ThreadSafeCheckpoint:
    """Thread-safe checkpoint manager using file locking."""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        self.lock_file = checkpoint_file.with_suffix('.lock')
    
    def save_results(self, results: List[Dict], append: bool = True):
        """Thread-safe save to checkpoint file."""
        if not results:
            return
        
        df = pd.DataFrame(results)
        
        with FileLock(str(self.lock_file), timeout=30):
            if append and self.checkpoint_file.exists():
                df.to_csv(self.checkpoint_file, mode='a', header=False, index=False)
            else:
                df.to_csv(self.checkpoint_file, index=False)
    
    def load_results(self) -> pd.DataFrame:
        """Load checkpoint file."""
        with FileLock(str(self.lock_file), timeout=30):
            if self.checkpoint_file.exists():
                return pd.read_csv(self.checkpoint_file)
        return pd.DataFrame()
    
    def get_completed_configs(self) -> set:
        """Get set of completed configurations."""
        df = self.load_results()
        if df.empty:
            return set()
        
        if 'n_communities' in df.columns and 'pref_attachment' in df.columns:
            return set(zip(df['n_communities'], df['pref_attachment']))
        return set()


class StrategicSeeding:
    """Calculate and select strategic seeding groups."""
    
    def __init__(self, characteristics: List[str], network_graphs: Dict):
        self.characteristics = sorted(characteristics)
        self.network_graphs = network_graphs
        self.strategies = {}
    
    def calculate_group_metrics(self, max_combinations: int = None) -> Dict[str, pd.DataFrame]:
        """Calculate seeding metrics for characteristic combinations."""
        combinations_list = list(self._get_combinations())
        if max_combinations:
            combinations_list = combinations_list[:max_combinations]
        
        for combo in combinations_list:
            group_cols = list(combo)
            char_str = '_'.join(group_cols)
            
            if char_str not in self.network_graphs:
                continue
            
            try:
                df_seeding = self._calc_combo_metrics(group_cols, char_str)
                self.strategies[char_str] = df_seeding
                gc.collect()
            except Exception:
                continue
        
        return self.strategies
    
    def _get_combinations(self):
        for r in range(1, len(self.characteristics) + 1):
            for combo in combinations(self.characteristics, r):
                yield combo
    
    def _calc_combo_metrics(self, group_cols: List[str], char_str: str) -> pd.DataFrame:
        df_edges = pd.read_csv(f"Data/aggregated/tab_werkschool_{char_str}.csv")
        df_nodes = pd.read_csv(f"Data/aggregated/tab_n_{char_str}.csv")
        
        df_group_id = pd.DataFrame([
            {col: val for col, val in key} | {'group_id': idx}
            for key, idx in self.network_graphs[char_str].attrs_to_group.items()
        ])
        
        df_edges['src_label'] = df_edges[[f"{c}_src" for c in group_cols]].astype(str).agg(','.join, axis=1)
        df_edges['dst_label'] = df_edges[[f"{c}_dst" for c in group_cols]].astype(str).agg(','.join, axis=1)
        df_nodes['node_label'] = df_nodes[group_cols].astype(str).agg(','.join, axis=1)
        df_group_id['node_label'] = df_group_id[group_cols].astype(str).agg(','.join, axis=1)
        
        df_nodes = df_nodes.merge(df_group_id[['node_label', 'group_id']], on='node_label')
        df_merged = df_edges.merge(df_nodes, left_on='src_label', right_on='node_label')
        
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
        
        df = pd.DataFrame(metrics)
        for col in ['internal_density', 'external_exposure_ratio']:
            max_val = df[col].max()
            df[f'{col}_norm'] = df[col] / max_val if max_val > 0 else 0
        
        df['combined_score'] = 0.9 * df['internal_density_norm'] + 0.1 * df['external_exposure_ratio_norm']
        
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


# ============================================================================
# Worker Functions (for parallel execution)
# ============================================================================

def process_single_config(n_comms: float, pref_att: float, 
                         sim_config: SimulationConfig,
                         net_config: NetworkConfig,
                         worker_id: int = 0) -> Optional[List[Dict]]:
    """Worker function to process a single parameter configuration.
    
    This function is designed to be called in parallel by multiple processes.
    """
    try:
        folder = Path(net_config.folder_path(n_comms, pref_att))
        
        if not folder.exists():
            return None
        
        # Load networks
        network_graphs = load_networks(str(folder), add_random=False)
        characteristic_groups = ["geslacht", "etngrp_geslacht_lft_oplniv", 
                               "geslacht_oplniv", "lft", "etngrp_oplniv"]
        
        networks = {key: network_graphs[key].graph 
                   for key in network_graphs 
                   if key in characteristic_groups}
        
        if not networks:
            return None
        
        # Run simulations
        results = []
        for name, G in networks.items():
            # Get ratio
            ratio = None
            if not name.startswith("Random"):
                try:
                    df_n = pd.read_csv(f"Data/aggregated/tab_n_{name}.csv")
                    ratio = df_n.n.max() / df_n.n.sum()
                    del df_n
                except FileNotFoundError:
                    continue
            
            # Run contagion
            sim = ContagionSimulator(G, name)
            n, initial = len(G), int(len(G) * sim_config.initial_infected_fraction)
            
            for i, tau in enumerate(sim_config.thresholds):
                ts_list = sim.complex_contagion(
                    threshold=tau, threshold_type=sim_config.threshold_type,
                    seeding='focal_neighbors', max_steps=sim_config.max_steps,
                    n_simulations=sim_config.n_simulations, initial_infected=initial
                )
                
                results.append({
                    'n_communities': n_comms,
                    'pref_attachment': pref_att,
                    'network': name,
                    'threshold_idx': i,
                    'threshold_value': tau,
                    'median_final_adoption': np.mean([ts[-1] for ts in ts_list]),
                    'internal_variance_adoption': np.var([ts[-1] for ts in ts_list]),
                    'ratio': ratio
                })
                
                del ts_list
            
            del sim
            gc.collect()
        
        # Cleanup
        del network_graphs, networks
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Worker {worker_id} error with config ({n_comms}, {pref_att}): {e}")
        return None


def process_config_with_strategies(n_comms: float, pref_att: float,
                                   sim_config: SimulationConfig,
                                   net_config: NetworkConfig,
                                   characteristics: List[str],
                                   strategies: List[str],
                                   worker_id: int = 0) -> Optional[List[Dict]]:
    """Worker function for parameter + strategy sweep."""
    try:
        folder = Path(net_config.folder_path(n_comms, pref_att))
        
        if not folder.exists():
            return None
        
        # Load networks
        network_graphs = load_networks(str(folder), add_random=False)
        characteristic_groups = ["geslacht", "etngrp_geslacht_lft_oplniv", 
                               "geslacht_oplniv", "lft"]
        
        networks = {key: network_graphs[key].graph 
                   for key in network_graphs 
                   if key in characteristic_groups}
        
        if not networks:
            return None
        
        # Calculate strategic seeding
        strategic = StrategicSeeding(characteristics, network_graphs)
        strategic.calculate_group_metrics(max_combinations=10)
        
        results = []
        
        for strategy in strategies:
            try:
                seeding_groups = strategic.select_seeding_groups(strategy=strategy)
                
                # Run simulations for this strategy
                for name, G in networks.items():
                    if name not in seeding_groups:
                        continue
                    
                    seedings = seeding_groups[name]
                    sim = ContagionSimulator(G, name)
                    initial = int(sim.n * sim_config.initial_infected_fraction)
                    
                    if len(seedings) < initial:
                        seedings = "random"
                    
                    for i, tau in enumerate(sim_config.thresholds):
                        ts_list = sim.complex_contagion(
                            threshold=tau, threshold_type=sim_config.threshold_type,
                            seeding=seedings, max_steps=sim_config.max_steps,
                            n_simulations=sim_config.n_simulations, initial_infected=initial
                        )
                        
                        results.append({
                            'n_communities': n_comms,
                            'pref_attachment': pref_att,
                            'strategy': strategy,
                            'network': name,
                            'threshold_idx': i,
                            'threshold_value': tau,
                            'median_final_adoption': np.median([ts[-1] for ts in ts_list])
                        })
                        
                        del ts_list
                    
                    del sim
                
                del seeding_groups
                gc.collect()
                
            except Exception as e:
                print(f"Worker {worker_id} error with strategy {strategy}: {e}")
                continue
        
        # Cleanup
        del network_graphs, networks, strategic
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Worker {worker_id} error with config ({n_comms}, {pref_att}): {e}")
        return None


# ============================================================================
# Parallel Analyzer
# ============================================================================

class ParallelContagionAnalyzer:
    """Parallelized contagion analysis with safe checkpointing."""
    
    def __init__(self, sim_config: SimulationConfig = None, 
                 net_config: NetworkConfig = None,
                 n_jobs: int = -1):
        """
        Args:
            sim_config: Simulation configuration
            net_config: Network configuration
            n_jobs: Number of parallel jobs (-1 = all CPUs, -2 = all but one)
        """
        self.sim = sim_config or SimulationConfig()
        self.net = net_config or NetworkConfig()
        self.n_jobs = n_jobs if n_jobs != -1 else mp.cpu_count()
        if n_jobs == -2:
            self.n_jobs = max(1, mp.cpu_count() - 1)
    
    def run_parameter_sweep(self, n_iterations: int = 10, 
                           shuffle: bool = True,
                           output_dir: str = "results/parallel_sweep",
                           resume: bool = True,
                           batch_size: int = None) -> pd.DataFrame:
        """Run parallelized parameter sweep.
        
        Args:
            n_iterations: Number of parameter combinations to test
            shuffle: Randomly shuffle parameter combinations
            output_dir: Output directory
            resume: Resume from checkpoint if available
            batch_size: Number of configs per batch (None = n_jobs * 2)
        """
        print("\n" + "="*70)
        print(f"PARALLEL CONTAGION SWEEP (using {self.n_jobs} workers)")
        print("="*70)
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_path / "checkpoint_parallel.csv"
        
        # Setup checkpoint manager
        checkpoint = ThreadSafeCheckpoint(checkpoint_file)
        
        # Get parameter combinations
        comms = self.net.community_values
        atts = self.net.attachment_values
        combinations_list = list(product(comms, atts))
        
        if shuffle:
            np.random.shuffle(combinations_list)
        
        params = combinations_list[:n_iterations]
        
        # Filter out completed
        if resume:
            completed = checkpoint.get_completed_configs()
            params = [(nc, pa) for nc, pa in params if (nc, pa) not in completed]
            print(f"Resuming: {len(completed)} completed, {len(params)} remaining")
        
        if not params:
            print("All configurations already completed!")
            return checkpoint.load_results()
        
        # Set batch size
        if batch_size is None:
            batch_size = max(1, self.n_jobs * 2)
        
        # Process in batches
        all_results = []
        for batch_start in range(0, len(params), batch_size):
            batch_params = params[batch_start:batch_start + batch_size]
            print(f"\nProcessing batch {batch_start//batch_size + 1} "
                  f"({len(batch_params)} configs)")
            
            # Parallel processing
            batch_results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=5)(
                delayed(process_single_config)(
                    n_comms, pref_att, self.sim, self.net, worker_id=idx
                )
                for idx, (n_comms, pref_att) in enumerate(batch_params)
            )
            
            # Flatten results and save checkpoint
            flat_results = []
            for result_list in batch_results:
                if result_list:
                    flat_results.extend(result_list)
            
            if flat_results:
                checkpoint.save_results(flat_results, append=True)
                all_results.extend(flat_results)
                print(f"Checkpoint saved: {len(flat_results)} results")
            
            gc.collect()
        
        # Load final results
        final_results = checkpoint.load_results()
        print(f"\nTotal results: {len(final_results)}")
        
        return final_results
    
    def run_parameter_strategy_sweep(self, n_iterations: int = 10,
                                    shuffle: bool = True,
                                    characteristics: List[str] = None,
                                    strategies: List[str] = None,
                                    output_dir: str = "results/parallel_strategy_sweep",
                                    resume: bool = True,
                                    batch_size: int = None) -> pd.DataFrame:
        """Run parallelized parameter + strategy sweep."""
        print("\n" + "="*70)
        print(f"PARALLEL PARAMETER + STRATEGY SWEEP (using {self.n_jobs} workers)")
        print("="*70)
        
        if characteristics is None:
            characteristics = ["geslacht", "lft", "etngrp", "oplniv"]
        if strategies is None:
            strategies = ['combined_score', 'internal_density', 'external_exposure_ratio', 
                         'random', 'size']
        
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = out_path / "checkpoint_parallel_strategy.csv"
        
        checkpoint = ThreadSafeCheckpoint(checkpoint_file)
        
        # Get parameter combinations
        comms = self.net.community_values
        atts = self.net.attachment_values
        combinations_list = list(product(comms, atts))
        
        if shuffle:
            np.random.shuffle(combinations_list)
        
        params = combinations_list[:n_iterations]
        
        # Filter completed
        if resume:
            completed = checkpoint.get_completed_configs()
            params = [(nc, pa) for nc, pa in params if (nc, pa) not in completed]
            print(f"Resuming: {len(completed)} completed, {len(params)} remaining")
        
        if not params:
            print("All configurations already completed!")
            return checkpoint.load_results()
        
        # Set batch size (smaller for strategy sweep)
        if batch_size is None:
            batch_size = max(1, self.n_jobs)
        
        # Process in batches
        all_results = []
        for batch_start in range(0, len(params), batch_size):
            batch_params = params[batch_start:batch_start + batch_size]
            print(f"\nProcessing batch {batch_start//batch_size + 1} "
                  f"({len(batch_params)} configs)")
            
            # Parallel processing
            batch_results = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=5)(
                delayed(process_config_with_strategies)(
                    n_comms, pref_att, self.sim, self.net, 
                    characteristics, strategies, worker_id=idx
                )
                for idx, (n_comms, pref_att) in enumerate(batch_params)
            )
            
            # Flatten and save
            flat_results = []
            for result_list in batch_results:
                if result_list:
                    flat_results.extend(result_list)
            
            if flat_results:
                checkpoint.save_results(flat_results, append=True)
                all_results.extend(flat_results)
                print(f"Checkpoint saved: {len(flat_results)} results")
            
            gc.collect()
        
        final_results = checkpoint.load_results()
        print(f"\nTotal results: {len(final_results)}")
        
        return final_results


# ============================================================================
# Visualization (unchanged from optimized version)
# ============================================================================

class Visualizer:
    """Create visualizations for contagion analysis results."""
    
    @staticmethod
    def plot_variance_vs_ratio(df: pd.DataFrame, out: str = "variance_analysis.png"):
        """Plot variance vs ratio."""
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
        gc.collect()


# ============================================================================
# Main Execution Functions
# ============================================================================

def run_parallel_sweep(n_iterations: int = 10,
                      n_jobs: int = -1,
                      output_dir: str = "results/parallel_sweep",
                      resume: bool = True):
    """Run parallelized parameter sweep."""
    analyzer = ParallelContagionAnalyzer(n_jobs=n_jobs)
    
    results = analyzer.run_parameter_sweep(
        n_iterations=n_iterations,
        output_dir=output_dir,
        resume=resume
    )
    
    # Save final
    out_path = Path(output_dir)
    results.to_csv(out_path / "sweep_results_final.csv", index=False)
    print(f"\nFinal results: {out_path / 'sweep_results_final.csv'}")
    
    # Visualize
    viz = Visualizer()
    viz.plot_variance_vs_ratio(results, str(out_path / "variance_analysis.png"))
    viz.plot_threshold_curves(results, str(out_path / "threshold_curves.png"))
    
    return results


def run_parallel_strategy_sweep(n_iterations: int = 10,
                               n_jobs: int = -1,
                               characteristics: List[str] = None,
                               strategies: List[str] = None,
                               output_dir: str = "results/parallel_strategy_sweep",
                               resume: bool = True):
    """Run parallelized parameter + strategy sweep."""
    if characteristics is None:
        characteristics = ["geslacht", "lft", "etngrp", "oplniv"]
    if strategies is None:
        strategies = ['combined_score', 'internal_density', 'random']
    
    analyzer = ParallelContagionAnalyzer(n_jobs=n_jobs)
    
    results = analyzer.run_parameter_strategy_sweep(
        n_iterations=n_iterations,
        characteristics=characteristics,
        strategies=strategies,
        output_dir=output_dir,
        resume=resume
    )
    
    # Save final
    out_path = Path(output_dir)
    results.to_csv(out_path / "strategy_results_final.csv", index=False)
    print(f"\nFinal results: {out_path / 'strategy_results_final.csv'}")
    
    # Visualize
    viz = Visualizer()
    viz.plot_threshold_curves(results, str(out_path / "threshold_curves.png"))
    if 'ratio' in results.columns:
        viz.plot_variance_vs_ratio(results, str(out_path / "variance_analysis.png"))
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    # Determine optimal number of workers
    n_cpus = mp.cpu_count()
    print(f"System has {n_cpus} CPUs")
    print(f"Recommended: use {max(1, n_cpus - 1)} workers (n_jobs=-2)")
    
    print("\n" + "="*70)
    print("PARALLEL PARAMETER SWEEP")
    print("="*70)
    
    results = run_parallel_sweep(
        n_iterations=100,
        n_jobs=3,  # Use all CPUs except one
        resume=True
    )
    
    print("\nSummary:")
    print(results.groupby('threshold_value')['median_final_adoption'].describe())