import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_strategy_performance_grid(df: pd.DataFrame, out: str = "strategy_grid"):
    """Create a comprehensive grid showing strategy performance across parameters."""
    if 'strategy' not in df.columns:
        print("No strategy column found in data")
        return
    
    thresholds = df['threshold_value'].unique()
    strategies = df['strategy'].unique()
    n_strategies = len(strategies)
    aggregations = df['network'].unique()
    n_aggregations = len(aggregations)
    
    # Create a separate plot for each threshold
    for threshold in thresholds:
        fig, axes = plt.subplots(n_aggregations, n_strategies, figsize=(16, 10))
        if n_aggregations == 1 or n_strategies == 1:
            axes = axes.reshape(n_aggregations, n_strategies)
        
        idx = 0
        for aggregation in aggregations:
            for count, strategy in enumerate(strategies):
                ax = axes.flatten()[idx]
                subset = df[(df['strategy'] == strategy) & 
                            (df['network'] == aggregation) &
                            (df["threshold_value"] == threshold)]
                
                if 'n_communities' in subset.columns and 'pref_attachment' in subset.columns:
                    # Pivot for heatmap
                    pivot = subset.pivot_table(
                        index='n_communities',
                        columns='pref_attachment', 
                        values='median_final_adoption',
                        aggfunc='mean'
                    )
                    
                    sns.heatmap(pivot, ax=ax, cmap='RdYlGn', 
                            cbar_kws={'label': 'Adoption'},
                            annot=False, fmt='.1f')
                    
                    if count == 0:
                        ax.set_title(f'{strategy}_{aggregation}', fontweight='bold')
                    else:
                        ax.set_title(f'{strategy}', fontweight='bold')
                    ax.set_xlabel('Preferential Attachment')
                    ax.set_ylabel('N Communities')
                else:
                    # Simple line plot if no parameter columns
                    if len(subset) > 0:
                        grouped = subset.groupby('threshold_value')['median_final_adoption'].mean()
                        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2)
                    ax.set_title(f'{strategy}_{aggregation}', fontweight='bold')
                    ax.set_xlabel('Threshold')
                    ax.set_ylabel('Adoption %')
                    ax.grid(True, alpha=0.3)
                
                idx += 1
        
        plt.tight_layout()
        filename = f"{out}_{round(threshold, 2)}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved to {filename}")
        plt.close()

df = pd.read_csv("results/parameter_strategy_sweep/param_strategy_results.csv")
plot_strategy_performance_grid(df)