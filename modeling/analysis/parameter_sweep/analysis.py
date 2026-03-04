"""
Simple Variance Analysis for Network Diffusion Data

Tests H1: Does variance increase with aggregation?

KEY CONCEPT:
Between-network variance includes variance across:
1. Different network realizations at the same structure parameters
2. Different network structure parameters (n_communities, pref_attachment)

This captures the TOTAL structural uncertainty from aggregation - 
you don't know the "right" structure, so variance includes all possibilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the CSV file"""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    
    # Check if order column exists
    if 'order' in df.columns:
        print(f"Order values found: {sorted(df['order'].unique())} (low=fine, high=coarse)")
        print(f"Network levels: {df.groupby('order')['network'].first().to_dict()}")
    else:
        print(f"Aggregation levels found: {sorted(df['network'].unique())}")
    
    df_mapping = pd.read_csv("group_charcteristic_mapping.csv")
    print(df.columns)
    df = df.merge(df_mapping, left_on="network", right_on="aggregation" )

    return df


def calculate_variance_metrics(df, use_order=True):
    """
    Calculate variance metrics for each aggregation level
    
    Calculates variance ACROSS all network structures (n_communities, pref_attachment)
    This captures total structural uncertainty from aggregation
    
    Parameters:
    -----------
    df : DataFrame
    use_order : bool
        If True, use 'order' column for aggregation level (low=fine, high=coarse)
        If False, use 'network' column
    """
    print("\nCalculating variance metrics...")
    df["order"] = df["hhi"]
    # Determine which column to use for aggregation level
    if use_order and 'order' in df.columns:
        agg_col = 'order'
        print("Using 'order' column (low=fine, high=coarse)")
    else:
        agg_col = 'network'
        print("Using 'network' column")
    
    results = []
    
    # Group by ONLY aggregation level and threshold
    # This calculates variance across ALL network structures
    for (agg_level, threshold), group in df.groupby([agg_col, 'threshold_value']):
        
        # Between-network variance: variance across ALL network realizations
        # This includes variance from different n_communities, pref_attachment, etc.
        outcomes = group['median_final_adoption'].values
        
        if len(outcomes) > 1:
            between_var = np.var(outcomes, ddof=1)
            between_sd = np.std(outcomes, ddof=1)
        else:
            between_var = 0
            between_sd = 0
        
        # Within-network variance: average of internal variances
        within_var = group['internal_variance_adoption'].mean()
        within_sd = np.sqrt(within_var)
        
        # Mean outcome
        mean_outcome = np.mean(outcomes)
        
        # Coefficient of variation (between-network only)
        if mean_outcome > 0:
            cv_between = between_sd / mean_outcome
            cv_within = within_sd / mean_outcome
        else:
            cv_between = np.nan
        
        # Total variance
        total_var = between_var + within_var
        total_sd = np.sqrt(total_var)
        
        # What % of variance is structural (between-network)?
        if total_var > 0:
            pct_structural = (between_var / total_var) * 100
        else:
            pct_structural = np.nan
        
        # Network structure diversity
        n_structures = len(group[['n_communities', 'pref_attachment']].drop_duplicates())
        
        # Get network name if using order
        if agg_col == 'order':
            network_name = group['network'].iloc[0]
        else:
            network_name = agg_level
        
        results.append({
            'aggregation_level': agg_level,
            'network_name': network_name,
            'threshold': threshold,
            'mean_outcome': mean_outcome,
            'between_sd': between_sd,
            'within_sd': within_sd,
            'total_sd': total_sd,
            'cv_between': cv_between,
            'cv_within': cv_within,
            'pct_structural': pct_structural,
            'n_networks': len(outcomes),
            'n_structures': n_structures
        })
    
    variance_df = pd.DataFrame(results)
    
    # For plotting, aggregation_level is already numeric if using order
    if agg_col == 'order':
        variance_df['agg_numeric'] = variance_df['aggregation_level']
    
    print(f"\nCalculated variance across {variance_df['n_networks'].mean():.0f} network realizations per condition")
    print(f"Including {variance_df['n_structures'].mean():.0f} different network structures per condition")
    
    return variance_df


def test_h1(variance_df):
    """
    Test H1: Does between-network variance (CV) increase with aggregation?
    Uses simple linear regression
    
    For order column: low order = fine, high order = coarse
    So positive slope = variance increases with aggregation (as expected)
    """
    print("\n" + "="*60)
    print("H1 TEST: Does variance increase with aggregation?")
    print("="*60)
    
    # Use agg_numeric if available, otherwise aggregation_level
    if 'agg_numeric' in variance_df.columns:
        x_col = 'agg_numeric'
    else:
        x_col = 'aggregation_level'

    for threshold in variance_df["threshold"].unique():
        variance_copy_df = variance_df[variance_df.threshold == threshold]
        # Remove NaN values

        data = variance_copy_df[[x_col, 'cv_between']].dropna()

        if len(data) < 3:
            print("Not enough data points for regression")
            # return None
        
        x = data[x_col].values
        y = data['cv_between'].values
        
        # Simple linear regression: y = slope * x + intercept
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Simple significance test
        std_err = np.sqrt(ss_res / (n - 2)) / np.sqrt(denominator)
        t_stat = slope / std_err

        # Rough p-value interpretation
        if abs(t_stat) > 3:
            p_value = "< 0.01"
            significant = True
        elif abs(t_stat) > 2:
            p_value = "< 0.05"
            significant = True
        else:
            p_value = "> 0.05"
            significant = False
        
        print(f"\nLinear Regression: CV_between ~ aggregation_level for threshold {threshold}")
        print(f"  Slope: {slope:.6f} (positive = variance increases with aggregation)")
        print(f"  R²: {r_squared:.4f}")
        print(f"  p-value: {p_value}")
        print(f"  Significant: {'YES' if significant else 'NO'}")
        print()
        
        if slope > 0 and significant:
            print("✓ HYPOTHESIS SUPPORTED")
            print("  Variance INCREASES with aggregation")
            print("  → Coarse data introduces structural uncertainty")
        elif slope < 0 and significant:
            print("✗ HYPOTHESIS REJECTED")
            print("  Variance DECREASES with aggregation")
            print("  → Large groups may average out extremes (Law of Large Numbers)")
        else:
            print("~ NO SIGNIFICANT TREND")
            print("  Variance is stable across aggregation levels")
    
    return {
        'slope': slope,
        'r_squared': r_squared,
        'p_value': p_value,
        'significant': significant
    }


def plot_results(variance_df, save_path='variance_plot.png'):
    """Create simple, beautiful plots"""
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bold colors
    colors = [
        '#F4E04D',  # Yellow
        '#4CAF50',  # Green
        '#42A5F5',  # Blue
        '#FF5722',  # Orange-red
    ]
    
    # Sort by aggregation level
    plot_df = variance_df.copy()
    plot_df = plot_df.sort_values('aggregation_level')
    
    # Get unique aggregation levels
    agg_levels = sorted(plot_df['aggregation_level'].unique())
    
    thresholds = ["low", "med-low", "med-high", "high"]
    # 1. Between-network CV
    ax = axes[0]
    plot_df['threshold'] = round(plot_df['threshold'], 2)
    for idx, threshold in enumerate(sorted(plot_df['threshold'].unique())):
        color = colors[idx % len(colors)]
        data = plot_df[plot_df['threshold'] == threshold]
        agg_avg = data.groupby('aggregation_level')['cv_between'].mean()
        threshold_label = thresholds[idx]
        # Regression line (subtle)
        if len(agg_levels) > 1:
            z = np.polyfit(agg_levels, agg_avg.values, 1)
            p = np.poly1d(z)
            ax.plot(agg_levels, p(agg_levels), 
                   linestyle='-', linewidth=1.5, color=color, alpha=0.25)
        
        # Scatter points
        ax.scatter(agg_levels, agg_avg.values, 
                  s=80, color=color, label=f'{threshold_label}', 
                  edgecolors='white', linewidth=2, zorder=3)
    
    ax.set_xlabel('Level of aggregation', fontsize=11)
    ax.set_ylabel('Between-Network CV', fontsize=11)
    # ax.set_title('Focal Seeding', fontsize=12, fontweight='bold')
    # ax.legend(frameon=False, title = 'contagion threshold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Within-network CV
    ax = axes[1]
    for idx, threshold in enumerate(sorted(plot_df['threshold'].unique())):
        color = colors[idx % len(colors)]
        data = plot_df[plot_df['threshold'] == threshold]
        agg_avg = data.groupby('aggregation_level')['cv_within'].mean()
        threshold_label = thresholds[idx]

        # Regression line (subtle)
        if len(agg_levels) > 1:
            z = np.polyfit(agg_levels, agg_avg.values, 1)
            p = np.poly1d(z)
            ax.plot(agg_levels, p(agg_levels), 
                   linestyle='-', linewidth=1.5, color=color, alpha=0.25)
        
        # Scatter points
        ax.scatter(agg_levels, agg_avg.values, 
                  s=80, color=color, label=f'{threshold_label}',
                  edgecolors='white', linewidth=2, zorder=3)
        

    ax.set_xlabel('Level of aggregation', fontsize=11)
    ax.set_ylabel('Within-Network CV', fontsize=11)
    # ax.set_title('Focal Seeding', fontsize=12, fontweight='bold')
    ax.legend(frameon=False, title = 'Contagion thresholds')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()
    
    # # 3. Between vs Within SD
    # ax = axes[1, 0]
    # agg_summary = plot_df.groupby('aggregation_level')[['between_sd', 'within_sd']].mean()
    # x = np.arange(len(agg_summary))
    # width = 0.35
    # ax.bar(x - width/2, agg_summary['between_sd'], width, 
    #        label='Between (structural)', color='steelblue')
    # ax.bar(x + width/2, agg_summary['within_sd'], width, 
    #        label='Within (stochastic)', color='coral')
    # ax.set_xlabel('Aggregation Level', fontsize=11, fontweight='bold')
    # ax.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
    # ax.set_title('Variance Components', fontsize=12, fontweight='bold')
    # ax.set_xticks(x)
    # ax.set_xticklabels(x_labels, fontsize=9)
    # ax.legend()
    # ax.grid(True, alpha=0.3, axis='y')
    
    # # 4. Mean outcome by aggregation
    # ax = axes[1, 1]
    # for threshold in sorted(plot_df['threshold'].unique()):
    #     data = plot_df[plot_df['threshold'] == threshold]
    #     agg_avg = data.groupby('aggregation_level')['mean_outcome'].mean()
    #     ax.plot(range(len(agg_avg)), agg_avg.values, 
    #             marker='^', label=f'θ={threshold}', linewidth=2, markersize=8)
    # ax.set_xticks(range(len(x_labels)))
    # ax.set_xticklabels(x_labels, fontsize=9)
    # ax.set_xlabel('Aggregation Level', fontsize=11, fontweight='bold')
    # ax.set_ylabel('Mean Final Adoption', fontsize=11, fontweight='bold')
    # ax.set_title('Mean Diffusion Outcomes', fontsize=12, fontweight='bold')
    # ax.legend()
    # ax.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"Plot saved to: {save_path}")
    # plt.show()


def print_summary(variance_df):
    """Print simple summary table"""
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print("(Averaged across all thresholds)")
    
    summary = variance_df.groupby('aggregation_level').agg({
        'mean_outcome': 'mean',
        'between_sd': 'mean',
        'within_sd': 'mean',
        'cv_between': 'mean',
        'pct_structural': 'mean',
        'n_networks': 'first',
        'n_structures': 'first'
    }).round(4)
    
    # Add network names if available
    if 'network_name' in variance_df.columns:
        network_names = variance_df.groupby('aggregation_level')['network_name'].first()
        summary['network_name'] = network_names
        # Reorder columns
        cols = ['network_name'] + [c for c in summary.columns if c != 'network_name']
        summary = summary[cols]
    
    # Sort by aggregation level (which is order value)
    summary = summary.sort_index()
    
    print("\nVariance across ALL network structures:")
    print("(low order = fine/granular, high order = coarse/aggregated)")
    print(summary.to_string())
    print()
    
    # Calculate change from fine to coarse (lowest to highest order)
    order_levels = sorted(summary.index)
    if len(order_levels) >= 2:
        first = order_levels[0]
        last = order_levels[-1]
        
        cv_change = ((summary.loc[last, 'cv_between'] / summary.loc[first, 'cv_between']) - 1) * 100
        pct_change = summary.loc[last, 'pct_structural'] - summary.loc[first, 'pct_structural']
        
        print(f"From order={first} (fine) → order={last} (coarse):")
        print(f"  CV change: {cv_change:+.1f}%")
        print(f"  Structural variance: {summary.loc[first, 'pct_structural']:.1f}% → {summary.loc[last, 'pct_structural']:.1f}% ({pct_change:+.1f}pp)")
        print(f"  → Variance includes {int(summary.loc[first, 'n_structures'])} different network structures")


def calculate_variance_by_structure(df, use_order=True):
    """
    OPTIONAL: Calculate variance separately for each network structure
    This shows whether effects are consistent across different structures
    """
    print("\n" + "="*60)
    print("VARIANCE BY NETWORK STRUCTURE")
    print("="*60)
    print("(Shows if effects are consistent across network types)")
    
    # Determine which column to use
    if use_order and 'order' in df.columns:
        agg_col = 'order'
    else:
        agg_col = 'network'
    
    results = []
    
    # Group by aggregation level, network structure, and threshold
    for (agg_level, n_comm, pref_att, threshold), group in df.groupby(
        [agg_col, 'n_communities', 'pref_attachment', 'threshold_value']
    ):
        outcomes = group['median_final_adoption'].values
        
        if len(outcomes) > 1:
            between_sd = np.std(outcomes, ddof=1)
        else:
            between_sd = 0
        
        mean_outcome = np.mean(outcomes)
        cv_between = between_sd / mean_outcome if mean_outcome > 0 else np.nan
        
        # Get network name if using order
        if agg_col == 'order':
            network_name = group['network'].iloc[0]
        else:
            network_name = agg_level
        
        results.append({
            'aggregation_level': agg_level,
            'network_name': network_name,
            'n_communities': n_comm,
            'pref_attachment': pref_att,
            'threshold': threshold,
            'cv_between': cv_between,
            'n_realizations': len(outcomes)
        })
    
    structure_df = pd.DataFrame(results)
    
    # Average across thresholds for cleaner display
    summary = structure_df.groupby(['aggregation_level', 'network_name', 'n_communities', 'pref_attachment']).agg({
        'cv_between': 'mean',
        'n_realizations': 'first'
    }).round(4)
    
    print("\nAverage CV by structure:")
    print(summary.to_string())
    
    return structure_df


def main(file_path, show_structure_breakdown=False):
    """
    Main analysis function
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
    show_structure_breakdown : bool
        If True, also show variance separately for each network structure
    """
    print("="*60)
    print("VARIANCE ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_data(file_path)
  
    df["order"] = df["ratio"]
    # Check if order column exists
    use_order = 'order' in df.columns
    if not use_order:
        print("\nWarning: 'order' column not found. Using 'network' column instead.")
        print("Note: Results may not be properly ordered from fine to coarse.")
    
    # Calculate metrics (variance across ALL network structures)
    variance_df = calculate_variance_metrics(df, use_order=use_order)
    
    # Test H1
    test_results = test_h1(variance_df)
    
    # Summary table
    print_summary(variance_df)
    
    # Optional: Show breakdown by structure
    if show_structure_breakdown:
        structure_df = calculate_variance_by_structure(df, use_order=use_order)
    else:
        structure_df = None
    
    # Plot
    plot_results(variance_df)
    
    # Save results
    variance_df.to_csv('variance_metrics.csv', index=False)
    print("\nResults saved to: variance_metrics.csv")
    
    if show_structure_breakdown and structure_df is not None:
        structure_df.to_csv('variance_by_structure.csv', index=False)
        print("Structure breakdown saved to: variance_by_structure.csv")
    
    return df, variance_df, test_results


if __name__ == "__main__":
    # Run analysis
    # Script automatically uses 'order' column if available
    # (low order = fine/granular, high order = coarse/aggregated)
    
    df, variance_df, results = main('results/parameter_sweep/focal_sweep_results.csv')

    
    # Optional: Show breakdown by network structure
    # df, variance_df, results = main('results.csv', show_structure_breakdown=True)