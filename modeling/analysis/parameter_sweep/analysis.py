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
    # df["order"] = df["effective_n"]
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
    Uses both linear and logarithmic regression
    
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

    all_results = []
    
    for threshold in variance_df["threshold"].unique():
        variance_copy_df = variance_df[variance_df.threshold == threshold]
        # Remove NaN values

        data = variance_copy_df[[x_col, 'cv_between']].dropna()

        if len(data) < 3:
            print(f"\nNot enough data points for regression (threshold {threshold})")
            continue
        
        x = data[x_col].values
        y = data['cv_between'].values
        
        # ============ LINEAR REGRESSION ============
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # R-squared
        y_pred_linear = slope * x + intercept
        ss_res_linear = np.sum((y - y_pred_linear) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared_linear = 1 - (ss_res_linear / ss_tot) if ss_tot > 0 else 0
        
        # Simple significance test
        std_err = np.sqrt(ss_res_linear / (n - 2)) / np.sqrt(denominator)
        t_stat = slope / std_err

        # Rough p-value interpretation
        if abs(t_stat) > 3:
            p_value_linear = "< 0.01"
            significant_linear = True
        elif abs(t_stat) > 2:
            p_value_linear = "< 0.05"
            significant_linear = True
        else:
            p_value_linear = "> 0.05"
            significant_linear = False
        
        # ============ LOGARITHMIC REGRESSION ============
        # Model: y = a + b * log(x)
        # Need x > 0 for log
        if np.any(x <= 0):
            print(f"\nWarning: Non-positive x values found for threshold {threshold}. Skipping log fit.")
            log_a = log_b = r_squared_log = None
            y_pred_log = None
        else:
            x_log = np.log(x)
            x_log_mean = np.mean(x_log)
            
            numerator_log = np.sum((x_log - x_log_mean) * (y - y_mean))
            denominator_log = np.sum((x_log - x_log_mean) ** 2)
            
            log_b = numerator_log / denominator_log
            log_a = y_mean - log_b * x_log_mean
            
            # R-squared for log model
            y_pred_log = log_a + log_b * x_log
            ss_res_log = np.sum((y - y_pred_log) ** 2)
            r_squared_log = 1 - (ss_res_log / ss_tot) if ss_tot > 0 else 0
        
        # ============ PRINT RESULTS ============
        print(f"\n{'='*50}")
        print(f"Threshold: {threshold}")
        print(f"{'='*50}")
        
        print(f"\n1. LINEAR MODEL: y = {slope:.6f}*x + {intercept:.6f}")
        print(f"   R²: {r_squared_linear:.4f}")
        print(f"   p-value: {p_value_linear}")
        print(f"   Significant: {'YES' if significant_linear else 'NO'}")
        
        if r_squared_log is not None:
            print(f"\n2. LOGARITHMIC MODEL: y = {log_a:.6f} + {log_b:.6f}*log(x)")
            print(f"   R²: {r_squared_log:.4f}")
            
            # Compare models
            print(f"\n   MODEL COMPARISON:")
            if r_squared_log > r_squared_linear:
                diff = r_squared_log - r_squared_linear
                print(f"   → Log model fits better (ΔR² = +{diff:.4f})")
                print(f"   → Suggests diminishing returns to aggregation")
            elif r_squared_linear > r_squared_log:
                diff = r_squared_linear - r_squared_log
                print(f"   → Linear model fits better (ΔR² = +{diff:.4f})")
                print(f"   → Suggests constant rate of variance increase")
            else:
                print(f"   → Models fit equally well")
        
        print()
        
        if slope > 0 and significant_linear:
            print("✓ HYPOTHESIS SUPPORTED")
            print("  Variance INCREASES with aggregation")
            print("  → Coarse data introduces structural uncertainty")
        elif slope < 0 and significant_linear:
            print("✗ HYPOTHESIS REJECTED")
            print("  Variance DECREASES with aggregation")
            print("  → Large groups may average out extremes (Law of Large Numbers)")
        else:
            print("~ NO SIGNIFICANT TREND")
            print("  Variance is stable across aggregation levels")
        
        all_results.append({
            'threshold': threshold,
            'linear_slope': slope,
            'linear_intercept': intercept,
            'linear_r_squared': r_squared_linear,
            'linear_p_value': p_value_linear,
            'linear_significant': significant_linear,
            'log_a': log_a,
            'log_b': log_b,
            'log_r_squared': r_squared_log,
            'better_model': 'log' if r_squared_log and r_squared_log > r_squared_linear else 'linear'
        })
    
    return all_results


def plot_results(variance_df, test_results=None, save_path='variance_plot.png'):
    """Create simple, beautiful plots with both linear and log fits"""
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
    
    # Prepare x values for smooth curves
    if len(agg_levels) > 1:
        x_smooth = np.linspace(min(agg_levels), max(agg_levels), 100)
    
    # 1. Between-network CV
    ax = axes[0]
    plot_df['threshold'] = round(plot_df['threshold'], 2)
    
    for idx, threshold in enumerate(sorted(plot_df['threshold'].unique())):
        color = colors[idx % len(colors)]
        data = plot_df[plot_df['threshold'] == threshold]
        agg_avg = data.groupby('aggregation_level')['cv_between'].mean()
        threshold_label = thresholds[idx]
        
        # Scatter points
        ax.scatter(agg_levels, agg_avg.values, 
                  s=80, color=color, label=f'{threshold_label}', 
                  edgecolors='white', linewidth=2, zorder=3)
        
        # Fit curves if we have test results
        if test_results and len(agg_levels) > 1:
            threshold_results = [r for r in test_results if round(r['threshold'], 2) == threshold]
            if threshold_results:
                result = threshold_results[0]
                
                # Linear fit (solid line)
                y_linear = result['linear_slope'] * x_smooth + result['linear_intercept']
                ax.plot(x_smooth, y_linear, 
                       linestyle='-', linewidth=1.5, color=color, alpha=0.3,
                       label=f'{threshold_label} (linear R²={result["linear_r_squared"]:.3f})')
                
                # Log fit (dashed line) if available
                if result['log_r_squared'] is not None:
                    y_log = result['log_a'] + result['log_b'] * np.log(x_smooth)
                    ax.plot(x_smooth, y_log, 
                           linestyle='--', linewidth=1.5, color=color, alpha=0.3,
                           label=f'{threshold_label} (log R²={result["log_r_squared"]:.3f})')
    
    ax.set_xlabel('Level of aggregation', fontsize=11)
    ax.set_ylabel('Between-Network CV', fontsize=11)
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
    ax.legend(frameon=False, title = 'Contagion thresholds')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.show()


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
    
    # Test H1 (now returns list of results for all thresholds)
    test_results = test_h1(variance_df)
    
    # Summary table
    print_summary(variance_df)
    
    # Optional: Show breakdown by structure
    if show_structure_breakdown:
        structure_df = calculate_variance_by_structure(df, use_order=use_order)
    else:
        structure_df = None
    
    # Plot (now with test results for curves)
    plot_results(variance_df, test_results=test_results)
    
    # Save results
    variance_df.to_csv('variance_metrics.csv', index=False)
    print("\nResults saved to: variance_metrics.csv")
    
    # Save regression results
    if test_results:
        results_df = pd.DataFrame(test_results)
        results_df.to_csv('regression_results.csv', index=False)
        print("Regression results saved to: regression_results.csv")
    
    if show_structure_breakdown and structure_df is not None:
        structure_df.to_csv('variance_by_structure.csv', index=False)
        print("Structure breakdown saved to: variance_by_structure.csv")
    
    return df, variance_df, test_results


if __name__ == "__main__":
    # Run analysis
    # Script automatically uses 'order' column if available
    # (low order = fine/granular, high order = coarse/aggregated)
    
    df, variance_df, results = main('modeling/analysis/parameter_sweep/checkpoint_sweep_0.1_4.csv')

    
    # Optional: Show breakdown by network structure
    # df, variance_df, results = main('results.csv', show_structure_breakdown=True)