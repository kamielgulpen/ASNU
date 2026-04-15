"""
Analyze network statistics from JSON files
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_all_json_files(base_dir='enriched'):
    """Load all JSON files from directory structure"""
    data = []
    
    base_path = Path(base_dir)
    json_files = list(base_path.rglob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                stats = json.load(f)
                
                # Extract folder and filename
                stats['folder'] = json_file.parent.name
                stats['filename'] = json_file.stem  # filename without extension
                stats['file_path'] = str(json_file)
                
                data.append(stats)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} network statistics")
    print(f"Unique filenames: {df['filename'].nunique()}")
    
    return df


def parse_params(df):
    """Parse parameter string into columns"""
    # Extract parameters from params string
    param_patterns = {
        'scale': r'scale=(\d+)',
        'comms': r'comms=(\d+)',
        'recip': r'recip=(\d+)',
        'trans': r'trans=(\d+)',
        'pa': r'pa=([\d.]+)',
        'bridge': r'bridge=([\d.]+)'
    }
    
    for param, pattern in param_patterns.items():
        df[param] = df['params'].str.extract(pattern).astype(float)
    
    return df


def analyze_by_filename(df):
    """Analyze statistics grouped by filename"""
    print("\n" + "="*60)
    print("BY FILENAME")
    print("="*60)
    
    grouped = df.groupby('filename').agg({
        'degree_skew': ['mean', 'std', 'min', 'max'],
        'transitivity': ['mean', 'std', 'min', 'max'],
        'reciprocity': ['mean', 'std'],
        'degree_mean': ['mean', 'std'],
        'nodes': ['mean', 'count']
    }).round(4)
    
    print(grouped.to_string())
    
    # Save detailed stats
    grouped.to_csv('stats_by_filename.csv')
    print("\nSaved: stats_by_filename.csv")
    
    return grouped


def plot_distributions_by_filename(df, save_dir='network_plots'):
    """Create distribution plots for each filename"""
    Path(save_dir).mkdir(exist_ok=True)
    
    filenames = sorted(df['filename'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(filenames)))
    
    # 1. Overall comparison plot (like the uploaded image)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Skewness distribution
    ax = axes[0, 0]
    for idx, fname in enumerate(filenames):
        data = df[df['filename'] == fname]['degree_skew']
        ax.hist(data, bins=20, alpha=0.5, label=fname[:20], color=colors[idx])
    ax.set_xlabel('Degree Skewness')
    ax.set_ylabel('Frequency')
    ax.set_title('Skewness Distribution by File')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Transitivity distribution
    ax = axes[0, 1]
    for idx, fname in enumerate(filenames):
        data = df[df['filename'] == fname]['transitivity']
        ax.hist(data, bins=20, alpha=0.5, label=fname[:20], color=colors[idx])
    ax.set_xlabel('Transitivity')
    ax.set_ylabel('Frequency')
    ax.set_title('Transitivity Distribution by File')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Skewness vs Transitivity
    ax = axes[0, 2]
    for idx, fname in enumerate(filenames):
        data = df[df['filename'] == fname]
        ax.scatter(data['degree_skew'], data['transitivity'], 
                  alpha=0.6, s=30, label=fname[:20], color=colors[idx])
    ax.set_xlabel('Degree Skewness')
    ax.set_ylabel('Transitivity')
    ax.set_title('Skewness vs Transitivity')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Skewness boxplot
    ax = axes[1, 0]
    skew_data = [df[df['filename'] == fname]['degree_skew'].values for fname in filenames]
    bp = ax.boxplot(skew_data, labels=[f[:15] for f in filenames], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Degree Skewness')
    ax.set_title('Skewness by File')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Transitivity boxplot
    ax = axes[1, 1]
    trans_data = [df[df['filename'] == fname]['transitivity'].values for fname in filenames]
    bp = ax.boxplot(trans_data, labels=[f[:15] for f in filenames], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Transitivity')
    ax.set_title('Transitivity by File')
    ax.tick_params(axis='x', rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Mean degree vs skewness
    ax = axes[1, 2]
    for idx, fname in enumerate(filenames):
        data = df[df['filename'] == fname]
        ax.scatter(data['degree_mean'], data['degree_skew'], 
                  alpha=0.6, s=30, label=fname[:20], color=colors[idx])
    ax.set_xlabel('Mean Degree')
    ax.set_ylabel('Degree Skewness')
    ax.set_title('Mean Degree vs Skewness')
    ax.legend(fontsize=7, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparison_by_filename.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/comparison_by_filename.png")
    plt.close()


def plot_individual_filenames(df, save_dir='network_plots'):
    """Create detailed plots for each filename"""
    filenames_dir = Path(save_dir) / 'by_filename'
    filenames_dir.mkdir(exist_ok=True, parents=True)
    
    for fname in df['filename'].unique():
        file_data = df[df['filename'] == fname]
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle(f'{fname}', fontsize=12, fontweight='bold')
        
        # Skewness distribution
        ax = axes[0, 0]
        ax.hist(file_data['degree_skew'], bins=15, color='#4CAF50', alpha=0.7, edgecolor='white')
        ax.axvline(file_data['degree_skew'].mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Degree Skewness')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Mean: {file_data["degree_skew"].mean():.3f}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Transitivity distribution
        ax = axes[0, 1]
        ax.hist(file_data['transitivity'], bins=15, color='#42A5F5', alpha=0.7, edgecolor='white')
        ax.axvline(file_data['transitivity'].mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Transitivity')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Mean: {file_data["transitivity"].mean():.4f}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Skewness vs Transitivity
        ax = axes[1, 0]
        ax.scatter(file_data['degree_skew'], file_data['transitivity'], 
                  alpha=0.6, s=50, color='#FF5722')
        ax.set_xlabel('Degree Skewness')
        ax.set_ylabel('Transitivity')
        ax.set_title('Skewness vs Transitivity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Network properties
        ax = axes[1, 1]
        ax.scatter(file_data['reciprocity'], file_data['transitivity'], 
                  alpha=0.6, s=50, color='#F4E04D')
        ax.set_xlabel('Reciprocity')
        ax.set_ylabel('Transitivity')
        ax.set_title('Reciprocity vs Transitivity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        safe_fname = fname.replace('/', '_').replace('\\', '_')
        plt.savefig(filenames_dir / f'{safe_fname}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual plots to: {filenames_dir}/")


def plot_overall_analysis(df, save_dir='network_plots'):
    """Overall analysis plots focusing on skewness and transitivity"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Overall Network Statistics', fontsize=14, fontweight='bold')
    
    # 1. Skewness distribution
    ax = axes[0, 0]
    ax.hist(df['degree_skew'], bins=30, color='#4CAF50', alpha=0.7, edgecolor='white')
    ax.axvline(df['degree_skew'].mean(), color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {df["degree_skew"].mean():.2f}')
    ax.set_xlabel('Degree Skewness')
    ax.set_ylabel('Frequency')
    ax.set_title('Degree Skewness Distribution')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Transitivity distribution
    ax = axes[0, 1]
    ax.hist(df['transitivity'], bins=30, color='#42A5F5', alpha=0.7, edgecolor='white')
    ax.axvline(df['transitivity'].mean(), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {df["transitivity"].mean():.4f}')
    ax.set_xlabel('Transitivity')
    ax.set_ylabel('Frequency')
    ax.set_title('Transitivity Distribution')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Skewness vs Transitivity
    ax = axes[0, 2]
    ax.scatter(df['degree_skew'], df['transitivity'], alpha=0.5, s=30, color='#FF5722')
    ax.set_xlabel('Degree Skewness')
    ax.set_ylabel('Transitivity')
    ax.set_title('Skewness vs Transitivity')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Reciprocity distribution
    ax = axes[1, 0]
    ax.hist(df['reciprocity'], bins=30, color='#F4E04D', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Reciprocity')
    ax.set_ylabel('Frequency')
    ax.set_title('Reciprocity Distribution')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 5. Mean degree vs Skewness
    ax = axes[1, 1]
    ax.scatter(df['degree_mean'], df['degree_skew'], alpha=0.5, s=30, color='#9C27B0')
    ax.set_xlabel('Mean Degree')
    ax.set_ylabel('Degree Skewness')
    ax.set_title('Mean Degree vs Skewness')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 6. Network size
    ax = axes[1, 2]
    ax.scatter(df['nodes'], df['edges'], alpha=0.5, s=30, color='#FF9800')
    ax.set_xlabel('Nodes')
    ax.set_ylabel('Edges')
    ax.set_title('Network Size')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/overall_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/overall_analysis.png")
    plt.close()


def summary_statistics(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    print(f"\nNetworks: {len(df)}")
    print(f"Unique filenames: {df['filename'].nunique()}")
    
    print("\nSkewness:")
    print(f"  Mean: {df['degree_skew'].mean():.3f} ± {df['degree_skew'].std():.3f}")
    print(f"  Range: [{df['degree_skew'].min():.3f}, {df['degree_skew'].max():.3f}]")
    
    print("\nTransitivity:")
    print(f"  Mean: {df['transitivity'].mean():.4f} ± {df['transitivity'].std():.4f}")
    print(f"  Range: [{df['transitivity'].min():.4f}, {df['transitivity'].max():.4f}]")
    
    print("\nReciprocity:")
    print(f"  Mean: {df['reciprocity'].mean():.4f} ± {df['reciprocity'].std():.4f}")


def main(base_dir='enriched'):
    """Main analysis function"""
    print("="*60)
    print("NETWORK STATISTICS ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_all_json_files(base_dir)
    df = df[df['nodes'] > 100000]

    df = parse_params(df)
    
    # Overall summary
    summary_statistics(df)
    
    # Analyze by filename
    analyze_by_filename(df)
    
    # Create plots
    plot_overall_analysis(df)
    plot_distributions_by_filename(df)
    plot_individual_filenames(df)
    
    # Save full dataset
    df.to_csv('network_statistics_full.csv', index=False)
    print(f"\nFull data saved to: network_statistics_full.csv")
    
    return df


if __name__ == "__main__":
    df = main('enriched')