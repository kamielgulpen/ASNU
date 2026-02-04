import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 100000

# 1. Log-normal with large sigma (extremely skewed)
sigma_values = [1, 2, 3, 4]
lognormal_data = {}

for sigma in sigma_values:
    data = np.random.lognormal(mean=0, sigma=sigma, size=n_samples)
    theoretical_skew = (np.exp(sigma**2) + 2) * np.sqrt(np.exp(sigma**2) - 1)
    empirical_skew = skew(data)
    lognormal_data[sigma] = {
        'data': data,
        'theoretical': theoretical_skew,
        'empirical': empirical_skew
    }

# 2. Power-law distribution (α = 5, just above threshold)
alpha = 5
xmin = 1
power_law_data = xmin * (1 - np.random.uniform(0, 1, n_samples)) ** (-1/(alpha-1))
# Theoretical skewness for power law: 2*sqrt(α-3)/(α-5) for α > 5
# For α = 5, it's at the boundary, so very large
power_law_skew = skew(power_law_data)

# 3. Exponential distribution (skewness = 2)
exponential_data = np.random.exponential(scale=1, size=n_samples)
exp_skew = skew(exponential_data)

# 4. Custom extreme distribution: mixture of normal and extreme tail
normal_part = np.random.normal(0, 1, int(n_samples * 0.95))
extreme_tail = np.random.exponential(10, int(n_samples * 0.05)) + 5
custom_data = np.concatenate([normal_part, extreme_tail])
custom_skew = skew(custom_data)

# Print results
print("=" * 60)
print("SKEWNESS COMPARISON")
print("=" * 60)

print("\n1. LOG-NORMAL DISTRIBUTIONS:")
print("-" * 60)
for sigma, info in lognormal_data.items():
    print(f"   σ = {sigma}:")
    print(f"   - Theoretical skewness: {info['theoretical']:.2f}")
    print(f"   - Empirical skewness:   {info['empirical']:.2f}")

print("\n2. POWER-LAW DISTRIBUTION (α = 5):")
print("-" * 60)
print(f"   - Empirical skewness: {power_law_skew:.2f}")
print(f"   - Note: At α = 5, skewness is theoretically infinite/undefined")

print("\n3. EXPONENTIAL DISTRIBUTION:")
print("-" * 60)
print(f"   - Theoretical skewness: 2.00")
print(f"   - Empirical skewness:   {exp_skew:.2f}")

print("\n4. CUSTOM EXTREME DISTRIBUTION:")
print("-" * 60)
print(f"   - Empirical skewness: {custom_skew:.2f}")

print("\n" + "=" * 60)
print(f"WINNER: Log-normal with σ = {max(sigma_values)}")
print(f"Skewness = {lognormal_data[max(sigma_values)]['theoretical']:.2f}")
print("=" * 60)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Most Skewed Distributions', fontsize=16, fontweight='bold')

# Plot 1-4: Log-normal distributions
for idx, sigma in enumerate(sigma_values[:4]):
    ax = axes[idx // 3, idx % 3]
    data = lognormal_data[sigma]['data']
    
    # Plot with clipping for visualization
    ax.hist(data[data < np.percentile(data, 99)], bins=100, 
            alpha=0.7, color='steelblue', edgecolor='black')
    ax.set_title(f'Log-normal (σ={sigma})\nSkewness = {lognormal_data[sigma]["empirical"]:.1f}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(alpha=0.3)

# Plot 5: Power-law
ax = axes[1, 1]
ax.hist(power_law_data[power_law_data < np.percentile(power_law_data, 99)], 
        bins=100, alpha=0.7, color='coral', edgecolor='black')
ax.set_title(f'Power-law (α=5)\nSkewness = {power_law_skew:.1f}')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.3)

# Plot 6: Custom extreme
ax = axes[1, 2]
ax.hist(custom_data, bins=100, alpha=0.7, color='mediumseagreen', edgecolor='black')
ax.set_title(f'Custom Extreme\nSkewness = {custom_skew:.1f}')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Create comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))

distributions = []
skewness_values = []

for sigma, info in lognormal_data.items():
    distributions.append(f'Log-normal\n(σ={sigma})')
    skewness_values.append(info['empirical'])

distributions.extend(['Power-law\n(α=5)', 'Exponential', 'Custom'])
skewness_values.extend([power_law_skew, exp_skew, custom_skew])

colors = ['steelblue'] * 4 + ['coral', 'orange', 'mediumseagreen']
bars = ax.bar(distributions, skewness_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

# Highlight the maximum
max_idx = np.argmax(skewness_values)
bars[max_idx].set_edgecolor('red')
bars[max_idx].set_linewidth(4)

ax.set_ylabel('Skewness', fontsize=12, fontweight='bold')
ax.set_title('Skewness Comparison - Most Skewed Distributions', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, max(skewness_values) * 1.1)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, skewness_values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}',
            ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.show()