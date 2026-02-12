"""
Analyze community structure from my_communities.json.

Shows community sizes and diversity metrics (unique groups, Shannon entropy,
Simpson's index, evenness) for each community.
"""
import json
import ast
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


def load_communities(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def shannon_entropy(counts):
    """Shannon entropy H = -sum(p * log(p)) over non-zero proportions."""
    total = sum(counts)
    if total == 0:
        return 0.0
    props = np.array([c / total for c in counts if c > 0])
    return -np.sum(props * np.log(props))


def simpson_index(counts):
    """Simpson's diversity index D = 1 - sum(p^2)."""
    total = sum(counts)
    if total <= 1:
        return 0.0
    props = np.array([c / total for c in counts])
    return 1.0 - np.sum(props ** 2)


def analyze(path="my_communities.json"):
    data = load_communities(path)
    num_communities = data["number_of_communities"]
    nodes_to_communities = data["nodes_to_communities"]
    communities_to_nodes = data["communities_to_nodes"]

    print(f"Number of communities: {num_communities}")
    print(f"Total nodes assigned: {len(nodes_to_communities)}")

    # --- Community sizes ---
    # Count nodes per community from nodes_to_communities
    comm_counts = Counter(nodes_to_communities.values())
    sizes = np.array([comm_counts.get(c, 0) for c in range(num_communities)])

    print(f"\n{'='*60}")
    print("COMMUNITY SIZES")
    print(f"{'='*60}")
    print(f"  Mean:   {np.mean(sizes):.1f}")
    print(f"  Std:    {np.std(sizes):.1f}")
    print(f"  Min:    {np.min(sizes)}")
    print(f"  Max:    {np.max(sizes)}")
    print(f"  Median: {np.median(sizes):.0f}")
    print(f"  CV:     {np.std(sizes) / np.mean(sizes):.3f}")

    empty = np.sum(sizes == 0)
    if empty:
        print(f"  Empty communities: {empty}")

    # --- Diversity per community ---
    # Build group composition per community from communities_to_nodes
    # Keys are strings like "(comm_id, group_id)"
    group_composition = {}  # comm_id -> Counter(group_id -> count)
    all_groups = set()
    for key_str, node_list in communities_to_nodes.items():
        comm_id, group_id = ast.literal_eval(key_str)
        count = len(node_list)
        if count == 0:
            continue
        if comm_id not in group_composition:
            group_composition[comm_id] = Counter()
        group_composition[comm_id][group_id] = count
        all_groups.add(group_id)

    n_groups_total = len(all_groups)
    max_entropy = np.log(n_groups_total) if n_groups_total > 1 else 1.0

    print(f"\n{'='*60}")
    print("COMMUNITY DIVERSITY")
    print(f"{'='*60}")
    print(f"  Total unique groups in data: {n_groups_total}")

    unique_groups_per_comm = []
    entropies = []
    simpsons = []
    evenness_vals = []

    for c in range(num_communities):
        comp = group_composition.get(c, Counter())

        counts = list(comp.values())
        n_unique = len(comp)
        unique_groups_per_comm.append(n_unique)

        h = shannon_entropy(counts)
        entropies.append(h)

        d = simpson_index(counts)
        simpsons.append(d)

        e = h / max_entropy if max_entropy > 0 else 0.0
        evenness_vals.append(e)

    unique_groups_per_comm = np.array(unique_groups_per_comm)
    unique_groups_per_comm_normalized = unique_groups_per_comm/n_groups_total
    entropies = np.array(entropies)
    simpsons = np.array(simpsons)
    evenness_vals = np.array(evenness_vals)

    print(f"\n  Unique groups per community:")
    print(f"    Mean:   {np.mean(unique_groups_per_comm):.1f}")
    print(f"    Std:    {np.std(unique_groups_per_comm):.1f}")
    print(f"    Min:    {np.min(unique_groups_per_comm)}")
    print(f"    Max:    {np.max(unique_groups_per_comm)}")

    print(f"\n  Unique groups per community normalized:")
    print(f"    Mean:   {np.mean(unique_groups_per_comm_normalized):.1f}")
    print(f"    Std:    {np.std(unique_groups_per_comm_normalized):.1f}")
    print(f"    Min:    {np.min(unique_groups_per_comm_normalized)}")
    print(f"    Max:    {np.max(unique_groups_per_comm_normalized)}")

    print(f"\n  Shannon entropy (H):")
    print(f"    Mean:   {np.mean(entropies):.3f}")
    print(f"    Std:    {np.std(entropies):.3f}")
    print(f"    Min:    {np.min(entropies):.3f}")
    print(f"    Max:    {np.max(entropies):.3f}")
    print(f"    Max possible (log {n_groups_total}): {max_entropy:.3f}")

    print(f"\n  normalized Shannon entropy (H):")
    print(f"    Mean:   {np.mean(entropies/max_entropy):.3f}")
    print(f"    Std:    {np.std(entropies/max_entropy):.3f}")
    print(f"    Min:    {np.min(entropies/max_entropy):.3f}")
    print(f"    Max:    {np.max(entropies/max_entropy):.3f}")
    print(f"    Max possible (log {n_groups_total}): {max_entropy:.3f}")

    print(f"\n  Pielou's evenness (H / H_max):")
    print(f"    Mean:   {np.mean(evenness_vals):.3f}")
    print(f"    Std:    {np.std(evenness_vals):.3f}")

    print(f"\n  Simpson's diversity index (1 - sum(p^2)):")
    print(f"    Mean:   {np.mean(simpsons):.3f}")
    print(f"    Std:    {np.std(simpsons):.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Community Analysis", fontsize=14)

    # 1. Size distribution
    ax = axes[0, 0]
    ax.hist(sizes, bins=25, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(sizes), color="red", linestyle="--", label=f"Mean={np.mean(sizes):.0f}")
    ax.set_xlabel("Community size (nodes)")
    ax.set_ylabel("Count")
    ax.set_title("Community size distribution")
    ax.legend()

    # 2. Unique groups per community
    ax = axes[0, 1]
    ax.hist(unique_groups_per_comm, bins=25, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Unique groups")
    ax.set_ylabel("Count")
    ax.set_title("Number of unique groups per community")

    # 3. Shannon entropy per community
    ax = axes[1, 0]
    ax.bar(range(num_communities), entropies, color="green", alpha=0.7)
    ax.axhline(max_entropy, color="red", linestyle="--", label=f"Max H={max_entropy:.2f}")
    ax.axhline(np.mean(entropies), color="blue", linestyle="--", label=f"Mean H={np.mean(entropies):.2f}")
    ax.set_xlabel("Community ID")
    ax.set_ylabel("Shannon entropy")
    ax.set_title("Shannon entropy per community")
    ax.legend()

    # 4. Size vs diversity scatter
    ax = axes[1, 1]
    sc = ax.scatter(sizes, simpsons, c=unique_groups_per_comm, cmap="viridis", alpha=0.7, edgecolors="black")
    ax.set_xlabel("Community size")
    ax.set_ylabel("Simpson's diversity index")
    ax.set_title("Size vs diversity")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Unique groups")

    plt.tight_layout()
    plt.savefig("analysis/community_analysis.png", dpi=150)
    plt.show()
    print("\nPlot saved to analysis/community_analysis.png")


if __name__ == "__main__":
    analyze()
