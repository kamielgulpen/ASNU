"""
Test script to demonstrate stratified sampling improvement.

This compares the old ceiling-based approach vs. the new stratified allocation.
"""

import numpy as np
import pandas as pd

def old_approach(group_sizes, scale):
    """Old approach: simple ceiling for each group."""
    allocations = {}
    for group_id, size in enumerate(group_sizes):
        allocations[group_id] = int(np.ceil(scale * size))
    return allocations

def new_stratified_approach(group_sizes, scale):
    """New approach: stratified allocation with remainder distribution."""
    total_pop = sum(group_sizes)
    target_total = int(scale * total_pop)

    # Allocate proportionally (floor)
    allocations = {}
    allocated_total = 0

    for group_id, size in enumerate(group_sizes):
        base_allocation = int(scale * size)
        allocations[group_id] = base_allocation
        allocated_total += base_allocation

    # Distribute remainder to largest groups
    remainder = target_total - allocated_total
    if remainder > 0:
        group_order = sorted(enumerate(group_sizes), key=lambda x: x[1], reverse=True)
        for i in range(remainder):
            group_id = group_order[i % len(group_order)][0]
            allocations[group_id] += 1

    return allocations

def analyze_approach(name, allocations, original_sizes, scale):
    """Analyze how well an approach preserves proportions."""
    print(f"\n{name}")
    print("=" * 60)

    total_original = sum(original_sizes)
    total_allocated = sum(allocations.values())

    print(f"Original total: {total_original}")
    print(f"Target total (scale={scale}): {int(scale * total_original)}")
    print(f"Actual allocated: {total_allocated}")
    print(f"Difference: {total_allocated - int(scale * total_original):+d}")

    print(f"\nGroup-by-group analysis:")
    print(f"{'Group':<8} {'Original':<10} {'Expected':<10} {'Allocated':<10} {'% Error':<10}")
    print("-" * 60)

    errors = []
    for group_id in sorted(allocations.keys()):
        original = original_sizes[group_id]
        expected = scale * original
        allocated = allocations[group_id]

        # Calculate percentage error (allocated vs original proportion)
        original_prop = original / total_original
        allocated_prop = allocated / total_allocated if total_allocated > 0 else 0
        pct_error = ((allocated_prop - original_prop) / original_prop * 100) if original_prop > 0 else 0

        errors.append(abs(pct_error))

        print(f"{group_id:<8} {original:<10} {expected:<10.2f} {allocated:<10} {pct_error:+.2f}%")

    print(f"\nMean absolute % error: {np.mean(errors):.2f}%")
    print(f"Max absolute % error: {np.max(errors):.2f}%")

    return errors

# Example: Real-world demographic groups
print("\n" + "="*60)
print("STRATIFIED SAMPLING COMPARISON")
print("="*60)

# Simulate a population with diverse group sizes (like age groups, education levels, etc.)
group_sizes = [1000, 500, 250, 100, 50, 25, 10, 5, 3, 2]
scale = 0.1

print(f"\nScenario: {len(group_sizes)} demographic groups")
print(f"Group sizes: {group_sizes}")
print(f"Scale factor: {scale}")

old_alloc = old_approach(group_sizes, scale)
new_alloc = new_stratified_approach(group_sizes, scale)

old_errors = analyze_approach("OLD APPROACH (ceiling)", old_alloc, group_sizes, scale)
new_errors = analyze_approach("NEW APPROACH (stratified)", new_alloc, group_sizes, scale)

# Summary comparison
print("\n" + "="*60)
print("IMPROVEMENT SUMMARY")
print("="*60)
print(f"Mean absolute error reduction: {np.mean(old_errors) - np.mean(new_errors):.2f}%")
print(f"Max error reduction: {np.max(old_errors) - np.max(new_errors):.2f}%")

# Show which approach over-allocates
old_total = sum(old_alloc.values())
new_total = sum(new_alloc.values())
target = int(scale * sum(group_sizes))

print(f"\nTotal nodes:")
print(f"  Target:        {target}")
print(f"  Old approach:  {old_total} ({old_total - target:+d} nodes)")
print(f"  New approach:  {new_total} ({new_total - target:+d} nodes)")
print(f"\nThe new approach maintains exact totals and preserves proportions better!")
