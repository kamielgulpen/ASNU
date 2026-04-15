import itertools
import pandas as pd

characteristics = sorted([
    "geslacht", "lft", "etngrp", "oplniv", "inkomensniveau", 
    "arbeidsstatus", "uitkeringstype", "burgerlijke_staat"
])

def get_all_permutations(characteristics):
    for r in range(1, len(characteristics) + 1):
        for perm in itertools.permutations(characteristics, r):
            yield perm

combos = get_all_permutations(characteristics)
base_path = "data/enriched/aggregated/pop_"
results = []

for combo in combos:
    group_cols = list(combo)
    char_str = '_'.join(group_cols)
    
    try:
        df = pd.read_csv(base_path + char_str + ".csv")
        
        # Calculate shares
        shares = df['n'] / df['n'].sum()
        
        # Calculate HHI
        hhi = (shares ** 2).sum()
        
        biggest_group = (df['n'] / df['n'].sum()).max()
        
        results.append({
            'aggregation': char_str,
            'n_groups': len(df),
            'hhi': hhi,
            'bg': biggest_group,
            'effective_n': 1 / hhi
        })
        
        print(f"Processed: {char_str} ({len(results)} done)")
        
    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"Error with {char_str}: {e}")
        continue

# Save results once at the end
results_df = pd.DataFrame(results)
results_df.to_csv("group_charcteristic_mapping.csv", index=False)
print(f"\nSaved {len(results)} results to group_charcteristic_mapping.csv")