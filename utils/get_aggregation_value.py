import itertools
import pandas as pd

characteristics = sorted(["geslacht", "lft", "etngrp", "oplniv"])

def get_combinations():
    for r in range(1, len(characteristics) + 1):
        for combo in itertools.combinations(characteristics, r):
            yield combo

combos = get_combinations()
base_path = "Data/aggregated/tab_n_"

results = []

for combo in combos:
    group_cols = list(combo)
    char_str = '_'.join(group_cols)
    df = pd.read_csv(base_path+char_str+".csv")
    
    # Calculate shares
    shares = df['n'] / df['n'].sum()
    
    # Calculate HHI
    hhi = (shares ** 2).sum()
    
    biggest_group = max(df['n']/df['n'].sum())

    results.append({
        'aggregation': char_str,
        'n_groups': len(df),
        'hhi': hhi,
        'bg': biggest_group,
        'effective_n': 1/hhi
    })

results_df = pd.DataFrame(results)
results_df.to_csv("group_charcteristic_mapping.csv")
print(results_df.sort_values("hhi"))