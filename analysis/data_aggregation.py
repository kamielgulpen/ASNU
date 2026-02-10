import pandas as pd
from itertools import combinations


# Read data
df = pd.read_csv('Data/tab_n_(with oplniv).csv')

characteristics = ["geslacht", "lft", "etngrp", "oplniv"]

# Loop through all possible subset sizes
for r in range(1, len(characteristics) + 1):
    # Get all combinations of size r
    for combo in combinations(characteristics, r):
        group_cols = list(combo)
        
        # Aggregate
        df_agg = df.groupby(group_cols, as_index=False)['n'].sum()
        
        # Save with descriptive filename
        filename = "_".join(group_cols) + ".csv"
        df_agg.to_csv(f'Data/aggregated/tab_n_{filename}', index=False)
        
        print(f"Created: {filename} - Total: {df_agg['n'].sum()}")

filenames = ["werkschool", "huishouden","familie", "buren"]
for file in filenames:
    df = pd.read_csv(f'Data/tab_{file}.csv')
    # Loop through all possible subset sizes (including keeping all)
    for r in range(1, len(characteristics) + 1):
        # Get all combinations of size r
        for combo in combinations(characteristics, r):
            # Build column names with _src and _dest suffixes
            src_cols = [f"{char}_src" for char in combo]
            dest_cols = [f"{char}_dst" for char in combo]
            group_cols = src_cols + dest_cols
            
            # Aggregate
            df_agg = df.groupby(group_cols, as_index=False)['n'].sum()
            
            # Save with descriptive filename
            filename = "_".join(combo) + ".csv"
            df_agg.to_csv(f'Data/aggregated/tab_{file}_{filename}', index=False)
            
            print(f"Created: {filename} - Total: {df_agg['n'].sum()}")