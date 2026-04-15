#!/usr/bin/env python3
"""
Combine task_0.csv through task_20.csv into a single dataset
"""

import pandas as pd
from pathlib import Path

def combine_task_files(output_file="combined_tasks.csv"):
    """
    Read and combine task_0.csv through task_20.csv
    
    Args:
        output_file: Name of the output CSV file
    """
    all_data = []
    
    # Read each task file
    for i in range(100):  # 0 to 20 inclusive
        filename = f"task_{i}.csv"
        
        try:
            df = pd.read_csv(filename)
            all_data.append(df)
            print(f"✓ Loaded {filename}: {len(df)} rows")
        except FileNotFoundError:
            print(f"✗ Warning: {filename} not found, skipping...")
        except Exception as e:
            print(f"✗ Error reading {filename}: {e}")
    
    if not all_data:
        print("No files were loaded successfully!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*50}")
    print(f"Total rows combined: {len(combined_df)}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    print(f"{'='*50}\n")
    
    # Save combined data
    combined_df.to_csv(output_file, index=False)
    print(f"✓ Saved to {output_file}")
    
    return combined_df

if __name__ == "__main__":
    # Run the combination
    df = combine_task_files()
    
    # Optional: Display first few rows
    if df is not None:
        print("\nFirst 5 rows of combined data:")
        print(df.head())