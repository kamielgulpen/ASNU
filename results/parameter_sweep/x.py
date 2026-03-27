import pandas as pd

df = pd.read_csv("results/parameter_sweep/checkpoint_sweep_0.1_2.csv")

df = df.sort_values("ratio")

df.to_csv("checkpoint_sweep_0.1_2.csv")