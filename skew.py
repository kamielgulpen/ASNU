import pandas as pd

df = pd.read_csv("community_edge_distribution.csv")
df2 = pd.read_csv("Data/tab_buren.csv")
df3 = pd.read_csv("community_node_distribution.csv")


print(df["n"].sum(), df2["n"].sum(), df3["n"].sum())
