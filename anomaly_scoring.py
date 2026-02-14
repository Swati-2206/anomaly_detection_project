import pandas as pd
import numpy as np

# load data
df = pd.read_csv("synthetic_trades.csv")

print("Total Trades:", len(df))

# create trade value
df["trade_value"] = df["price"] * df["quantity"]

# calculate mean and std
mean_value = df["trade_value"].mean()
std_value = df["trade_value"].std()

# calculate z-score
df["z_score"] = (df["trade_value"] - mean_value) / std_value

# mark anomalies
df["stat_anomaly"] = df["z_score"].abs() > 3

# print results
print("Anomalies Detected:", df["stat_anomaly"].sum())

print("\nSample Anomalies:")
print(df[df["stat_anomaly"]].head())
