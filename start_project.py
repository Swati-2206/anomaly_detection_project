import pandas as pd

df = pd.read_csv("synthetic_trades.csv")

print(df.head())
print("\nTotal trades:", len(df))
print("Unique accounts:", df["account_id"].nunique())
print("Symbols:", df["symbol"].unique())
