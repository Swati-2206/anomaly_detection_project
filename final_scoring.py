import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

df = pd.read_csv("synthetic_trades.csv")

# Feature engineering
df["trade_value"] = df["price"] * df["quantity"]

# --- Statistical Score ---
df["z_score"] = (df["trade_value"] - df["trade_value"].mean()) / df["trade_value"].std()
df["stat_anomaly"] = np.where(abs(df["z_score"]) > 3, 1, 0)

# --- ML Score ---
features = df[["price", "quantity", "trade_value"]]
model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
df["ml_score"] = model.fit_predict(features)
df["ml_anomaly"] = df["ml_score"].apply(lambda x: 1 if x == -1 else 0)

# --- Final Risk Score ---
df["risk_score"] = df["stat_anomaly"] + df["ml_anomaly"]

df["priority"] = df["risk_score"].apply(
    lambda x: "HIGH" if x == 2 else ("MEDIUM" if x == 1 else "LOW")
)

print("High Priority Alerts:", len(df[df["priority"] == "HIGH"]))
print("Medium Priority Alerts:", len(df[df["priority"] == "MEDIUM"]))
print("Low Risk Trades:", len(df[df["priority"] == "LOW"]))

print("\nTop High Priority Cases:")
print(df[df["priority"] == "HIGH"].head())
