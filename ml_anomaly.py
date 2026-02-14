import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv("synthetic_trades.csv")

# Feature engineering
df["trade_value"] = df["price"] * df["quantity"]

features = df[["price", "quantity", "trade_value"]]

# Isolation Forest Model
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,  # 2% anomalies expected
    random_state=42
)

df["ml_score"] = model.fit_predict(features)

# Convert (-1 = anomaly, 1 = normal)
df["ml_anomaly"] = df["ml_score"].apply(lambda x: 1 if x == -1 else 0)

print("ML Anomalies Detected:", df["ml_anomaly"].sum())

print("\nSample ML Anomalies:")
print(df[df["ml_anomaly"] == 1].head())
