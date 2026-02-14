import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

accounts = [f"ACC_{i}" for i in range(20)]
symbols = ["AAPL", "TSLA", "MSFT", "AMZN"]

rows = []

for i in range(1000):

    # Normal trades
    price = round(np.random.normal(100, 5), 2)
    quantity = random.randint(1, 100)

    # Inject fraud in 2% of trades
    if random.random() < 0.02:
        quantity = random.randint(500, 2000)  # huge suspicious trades

    rows.append({
        "trade_id": i,
        "account_id": random.choice(accounts),
        "symbol": random.choice(symbols),
        "price": price,
        "quantity": quantity,
        "side": random.choice(["BUY", "SELL"]),
        "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 10000))
    })

df = pd.DataFrame(rows)
df.to_csv("synthetic_trades.csv", index=False)

print("New dataset with fraud injected created.")
