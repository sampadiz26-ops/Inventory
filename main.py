import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from src.optimization import calculate_eoq, safety_stock

# =========================
# Load + preprocess
# =========================
df = pd.read_csv("data/train.csv")

df = df[(df['family'] == 'GROCERY I') & (df['store_nbr'] == 1)]

df['date'] = pd.to_datetime(df['date'])
df = df.groupby('date')['sales'].sum().reset_index()
df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# =========================
# Train Prophet
# =========================
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# =========================
# Evaluation
# =========================
merged = df.merge(forecast[['ds', 'yhat']], on='ds')
mae = mean_absolute_error(merged['y'], merged['yhat'])

# =========================
# Optimization
# =========================
future_demand = forecast.tail(30)['yhat'].values
total_demand = np.sum(future_demand)

eoq = calculate_eoq(D=total_demand, S=50, H=2)

std_dev = np.std(df['y'])
ss = safety_stock(std_dev, lead_time=7)

reorder_point = ss + np.mean(future_demand) * 7

# =========================
# Output
# =========================
print("\n===== MODEL PERFORMANCE =====")
print(f"MAE: {mae}")

print("\n===== INVENTORY DECISIONS =====")
print(f"Total Demand (30 days): {total_demand}")
print(f"EOQ: {eoq}")
print(f"Safety Stock: {ss}")
print(f"Reorder Point: {reorder_point}")