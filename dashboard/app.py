# =========================
# FIX IMPORT PATH (IMPORTANT)
# =========================
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# =========================
# IMPORTS
# =========================
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

from src.optimization import calculate_eoq, safety_stock

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Inventory Dashboard", layout="wide")

# =========================
# TITLE
# =========================
st.title("📦 Inventory Demand Forecasting Dashboard")

# =========================
# LOAD DATA (FIXED PATH)
# =========================
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
df = pd.read_csv(data_path)

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

store = st.sidebar.selectbox("Select Store", sorted(df['store_nbr'].unique()))
family = st.sidebar.selectbox("Select Product", sorted(df['family'].unique()))

df = df[(df['store_nbr'] == store) & (df['family'] == family)]

# =========================
# PREPROCESS
# =========================
df['date'] = pd.to_datetime(df['date'])
df = df.groupby('date')['sales'].sum().reset_index()
df = df.rename(columns={'date': 'ds', 'sales': 'y'})

# =========================
# MODEL
# =========================
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# =========================
# FORECAST GRAPH
# =========================
st.subheader("📊 Demand Forecast")

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df['ds'], df['y'], label="Actual")
ax.plot(forecast['ds'], forecast['yhat'], label="Forecast")
ax.legend()
ax.set_title("Demand Forecast")

st.pyplot(fig)

# =========================
# SIDEBAR PARAMETERS
# =========================
st.sidebar.header("Inventory Parameters")

ordering_cost = st.sidebar.slider("Ordering Cost", 10, 200, 50)
holding_cost = st.sidebar.slider("Holding Cost", 1, 10, 2)
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
current_inventory = st.sidebar.slider("Current Inventory", 0, 50000, 20000)

# =========================
# OPTIMIZATION
# =========================
future_demand = forecast.tail(30)['yhat'].values
total_demand = np.sum(future_demand)

eoq = calculate_eoq(D=total_demand, S=ordering_cost, H=holding_cost)

std_dev = np.std(df['y'])
ss = safety_stock(std_dev, lead_time)

reorder_point = ss + np.mean(future_demand) * lead_time

# =========================
# METRICS (CLEAN LAYOUT)
# =========================
st.subheader("📦 Inventory Recommendations")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Total Demand (30 days)", f"{int(total_demand)}")
col2.metric("EOQ", f"{int(eoq)}")
col3.metric("Safety Stock", f"{int(ss)}")
col4.metric("Reorder Point", f"{int(reorder_point)}")

# =========================
# ALERT SYSTEM
# =========================
st.subheader("🚨 Inventory Status")

if current_inventory < reorder_point:
    st.error("🚨 URGENT: Inventory below reorder point!")
    st.warning(f"Recommended Order Quantity: {int(eoq)} units")
else:
    st.success("✅ Inventory is healthy")

# =========================
# INSIGHTS
# =========================
st.subheader("💡 Business Insights")

st.write(f"""
- Expected demand next month is **{int(total_demand)} units**
- Optimal order size is **{int(eoq)} units**
- Maintain at least **{int(ss)} units** as safety stock
- Reorder when inventory reaches **{int(reorder_point)} units**
""")