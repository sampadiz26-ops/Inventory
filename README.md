# 📦 Inventory Demand Forecasting & Optimization System

## 🌐 Live Demo

👉 https://your-app-name.streamlit.app

---

## 📸 Dashboard Preview


(dashboard/image.png)

---

## 🚀 Overview

This project builds an **end-to-end inventory management system** that forecasts product demand and converts predictions into actionable inventory decisions.

It combines **time-series forecasting (Prophet, LSTM)** with **operations research techniques (EOQ, Safety Stock)** to optimize inventory levels and reduce stockouts.

---

## 💡 Why This Project?

Retail businesses often struggle with balancing inventory:

* 📉 Overstock → high holding costs
* 📈 Understock → lost sales

This system uses **machine learning + optimization** to make **data-driven inventory decisions**, similar to real-world supply chain systems.

---

## 🧠 Models Used

### 🔹 Prophet

* Captures trend and seasonality effectively
* Achieved ~11% MAE on dataset
* Best performing model in this project

### 🔹 LSTM

* Deep learning model for time-series
* Used for comparison
* Showed underfitting due to limited feature engineering

---

## 📊 Dataset

* Retail sales dataset (Kaggle)
* Filtered for:

  * Store 1
  * Product category: GROCERY I
* Reduced dataset used for deployment

---

## ⚙️ Pipeline

1. Data preprocessing
2. Demand forecasting (Prophet & LSTM)
3. Model evaluation (MAE)
4. Inventory optimization:

   * EOQ (Economic Order Quantity)
   * Safety Stock
   * Reorder Point
5. Interactive dashboard visualization

---

## 📦 Inventory Optimization

* **EOQ** → Optimal order size minimizing cost
* **Safety Stock** → Buffer for demand uncertainty
* **Reorder Point** → When to place new order

---

## 📈 Results

* 📊 MAE: ~280 (~11% error)
* 📦 Monthly Demand: ~78,000 units
* 📦 EOQ: ~1,975 units
* 📦 Safety Stock: ~3,400 units
* 📦 Reorder Point: ~21,600 units

---

## 🚀 Features

* 📊 Demand forecasting using Prophet
* 🤖 LSTM model comparison
* 📦 Inventory optimization (EOQ + Safety Stock)
* ⚠️ Real-time inventory alerts
* 🎛️ Interactive dashboard (Streamlit)
* 🌐 Deployed web application

---

## 🧱 Project Structure

```
inventory-forecasting/
│
├── dashboard/          # Streamlit app
├── data/               # Dataset (small version for deployment)
├── src/                # Core modules
├── main.py             # Pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## 🧑‍💻 Tech Stack

* Python
* Pandas, NumPy
* Prophet
* PyTorch
* Matplotlib
* Streamlit

---

## 🔮 Future Improvements

* Multi-store & multi-product forecasting
* Feature engineering for LSTM
* Real-time data pipeline
* Cloud-based deployment with APIs
* Advanced ensemble models

---

## 📌 Conclusion

This project demonstrates how **machine learning + optimization** can be combined to build intelligent inventory systems that provide **actionable business decisions**, not just predictions.

---

## 👤 Author

Samrat Roy Choudhury
