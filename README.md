# 📦 Inventory Demand Forecasting & Optimization System

## 🚀 Overview

This project builds an end-to-end inventory management system that forecasts product demand and converts predictions into actionable inventory decisions.

It combines time-series forecasting with classical inventory optimization techniques to reduce stockouts and holding costs.

---

## 🎯 Objectives

* Predict future demand using time-series models
* Compare ML (LSTM) and statistical (Prophet) approaches
* Optimize inventory using EOQ and Safety Stock
* Provide actionable reorder decisions

---

## 🧠 Models Used

### 🔹 Prophet

* Captures trend and seasonality effectively
* Achieved ~11% MAE on real retail dataset

### 🔹 LSTM

* Deep learning approach for sequence modeling
* Compared against Prophet to analyze performance differences

---

## 📊 Dataset

* Retail sales dataset (Kaggle)
* Filtered for:

  * Store 1
  * Product category: GROCERY I

---

## ⚙️ Pipeline

1. Data preprocessing
2. Demand forecasting (Prophet & LSTM)
3. Model evaluation (MAE)
4. Inventory optimization:

   * Economic Order Quantity (EOQ)
   * Safety Stock
   * Reorder Point

---

## 📦 Inventory Optimization

* **EOQ**: Determines optimal order size
* **Safety Stock**: Handles demand uncertainty
* **Reorder Point**: Indicates when to reorder

---

## 📈 Results

* MAE: ~280 (~11% error)
* Monthly Demand: ~78,000 units
* EOQ: ~1,975 units
* Safety Stock: ~3,400 units
* Reorder Point: ~21,600 units

---

## 🧱 Project Structure

```
inventory-forecasting/
│
├── data/
├── src/
├── main.py
├── requirements.txt
└── README.md
```

---

## 💡 Key Insights

* Prophet outperformed LSTM for this dataset due to strong seasonality
* LSTM showed underfitting without feature engineering
* Inventory optimization adds real-world business value beyond prediction

---

## 🔥 Future Improvements

* Multi-product and multi-store support
* Real-time demand forecasting API
* Dashboard for visualization
* Feature engineering for LSTM improvement

---

## 🧑‍💻 Tech Stack

* Python
* Pandas, NumPy
* Prophet
* PyTorch
* Matplotlib

---

## 📌 Conclusion

This project demonstrates how machine learning can be integrated with operations research to build intelligent inventory systems.
