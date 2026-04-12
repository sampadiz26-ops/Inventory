import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# =========================
# STEP 1: Load + preprocess
# =========================
df = pd.read_csv("data/train.csv")

# Filter one store + product
df = df[(df['family'] == 'GROCERY I') & (df['store_nbr'] == 1)]

df['date'] = pd.to_datetime(df['date'])
df = df.groupby('date')['sales'].sum().reset_index()

data = df['sales'].values.reshape(-1, 1)

# =========================
# STEP 2: Scaling
# =========================
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# =========================
# STEP 3: Create sequences
# =========================
def create_sequences(data, seq_length=14):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, seq_length=14)

# Convert to tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# =========================
# STEP 4: Train-Test Split
# =========================
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# =========================
# STEP 5: LSTM Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# STEP 6: Training
# =========================
epochs = 30

for epoch in range(epochs):
    model.train()

    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# =========================
# STEP 7: Prediction
# =========================
model.eval()

with torch.no_grad():
    pred = model(X_test).numpy()

# Inverse scaling
pred = scaler.inverse_transform(pred.reshape(-1, 1))
actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# =========================
# STEP 8: Plot
# =========================
plt.figure(figsize=(10,5))
plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("LSTM Prediction (Test Data)")
plt.show()