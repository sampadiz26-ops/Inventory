import pandas as pd

df = pd.read_csv("data/train.csv")

# Take small subset
df_small = df.sample(50000, random_state=42)

df_small.to_csv("data/train_small.csv", index=False)

print("Small dataset created successfully!")