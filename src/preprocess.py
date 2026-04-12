import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])

    # Sort
    df = df.sort_values('Date')

    # Aggregate sales per day
    df = df.groupby('Date')['Weekly_Sales'].sum().reset_index()

    # Rename for Prophet
    df = df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})

    return df