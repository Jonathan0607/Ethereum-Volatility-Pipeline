import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_features(df):
    # 1. Log Returns (Price Stationarity)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Realized Volatility (Rolling Window)
    # Annualized volatility for a 24-hour window
    window = 24
    df['volatility'] = df['log_ret'].rolling(window=window).std() * np.sqrt(365 * 24)
    
    # Drop NaNs created by rolling window
    df.dropna(inplace=True)
    return df

def check_stationarity(series):
    """
    Performs Augmented Dickey-Fuller (ADF) test.
    Returns True if stationary (p-value < 0.05).
    """
    result = adfuller(series.values)
    p_value = result[1]
    print(f"ADF P-Value: {p_value:.5f}")
    if p_value < 0.05:
        print("Result: Data is STATIONARY (Ready for LSTM)")
        return True
    else:
        print("Result: Data is NON-STATIONARY (Needs differencing)")
        return False

if __name__ == "__main__":
    # Create dummy data if file doesn't exist yet
    try:
        df = pd.read_csv('data/eth_hourly.csv')
    except FileNotFoundError:
        print("Run fetch_data.py first!")
        exit()
        
    df = calculate_features(df)
    
    print("Checking Close Price Stationarity:")
    check_stationarity(df['close'])
    
    print("\nChecking Log-Returns Stationarity:")
    check_stationarity(df['log_ret'])