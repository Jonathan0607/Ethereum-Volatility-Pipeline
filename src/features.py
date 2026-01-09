import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_features(df):
    """
    Adds Log Returns and Realized Volatility to the dataframe.
    """
    df = df.copy()
    
    # 1. Log Returns (Price Stationarity)
    # Formula: ln(Price_t / Price_t-1)
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
    
    print(f"   [ADF Test] P-Value: {p_value:.5f}")
    if p_value < 0.05:
        print("   [Result] Data is STATIONARY (Ready for LSTM)")
        return True
    else:
        print("   [Result] Data is NON-STATIONARY (Needs differencing)")
        return False