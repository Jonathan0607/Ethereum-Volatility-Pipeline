import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def calculate_features(df):
    """
    Adds Log Returns and DOWNSIDE Volatility (Sortino Logic).
    """
    df = df.copy()
    
    # 1. Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Downside Volatility ONLY
    # We clip positive returns to 0. We don't care if price jumps UP.
    window = 24
    negative_rets = df['log_ret'].clip(upper=0)
    
    # This metric will stay LOW during a pump, but spike HIGH during a crash
    df['volatility'] = negative_rets.rolling(window=window).std() * np.sqrt(365 * 24)
    
    # Drop NaNs
    df.dropna(inplace=True)
    return df

def check_stationarity(series):
    result = adfuller(series.values)
    p_value = result[1]
    return p_value < 0.05