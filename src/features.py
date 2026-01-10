import pandas as pd
import numpy as np
from arch import arch_model
import warnings

# Suppress harmless warnings from the arch library
warnings.filterwarnings("ignore")

def calculate_features(df):
    """
    Feature Engineering:
    - Log Returns
    - GARCH(1,1) Conditional Volatility (The 'Smart' Risk Metric)
    - Momentum features for the LSTM
    """
    df = df.copy()
    
    # 1. Log Returns (Required for GARCH)
    # We use Log returns because they are time-additive and symmetric
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # Drop the first NaN created by shifting
    df.dropna(inplace=True)
    
    # 2. GARCH(1,1) Implementation
    print("Fitting GARCH(1,1) Model... (This might take a moment)")
    
    # NOTE: GARCH models struggle to converge with tiny numbers (like 0.0002).
    # We scale returns by 100 to make them "percent" (e.g., 0.2 instead of 0.002).
    # This helps the optimizer find the solution faster.
    returns = df['log_ret'] * 100
    
    # Define the model: Constant Mean, GARCH Volatility, Normal Distribution
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    
    # Fit the model
    # disp='off' suppresses the console output during fitting
    res = model.fit(disp='off')
    
    # 3. Extract Conditional Volatility
    # This is the estimated "Risk" for each specific hour based on the model.
    # We divide by 100 to scale it back down to the original unit.
    df['garch_vol'] = res.conditional_volatility / 100
    
    # 4. Feature Selection
    # We override the old 'volatility' column with our new smart GARCH metric
    df['volatility'] = df['garch_vol']
    
    # Keep the momentum feature for the LSTM
    df['returns'] = df['log_ret']
    
    # 5. Optional: Add a simple Rolling Volatility for comparison/fallback
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()
    
    # Drop any NaNs created by the rolling window or GARCH
    df.dropna(inplace=True)
    
    print("Feature Engineering Complete.")
    return df