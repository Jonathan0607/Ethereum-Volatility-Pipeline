import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def detect_regimes(df):
    """
    Uses Gaussian Mixture Models (GMM) to cluster market conditions
    into 'Low Volatility' and 'High Volatility' regimes.
    """
    # Reshape for Sklearn
    X = df['volatility'].values.reshape(-1, 1)
    
    # Init GMM with 2 components (Stable vs Turbulent)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    
    # Predict regimes
    regimes = gmm.predict(X)
    df['regime'] = regimes
    
    print("Regime Detection Complete.")
    print(df['regime'].value_counts())
    return df

if __name__ == "__main__":
    # Load data (Assuming you ran fetch_data.py and features.py)
    try:
        df = pd.read_csv('data/eth_hourly.csv')
        # Re-calculate volatility since we didn't save it to CSV last time
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=24).std()
        df.dropna(inplace=True)
        
        detect_regimes(df)
        print("Regimes tagged. Ready for analysis.")
        
    except FileNotFoundError:
        print("Data not found. Run fetch_data.py first.")