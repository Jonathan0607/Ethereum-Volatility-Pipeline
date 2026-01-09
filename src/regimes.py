import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

def detect_regimes(df, n_components=2):
    """
    Uses Gaussian Mixture Models (GMM) to cluster market conditions.
    Regime 0: Low Volatility (Safe)
    Regime 1: High Volatility (Risk)
    """
    df = df.copy()
    
    # Reshape for Sklearn
    X = df['volatility'].values.reshape(-1, 1)
    
    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    
    # Predict Regimes
    regimes = gmm.predict(X)
    df['regime'] = regimes
    
    # --- THE FIX: LABEL ENFORCEMENT ---
    # We must ensure that Regime 1 is ALWAYS the "High Volatility" regime.
    # Otherwise, the model might randomly label "Safe" as 1 and "Risk" as 0.
    
    vol_avg_0 = df[df['regime'] == 0]['volatility'].mean()
    vol_avg_1 = df[df['regime'] == 1]['volatility'].mean()
    
    if vol_avg_0 > vol_avg_1:
        print("   [Info] GMM labels were flipped. Swapping them to ensure 1 = High Vol.")
        # Swap 0s and 1s
        df['regime'] = 1 - df['regime']
    
    print(f"   [Result] Regime Detection Complete.")
    print(f"   - Low Vol (Safe) Count: {len(df[df['regime']==0])}")
    print(f"   - High Vol (Risk) Count: {len(df[df['regime']==1])}")
    
    return df

def plot_regimes(df):
    """
    Visualizes the regimes.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot Price
    plt.plot(df.index, df['close'], color='gray', alpha=0.5, label='ETH Price')
    
    # Highlight High Volatility Regimes (Regime 1) in RED
    # We use a scatter plot on top of the line
    risk_df = df[df['regime'] == 1]
    plt.scatter(risk_df.index, risk_df['close'], color='red', s=10, label='High Volatility (Risk)')
    
    plt.title('Market Regimes: Red Dots = High Risk (GMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save Plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'regime_plot.png')
    plt.savefig(output_path)
    print(f"   [Visual] Saved to {output_path}")

if __name__ == "__main__":
    # Test Block
    from features import calculate_features
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = calculate_features(df)
        df = detect_regimes(df)
        plot_regimes(df)
    else:
        print("Data file not found.")