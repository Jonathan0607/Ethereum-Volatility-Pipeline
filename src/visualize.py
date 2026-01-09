import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import calculate_features
from regimes import detect_regimes

def plot_regimes():
    # --- ROBUST PATH FIX ---
    # Finds the directory where this script lives, then looks one level up ('..') for data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')

    print(f"Looking for data at: {data_path}")

    try:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure time is readable
    except FileNotFoundError:
        print("Error: Data file not found. Please run src/fetch_data.py first.")
        return

    # Apply Logic
    df = calculate_features(df)
    df = detect_regimes(df)
    
    # --- PLOTTING LOGIC ---
    plt.figure(figsize=(14, 7))
    
    # Subplot 1: Price colored by Regime
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='ETH Price', color='gray', alpha=0.5)
    
    # Since we enforced consistency in regimes.py, we know:
    # 0 = Stable
    # 1 = High Volatility
    
    stable_df = df[df['regime'] == 0]
    volatile_df = df[df['regime'] == 1]
    
    plt.scatter(stable_df['timestamp'], stable_df['close'], color='green', label='Stable Regime', s=10)
    plt.scatter(volatile_df['timestamp'], volatile_df['close'], color='red', label='High Volatility (Risk)', s=10)
    
    plt.title('Ethereum Market Regimes (Green=Safe, Red=Risk)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Volatility
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['volatility'], color='blue', label='Rolling Volatility (24h)')
    plt.title('Market Volatility Feature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to the Project Root (one level up)
    save_path = os.path.join(current_dir, '..', 'regime_plot.png')
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    plot_regimes()