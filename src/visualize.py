import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from features import calculate_features
from regimes import detect_regimes

def plot_regimes():
    # Load and prep data
    try:
        df = pd.read_csv('data/eth_hourly.csv')
    except FileNotFoundError:
        print("Run fetch_data.py first!")
        return

    df = calculate_features(df)
    df = detect_regimes(df)
    
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot Price
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['close'], label='ETH Price', color='gray', alpha=0.5)
    
    # Highlight Regimes
    # Regime 0 (Usually Low Vol) = Green, Regime 1 (High Vol) = Red
    # Note: GMM labels are random, so we assume the regime with higher std dev is "Volatile"
    
    regime_0_std = df[df['regime'] == 0]['log_ret'].std()
    regime_1_std = df[df['regime'] == 1]['log_ret'].std()
    
    volatile_label = 1 if regime_1_std > regime_0_std else 0
    stable_label = 0 if volatile_label == 1 else 1
    
    # Scatter plot for regimes
    stable_df = df[df['regime'] == stable_label]
    volatile_df = df[df['regime'] == volatile_label]
    
    plt.scatter(stable_df['timestamp'], stable_df['close'], color='green', label='Stable Regime', s=10)
    plt.scatter(volatile_df['timestamp'], volatile_df['close'], color='red', label='Volatile Regime', s=10)
    
    plt.title('Ethereum Price Colored by Market Regime (GMM)')
    plt.legend()
    
    # Plot Volatility
    plt.subplot(2, 1, 2)
    plt.plot(df['timestamp'], df['volatility'], color='blue', label='Rolling Volatility')
    plt.title('Rolling Volatility Feature')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('regime_plot.png')
    print("Plot saved as regime_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_regimes()