import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import calculate_features
from regimes import detect_regimes

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def run_backtest(df):
    """
    Simulates: Regime + Trend (EMA) + Volume Confirmation (The False Signal Filter)
    """
    print("Running Backtest with Volume Filter...")

    # 1. Pipeline Logic
    df = calculate_features(df)
    df = detect_regimes(df)
    
    # 2. Market Returns
    df['market_returns'] = df['close'].pct_change()
    
    # 3. Technical Indicators
    # Trend: EMA 24 (Fast reaction)
    df['fast_trend'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # Volume: 24-Hour Average Volume
    df['vol_ma'] = df['volume'].rolling(window=24).mean()
    
    # 4. THE SHARPE BOOSTER LOGIC
    # Condition A: Regime is Safe (0)
    # Condition B: Price is in an Uptrend (Price > EMA)
    # Condition C: Volume is Strong (Current Vol > Average Vol) -> NEW!
    
    long_condition = (
        (df['regime'] == 0) & 
        (df['close'] > df['fast_trend']) & 
        (df['volume'] > df['vol_ma'])  # Only buy with conviction
    )
    
    # Set Signal
    df['signal'] = np.where(long_condition, 1, 0)
    
    # 5. Returns
    df['strategy_returns'] = df['market_returns'] * df['signal'].shift(1)
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    return df
def calculate_metrics(df):
    # Annualized Sharpe Ratio
    risk_free_rate = 0.0
    strategy_mean = df['strategy_returns'].mean() * 24 * 365
    strategy_std = df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)
    
    # Max Drawdown
    cumulative = df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    print("\n=== AI-POWERED BACKTEST RESULTS ===")
    print(f"Market Return:   {(df['cumulative_market'].iloc[-1] - 1)*100:.2f}%")
    print(f"Strategy Return: {(df['cumulative_strategy'].iloc[-1] - 1)*100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_drawdown*100:.2f}%")

def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='AI Strategy (Aggressive)', color='blue', linewidth=1.5)
    
    plt.title('Strategy Performance: GMM Regime + EMA 24 Filter')
    plt.ylabel('Cumulative Return ($1 invested)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    try:
        data = load_data()
        results = run_backtest(data)
        calculate_metrics(results)
        plot_results(results)
    except Exception as e:
        print(f"Error: {e}")