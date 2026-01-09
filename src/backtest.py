import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def load_data():
    # Load your processed data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def run_backtest(df):
    """
    Simulates a strategy where we:
    - BUY when volatility is Low (Stable Regime)
    - SELL/CASH when volatility is High (Risk Regime)
    """
    print("Running Backtest...")

    # 1. Calculate Market Returns (Buy & Hold)
    # We use 'close' price to calculate hourly percentage change
    df['market_returns'] = df['close'].pct_change()
    
    # 2. Simulate a Signal (If you haven't saved model predictions yet, we simulate a proxy)
    # REALITY CHECK: In the future, replace this with: df['signal'] = model.predict(X)
    # FOR NOW: We use Rolling Volatility as a proxy for the GMM Regime to test the engine
    df['rolling_vol'] = df['close'].pct_change().rolling(window=24).std()
    threshold = df['rolling_vol'].mean() + 0.5 * df['rolling_vol'].std()
    
    # Strategy: If Volatility is HIGH, go to Cash (0). Else, go Long (1).
    df['signal'] = np.where(df['rolling_vol'] > threshold, 0, 1)
    
    # 3. Calculate Strategy Returns
    # We shift signal by 1 because we act on the signal *next* hour
    df['strategy_returns'] = df['market_returns'] * df['signal'].shift(1)
    
    # 4. Cumulative Returns (The "Equity Curve")
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    return df

def calculate_metrics(df):
    # Annualized Sharpe Ratio (assuming 24/7 crypto trading)
    risk_free_rate = 0.0
    strategy_mean = df['strategy_returns'].mean() * 24 * 365
    strategy_std = df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)
    
    # Max Drawdown
    cumulative = df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    print("\n--- Performance Metrics ---")
    print(f"Total Return (Market):   {(df['cumulative_market'].iloc[-1] - 1)*100:.2f}%")
    print(f"Total Return (Strategy): {(df['cumulative_strategy'].iloc[-1] - 1)*100:.2f}%")
    print(f"Sharpe Ratio:            {sharpe:.2f}")
    print(f"Max Drawdown:            {max_drawdown*100:.2f}%")

def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='Regime Strategy', color='blue')
    
    plt.title('Strategy Performance: Risk-Off during High Volatility')
    plt.ylabel('Cumulative Return ($1 invested)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path)
    print(f"\nBacktest chart saved to: {output_path}")

if __name__ == "__main__":
    try:
        data = load_data()
        results = run_backtest(data)
        calculate_metrics(results)
        plot_results(results)
    except Exception as e:
        print(f"Error: {e}")