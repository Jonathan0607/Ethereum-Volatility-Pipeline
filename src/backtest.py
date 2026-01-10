import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import calculate_features
from regimes import detect_regimes
from model import LSTMModel

# --- CONFIG ---
SEQ_LENGTH = 60

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def get_lstm_predictions(df):
    print("Generating LSTM Volatility Forecasts...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    import ast
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    model_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    
    # Defaults
    hidden_dim, num_layers, dropout = 64, 2, 0.2
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = ast.literal_eval(f.read())
            hidden_dim = params.get('hidden_dim', 64)
            num_layers = params.get('num_layers', 2)
            dropout = params.get('dropout', 0.2)
            
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        return pd.Series(0, index=df.index)

    model.to(device)
    model.eval()
    
    # Prepare Data
    data = df['volatility'].values.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    predictions = [np.nan] * SEQ_LENGTH 
    inputs = []
    for i in range(len(data_scaled) - SEQ_LENGTH):
        inputs.append(data_scaled[i:i+SEQ_LENGTH])
    
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).unsqueeze(-1).to(device)
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            output = model(batch)
            preds = output.cpu().numpy().flatten() * (std + 1e-8) + mean
            predictions.extend(preds)
            
    return pd.Series(predictions, index=df.index)

def run_backtest(df):
    """
    STRATEGY V6: 'THE SHARPE BREAKER'
    1. Faster Entry: EMA 20 (Standard Bollinger Mean).
    2. Stricter Safety: LSTM Threshold lowered to 1.8x.
    3. Faster Volume: Volume MA lowered to 12 (React to breakout volume faster).
    """
    print("Running Backtest with Sharpe Optimization...")

    df = calculate_features(df)
    df = detect_regimes(df) # Uses 2-component GMM (Safe vs Risk)
    df['lstm_pred_vol'] = get_lstm_predictions(df)
    
    df['market_returns'] = df['close'].pct_change()
    
    # --- TECHNICALS TUNING ---
    # Tweak 1: EMA 20 is slightly faster than 24.
    df['fast_trend'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Tweak 2: EMA 50 (Keep robust baseline for Parabolic check)
    df['slow_trend'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Tweak 3: Volume MA 12 (Reacts faster to pump volume than 24)
    df['vol_ma'] = df['volume'].rolling(window=12).mean()
    
    # --- LOGIC ---
    
    # Base Signal
    base_signal = (
        (df['regime'] == 0) & 
        (df['close'] > df['fast_trend']) &
        (df['volume'] > df['vol_ma'])
    )
    
    # Tweak 4: Stricter LSTM Threshold (1.8 instead of 2.0)
    # Filters out slightly more "medium-high" risk trades to lower Drawdown.
    vol_threshold = df['volatility'].rolling(24).mean() * 1.8
    lstm_risk_on = (df['lstm_pred_vol'] > vol_threshold)
    
    # Parabolic Exception (Keep this, it works great)
    parabolic_move = (df['close'] > (df['slow_trend'] * 1.02))
    
    long_condition = base_signal & (~lstm_risk_on | parabolic_move)
    
    df['signal'] = np.where(long_condition, 1, 0)
    
    df['strategy_returns'] = df['market_returns'] * df['signal'].shift(1)
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    return df

def calculate_metrics(df):
    risk_free_rate = 0.0
    strategy_mean = df['strategy_returns'].mean() * 24 * 365
    strategy_std = df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)
    
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
    plt.plot(df.index, df['cumulative_strategy'], label='Hybrid AI (Sharpe Optimized)', color='blue', linewidth=1.5)
    plt.title('Strategy Performance: Optimized AI Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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