import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import sys
import torch
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import calculate_features, calculate_features_test, split_data
from strategy import predict_regimes, LSTMModel

SEQ_LENGTH = 60

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def get_lstm_predictions(df):
    print("Generating LSTM Volatility Forecasts (Strict Test Set)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hidden_dim, num_layers, dropout = 64, 2, 0.2

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = ast.literal_eval(f.read())
            hidden_dim = params.get('hidden_dim', 64)
            num_layers = params.get('num_layers', 2)
            dropout    = params.get('dropout', 0.2)

    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    model_path = os.path.join(current_dir, '..', 'lstm_model.pth')

    if not os.path.exists(model_path):
        print("   [WARNING] Model not found. Returning flat line.")
        return pd.Series(0, index=df.index)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    mean_path = os.path.join(current_dir, '..', 'scaler_mean.npy')
    std_path  = os.path.join(current_dir, '..', 'scaler_std.npy')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        print("   [WARNING] Scaler stats not found! Dashboard may look skewed.")
        data = df['volatility'].values
        mean, std = np.mean(data), np.std(data)
    else:
        mean = np.load(mean_path)
        std  = np.load(std_path)

    data = df['volatility'].values.astype(np.float32)
    data_scaled = (data - mean) / (std + 1e-8)

    predictions = [np.nan] * SEQ_LENGTH
    inputs = []

    for i in range(len(data_scaled) - SEQ_LENGTH):
        inputs.append(data_scaled[i:i + SEQ_LENGTH])

    if len(inputs) == 0:
        return pd.Series(0, index=df.index)

    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).unsqueeze(-1).to(device)

    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch  = inputs[i:i + batch_size]
            output = model(batch)
            preds  = output.cpu().numpy().flatten() * (std + 1e-8) + mean
            predictions.extend(preds)

    return pd.Series(predictions, index=df.index)

def run_backtest(df):
    print("Running Backtest with Strict Chronological Split...")
    train_raw, test_raw = split_data(df, verbose=False)

    if test_raw.empty:
        raise ValueError("Test set is empty! Check TEST_START_DATE in data_split.py")

    train_df = calculate_features(train_raw, train_df=train_raw)
    test_df = calculate_features_test(test_raw)

    print(f"--- BACKTEST STARTING ON {test_df.index[0]} ---")
    test_df = predict_regimes(test_df)
    test_df['lstm_pred_vol'] = get_lstm_predictions(test_df)
    test_df['market_returns'] = test_df['close'].pct_change()
    test_df['fast_trend'] = test_df['close'].ewm(span=20, adjust=False).mean()
    test_df['slow_trend'] = test_df['close'].ewm(span=50, adjust=False).mean()
    test_df['vol_ma']     = test_df['volume'].rolling(window=12).mean()

    base_signal = ((test_df['regime_gmm'] == 0) & (test_df['close'] > test_df['fast_trend']) & (test_df['volume'] > test_df['vol_ma']))
    vol_threshold  = test_df['volatility'].rolling(24).mean() * 1.8
    lstm_risk_on   = test_df['lstm_pred_vol'] > vol_threshold
    parabolic_move = test_df['close'] > (test_df['slow_trend'] * 1.02)

    long_condition = base_signal & (~lstm_risk_on | parabolic_move)
    test_df['signal'] = np.where(long_condition, 1, 0)
    test_df['strategy_returns']    = test_df['market_returns'] * test_df['signal'].shift(1)
    test_df['cumulative_market']   = (1 + test_df['market_returns']).cumprod()
    test_df['cumulative_strategy'] = (1 + test_df['strategy_returns']).cumprod()

    return test_df

def calculate_metrics(df):
    risk_free_rate = 0.0
    strategy_mean  = df['strategy_returns'].mean() * 24 * 365
    strategy_std   = df['strategy_returns'].std()  * np.sqrt(24 * 365)
    sharpe         = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)

    cumulative   = df['cumulative_strategy']
    peak         = cumulative.cummax()
    drawdown     = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    market_ret  = (df['cumulative_market'].iloc[-1] - 1) * 100
    strat_ret   = (df['cumulative_strategy'].iloc[-1] - 1) * 100

    print("\n=== AI-POWERED BACKTEST RESULTS (OUT OF SAMPLE) ===")
    print(f"Market Return:   {market_ret:.2f}%")
    print(f"Strategy Return: {strat_ret:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_drawdown * 100:.2f}%")

    return {
        'market_return': round(market_ret, 2),
        'strategy_return': round(strat_ret, 2),
        'sharpe': round(sharpe, 2),
        'max_drawdown': round(max_drawdown * 100, 2),
    }

def export_json(df, metrics):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    step = max(1, len(df) // 400)
    sampled = df.iloc[::step].copy()
    sampled = sampled.dropna(subset=['cumulative_market', 'cumulative_strategy'])

    records = []
    for ts, row in sampled.iterrows():
        records.append({
            'timestamp': ts.isoformat(),
            'close': round(float(row['close']), 2),
            'cumulative_market': round(float(row['cumulative_market']), 4),
            'cumulative_strategy': round(float(row['cumulative_strategy']), 4),
            'signal': int(row['signal']),
        })

    payload = {'metrics': metrics, 'series': records}
    out_path = os.path.join(current_dir, '..', 'backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    print(f"Backtest JSON exported → {out_path}")

def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'],   label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='Hybrid AI (Test Set)', color='blue', linewidth=1.5)
    plt.title('Strategy Performance: Out-of-Sample Test')
    plt.legend()
    plt.grid(True, alpha=0.3)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

def plot_dashboard(df):
    print("   [Visual] Rendering Final Portfolio Dashboard...")
    subset = df.tail(2000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    ax1.plot(subset.index, subset['close'], color='gray', alpha=0.3, label='Price')
    safe = subset[subset['regime'] == 0]
    risk = subset[subset['regime'] == 1]
    
    ax1.scatter(safe.index, safe['close'], color='green', s=10, alpha=0.6, label='Safe Regime')
    ax1.scatter(risk.index, risk['close'], color='red', s=10, alpha=0.6, label='Risk Regime')
    
    ax1.set_title('Market Regimes & AI Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.15)
    
    if 'lstm_pred_vol' in subset.columns:
        lstm_col = 'lstm_pred_vol'
    else:
        lstm_col = 'lstm_pred'
        
    ax2.plot(subset.index, subset['volatility'], color='blue', alpha=0.5, label='Actual Volatility')
    
    if lstm_col in subset.columns:
        ax2.plot(subset.index, subset[lstm_col], color='magenta', linestyle='--', linewidth=1.5, label='AI Prediction')
        veto_line = subset['volatility'].rolling(24).mean() * 1.8
        ax2.plot(subset.index, veto_line, color='black', linestyle=':', alpha=0.6, label='Risk Threshold (1.8σ)')
        ax2.fill_between(subset.index, 0, subset['volatility'].max(), 
                         where=(subset[lstm_col] > veto_line), 
                         color='red', alpha=0.1, label='AI Signal: CASH')

    ax2.set_title('AI Risk Detection (LSTM)', fontsize=12)
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.15)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_dashboard.png')
    plt.savefig(output_path, dpi=300)
    print(f"   [Success] Dashboard saved to {output_path}")

def run_visualizer():
    df = load_data()
    df = calculate_features(df)
    df = predict_regimes(df)
    df['lstm_pred_vol'] = get_lstm_predictions(df)
    plot_dashboard(df)

if __name__ == "__main__":
    try:
        data    = load_data()
        results = run_backtest(data)
        metrics = calculate_metrics(results)
        export_json(results, metrics)
        plot_results(results)
        plot_dashboard(results)
    except Exception as e:
        print(f"Error: {e}")