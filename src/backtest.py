import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import sys
import torch
import ast
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import calculate_features, calculate_features_test, split_data, FEATURE_COLS
from strategy import ProgressiveModel

SEQ_LENGTH = 60

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
TARGET_VOLATILITY = 0.06  # Must match api.py

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def get_lstm_predictions(df):
    """
    Generates forward 24h volatility forecasts using the ProgressiveModel
    with 5-feature input and StandardScaler.
    """
    print("Generating Volatility Forecasts (ProgressiveModel, 5-feature)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hidden_dim, dropout = 64, 0.2
    input_dim = len(FEATURE_COLS)

    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            content = f.read()
            try:
                params = json.loads(content)
            except Exception:
                params = ast.literal_eval(content)
            hidden_dim = params.get('hidden_dim', 64)
            dropout = params.get('dropout', 0.2)
            input_dim = params.get('input_dim', len(FEATURE_COLS))

    model = ProgressiveModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, dropout=dropout)
    model_path = os.path.join(current_dir, '..', 'lstm_model.pth')

    if not os.path.exists(model_path):
        print("   [WARNING] Model not found. Returning flat line.")
        return pd.Series(0, index=df.index)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load scalers
    scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')
    tgt_scaler_path = os.path.join(current_dir, '..', 'target_scaler.pkl')

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            feat_scaler = pickle.load(f)
    else:
        print("   [WARNING] scaler.pkl not found! Using raw features.")
        feat_scaler = None

    tgt_scaler = None
    if os.path.exists(tgt_scaler_path):
        with open(tgt_scaler_path, 'rb') as f:
            tgt_scaler = pickle.load(f)

    # Extract and scale features
    feature_data = df[FEATURE_COLS].values.astype(np.float32)

    if feat_scaler is not None:
        feature_scaled = feat_scaler.transform(feature_data)
    else:
        feature_scaled = feature_data

    predictions = [np.nan] * SEQ_LENGTH
    inputs = []

    for i in range(len(feature_scaled) - SEQ_LENGTH):
        inputs.append(feature_scaled[i:i + SEQ_LENGTH])

    if len(inputs) == 0:
        return pd.Series(0, index=df.index)

    inputs = np.array(inputs)  # [N, 60, 5]
    inputs = torch.from_numpy(inputs).to(device)

    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            output = model(batch)
            preds_scaled = output.cpu().numpy().flatten()

            if tgt_scaler is not None:
                preds = tgt_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
            else:
                preds = preds_scaled

            predictions.extend(preds)

    return pd.Series(predictions, index=df.index)

def run_backtest(df, target_volatility=TARGET_VOLATILITY):
    print("Running Backtest with Volatility-Scaled Sizing...")
    
    # Load parameters
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            content = f.read()
            try:
                best_params = json.loads(content)
            except Exception:
                best_params = ast.literal_eval(content)
    else:
        best_params = {}
        
    z_window        = best_params.get('z_window', 20)
    z_buy           = best_params.get('z_buy', -2.0)
    z_sell          = best_params.get('z_sell', 0.5)
    hurst_threshold = best_params.get('hurst_threshold', 0.45)
    hurst_window    = best_params.get('hurst_window', 48)
    sl_mult         = best_params.get('sl_mult', 1.5)
    cooldown_hours  = best_params.get('cooldown_hours', 3)
    vol_mult        = best_params.get('vol_mult', 1.8)
    use_partial_sell = best_params.get('use_partial_sell', True)
    partial_sell_ratio = best_params.get('partial_sell_ratio', 0.5)
    use_ema_exit    = best_params.get('use_ema_exit', False)

    # Calculate Hurst Exponent and EMA-200 on full df before splitting to prevent lookback cold-start NaNs
    df = df.copy()
    df['hurst'] = df['close'].rolling(window=hurst_window).apply(lambda x: get_hurst_exponent(x), raw=True)
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    train_raw, test_raw = split_data(df, verbose=False)

    if test_raw.empty:
        raise ValueError("Test set is empty!")

    train_df = calculate_features(train_raw, train_df=train_raw)
    test_df = calculate_features_test(test_raw)

    print(f"--- BACKTEST STARTING ON {test_df.index[0]} ---")
    test_df['forecasted_vol'] = get_lstm_predictions(test_df)
    test_df['market_returns'] = test_df['close'].pct_change()

    # Calculate rolling Z-Score
    roll_mean = test_df['close'].rolling(window=z_window).mean()
    roll_std  = test_df['close'].rolling(window=z_window).std()
    test_df['z_score'] = (test_df['close'] - roll_mean) / (roll_std + 1e-8)

    # Volatility threshold
    vol_thresholds = test_df['volatility'].rolling(24).mean() * vol_mult

    # Execution Loop (Stop Loss, Cooldown, Soft Exit, Macro EMA-200 Filter)
    signals = []
    position_multipliers = []
    position = 0  # 0: FLAT, 1: LONG
    entry_price = 0.0
    current_mult = 0.0
    cooldown_until_idx = -1
    has_partial_sold = False

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        price = row['close']
        forecasted_vol = row['forecasted_vol']
        z_score = row['z_score']
        hurst = row['hurst']
        vol_threshold = vol_thresholds.iloc[i]
        
        is_safe_vol = forecasted_vol < vol_threshold
        is_mean_reverting = hurst < hurst_threshold
        is_macro_uptrend = price > row['ema_200']
        
        # Dynamic Z-Score Band
        dynamic_z_buy = z_buy - (forecasted_vol * 10)
        
        # Check cooldown
        in_cooldown = i < cooldown_until_idx
        
        action = "HOLDING" if position == 1 else "FLAT"
        
        # Stop Loss Check (Gate 4)
        if position == 1:
            stop_loss_pct = forecasted_vol * sl_mult
            if price < entry_price * (1.0 - stop_loss_pct):
                action = "CASH"
                cooldown_until_idx = i + cooldown_hours
            elif use_ema_exit and price < row['ema_200']:
                action = "CASH"
                cooldown_until_idx = i + cooldown_hours
                
        # If not stopped out and not in cooldown
        if action not in ["CASH"] and not in_cooldown:
            if is_mean_reverting and is_safe_vol and is_macro_uptrend and z_score <= dynamic_z_buy:
                action = "BUY"
            elif z_score >= z_sell or not is_safe_vol:
                action = "CASH"
            elif use_partial_sell and z_score >= 0.0 and position == 1 and not has_partial_sold:
                action = "PARTIAL_SELL"
                
        # State machine updates
        if action == "BUY":
            if position == 0:
                entry_price = price
                current_mult = 1.0
                has_partial_sold = False
            position = 1
            signals.append(1)
        elif action == "PARTIAL_SELL":
            current_mult = current_mult * partial_sell_ratio
            position = 1
            has_partial_sold = True
            signals.append(1)
        elif action == "CASH":
            if position == 1 and price < entry_price:
                cooldown_until_idx = i + cooldown_hours
            position = 0
            entry_price = 0.0
            current_mult = 0.0
            has_partial_sold = False
            signals.append(0)
        elif action == "HOLDING":
            signals.append(1)
        else:  # FLAT
            position = 0
            entry_price = 0.0
            current_mult = 0.0
            has_partial_sold = False
            signals.append(0)
            
        position_multipliers.append(current_mult)

    test_df['signal'] = signals

    # Convert forecasted hourly volatility to daily volatility (since TARGET is likely daily risk)
    daily_forecasted_vol = test_df['forecasted_vol'] * np.sqrt(24)

    # Volatility-Scaled Position Sizing
    uncapped_size = target_volatility / (daily_forecasted_vol + 1e-8)
    base_position_size = uncapped_size.clip(upper=1.0)
    test_df['position_size'] = base_position_size * position_multipliers
    
    print(f"   [Debug] Avg Uncapped Position Size: {uncapped_size.mean():.4f}")

    # Weighted strategy returns
    test_df['strategy_returns'] = (
        test_df['market_returns'] *
        test_df['signal'].shift(1) *
        test_df['position_size'].shift(1)
    )
    test_df['cumulative_market'] = (1 + test_df['market_returns']).cumprod()
    test_df['cumulative_strategy'] = (1 + test_df['strategy_returns']).cumprod()

    # Backward compat alias
    test_df['lstm_pred_vol'] = test_df['forecasted_vol']

    return test_df

def calculate_metrics(df, verbose=True):
    risk_free_rate = 0.0
    strategy_mean = df['strategy_returns'].mean() * 24 * 365
    strategy_std = df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)

    cumulative = df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    market_ret = (df['cumulative_market'].iloc[-1] - 1) * 100
    strat_ret = (df['cumulative_strategy'].iloc[-1] - 1) * 100

    if verbose:
        print("\n=== VOLATILITY-SCALED BACKTEST RESULTS (OUT OF SAMPLE) ===")
        print(f"Market Return:   {market_ret:.2f}%")
        print(f"Strategy Return: {strat_ret:.2f}%")
        print(f"Sharpe Ratio:    {sharpe:.2f}")
        print(f"Max Drawdown:    {max_drawdown * 100:.2f}%")
        try:
            target_vol_val = df['position_size'].iloc[-1] # Approximation just for printing if available
            print(f"Sizing Mode:     Volatility-Scaled")
        except:
            pass

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
            'position_size': round(float(row.get('position_size', 1.0)), 4),
        })

    payload = {'metrics': metrics, 'series': records}
    out_path = os.path.join(current_dir, '..', 'backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f)
    print(f"Backtest JSON exported → {out_path}")

def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='Vol-Scaled AI (Test Set)', color='blue', linewidth=1.5)
    plt.title('Volatility-Scaled Strategy Performance: Out-of-Sample')
    plt.legend()
    plt.grid(True, alpha=0.3)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

def plot_dashboard(df):
    print("   [Visual] Rendering Final Portfolio Dashboard...")
    
    # Load parameters to get hurst_threshold
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hurst_threshold = 0.45
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                content = f.read()
                try:
                    best_params = json.loads(content)
                except Exception:
                    best_params = ast.literal_eval(content)
                hurst_threshold = best_params.get('hurst_threshold', 0.45)
        except Exception:
            pass

    subset = df.tail(2000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(subset.index, subset['close'], color='gray', alpha=0.3, label='Price')
    
    # Color coding based on Hurst regime filter
    mean_revert = subset[subset['hurst'] < hurst_threshold]
    trending = subset[subset['hurst'] >= hurst_threshold]

    ax1.scatter(mean_revert.index, mean_revert['close'], color='green', s=10, alpha=0.6, label=f'Mean-Reverting (Hurst < {hurst_threshold})')
    ax1.scatter(trending.index, trending['close'], color='red', s=10, alpha=0.6, label=f'Trending (Hurst >= {hurst_threshold})')

    ax1.set_title('Market Regimes & Vol-Scaled AI Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.15)

    lstm_col = 'forecasted_vol' if 'forecasted_vol' in subset.columns else 'lstm_pred_vol'

    ax2.plot(subset.index, subset['volatility'], color='blue', alpha=0.5, label='Actual Volatility')

    if lstm_col in subset.columns:
        ax2.plot(subset.index, subset[lstm_col], color='magenta', linestyle='--', linewidth=1.5, label='AI Forecast')
        veto_line = subset['volatility'].rolling(24).mean() * 1.8
        ax2.plot(subset.index, veto_line, color='black', linestyle=':', alpha=0.6, label='Risk Threshold (1.8σ)')
        ax2.fill_between(subset.index, 0, subset['volatility'].max(),
                         where=(subset[lstm_col] > veto_line),
                         color='red', alpha=0.1, label='AI Signal: CASH')

    ax2.set_title('AI Risk Detection (ProgressiveModel)', fontsize=12)
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
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hurst_window = 48
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                content = f.read()
                try:
                    best_params = json.loads(content)
                except Exception:
                    best_params = ast.literal_eval(content)
                hurst_window = best_params.get('hurst_window', 48)
        except:
            pass
            
    df['hurst'] = df['close'].rolling(window=hurst_window).apply(lambda x: get_hurst_exponent(x), raw=True)
    df = calculate_features(df)
    df['forecasted_vol'] = get_lstm_predictions(df)
    df['lstm_pred_vol'] = df['forecasted_vol']
    plot_dashboard(df)

if __name__ == "__main__":
    try:
        data = load_data()
        results = run_backtest(data)
        metrics = calculate_metrics(results)
        export_json(results, metrics)
        plot_results(results)
        plot_dashboard(results)
    except Exception as e:
        print(f"Error: {e}")