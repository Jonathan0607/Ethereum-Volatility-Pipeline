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
from strategy import ProgressiveModel, get_gmm_state
from hmm_engine import get_high_vol_probability

SEQ_LENGTH = 60

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
TARGET_VOLATILITY = 0.06  # Must match api.py
FEE_PCT = 0.0  # Fees disabled for backtesting

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
        
    breakout_window = best_params.get('breakout_window', 48)
    hmm_chop_max    = best_params.get('hmm_chop_max', 0.40)
    hmm_trend_min   = best_params.get('hmm_trend_min', 0.60)
    gmm_z_buy       = best_params.get('gmm_z_buy', -1.5)
    gmm_z_sell      = best_params.get('gmm_z_sell', 0.5)
    vol_shock_mult  = best_params.get('vol_shock_mult', 1.5)
    rebalance_threshold = best_params.get('rebalance_threshold', 0.15)
    gmm_max_vol     = best_params.get('gmm_max_vol', 0.020)

    # Calculate HMM Probability on full df before splitting to prevent NaNs
    df = df.copy()
    
    # Check HMM cache
    import pickle
    cache_path = os.path.join(current_dir, '..', 'data', 'hmm_features_cache.pkl')
    hmm_loaded = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            if cached_df.index[-1] == df.index[-1] and len(cached_df) == len(df):
                df['prob_high_vol'] = cached_df['prob_high_vol']
                print("[Cache] Loaded HMM prob_high_vol from cache.")
                hmm_loaded = True
        except Exception as e:
            print(f"[Cache] Cache load error: {e}. Recomputing...")
            
    if not hmm_loaded:
        print("Computing rolling HMM high-vol probability (this might take a few minutes if not cached)...")
        df['prob_high_vol'] = df['close'].rolling(window=500).apply(
            lambda x: get_high_vol_probability(x) if not np.isnan(x).any() else 0.0, raw=True
        )
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df[['close', 'prob_high_vol']], f)
            print("[Cache] Saved computed HMM prob_high_vol to cache.")
        except Exception as e:
            print(f"[Cache] Save error: {e}")
            
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    train_raw, test_raw = split_data(df, verbose=False)

    if test_raw.empty:
        raise ValueError("Test set is empty!")

    train_df = calculate_features(train_raw, train_df=train_raw)
    test_df = calculate_features_test(test_raw)

    print(f"--- BACKTEST STARTING ON {test_df.index[0]} ---")
    test_df['forecasted_vol'] = get_lstm_predictions(test_df)
    test_df['market_returns'] = test_df['close'].pct_change()

    # Compression Breakout Filters
    test_df['rolling_max'] = test_df['close'].rolling(window=breakout_window).max().shift(1)
    test_df['rolling_min'] = test_df['close'].rolling(window=breakout_window).min().shift(1)
    test_df['rolling_mean'] = test_df['close'].rolling(window=24).mean()
    
    # Realized Volatility Overlay (Circuit Breaker)
    test_df['log_ret'] = np.log(test_df['close'] / test_df['close'].shift(1))
    test_df['vol_24h'] = test_df['log_ret'].rolling(24).std()
    test_df['vol_168h'] = test_df['log_ret'].rolling(168).std()
    vol_shock = (test_df['vol_24h'] > (test_df['vol_168h'] * vol_shock_mult)).fillna(False)

    # Fast Z-Score
    roll_mean = test_df['close'].rolling(window=20).mean()
    roll_std = test_df['close'].rolling(window=20).std()
    test_df['z_score'] = (test_df['close'] - roll_mean) / (roll_std + 1e-8)

    # Bi-directional HMM/GMM Hierarchical Logic
    test_df['signal'] = np.nan
    
    # AGENT 1: Breakout Sub-Agent (SHORT ONLY)
    trend_active = (test_df['prob_high_vol'] > hmm_trend_min) | vol_shock
    test_df.loc[trend_active & (test_df['close'] < test_df['rolling_min']), 'signal'] = -1
    
    # AGENT 2: GMM Mean-Reversion (LONG ONLY + Low Volatility Gate)
    chop_active = (test_df['prob_high_vol'] < hmm_chop_max) & ~vol_shock
    
    # The GMM can only enter if the current 24h volatility is below the threshold
    test_df.loc[chop_active & (test_df['z_score'] < gmm_z_buy) & (test_df['vol_24h'] < gmm_max_vol), 'signal'] = 1
    
    # 3. Master Exits
    # Exit Longs when overbought, Exit Shorts when trend breaks upward
    test_df.loc[chop_active & (test_df['z_score'] > gmm_z_sell), 'signal'] = 0
    test_df.loc[trend_active & (test_df['close'] > test_df['rolling_max']), 'signal'] = 0
    
    # Force Cash in the Transition Zone
    test_df.loc[(test_df['prob_high_vol'] >= hmm_chop_max) & (test_df['prob_high_vol'] <= hmm_trend_min) & ~vol_shock, 'signal'] = 0
    
    test_df['signal'] = test_df['signal'].ffill().fillna(0)

    # Track Active Regimes in Dataframe
    test_df['active_agent'] = 'CASH'
    test_df.loc[trend_active, 'active_agent'] = 'Breakout'
    test_df.loc[chop_active, 'active_agent'] = 'GMM'


    # Convert forecasted hourly volatility to daily volatility (since TARGET is likely daily risk)
    daily_forecasted_vol = test_df['forecasted_vol'] * np.sqrt(24)

    # Volatility-Scaled Position Sizing (signed: +1 long, -1 short)
    uncapped_size = target_volatility / (daily_forecasted_vol + 1e-8)
    base_position_size = uncapped_size.clip(upper=1.0)
    test_df['target_size'] = base_position_size * test_df['signal']

    # 4. Friction Filter (Rebalance Threshold Loop)
    raw_targets = test_df['target_size'].values
    active_positions = np.zeros(len(raw_targets))
    current_pos = 0.0
    
    for i in range(len(raw_targets)):
        target = raw_targets[i]
        if np.isnan(target):
            active_positions[i] = current_pos
            continue
        if target == 0.0 or abs(target - current_pos) > rebalance_threshold:
            current_pos = target
        active_positions[i] = current_pos
        
    test_df['position_size'] = active_positions
    
    n_long = (test_df['position_size'] > 0).sum()
    n_short = (test_df['position_size'] < 0).sum()
    n_flat = (test_df['position_size'] == 0).sum()
    print(f"   [Debug] Sized Positions: LONG={n_long} | SHORT={n_short} | FLAT={n_flat}")

    # 1. Shift the position size (because we enter the position at the CLOSE of the signal candle)
    # The 'position_size' column is directional (positive for Long, negative for Short, 0 for Flat)
    test_df['active_position'] = test_df['position_size'].shift(1).fillna(0)

    # 2. Calculate Gross Return based on the scaled position
    test_df['gross_strategy_returns'] = test_df['market_returns'] * test_df['active_position']

    # 3. Calculate Transaction Friction (Fees)
    # We pay a fee every time the position size changes.
    test_df['position_change'] = test_df['active_position'].diff().abs().fillna(0)
    test_df['transaction_costs'] = test_df['position_change'] * FEE_PCT

    # 4. Calculate Net Strategy Return
    test_df['strategy_returns'] = test_df['gross_strategy_returns'] - test_df['transaction_costs']

    test_df['cumulative_market'] = (1 + test_df['market_returns']).cumprod()
    test_df['cumulative_strategy'] = (1 + test_df['strategy_returns']).cumprod()

    # Backward compat alias
    test_df['lstm_pred_vol'] = test_df['forecasted_vol']

    # Calculate PNL Attribution
    # We attribute the strategy return of each hour to the agent that was active at the START of the position
    # Forward-fill the active_agent to track who "owns" the current trade
    test_df['trade_owner'] = test_df['active_agent'].replace('CASH', np.nan).ffill()
    
    # Sum the returns based on the trade owner
    gmm_returns = test_df.loc[test_df['trade_owner'] == 'GMM', 'strategy_returns'].sum()
    breakout_returns = test_df.loc[test_df['trade_owner'] == 'Breakout', 'strategy_returns'].sum()
    
    print("\n=== AGENT PNL ATTRIBUTION ===")
    print(f"GMM Sub-Agent Contribution:      {gmm_returns * 100:.2f}%")
    print(f"Breakout Sub-Agent Contribution: {breakout_returns * 100:.2f}%\n")

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
        if 'trade_owner' in df.columns:
            gmm_contrib = df.loc[df['trade_owner'] == 'GMM', 'strategy_returns'].sum() * 100
            breakout_contrib = df.loc[df['trade_owner'] == 'Breakout', 'strategy_returns'].sum() * 100
            print(f"GMM Sub-Agent Contribution:      {gmm_contrib:.2f}%")
            print(f"Breakout Sub-Agent Contribution: {breakout_contrib:.2f}%")
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
    
    # Shade active regimes
    ax = plt.gca()
    if len(df) > 0 and 'active_agent' in df.columns:
        import matplotlib.patches as mpatches
        
        # Group contiguous blocks of active_agent to speed up plotting
        changes = df['active_agent'] != df['active_agent'].shift(1)
        change_indices = df.index[changes].tolist()
        
        # Calculate time diff to extend the last block
        time_diff = df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta(hours=1)
        end_of_df = df.index[-1] + time_diff
        change_indices.append(end_of_df)
        
        for k in range(len(change_indices) - 1):
            start_time = change_indices[k]
            end_time = change_indices[k+1]
            agent = df.loc[start_time, 'active_agent']
            if isinstance(agent, pd.Series):
                agent = agent.iloc[0]
                
            if agent == 'Breakout':
                ax.axvspan(start_time, end_time, color='red', alpha=0.08)
            elif agent == 'GMM':
                ax.axvspan(start_time, end_time, color='green', alpha=0.08)
            else:
                ax.axvspan(start_time, end_time, color='gray', alpha=0.05)
                
        # Create custom legend entries for the shaded regions
        breakout_patch = mpatches.Patch(color='red', alpha=0.08, label='Red Shading = Momentum Breakout Sub-Agent')
        gmm_patch = mpatches.Patch(color='green', alpha=0.08, label='Green Shading = GMM Mean-Reversion Sub-Agent')
        cash_patch = mpatches.Patch(color='gray', alpha=0.05, label='Gray Shading = Meta-Controller Transition Zone (CASH)')
        
        handles, labels = ax.get_legend_handles_labels()
        handles.extend([breakout_patch, gmm_patch, cash_patch])
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
        
    plt.title('Volatility-Scaled Strategy Performance: Out-of-Sample')
    plt.grid(True, alpha=0.3)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Chart saved to {output_path}")

def plot_dashboard(df):
    print("   [Visual] Rendering Final Portfolio Dashboard...")
    
    # Load parameters to get hmm_threshold
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hmm_threshold = 0.85
    if os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                content = f.read()
                try:
                    best_params = json.loads(content)
                except Exception:
                    best_params = ast.literal_eval(content)
                hmm_threshold = best_params.get('hmm_threshold', 0.85)
        except Exception:
            pass

    subset = df.tail(2000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(subset.index, subset['close'], color='gray', alpha=0.3, label='Price')
    
    # Color coding based on HMM Probability
    high_vol = subset[subset['prob_high_vol'] > hmm_threshold]
    low_vol = subset[subset['prob_high_vol'] <= hmm_threshold]

    ax1.scatter(high_vol.index, high_vol['close'], color='red', s=10, alpha=0.6, label=f'HMM High-Vol (Prob > {hmm_threshold:.2f})')
    ax1.scatter(low_vol.index, low_vol['close'], color='blue', s=10, alpha=0.6, label=f'HMM Low-Vol (Prob <= {hmm_threshold:.2f})')

    ax1.set_title('Market Regimes & HMM Breakout Strategy', fontsize=14, fontweight='bold')
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
    # Load HMM probability from cache or compute it
    import pickle
    cache_path = os.path.join(current_dir, '..', 'data', 'hmm_features_cache.pkl')
    hmm_loaded = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            if cached_df.index[-1] == df.index[-1] and len(cached_df) == len(df):
                df['prob_high_vol'] = cached_df['prob_high_vol']
                print("[Cache] Loaded HMM prob_high_vol from cache.")
                hmm_loaded = True
        except Exception as e:
            print(f"[Cache] Cache load error: {e}. Recomputing...")
            
    if not hmm_loaded:
        print("Computing rolling HMM high-vol probability...")
        df['prob_high_vol'] = df['close'].rolling(window=500).apply(
            lambda x: get_high_vol_probability(x.values) if len(x.dropna()) == 500 else 0.0, raw=True
        )
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df[['close', 'prob_high_vol']], f)
        except Exception as e:
            pass

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