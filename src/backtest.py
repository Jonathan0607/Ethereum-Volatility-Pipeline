import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import sys
import ast
import pickle
import logging

# Tell hmmlearn to shut up unless it's a fatal error
logging.getLogger("hmmlearn").setLevel(logging.WARNING)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import calculate_features, calculate_features_test, split_data, FEATURE_COLS
from strategy import get_gmm_state
from hmm_engine import get_high_vol_probability

SEQ_LENGTH = 60

def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0
TARGET_VOLATILITY = 0.06  # Must match api.py
FEE_PCT = 0.0010  # Fees disabled for backtesting

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def run_backtest(df, target_volatility=TARGET_VOLATILITY):
    # print("Running Backtest with Volatility-Scaled Sizing...")
    
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
        
    rolling_min_window = best_params.get('rolling_min_window', 48)
    rolling_max_window = best_params.get('rolling_max_window', 48)
    hmm_chop_max    = best_params.get('hmm_chop_max', 0.40)
    hmm_trend_min   = best_params.get('hmm_trend_min', 0.60)
    gmm_z_buy       = best_params.get('gmm_z_buy', -1.5)
    gmm_z_sell      = best_params.get('gmm_z_sell', 0.5)
    vol_shock_mult  = best_params.get('vol_shock_mult', 1.5)
    rebalance_threshold = best_params.get('rebalance_threshold', 0.15)

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
                # print("[Cache] Loaded HMM prob_high_vol from cache.")
                hmm_loaded = True
        except Exception as e:
            print(f"[Cache] Cache load error: {e}. Recomputing...")
            
    if not hmm_loaded:
        # print("Computing rolling HMM high-vol probability (this might take a few minutes if not cached)...")
        df['prob_high_vol'] = df['close'].rolling(window=500).apply(
            lambda x: get_high_vol_probability(x) if not np.isnan(x).any() else 0.0, raw=True
        )
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df[['close', 'prob_high_vol']], f)
            # print("[Cache] Saved computed HMM prob_high_vol to cache.")
        except Exception as e:
            print(f"[Cache] Save error: {e}")
            
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    train_raw, test_raw = split_data(df, verbose=False)

    if test_raw.empty:
        raise ValueError("Test set is empty!")

    train_df = calculate_features(train_raw, train_df=train_raw)
    test_df = calculate_features_test(test_raw, train_df=train_raw)

    # print(f"--- BACKTEST STARTING ON {test_df.index[0]} ---")
    test_df['market_returns'] = test_df['close'].pct_change()
    test_df['log_ret'] = np.log(test_df['close'] / test_df['close'].shift(1))
    test_df['vol_24h'] = test_df['log_ret'].rolling(24).std()
    test_df['vol_168h'] = test_df['log_ret'].rolling(168).std()
    
    test_df['forecasted_vol'] = test_df['vol_24h']

    # Compression Breakout Filters
    test_df['rolling_max'] = test_df['close'].rolling(window=rolling_max_window).max().shift(1)
    test_df['rolling_min'] = test_df['close'].rolling(window=rolling_min_window).min().shift(1)
    test_df['rolling_mean'] = test_df['close'].rolling(window=24).mean()
    
    vol_shock = (test_df['vol_24h'] > (test_df['vol_168h'] * vol_shock_mult)).fillna(False)

    # Fast Z-Score
    roll_mean = test_df['close'].rolling(window=20).mean()
    roll_std = test_df['close'].rolling(window=20).std()
    test_df['z_score'] = (test_df['close'] - roll_mean) / (roll_std + 1e-8)

    # 1. Evaluate BOTH sub-agents independently on every tick (stateful)
    s_breakout_arr = np.zeros(len(test_df))
    s_gmm_arr = np.zeros(len(test_df))
    
    current_s_breakout = 0.0
    current_s_gmm = 0.0
    
    closes = test_df['close'].values
    rolling_mins = test_df['rolling_min'].values
    rolling_maxs = test_df['rolling_max'].values
    z_scores = test_df['z_score'].values
    
    for i in range(len(test_df)):
        close = closes[i]
        r_min = rolling_mins[i]
        r_max = rolling_maxs[i]
        z_score = z_scores[i]
        
        # S_breakout logic
        if not np.isnan(r_min) and close < r_min:
            current_s_breakout = -1.0
        elif not np.isnan(r_max) and close > r_max:
            current_s_breakout = 0.0
            
        # S_gmm logic
        if not np.isnan(z_score) and z_score < gmm_z_buy:
            current_s_gmm = 1.0
        elif not np.isnan(z_score) and z_score > gmm_z_sell:
            current_s_gmm = 0.0
            
        s_breakout_arr[i] = current_s_breakout
        s_gmm_arr[i] = current_s_gmm

    test_df['S_breakout'] = s_breakout_arr
    test_df['S_gmm'] = s_gmm_arr

    # 2. Dynamic Weighting Math
    p = test_df['prob_high_vol'].values
    vol_shock_arr = vol_shock.values
    p_blended = np.where(vol_shock_arr, 1.0, p)
    
    w_breakout = p_blended ** 2
    w_gmm = 1.0 - w_breakout
    
    # 3. Target Position Formula (Blended & Scaled)
    raw_target = (w_breakout * s_breakout_arr) + (w_gmm * s_gmm_arr)
    
    daily_forecasted_vol = test_df['forecasted_vol'].values * np.sqrt(24)
    vol_scaler = target_volatility / (daily_forecasted_vol + 1e-8)
    ideal_size = np.clip(raw_target * vol_scaler, -1.0, 1.0)
    test_df['target_size'] = ideal_size

    # Backwards compatibility signals and tracking
    test_df['signal'] = np.where(ideal_size > 0.0, 1.0, np.where(ideal_size < 0.0, -1.0, 0.0))

    active_agents = []
    for w_b, s_b, w_g, s_g in zip(w_breakout, s_breakout_arr, w_gmm, s_gmm_arr):
        b_val = w_b * abs(s_b)
        g_val = w_g * abs(s_g)
        if b_val > g_val:
            active_agents.append('Breakout')
        elif g_val > b_val:
            active_agents.append('GMM')
        else:
            active_agents.append('CASH')
    test_df['active_agent'] = active_agents

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
    # print(f"   [Debug] Sized Positions: LONG={n_long} | SHORT={n_short} | FLAT={n_flat}")

    # 1. Shift the position size (because we enter the position at the CLOSE of the signal candle)
    # The 'position_size' column is directional (positive for Long, negative for Short, 0 for Flat)
    test_df['active_position'] = test_df['position_size'].shift(1).fillna(0)

    # 2. Calculate Gross Return based on the scaled position
    test_df['gross_strategy_returns'] = test_df['market_returns'] * test_df['active_position']

    # 3. Calculate Transaction Friction (Dynamic Square-Root Market Impact Slippage)
    market_returns = test_df['market_returns'].values
    active_position = test_df['active_position'].values
    closes = test_df['close'].values
    volumes = test_df['volume'].values
    vols = test_df['vol_24h'].values
    
    strategy_returns = np.zeros(len(test_df))
    slippage_pcts = np.zeros(len(test_df))
    equity = np.zeros(len(test_df))
    
    # Calculate median non-zero hourly volume to use as fallback
    non_zero_vols = volumes[volumes > 0]
    median_vol = np.median(non_zero_vols) if len(non_zero_vols) > 0 else 3.38e8
    
    current_equity = 10000.0
    gamma = 0.5
    
    for t in range(len(test_df)):
        r_t = market_returns[t]
        w_t = active_position[t]
        vol_t = vols[t] if not np.isnan(vols[t]) else 0.0
        v_t = volumes[t] if not np.isnan(volumes[t]) else 0.0
        
        # Calculate position change
        w_prev = active_position[t-1] if t > 0 else 0.0
        pos_change = abs(w_t - w_prev)
        
        if pos_change > 0.0 and t > 0:
            # Trade value in USD
            trade_val_usd = current_equity * pos_change
            
            # Fallback to median volume if reported volume is 0
            v_t_usd = v_t if v_t > 0.0 else median_vol
            
            # Slippage percent using Square-Root Law (dimensionless ratio in USD)
            slip = gamma * vol_t * np.sqrt(trade_val_usd / v_t_usd)
            slip = min(slip, 0.05)
            
            # Apply broker fee floor of 10 bps (0.0010)
            friction = max(0.0010, slip)
        else:
            friction = 0.0
            
        slippage_pcts[t] = friction
        
        # Gross return
        gross_ret = r_t * w_t
        if np.isnan(gross_ret):
            gross_ret = 0.0
            
        # Net return
        net_ret = gross_ret - friction
        strategy_returns[t] = net_ret
        
        # Update equity
        current_equity = current_equity * (1.0 + net_ret)
        current_equity = max(current_equity, 0.0)
        equity[t] = current_equity
        
    test_df['slippage_costs'] = slippage_pcts
    test_df['transaction_costs'] = slippage_pcts
    test_df['strategy_returns'] = strategy_returns
    test_df['position_change'] = test_df['active_position'].diff().abs().fillna(0)

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

    # Calculate sub-agent contributions and drawdowns
    if 'active_agent' in df.columns:
        df['trade_owner'] = df['active_agent'].replace('CASH', np.nan).ffill()
        
        gmm_returns = df.loc[df['trade_owner'] == 'GMM', 'strategy_returns'].sum()
        breakout_returns = df.loc[df['trade_owner'] == 'Breakout', 'strategy_returns'].sum()
        
        strat_rets = df['strategy_returns'].fillna(0.0)
        trade_owners = df['trade_owner'].fillna('CASH')
        
        gmm_series = np.where(trade_owners == 'GMM', strat_rets, 0.0)
        gmm_cum = np.cumprod(1 + gmm_series)
        gmm_peak = np.maximum.accumulate(gmm_cum)
        gmm_drawdown = (gmm_cum - gmm_peak) / gmm_peak
        gmm_max_dd = np.min(gmm_drawdown) * 100
        
        breakout_series = np.where(trade_owners == 'Breakout', strat_rets, 0.0)
        breakout_cum = np.cumprod(1 + breakout_series)
        breakout_peak = np.maximum.accumulate(breakout_cum)
        breakout_drawdown = (breakout_cum - breakout_peak) / breakout_peak
        breakout_max_dd = np.min(breakout_drawdown) * 100

    market_ret = (df['cumulative_market'].iloc[-1] - 1) * 100
    strat_ret = (df['cumulative_strategy'].iloc[-1] - 1) * 100

    # Calculate Deflated Sharpe Ratio (DSR)
    from scipy.stats import norm, skew, kurtosis
    returns = df['strategy_returns'].dropna()
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    
    dsr = 0.0
    if std_ret > 1e-9:
        sr_observed = mean_ret / std_ret
        n = len(returns)
        skewness = skew(returns)
        excess_kurt = kurtosis(returns, fisher=True)
        
        std_sr_unann = np.sqrt((1.0 + 0.5 * sr_observed**2 - skewness * sr_observed + (excess_kurt / 4.0) * sr_observed**2) / (n - 1.0))
        std_sr_ann = std_sr_unann * np.sqrt(24 * 365)
        
        n_trials = 500
        var_sharpes = 0.5
        gamma_const = 0.57721566490153286
        expected_max_ann = 0.0 + np.sqrt(var_sharpes) * (
            (1.0 - gamma_const) * norm.ppf(1.0 - 1.0 / n_trials) +
            gamma_const * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
        
        dsr = norm.cdf((sharpe - expected_max_ann) / (std_sr_ann + 1e-9))

    if verbose:
        market_return = market_ret / 100
        strategy_return = strat_ret / 100
        sharpe_ratio = sharpe
        gmm_max_dd_val = gmm_max_dd / 100 if 'active_agent' in df.columns else 0.0
        breakout_max_dd_val = breakout_max_dd / 100 if 'active_agent' in df.columns else 0.0

        print("\n=== VOLATILITY-SCALED BACKTEST RESULTS (OUT OF SAMPLE) ===")
        
        # Calculate sized positions string (adjust variable names to match your existing dataframe columns)
        longs = (df['position_size'] > 0).sum()
        shorts = (df['position_size'] < 0).sum()
        flat = (df['position_size'] == 0).sum()
        print(f"Sized Positions: LONG={longs} | SHORT={shorts} | FLAT={flat}")
        
        print(f"Market Return:   {market_return * 100:.2f}%")
        print(f"Strategy Return: {strategy_return * 100:.2f}%")
        print(f"Raw Sharpe:      {sharpe_ratio:.2f}")
        print(f"Deflated Sharpe: {dsr:.4f} ({dsr * 100:.2f}% Prob of Gen Alpha)")
        print(f"Max Drawdown:    {max_drawdown * 100:.2f}%")
        
        if 'trade_owner' in df.columns:
            print("\n=====Sub-Agent PNL========")
            print(f"GMM  Contribution:      {gmm_returns * 100:.2f}%")
            print(f"GMM  Max Drawdown:      {gmm_max_dd_val * 100:.2f}%")
            print(f"Breakout  Contribution: {breakout_returns * 100:.2f}%")
            print(f"Breakout  Max Drawdown: {breakout_max_dd_val * 100:.2f}%")

    return {
        'market_return': round(market_ret, 2),
        'strategy_return': round(strat_ret, 2),
        'sharpe': round(sharpe, 2),
        'deflated_sharpe': round(dsr, 4),
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
    # print(f"Backtest JSON exported → {out_path}")

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
    # print(f"Chart saved to {output_path}")

def plot_dashboard(df):
    # print("   [Visual] Rendering Final Portfolio Dashboard...")
    
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
        ax2.plot(subset.index, subset[lstm_col], color='magenta', linestyle='--', linewidth=1.5, label='Rolling Vol (24h)')
        veto_line = subset['volatility'].rolling(24).mean() * 1.8
        ax2.plot(subset.index, veto_line, color='black', linestyle=':', alpha=0.6, label='Risk Threshold (1.8σ)')
        ax2.fill_between(subset.index, 0, subset['volatility'].max(),
                         where=(subset[lstm_col] > veto_line),
                         color='red', alpha=0.1, label='Risk Signal: CASH')

    ax2.set_title('Risk Detection & Regimes', fontsize=12)
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.15)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_dashboard.png')
    plt.savefig(output_path, dpi=300)
    # print(f"   [Success] Dashboard saved to {output_path}")

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
                # print("[Cache] Loaded HMM prob_high_vol from cache.")
                hmm_loaded = True
        except Exception as e:
            print(f"[Cache] Cache load error: {e}. Recomputing...")
            
    if not hmm_loaded:
        # print("Computing rolling HMM high-vol probability...")
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
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['vol_24h'] = df['log_ret'].rolling(24).std()
    df['forecasted_vol'] = df['vol_24h']
    df['lstm_pred_vol'] = df['vol_24h']
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