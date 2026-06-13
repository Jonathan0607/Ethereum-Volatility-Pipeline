import os
import sys
import pandas as pd
import numpy as np
import json
import ast
import pickle

PROJECT_ROOT = "/Users/kingebenezer/Desktop/Coding/Projects/Ethereum Tracker"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data import calculate_features, calculate_features_test, split_data

def run_simulation_with_exit(df, breakout_exit_type):
    params_path = os.path.join(PROJECT_ROOT, 'best_params.txt')
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
    hmm_trend_min   = best_params.get('hmm_trend_min', 0.60)
    gmm_z_buy       = best_params.get('gmm_z_buy', -3.65)
    gmm_z_sell      = best_params.get('gmm_z_sell', 0.5)
    vol_shock_mult  = best_params.get('vol_shock_mult', 1.5)
    rebalance_threshold = best_params.get('rebalance_threshold', 0.48)
    gmm_ema_mult    = best_params.get('gmm_ema_mult', 1.03)

    test_df = df.copy()
    test_df['market_returns'] = test_df['close'].pct_change()
    test_df['log_ret'] = np.log(test_df['close'] / test_df['close'].shift(1))
    test_df['vol_24h'] = test_df['log_ret'].rolling(24).std()
    test_df['vol_168h'] = test_df['log_ret'].rolling(168).std()
    
    test_df['rolling_max'] = test_df['close'].rolling(window=rolling_max_window).max().shift(1)
    test_df['rolling_min'] = test_df['close'].rolling(window=rolling_min_window).min().shift(1)
    test_df['rolling_mean_24'] = test_df['close'].rolling(window=24).mean()
    
    vol_shock = (test_df['vol_24h'] > (test_df['vol_168h'] * vol_shock_mult)).fillna(False)
    
    roll_mean = test_df['close'].rolling(window=20).mean()
    roll_std = test_df['close'].rolling(window=20).std()
    test_df['z_score'] = (test_df['close'] - roll_mean) / (roll_std + 1e-8)
    
    p = test_df['prob_high_vol'].values
    vol_shock_arr = vol_shock.values
    p_blended = np.where(vol_shock_arr, 1.0, p)
    
    s_breakout_arr = np.zeros(len(test_df))
    s_gmm_arr = np.zeros(len(test_df))
    
    current_s_breakout = 0.0
    current_s_gmm = 0.0
    
    closes = test_df['close'].values
    rolling_mins = test_df['rolling_min'].values
    rolling_maxs = test_df['rolling_max'].values
    z_scores = test_df['z_score'].values
    ema_200s = test_df['ema_200'].values
    rolling_means_24 = test_df['rolling_mean_24'].values
    
    for i in range(len(test_df)):
        close = closes[i]
        r_min = rolling_mins[i]
        r_max = rolling_maxs[i]
        z_score = z_scores[i]
        
        # S_breakout logic
        if not np.isnan(r_min) and close < r_min:
            trend_active = p_blended[i] > hmm_trend_min
            if trend_active:
                current_s_breakout = -1.0
        else:
            # Check exit conditions
            if current_s_breakout == -1.0:
                is_exit = False
                if not np.isnan(r_max) and close > r_max:
                    is_exit = True
                
                if breakout_exit_type == 'r_min' and not np.isnan(r_min) and close > r_min:
                    is_exit = True
                elif breakout_exit_type == 'r_min_1pct' and not np.isnan(r_min) and close > r_min * 1.01:
                    is_exit = True
                elif breakout_exit_type == 'mean_24' and not np.isnan(rolling_means_24[i]) and close > rolling_means_24[i]:
                    is_exit = True
                
                if is_exit:
                    current_s_breakout = 0.0
            
        # S_gmm logic
        if not np.isnan(z_score) and z_score < gmm_z_buy:
            if close < ema_200s[i] * gmm_ema_mult:
                current_s_gmm = 1.0
        elif not np.isnan(z_score) and z_score > gmm_z_sell:
            current_s_gmm = 0.0
            
        s_breakout_arr[i] = current_s_breakout
        s_gmm_arr[i] = current_s_gmm

    test_df['S_breakout'] = s_breakout_arr
    test_df['S_gmm'] = s_gmm_arr
    
    w_breakout = p_blended ** 2
    w_gmm = 1.0 - w_breakout
    
    raw_target = (w_breakout * s_breakout_arr) + (w_gmm * s_gmm_arr)
    
    daily_forecasted_vol = test_df['vol_24h'].values * np.sqrt(24)
    vol_scaler = 0.06 / (daily_forecasted_vol + 1e-8)
    ideal_size = np.clip(raw_target * vol_scaler, -1.0, 1.0)
    test_df['target_size'] = ideal_size

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
    test_df['active_position'] = test_df['position_size'].shift(1).fillna(0)
    
    market_returns = test_df['market_returns'].values
    active_position = test_df['active_position'].values
    volumes = test_df['volume'].values
    vols = test_df['vol_24h'].values
    
    strategy_returns = np.zeros(len(test_df))
    slippage_pcts = np.zeros(len(test_df))
    current_equity = 10000.0
    gamma = 0.5
    median_vol = 3.38e8
    
    for t in range(len(test_df)):
        r_t = market_returns[t]
        w_t = active_position[t]
        vol_t = vols[t] if not np.isnan(vols[t]) else 0.0
        v_t = volumes[t] if not np.isnan(volumes[t]) else 0.0
        
        w_prev = active_position[t-1] if t > 0 else 0.0
        pos_change = abs(w_t - w_prev)
        
        if pos_change > 0.0 and t > 0:
            trade_val_usd = current_equity * pos_change
            v_t_usd = v_t if v_t > 0.0 else median_vol
            slip = gamma * vol_t * np.sqrt(trade_val_usd / v_t_usd)
            slip = min(slip, 0.05)
            friction = max(0.0010, slip)
        else:
            friction = 0.0
            
        slippage_pcts[t] = friction
        gross_ret = r_t * w_t
        if np.isnan(gross_ret):
            gross_ret = 0.0
            
        net_ret = gross_ret - friction
        strategy_returns[t] = net_ret
        current_equity = current_equity * (1.0 + net_ret)
        current_equity = max(current_equity, 0.0)
        
    test_df['strategy_returns'] = strategy_returns
    test_df['cumulative_strategy'] = (1 + test_df['strategy_returns']).cumprod()
    
    # Slice the period: Feb 5, 2026 to May 15, 2026
    start_date = pd.to_datetime('2026-02-05', utc=True)
    end_date = pd.to_datetime('2026-05-15', utc=True)
    if test_df.index.tz is None:
        start_date = start_date.tz_localize(None)
        end_date = end_date.tz_localize(None)
    slice_df = test_df.loc[start_date:end_date]
    
    strat_ret_slice = (slice_df['cumulative_strategy'].iloc[-1] / slice_df['cumulative_strategy'].iloc[0] - 1) * 100
    
    # Overall metrics
    risk_free_rate = 0.0
    strategy_mean = test_df['strategy_returns'].mean() * 24 * 365
    strategy_std = test_df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)

    cumulative = test_df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    strat_ret = (cumulative.iloc[-1] - 1) * 100
    
    return strat_ret, sharpe, max_drawdown * 100, strat_ret_slice

def main():
    # Load data
    data_path = os.path.join(PROJECT_ROOT, 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    
    # Load HMM Cache
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'hmm_features_cache.pkl')
    df['prob_high_vol'] = np.nan
    with open(cache_path, 'rb') as f:
        cached_df = pickle.load(f)
    common_idx = df.index.intersection(cached_df.index)
    df.loc[common_idx, 'prob_high_vol'] = cached_df.loc[common_idx, 'prob_high_vol']
    
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    train_raw, test_raw = split_data(df, verbose=False)
    test_df = calculate_features_test(test_raw, train_df=train_raw)
    test_df['ema_200'] = df['ema_200'].loc[test_df.index]
    
    print("Evaluating exit conditions for breakout shorts...")
    
    # Baseline
    ret, sharpe, dd, slice_ret = run_simulation_with_exit(test_df, 'baseline')
    print(f"Baseline (r_max only) -> Full Return: {ret:.2f}% | Sharpe: {sharpe:.4f} | Max DD: {dd:.2f}% | Slice Return: {slice_ret:.2f}%")
    
    # Stop-out at r_min
    ret, sharpe, dd, slice_ret = run_simulation_with_exit(test_df, 'r_min')
    print(f"Stop at r_min (breakdown line) -> Full Return: {ret:.2f}% | Sharpe: {sharpe:.4f} | Max DD: {dd:.2f}% | Slice Return: {slice_ret:.2f}%")
    
    # Stop at r_min * 1.01
    ret, sharpe, dd, slice_ret = run_simulation_with_exit(test_df, 'r_min_1pct')
    print(f"Stop at r_min * 1.01 -> Full Return: {ret:.2f}% | Sharpe: {sharpe:.4f} | Max DD: {dd:.2f}% | Slice Return: {slice_ret:.2f}%")
    
    # Stop at 24h close mean
    ret, sharpe, dd, slice_ret = run_simulation_with_exit(test_df, 'mean_24')
    print(f"Stop at 24h Mean -> Full Return: {ret:.2f}% | Sharpe: {sharpe:.4f} | Max DD: {dd:.2f}% | Slice Return: {slice_ret:.2f}%")

if __name__ == "__main__":
    main()
