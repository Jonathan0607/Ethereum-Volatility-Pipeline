import os
import sys
import pandas as pd
import numpy as np
import json
import ast
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = "/Users/kingebenezer/Desktop/Coding/Projects/Ethereum Tracker"
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from data import calculate_features, calculate_features_test, split_data

def run_simulation(df, option_type):
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
    short_entry_price = 0.0
    short_hold_duration = 0
    
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
                if current_s_breakout == 0.0:
                    short_entry_price = close
                    short_hold_duration = 0
                current_s_breakout = -1.0
        else:
            # Check exit conditions
            if current_s_breakout == -1.0:
                is_exit = False
                if not np.isnan(r_max) and close > r_max:
                    is_exit = True
                
                # Option A: 24h Mean Exit
                if option_type == 'option_a':
                    if not np.isnan(rolling_means_24[i]) and close > rolling_means_24[i]:
                        is_exit = True
                
                # Option B: 4% Stop Loss
                elif option_type == 'option_b':
                    if short_entry_price > 0.0 and close > short_entry_price * 1.04:
                        is_exit = True
                
                # Option C: 72h Time Stop
                elif option_type == 'option_c':
                    if short_hold_duration >= 72:
                        is_exit = True
                
                if is_exit:
                    current_s_breakout = 0.0
                    short_entry_price = 0.0
                    short_hold_duration = 0
            
        if current_s_breakout == -1.0:
            short_hold_duration += 1
            
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
    
    # Active Agent determination for chart shading
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
    test_df['cumulative_market'] = (1 + test_df['market_returns'].fillna(0.0)).cumprod()
    
    return test_df

def plot_and_save(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='Vol-Scaled AI (Test Set)', color='blue', linewidth=1.5)
    
    # Shade active regimes
    ax = plt.gca()
    if len(df) > 0 and 'active_agent' in df.columns:
        changes = df['active_agent'] != df['active_agent'].shift(1)
        change_indices = df.index[changes].tolist()
        
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
        
    plt.title(title, fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(PROJECT_ROOT, filename)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {output_path}")

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
    
    print("Simulating Option A (24h MA Exit)...")
    df_a = run_simulation(test_df, 'option_a')
    plot_and_save(df_a, 'Volatility-Scaled Strategy: Option A (24h MA Exit)', 'backtest_results_option_a.png')
    
    print("Simulating Option B (4% Stop Loss)...")
    df_b = run_simulation(test_df, 'option_b')
    plot_and_save(df_b, 'Volatility-Scaled Strategy: Option B (4% Stop Loss)', 'backtest_results_option_b.png')
    
    print("Simulating Option C (72h Time Stop)...")
    df_c = run_simulation(test_df, 'option_c')
    plot_and_save(df_c, 'Volatility-Scaled Strategy: Option C (72h Time Stop)', 'backtest_results_option_c.png')
    
    print("All option plots generated successfully.")

if __name__ == "__main__":
    main()
