#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  WALK-FORWARD ANALYSIS (WFA) BACKTESTER                              ║
║  Iteratively tunes on rolling 180d, tests out-of-sample on next 30d. ║
║  Generates sanitized out-of-sample performance metrics.              ║
║                                                                      ║
║  Usage:  python research/walk_forward_backtest.py                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import pickle
import ast
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import norm, skew, kurtosis
from arch import arch_model
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data import FEATURE_COLS, calculate_features_test
from strategy import get_gmm_state
from hmm_engine import get_high_vol_probability

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')
STUDY_DB = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'walk_forward_study.db')}"

# Load current best params at startup for Anchored Neighborhood Search
current_params = {}
if os.path.exists(PARAMS_PATH):
    with open(PARAMS_PATH, 'r') as f:
        content = f.read()
        try:
            current_params = json.loads(content)
        except Exception:
            try:
                current_params = ast.literal_eval(content)
            except Exception:
                current_params = {}

def suggest_neighborhood_float(trial, name, default_center, low_factor=0.8, high_factor=1.2, min_val=None, max_val=None):
    center = current_params.get(name, default_center)
    v1 = center * low_factor
    v2 = center * high_factor
    low = min(v1, v2)
    high = max(v1, v2)
    if min_val is not None:
        low = max(low, min_val)
    if max_val is not None:
        high = min(high, max_val)
    if low >= high:
        low = high - 1e-5
    return trial.suggest_float(name, float(low), float(high))

def suggest_neighborhood_int(trial, name, default_center, low_factor=0.8, high_factor=1.2, min_val=1, max_val=None):
    center = current_params.get(name, default_center)
    v1 = int(round(center * low_factor))
    v2 = int(round(center * high_factor))
    low = min(v1, v2)
    high = max(v1, v2)
    if min_val is not None:
        low = max(low, min_val)
    if max_val is not None:
        high = min(high, max_val)
    if low >= high:
        low = high - 1
    return trial.suggest_int(name, int(low), int(high))

def load_eth_data() -> pd.DataFrame:
    data_path = os.path.join(PROJECT_ROOT, 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def load_hmm_features(df):
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'hmm_features_cache.pkl')
    hmm_loaded = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            if df.index.isin(cached_df.index).all():
                df['prob_high_vol'] = cached_df.loc[df.index, 'prob_high_vol']
                hmm_loaded = True
        except Exception:
            pass

    if not hmm_loaded:
        print("Computing rolling HMM high-vol probability...")
        df['prob_high_vol'] = df['close'].rolling(window=500).apply(
            lambda x: get_high_vol_probability(x) if not np.isnan(x).any() else 0.0, raw=True
        )
    return df

def simulate_window(df: pd.DataFrame, params: dict, initial_state: dict = None) -> tuple:
    """Simulates trading on a slice of data, carrying state forward chronologically."""
    bt = df.copy()
    bt['market_returns'] = bt['close'].pct_change()

    # Pre-calculate rolling channels & indicators
    bt['rolling_max'] = bt['close'].rolling(window=int(params['rolling_max_window'])).max().shift(1)
    bt['rolling_min'] = bt['close'].rolling(window=int(params['rolling_min_window'])).min().shift(1)
    
    bt['log_ret'] = np.log(bt['close'] / bt['close'].shift(1))
    bt['vol_24h'] = bt['log_ret'].rolling(24).std()
    bt['vol_168h'] = bt['log_ret'].rolling(168).std()
    vol_shock = (bt['vol_24h'] > (bt['vol_168h'] * params['vol_shock_mult'])).fillna(False)

    roll_mean = bt['close'].rolling(window=20).mean()
    roll_std = bt['close'].rolling(window=20).std()
    bt['z_score'] = (bt['close'] - roll_mean) / (roll_std + 1e-8)

    p = bt['prob_high_vol'].values
    vol_shock_arr = vol_shock.values
    p_blended = np.where(vol_shock_arr, 1.0, p)

    s_breakout_arr = np.zeros(len(bt))
    s_gmm_arr = np.zeros(len(bt))
    
    # Initialize state
    if initial_state:
        current_s_breakout = initial_state['s_breakout']
        current_s_gmm = initial_state['s_gmm']
        short_hold_duration = initial_state['short_hold_duration']
        current_pos = initial_state['current_pos']
    else:
        current_s_breakout = 0.0
        current_s_gmm = 0.0
        short_hold_duration = 0
        current_pos = 0.0

    closes = bt['close'].values
    rolling_mins = bt['rolling_min'].values
    rolling_maxs = bt['rolling_max'].values
    z_scores = bt['z_score'].values
    ema_200s = bt['ema_200'].values
    
    gmm_z_buy = params['gmm_z_buy']
    gmm_z_sell = params['gmm_z_sell']
    gmm_ema_mult = params.get('gmm_ema_mult', 1.03)
    breakout_time_stop_hours = params.get('breakout_time_stop_hours', 72)
    short_dip_thresh         = params.get('short_dip_thresh', 0.12)
    
    for i in range(len(bt)):
        close = closes[i]
        r_min = rolling_mins[i]
        r_max = rolling_maxs[i]
        z_score = z_scores[i]
        
        # S_breakout logic
        if current_s_breakout == -1.0:
            short_hold_duration += 1
            is_exit = False
            if not np.isnan(r_max) and close > r_max:
                is_exit = True
            elif short_hold_duration >= breakout_time_stop_hours:
                is_exit = True
                
            if is_exit:
                current_s_breakout = 0.0
                short_hold_duration = 0
        else:
            if not np.isnan(r_min) and close < r_min:
                trend_active = p_blended[i] > params.get('hmm_trend_min', 0.60)
                if trend_active:
                    if close >= ema_200s[i] * (1.0 - short_dip_thresh):
                        current_s_breakout = -1.0
                        short_hold_duration = 0
            
        # S_gmm logic
        if not np.isnan(z_score) and z_score < gmm_z_buy:
            if close < ema_200s[i] * gmm_ema_mult:
                current_s_gmm = 1.0
        elif not np.isnan(z_score) and z_score > gmm_z_sell:
            current_s_gmm = 0.0
            
        s_breakout_arr[i] = current_s_breakout
        s_gmm_arr[i] = current_s_gmm

    w_breakout = p_blended ** 2
    w_gmm = 1.0 - w_breakout
    raw_target = (w_breakout * s_breakout_arr) + (w_gmm * s_gmm_arr)
    
    daily_forecasted_vol = bt['vol_24h'].values * np.sqrt(24)
    vol_scaler = 0.06 / (daily_forecasted_vol + 1e-8)
    ideal_size = np.clip(raw_target * vol_scaler, -1.0, 1.0)
    bt['target_size'] = ideal_size

    # Friction Filter (Rebalance Threshold Loop)
    raw_targets = bt['target_size'].values
    active_positions = np.zeros(len(raw_targets))
    rebalance_thresh = params['rebalance_threshold']
    
    for i in range(len(raw_targets)):
        target = raw_targets[i]
        if np.isnan(target):
            active_positions[i] = current_pos
            continue
        if target == 0.0 or abs(target - current_pos) > rebalance_thresh:
            current_pos = target
        active_positions[i] = current_pos
        
    bt['position_size'] = active_positions
    bt['active_position'] = bt['position_size'].shift(1).fillna(0)
    
    # Carry initial active position state if provided
    if initial_state and len(bt) > 0:
        bt['active_position'].iloc[0] = initial_state['current_pos']

    bt['gross_strategy_returns'] = bt['market_returns'] * bt['active_position']
    bt['position_change'] = bt['active_position'].diff().abs().fillna(0)
    
    FEE_PCT = 0.0020  # 20 bps fee per trade
    bt['transaction_costs'] = bt['position_change'] * FEE_PCT
    bt['strategy_returns'] = bt['gross_strategy_returns'] - bt['transaction_costs']
    bt.dropna(subset=['strategy_returns'], inplace=True)
    
    final_state = {
        's_breakout': current_s_breakout,
        's_gmm': current_s_gmm,
        'short_hold_duration': short_hold_duration,
        'current_pos': current_pos
    }
    
    return bt, final_state

def objective(trial, train_df):
    params = {
        'hmm_chop_max': suggest_neighborhood_float(trial, 'hmm_chop_max', 0.40, low_factor=0.8, high_factor=1.2, min_val=0.10, max_val=0.50),
        'hmm_trend_min': suggest_neighborhood_float(trial, 'hmm_trend_min', 0.60, low_factor=0.8, high_factor=1.2, min_val=0.50, max_val=0.95),
        'gmm_z_buy': suggest_neighborhood_float(trial, 'gmm_z_buy', -3.65, low_factor=0.8, high_factor=1.2, min_val=-4.50, max_val=-1.00),
        'gmm_z_sell': suggest_neighborhood_float(trial, 'gmm_z_sell', 0.5, low_factor=0.8, high_factor=1.2, min_val=0.0, max_val=1.5),
        'rolling_min_window': suggest_neighborhood_int(trial, 'rolling_min_window', 52, low_factor=0.7, high_factor=1.3, min_val=12, max_val=150),
        'rolling_max_window': suggest_neighborhood_int(trial, 'rolling_max_window', 52, low_factor=0.7, high_factor=1.3, min_val=12, max_val=150),
        'vol_shock_mult': suggest_neighborhood_float(trial, 'vol_shock_mult', 1.50, low_factor=0.8, high_factor=1.2, min_val=1.10, max_val=2.50),
        'rebalance_threshold': suggest_neighborhood_float(trial, 'rebalance_threshold', 0.48, low_factor=0.8, high_factor=1.2, min_val=0.10, max_val=0.85),
        'gmm_ema_mult': suggest_neighborhood_float(trial, 'gmm_ema_mult', 1.03, low_factor=0.8, high_factor=1.2, min_val=0.90, max_val=1.20),
        'breakout_time_stop_hours': suggest_neighborhood_int(trial, 'breakout_time_stop_hours', 72, low_factor=0.8, high_factor=1.2, min_val=24, max_val=168),
        'short_dip_thresh': suggest_neighborhood_float(trial, 'short_dip_thresh', 0.12, low_factor=0.8, high_factor=1.2, min_val=0.01, max_val=0.30),
    }

    # Simulate metrics in-sample
    bt, _ = simulate_window(train_df, params)
    
    if len(bt) < 100 or bt['strategy_returns'].std() < 1e-10:
        return -999.0, -999.0

    total_trades = (bt['active_position'].diff().abs() > 0).sum()
    if total_trades == 0:
        return -999.0, -999.0
        
    total_net_return = bt['strategy_returns'].sum()
    avg_trade_profit = total_net_return / total_trades
    if avg_trade_profit < 0.0020:
        return -999.0, -999.0
        
    mean_ret = bt['strategy_returns'].mean() * 24 * 365
    negative_returns = np.minimum(bt['strategy_returns'].values, 0.0)
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ret / (downside_dev + 1e-9)
    
    cumulative_returns = (1 + bt['strategy_returns']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    
    dataset_days = len(bt) / 24
    normalized_trades = total_trades * (150 / dataset_days)
    if normalized_trades > 250:
        return -999.0, -999.0
        
    return float(sortino), float(max_dd)

def main():
    print("=" * 60)
    print("  WALK-FORWARD ANALYSIS (WFA) RUNNER")
    print("  Train Window: 180 Days | Test Window: 30 Days | Trials/Step: 50")
    print("=" * 60)

    # 1. Load data and setup features globally to ensure consistency
    df = load_eth_data()
    df = load_hmm_features(df)
    
    # Calculate stationary features
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_ret'], inplace=True)
    
    print("Computing conditional volatility (GARCH)...")
    garch = arch_model(df['log_ret'] * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    res = garch.fit(update_freq=0, disp='off')
    df['garch_vol'] = res.conditional_volatility / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()

    # Stationary features
    df['log_return']      = np.log(df['close'] / df['close'].shift(1))
    df['rolling_vol_24h'] = df['log_return'].rolling(window=24).std()
    sma_20 = df['close'].rolling(window=20).mean()
    sma_50 = df['close'].rolling(window=50).mean()
    df['ma_20_dist'] = (df['close'] - sma_20) / sma_20
    df['ma_50_dist'] = (df['close'] - sma_50) / sma_50
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df.dropna(subset=FEATURE_COLS, inplace=True)
    
    # 2. Setup Walk-Forward schedule
    start_test_date = pd.to_datetime('2025-01-01', utc=True)
    end_test_date = df.index.max()
    
    step_days = 30
    train_days = 180
    n_trials = 50
    
    current_test_start = start_test_date
    oos_results = []
    
    initial_state = {
        's_breakout': 0.0,
        's_gmm': 0.0,
        'short_hold_duration': 0,
        'current_pos': 0.0
    }
    
    step_num = 1
    
    # Keep track of active study name to avoid conflicts
    db_conn_str = STUDY_DB
    
    while current_test_start + pd.Timedelta(days=step_days) <= end_test_date:
        current_test_end = current_test_start + pd.Timedelta(days=step_days)
        current_train_start = current_test_start - pd.Timedelta(days=train_days)
        current_train_end = current_test_start
        
        print(f"\n--- WFA STEP {step_num} ---")
        print(f"  Training: {current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')}")
        print(f"  Testing:  {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")
        
        train_df = df[(df.index >= current_train_start) & (df.index < current_train_end)]
        test_df = df[(df.index >= current_test_start) & (df.index < current_test_end)]
        
        if len(train_df) < 2000 or len(test_df) < 500:
            print("  [Warning] Insufficient data for this step. Skipping.")
            current_test_start = current_test_end
            step_num += 1
            continue
            
        # Create dynamic study for this step
        study_name = f"wfa_step_{step_num}_{int(time.time())}"
        study = optuna.create_study(
            study_name=study_name,
            directions=["maximize", "maximize"],
            storage=db_conn_str,
            load_if_exists=False
        )
        
        # Warm start using current_params from best_params.txt if available
        if current_params:
            study.enqueue_trial({
                'hmm_chop_max': current_params.get('hmm_chop_max', 0.40),
                'hmm_trend_min': current_params.get('hmm_trend_min', 0.60),
                'gmm_z_buy': current_params.get('gmm_z_buy', -3.65),
                'gmm_z_sell': current_params.get('gmm_z_sell', 0.5),
                'rolling_min_window': current_params.get('rolling_min_window', 52),
                'rolling_max_window': current_params.get('rolling_max_window', 52),
                'vol_shock_mult': current_params.get('vol_shock_mult', 1.50),
                'rebalance_threshold': current_params.get('rebalance_threshold', 0.48),
                'gmm_ema_mult': current_params.get('gmm_ema_mult', 1.03),
                'breakout_time_stop_hours': current_params.get('breakout_time_stop_hours', 72),
                'short_dip_thresh': current_params.get('short_dip_thresh', 0.12)
            })
            
        # Optimize on training slice
        study.optimize(lambda trial: objective(trial, train_df), n_trials=n_trials)
        
        best_trials = study.best_trials
        if not best_trials:
            print("  [Error] No completed trials found. Skipping step.")
            current_test_start = current_test_end
            step_num += 1
            continue
            
        # Choose trial with highest Sortino ratio
        best_trial = max(best_trials, key=lambda t: t.values[0])
        best_params_step = best_trial.params
        
        print(f"  [Optimizer] Best Train Sortino: {best_trial.values[0]:.4f} | Max DD: {best_trial.values[1]*100:.2f}%")
        
        # Simulate out-of-sample test window with optimal parameters and carry state forward
        test_results_df, next_state = simulate_window(test_df, best_params_step, initial_state)
        initial_state = next_state # Set initial state for next window to final state of current window
        
        # Dynamically shift the neighborhood anchor to these new parameters for the next step
        current_params.update(best_params_step)
        
        # Save step results
        test_results_df['step'] = step_num
        oos_results.append(test_results_df)
        
        current_test_start = current_test_end
        step_num += 1

    if not oos_results:
        print("[Error] No out-of-sample results generated.")
        return

    # 3. Compile out-of-sample series
    final_oos_df = pd.concat(oos_results)
    final_oos_df.sort_index(inplace=True)
    
    # Remove duplicates from window overlaps if any
    final_oos_df = final_oos_df[~final_oos_df.index.duplicated(keep='first')]

    # Calculate metrics on out-of-sample data
    final_oos_df['cumulative_market'] = (1 + final_oos_df['market_returns']).cumprod()
    final_oos_df['cumulative_strategy'] = (1 + final_oos_df['strategy_returns']).cumprod()

    # Calculate out-of-sample stats
    market_ret = (final_oos_df['cumulative_market'].iloc[-1] - 1) * 100
    strat_ret = (final_oos_df['cumulative_strategy'].iloc[-1] - 1) * 100
    
    returns = final_oos_df['strategy_returns'].dropna()
    mean_ret = returns.mean() * 24 * 365
    std_ret = returns.std() * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)
    
    peak = final_oos_df['cumulative_strategy'].cummax()
    drawdown = (final_oos_df['cumulative_strategy'] - peak) / peak
    max_dd = drawdown.min() * 100

    # Calculate Deflated Sharpe Ratio (DSR)
    dsr = 0.0
    if returns.std() > 1e-9:
        sr_observed = returns.mean() / returns.std()
        n = len(returns)
        skewness = skew(returns)
        excess_kurt = kurtosis(returns, fisher=True)
        
        std_sr_unann = np.sqrt((1.0 + 0.5 * sr_observed**2 - skewness * sr_observed + (excess_kurt / 4.0) * sr_observed**2) / (n - 1.0))
        std_sr_ann = std_sr_unann * np.sqrt(24 * 365)
        
        n_trials = step_num * n_trials
        var_sharpes = 0.5
        gamma_const = 0.57721566490153286
        expected_max_ann = 0.0 + np.sqrt(var_sharpes) * (
            (1.0 - gamma_const) * norm.ppf(1.0 - 1.0 / n_trials) +
            gamma_const * norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
        dsr = norm.cdf((sharpe - expected_max_ann) / (std_sr_ann + 1e-9))

    print("\n" + "=" * 60)
    print("  WALK-FORWARD ANALYSIS (WFA) RESULTS (SANIZED OOS)")
    print("=" * 60)
    print(f"OOS Start Date:   {final_oos_df.index.min().strftime('%Y-%m-%d')}")
    print(f"OOS End Date:     {final_oos_df.index.max().strftime('%Y-%m-%d')}")
    print(f"Market Return:    {market_ret:.2f}%")
    print(f"Strategy Return:  {strat_ret:.2f}%")
    print(f"OOS Sharpe Ratio: {sharpe:.2f}")
    print(f"DSR Score:        {dsr * 100:.2f}% Prob of Gen Alpha")
    print(f"Max Drawdown:     {max_dd:.2f}%")
    print("=" * 60)

    # 4. Save results to JSON
    out_payload = {
        'metrics': {
            'market_return': round(float(market_ret), 2),
            'strategy_return': round(float(strat_ret), 2),
            'sharpe': round(float(sharpe), 2),
            'deflated_sharpe': round(float(dsr), 4),
            'max_drawdown': round(float(max_dd), 2)
        }
    }
    json_path = os.path.join(PROJECT_ROOT, 'walk_forward_results.json')
    with open(json_path, 'w') as f:
        json.dump(out_payload, f)
    print(f"\n[WFA] walk_forward_results.json written → {json_path}")

    # 5. Plot walk-forward results
    plt.figure(figsize=(12, 6))
    plt.plot(final_oos_df.index, final_oos_df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(final_oos_df.index, final_oos_df['cumulative_strategy'], label='WFA Vol-Scaled Strategy (OOS)', color='blue', linewidth=1.5)
    plt.title('Walk-Forward Analysis (WFA) Performance: Out-of-Sample')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(PROJECT_ROOT, 'walk_forward_results.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"[WFA] Performance chart saved → {plot_path}")

if __name__ == "__main__":
    main()
