#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX (CONSTRAINED V2)             ║
║  Maximizes simulated Sharpe Ratio over historical ETH data.          ║
║  Auto-writes optimal params → best_params.txt on completion.         ║
║                                                                      ║
║  Usage:  python research/optuna_tuner.py                             ║
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
from sklearn.preprocessing import StandardScaler
from arch import arch_model
import optuna

warnings.filterwarnings("ignore")

# Force unbuffered stdout for immediate logging
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data import FEATURE_COLS, split_data, calculate_features, calculate_features_test
from strategy import get_gmm_state
from hmm_engine import get_high_vol_probability

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')

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

STUDY_DB    = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'optuna_study.db')}"
N_TRIALS    = 500

def load_eth_data() -> pd.DataFrame:
    """Load ETH-USD hourly data from static CSV."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"[Sandbox] Loaded {len(df)} rows from CSV.")
    return df

def load_hmm_features(df):
    """Loads HMM prob_high_vol from cache or computes it."""
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'hmm_features_cache.pkl')
    hmm_loaded = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_df = pickle.load(f)
            if df.index.isin(cached_df.index).all():
                df['prob_high_vol'] = cached_df.loc[df.index, 'prob_high_vol']
                print("[Cache] Loaded HMM prob_high_vol from cache.")
                hmm_loaded = True
        except Exception as e:
            print(f"[Cache] Cache load error: {e}. Recomputing...")

    if not hmm_loaded:
        print("Computing rolling HMM high-vol probability...")
        df['prob_high_vol'] = df['close'].rolling(window=500).apply(
            lambda x: get_high_vol_probability(x) if not np.isnan(x).any() else 0.0, raw=True
        )
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(df[['close', 'prob_high_vol']], f)
            print("[Cache] Saved HMM features to cache.")
        except Exception as e:
            print(f"[Cache] Save error: {e}")
    return df

def simulate_sharpe(test_df: pd.DataFrame, params: dict) -> float:
    """Simulate the V3 HMM + GMM blended stateful strategy and calculate Sharpe."""
    bt = test_df.copy()
    bt['market_returns'] = bt['close'].pct_change()

    # 1. Pre-calculate metrics
    bt['rolling_max'] = bt['close'].rolling(window=params['rolling_max_window']).max().shift(1)
    bt['rolling_min'] = bt['close'].rolling(window=params['rolling_min_window']).min().shift(1)
    
    # Realized Volatility Overlay (Circuit Breaker)
    bt['log_ret'] = np.log(bt['close'] / bt['close'].shift(1))
    bt['vol_24h'] = bt['log_ret'].rolling(24).std()
    bt['vol_168h'] = bt['log_ret'].rolling(168).std()
    vol_shock = (bt['vol_24h'] > (bt['vol_168h'] * params['vol_shock_mult'])).fillna(False)

    # Fast Z-Score for the GMM condition
    roll_mean = bt['close'].rolling(window=20).mean()
    roll_std = bt['close'].rolling(window=20).std()
    bt['z_score'] = (bt['close'] - roll_mean) / (roll_std + 1e-8)

    # 3. Dynamic Weighting Math
    p = bt['prob_high_vol'].values
    vol_shock_arr = vol_shock.values
    p_blended = np.where(vol_shock_arr, 1.0, p)

    # 2. Evaluate BOTH sub-agents independently on every tick (stateful)
    s_breakout_arr = np.zeros(len(bt))
    s_gmm_arr = np.zeros(len(bt))
    
    current_s_breakout = 0.0
    current_s_gmm = 0.0
    
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
    
    short_hold_duration = 0
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

    # 3. Dynamic Weighting Math (re-use p_blended)
    p = bt['prob_high_vol'].values
    vol_shock_arr = vol_shock.values
    p_blended = np.where(vol_shock_arr, 1.0, p)
    
    w_breakout = p_blended ** 2
    w_gmm = 1.0 - w_breakout
    
    # 4. Target Position Formula (Blended & Scaled)
    raw_target = (w_breakout * s_breakout_arr) + (w_gmm * s_gmm_arr)
    
    daily_forecasted_vol = bt['vol_24h'].values * np.sqrt(24)
    vol_scaler = 0.06 / (daily_forecasted_vol + 1e-8)
    ideal_size = np.clip(raw_target * vol_scaler, -1.0, 1.0)
    bt['target_size'] = ideal_size

    # 5. Friction Filter (Rebalance Threshold Loop)
    raw_targets = bt['target_size'].values
    active_positions = np.zeros(len(raw_targets))
    current_pos = 0.0
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

    # 6. Shift the position size (because we enter the position at the CLOSE of the signal candle)
    bt['active_position'] = bt['position_size'].shift(1).fillna(0)

    # 7. Calculate Gross Return
    bt['gross_strategy_returns'] = bt['market_returns'] * bt['active_position']

    # 8. Calculate Transaction Friction (Enforce 20 bps fee per trade)
    bt['position_change'] = bt['active_position'].diff().abs().fillna(0)
    
    FEE_PCT = 0.0020  # 20 bps fee per trade
    bt['transaction_costs'] = bt['position_change'] * FEE_PCT

    # 9. Calculate Net Strategy Return
    bt['strategy_returns'] = bt['gross_strategy_returns'] - bt['transaction_costs']
    bt.dropna(subset=['strategy_returns'], inplace=True)

    if len(bt) < 100 or bt['strategy_returns'].std() < 1e-10:
        return -999.0

    total_trades = (bt['active_position'].diff().abs() > 0).sum()
    
    # Annualized Sharpe
    mean_ret = bt['strategy_returns'].mean() * 24 * 365
    std_ret  = bt['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe   = mean_ret / (std_ret + 1e-9)
    
    # Calculate Maximum Drawdown
    cumulative_returns = (1 + bt['strategy_returns']).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_dd = drawdown.min()
    
    # Normalize trade count
    dataset_days = len(bt) / 24
    normalized_trades = total_trades * (150 / dataset_days)
    
    # PENALTY 1: Hyper-activity (avoid whipsaws)
    if normalized_trades > 250:
        return -5.0
        
    # PENALTY 2: Severe drawdown violation (Drawdown > -25%)
    if max_dd < -0.25:
        # Subtract a heavy penalty based on violation magnitude
        penalty = abs(max_dd + 0.25) * 100.0
        return float(sharpe - penalty)
        
    return float(sharpe)

def objective(trial, test_df):
    default_min = current_params.get('rolling_min_window', 52)
    default_max = current_params.get('rolling_max_window', 52)
    
    params = {
        'hmm_chop_max': suggest_neighborhood_float(trial, 'hmm_chop_max', 0.40, low_factor=0.8, high_factor=1.2, min_val=0.10, max_val=0.50),
        'hmm_trend_min': suggest_neighborhood_float(trial, 'hmm_trend_min', 0.60, low_factor=0.8, high_factor=1.2, min_val=0.50, max_val=0.95),
        'gmm_z_buy': suggest_neighborhood_float(trial, 'gmm_z_buy', -3.65, low_factor=0.8, high_factor=1.2, min_val=-4.50, max_val=-1.00),
        'gmm_z_sell': suggest_neighborhood_float(trial, 'gmm_z_sell', 0.5, low_factor=0.8, high_factor=1.2, min_val=0.0, max_val=1.5),
        
        # Optimize support and resistance channels independently
        'rolling_min_window': suggest_neighborhood_int(trial, 'rolling_min_window', default_min, low_factor=0.7, high_factor=1.3, min_val=12, max_val=150),
        'rolling_max_window': suggest_neighborhood_int(trial, 'rolling_max_window', default_max, low_factor=0.7, high_factor=1.3, min_val=12, max_val=150),
        
        'vol_shock_mult': suggest_neighborhood_float(trial, 'vol_shock_mult', 1.50, low_factor=0.8, high_factor=1.2, min_val=1.10, max_val=2.50),
        'rebalance_threshold': suggest_neighborhood_float(trial, 'rebalance_threshold', 0.48, low_factor=0.8, high_factor=1.2, min_val=0.10, max_val=0.85),
        'gmm_ema_mult': suggest_neighborhood_float(trial, 'gmm_ema_mult', 1.03, low_factor=0.8, high_factor=1.2, min_val=0.90, max_val=1.20),
        'breakout_time_stop_hours': suggest_neighborhood_int(trial, 'breakout_time_stop_hours', 72, low_factor=0.8, high_factor=1.2, min_val=24, max_val=168),
        'short_dip_thresh': suggest_neighborhood_float(trial, 'short_dip_thresh', 0.12, low_factor=0.8, high_factor=1.2, min_val=0.01, max_val=0.30),
    }

    sharpe = simulate_sharpe(test_df, params)
    return sharpe

def write_best_params(study, current_params):
    best = study.best_trial.params
    current_params['rolling_min_window'] = int(best['rolling_min_window'])
    current_params['rolling_max_window'] = int(best['rolling_max_window'])
    current_params['hmm_chop_max']    = float(best['hmm_chop_max'])
    current_params['hmm_trend_min']   = float(best['hmm_trend_min'])
    current_params['gmm_z_buy']       = float(best['gmm_z_buy'])
    current_params['gmm_z_sell']      = float(best['gmm_z_sell'])
    current_params['vol_shock_mult']      = float(best['vol_shock_mult'])
    current_params['rebalance_threshold'] = float(best['rebalance_threshold'])
    current_params['gmm_ema_mult']        = float(best['gmm_ema_mult'])
    current_params['breakout_time_stop_hours'] = int(best['breakout_time_stop_hours'])
    current_params['short_dip_thresh']         = float(best['short_dip_thresh'])
    
    # Remove obsolete single breakout_window to avoid confusion
    if 'breakout_window' in current_params:
        del current_params['breakout_window']

    with open(PARAMS_PATH, 'w') as f:
        json.dump(current_params, f, indent=2)

    print(f"\n[Air-Gap] best_params.txt updated → {PARAMS_PATH}")
    print(f"[Air-Gap] Best Sharpe: {study.best_value:+.4f}")
    print(f"[Air-Gap] Updated Config:\n{json.dumps(current_params, indent=2)}")

def main():
    print("=" * 60)
    print("  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX (CONSTRAINED V2)")
    print("  Metric: Maximize Simulated Sharpe Ratio")
    print(f"  Trials: {N_TRIALS}  |  Study DB: {STUDY_DB}")
    print("=" * 60)

    # Load current best params
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, 'r') as f:
            content = f.read()
            try:
                best_params = json.loads(content)
            except Exception:
                best_params = ast.literal_eval(content)
    else:
        best_params = {}

    df = load_eth_data()
    df = load_hmm_features(df)
    
    # Calculate stationary features
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(subset=['log_ret'], inplace=True)
    
    # Fit GARCH on full set or training set to align features
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

    # In main(): Replace the rigid train split with a rolling production lookback
    # Ensure the data passed to the optimizer represents the immediate trailing 90 or 180 days
    optimization_window_days = 180
    cutoff_date = df.index.max() - pd.Timedelta(days=optimization_window_days)
    production_train_df = df[df.index >= cutoff_date]

    # Complete feature preparation on the designated window
    train_df = calculate_features_test(production_train_df)
    train_df['forecasted_vol'] = train_df['rolling_vol_24h']
    train_df.dropna(subset=['forecasted_vol'], inplace=True)

    print(f"\n[Sandbox] Feature engineering & pre-computation complete.")
    print(f"[Sandbox] Beginning {N_TRIALS} fast Optuna trials...\n")

    import time

    # Create a unique timestamped study name for a completely clean canvas
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    STUDY_NAME = f"eth_hmm_breakout_v4_{timestamp}"

    print(f"[Sandbox] Initializing clean optimization study: {STUDY_NAME}")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=False, # Safe now because the name is unique every run
    )

    study.optimize(lambda trial: objective(trial, train_df), n_trials=N_TRIALS)

    try:
        write_best_params(study, best_params)
    except ValueError:
        print("[Sandbox] No completed trials found. best_params.txt unchanged.")

    print("\n[Sandbox] Fast Optimization Done.")

if __name__ == "__main__":
    main()
