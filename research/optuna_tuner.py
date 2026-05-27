#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX                              ║
║  Maximizes simulated Sharpe Ratio over historical ETH data.          ║
║  Auto-writes optimal params → best_params.txt on completion.         ║
║  Loads HMM cache and pre-computes LSTM forecasts for ultra-speed!    ║
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
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import torch
import torch.nn as nn
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
from strategy import ProgressiveModel, get_gmm_state
from hmm_engine import get_high_vol_probability

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')
STUDY_DB    = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'optuna_study.db')}"
N_TRIALS    = 150
SEQ_LENGTH  = 60
FEE_PCT     = 0.0  # Fees disabled for backtesting

def load_eth_data() -> pd.DataFrame:
    """Load ETH-USD hourly data from static CSV."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"[Sandbox] Loaded {len(df)} rows from CSV.")
    return df

def get_lstm_predictions(df, best_params):
    """Generates forward 24h volatility forecasts using the ProgressiveModel."""
    print("Pre-generating Volatility Forecasts via ProgressiveModel...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    hidden_dim = best_params.get('hidden_dim', 64)
    dropout = best_params.get('dropout', 0.2)
    input_dim = len(FEATURE_COLS)

    model = ProgressiveModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, dropout=dropout)
    model_path = os.path.join(PROJECT_ROOT, 'lstm_model.pth')

    if not os.path.exists(model_path):
        print("   [WARNING] Model not found. Returning flat line.")
        return pd.Series(0, index=df.index)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load scalers
    scaler_path = os.path.join(PROJECT_ROOT, 'scaler.pkl')
    tgt_scaler_path = os.path.join(PROJECT_ROOT, 'target_scaler.pkl')

    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            feat_scaler = pickle.load(f)
    else:
        feat_scaler = None

    tgt_scaler = None
    if os.path.exists(tgt_scaler_path):
        with open(tgt_scaler_path, 'rb') as f:
            tgt_scaler = pickle.load(f)

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

    inputs = np.array(inputs)
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

def load_hmm_features(df):
    """Loads HMM prob_high_vol from cache or computes it."""
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'hmm_features_cache.pkl')
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
    """Ultra-fast hierarchical strategy simulation on pre-computed volatility forecasts."""
    bt = test_df.copy()
    bt['market_returns'] = bt['close'].pct_change()

    # 1. Pre-calculate metrics
    bt['rolling_max'] = bt['close'].rolling(window=params['breakout_window']).max().shift(1)
    bt['rolling_min'] = bt['close'].rolling(window=params['breakout_window']).min().shift(1)
    
    # Realized Volatility Overlay (Circuit Breaker)
    bt['log_ret'] = np.log(bt['close'] / bt['close'].shift(1))
    bt['vol_24h'] = bt['log_ret'].rolling(24).std()
    bt['vol_168h'] = bt['log_ret'].rolling(168).std()
    vol_shock = (bt['vol_24h'] > (bt['vol_168h'] * params['vol_shock_mult'])).fillna(False)

    # Fast Z-Score for the vectorized backtester
    roll_mean = bt['close'].rolling(window=20).mean()
    roll_std = bt['close'].rolling(window=20).std()
    bt['z_score'] = (bt['close'] - roll_mean) / (roll_std + 1e-8)

    # 2. Hierarchical Routing
    bt['signal'] = np.nan
    
    # AGENT 1: Momentum Breakout (Activated when HMM > trend_min OR vol_shock)
    trend_active = (bt['prob_high_vol'] > params['hmm_trend_min']) | vol_shock
    bt.loc[trend_active & (bt['close'] > bt['rolling_max']), 'signal'] = 1
    bt.loc[trend_active & (bt['close'] < bt['rolling_min']), 'signal'] = -1
    
    # AGENT 2: GMM Mean-Reversion (Activated when HMM < chop_max AND NOT vol_shock)
    chop_active = (bt['prob_high_vol'] < params['hmm_chop_max']) & ~vol_shock
    bt.loc[chop_active & (bt['z_score'] < params['gmm_z_buy']), 'signal'] = 1
    bt.loc[chop_active & (bt['z_score'] > params['gmm_z_sell']), 'signal'] = 0
    
    # 3. Master Exits
    bt.loc[(bt['prob_high_vol'] >= params['hmm_chop_max']) & (bt['prob_high_vol'] <= params['hmm_trend_min']) & ~vol_shock, 'signal'] = 0
    
    bt['signal'] = bt['signal'].ffill().fillna(0)

    # Volatility-Scaled Sizing (abs signal for sizing, sign for direction)
    daily_forecasted_vol = bt['forecasted_vol'] * np.sqrt(24)
    base_size = (0.06 / (daily_forecasted_vol + 1e-8)).clip(upper=1.0)
    bt['target_size'] = base_size * bt['signal']

    # 4. Friction Filter (Rebalance Threshold Loop)
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

    # 5. Shift the position size (because we enter the position at the CLOSE of the signal candle)
    bt['active_position'] = bt['position_size'].shift(1).fillna(0)

    # 6. Calculate Gross Return based on the scaled position
    bt['gross_strategy_returns'] = bt['market_returns'] * bt['active_position']

    # 7. Calculate Transaction Friction (Fees)
    bt['position_change'] = bt['active_position'].diff().abs().fillna(0)
    bt['transaction_costs'] = bt['position_change'] * FEE_PCT

    # 8. Calculate Net Strategy Return
    bt['strategy_returns'] = bt['gross_strategy_returns'] - bt['transaction_costs']
    bt.dropna(subset=['strategy_returns'], inplace=True)

    if len(bt) < 100 or bt['strategy_returns'].std() < 1e-10:
        return -999.0

    # Annualized Sharpe
    mean_ret = bt['strategy_returns'].mean() * 24 * 365
    std_ret  = bt['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe   = mean_ret / (std_ret + 1e-9)
    return float(sharpe)

def objective(trial, test_df):
    params = {
        'hmm_chop_max': trial.suggest_float('hmm_chop_max', 0.30, 0.45),
        'hmm_trend_min': trial.suggest_float('hmm_trend_min', 0.55, 0.70),
        'gmm_z_buy': trial.suggest_float('gmm_z_buy', -2.5, -1.0),
        'gmm_z_sell': trial.suggest_float('gmm_z_sell', 0.0, 1.5),
        'breakout_window': trial.suggest_int('breakout_window', 12, 48),
        'vol_shock_mult': trial.suggest_float('vol_shock_mult', 1.3, 2.0),
        'rebalance_threshold': trial.suggest_float('rebalance_threshold', 0.10, 0.25),
    }

    sharpe = simulate_sharpe(test_df, params)
    return sharpe

def write_best_params(study, current_params):
    best = study.best_trial.params
    current_params['breakout_window'] = int(best['breakout_window'])
    current_params['hmm_chop_max']    = float(best['hmm_chop_max'])
    current_params['hmm_trend_min']   = float(best['hmm_trend_min'])
    current_params['gmm_z_buy']       = float(best['gmm_z_buy'])
    current_params['gmm_z_sell']      = float(best['gmm_z_sell'])
    current_params['vol_shock_mult']      = float(best['vol_shock_mult'])
    current_params['rebalance_threshold'] = float(best['rebalance_threshold'])

    with open(PARAMS_PATH, 'w') as f:
        json.dump(current_params, f, indent=2)

    print(f"\n[Air-Gap] best_params.txt updated → {PARAMS_PATH}")
    print(f"[Air-Gap] Best Sharpe: {study.best_value:+.4f}")
    print(f"[Air-Gap] Updated Config:\n{json.dumps(current_params, indent=2)}")

def main():
    print("=" * 60)
    print("  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX (FAST MODE)")
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
        best_params = {"hidden_dim": 64, "lr": 0.001, "dropout": 0.2, "input_dim": 5}

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

    # Split data to get the train set
    train_raw, _ = split_data(df, verbose=True)
    
    # Feature engineering for train set
    train_df = calculate_features_test(train_raw)
    
    # Pre-generate forecasts
    train_df['forecasted_vol'] = get_lstm_predictions(train_df, best_params)
    train_df.dropna(subset=['forecasted_vol'], inplace=True)

    print(f"\n[Sandbox] Feature engineering & pre-computation complete.")
    print(f"[Sandbox] Beginning {N_TRIALS} fast Optuna trials...\n")

    study = optuna.create_study(
        study_name="eth_hmm_breakout_opt",
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, train_df), n_trials=N_TRIALS)

    try:
        write_best_params(study, best_params)
    except ValueError:
        print("[Sandbox] No completed trials found. best_params.txt unchanged.")

    print("\n[Sandbox] Fast Optimization Done.")

if __name__ == "__main__":
    main()
