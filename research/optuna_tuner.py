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
from strategy import ProgressiveModel
from hmm_engine import get_high_vol_probability

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')
STUDY_DB    = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'optuna_study.db')}"
N_TRIALS    = 150
SEQ_LENGTH  = 60

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

def simulate_sharpe(test_df: pd.DataFrame, breakout_window: int, hmm_threshold: float) -> float:
    """Ultra-fast bi-directional strategy simulation on pre-computed volatility forecasts."""
    bt = test_df.copy()
    bt['market_returns'] = bt['close'].pct_change()

    # Compression Breakout Filters
    bt['rolling_max'] = bt['close'].rolling(window=breakout_window).max().shift(1)
    bt['rolling_min'] = bt['close'].rolling(window=breakout_window).min().shift(1)
    bt['rolling_mean'] = bt['close'].rolling(window=24).mean()

    # Bi-directional HMM Breakout Logic
    long_cond = (bt['close'] > bt['rolling_max']) & (bt['prob_high_vol'] > hmm_threshold)
    short_cond = (bt['close'] < bt['rolling_min']) & (bt['prob_high_vol'] > hmm_threshold)

    exit_long = (bt['prob_high_vol'] < 0.40) | (bt['close'] < bt['rolling_mean'])
    exit_short = (bt['prob_high_vol'] < 0.40) | (bt['close'] > bt['rolling_mean'])

    # Vectorized state evaluation
    bt['signal'] = np.nan
    bt.loc[long_cond, 'signal'] = 1
    bt.loc[short_cond, 'signal'] = -1

    # Forward fill the active signal, then apply exits based on current position
    bt['signal'] = bt['signal'].ffill()
    bt.loc[(bt['signal'] == 1) & exit_long, 'signal'] = 0
    bt.loc[(bt['signal'] == -1) & exit_short, 'signal'] = 0
    bt['signal'] = bt['signal'].ffill().fillna(0)

    # Volatility-Scaled Sizing (abs signal for sizing, sign for direction)
    daily_forecasted_vol = bt['forecasted_vol'] * np.sqrt(24)
    base_size = (0.06 / (daily_forecasted_vol + 1e-8)).clip(upper=1.0)
    bt['position_size'] = base_size * bt['signal']  # Signed: +1 long, -1 short

    # Strategy returns: signal * market_returns handles short P&L correctly
    bt['strategy_returns'] = bt['market_returns'] * bt['signal'].shift(1)
    bt.dropna(subset=['strategy_returns'], inplace=True)

    if len(bt) < 100 or bt['strategy_returns'].std() < 1e-10:
        return -999.0

    # Annualized Sharpe
    mean_ret = bt['strategy_returns'].mean() * 24 * 365
    std_ret  = bt['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe   = mean_ret / (std_ret + 1e-9)
    return float(sharpe)

def objective(trial, test_df):
    breakout_window = trial.suggest_int('breakout_window', 12, 48)  # Micro-breakouts
    hmm_threshold   = trial.suggest_float('hmm_threshold', 0.50, 0.95)

    sharpe = simulate_sharpe(test_df, breakout_window, hmm_threshold)
    return sharpe

def write_best_params(study, current_params):
    best = study.best_trial.params
    current_params['breakout_window'] = int(best['breakout_window'])
    current_params['hmm_threshold']   = float(best['hmm_threshold'])

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

    # Split data to get the test set
    _, test_raw = split_data(df, verbose=True)
    
    # Feature engineering for test set
    test_df = calculate_features_test(test_raw)
    
    # Pre-generate forecasts
    test_df['forecasted_vol'] = get_lstm_predictions(test_df, best_params)
    test_df.dropna(subset=['forecasted_vol'], inplace=True)

    print(f"\n[Sandbox] Feature engineering & pre-computation complete.")
    print(f"[Sandbox] Beginning {N_TRIALS} fast Optuna trials...\n")

    study = optuna.create_study(
        study_name="eth_hmm_breakout_opt",
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, test_df), n_trials=N_TRIALS)

    try:
        write_best_params(study, best_params)
    except ValueError:
        print("[Sandbox] No completed trials found. best_params.txt unchanged.")

    print("\n[Sandbox] Fast Optimization Done.")

if __name__ == "__main__":
    main()
