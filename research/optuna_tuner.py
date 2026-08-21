#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX                              ║
║  Cointegrated Statistical Arbitrage (ETH/BTC):                       ║
║  - 72h Rolling OLS Cointegration (Z-Scores & Dynamic Beta)           ║
║  - 20D Lead-Lag Signatures (4h Residual Drift Prediction)            ║
║  - Discrete Mean-Reversion State Machine (Entry, TP, SL)             ║
║  - MS-GARCH-X Spread Volatility Scaling                              ║
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
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.stats import norm, skew, kurtosis
import optuna

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data import split_data, load_pairs_data
from cointegration import compute_rolling_cointegration
from rough_paths import lead_lag_transform, compute_signatures, SignatureDriftPredictor
from ms_garch_engine import MSGARCHX, compute_garman_klass_volatility, compute_volume_shock
from kelly_execution import execute_discrete_state_machine, compute_volatility_multiplier

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')
STUDY_DB    = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'optuna_study.db')}"
N_TRIALS    = 300
FEE_PCT     = 0.0020

def simulate_returns(X_scaled, y_4h_delta_eps, z_scores, betas, sigmas_spread, r_eth_1h, r_btc_1h, alpha, entry_threshold, stop_loss_threshold, target_vol):
    if len(X_scaled) < 100:
        return None

    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X_scaled, y_4h_delta_eps)
    raw_mu = ridge.predict(X_scaled)
    mu_residuals = pd.Series(raw_mu).ewm(span=3, adjust=False).mean().values

    pos_eth, pos_btc, directions, _ = execute_discrete_state_machine(
        z_scores=z_scores,
        mu_residuals=mu_residuals,
        betas=betas,
        sigmas_spread=sigmas_spread,
        entry_threshold=entry_threshold,
        stop_loss_threshold=stop_loss_threshold,
        target_vol=target_vol
    )

    pos_eth_exec = np.roll(pos_eth, 1)
    pos_btc_exec = np.roll(pos_btc, 1)
    pos_eth_exec[0] = 0.0
    pos_btc_exec[0] = 0.0

    gross_returns = pos_eth_exec * r_eth_1h + pos_btc_exec * r_btc_1h
    turnover = np.abs(np.diff(pos_eth_exec, prepend=0.0)) + np.abs(np.diff(pos_btc_exec, prepend=0.0))
    trans_costs = turnover * FEE_PCT
    net_returns = gross_returns - trans_costs
    return net_returns, directions

def objective(trial, X_scaled, y_4h_delta_eps, z_scores, betas, sigmas_spread, r_eth_1h, r_btc_1h):
    entry_threshold = trial.suggest_float('entry_threshold', 1.25, 2.25)
    stop_loss_threshold = trial.suggest_float('stop_loss_threshold', 3.00, 4.50)
    target_vol = trial.suggest_float('target_vol', 0.10, 0.30)
    alpha = trial.suggest_float('alpha', 1e-2, 1e3, log=True)

    sim = simulate_returns(
        X_scaled, y_4h_delta_eps, z_scores, betas, sigmas_spread, r_eth_1h, r_btc_1h,
        alpha, entry_threshold, stop_loss_threshold, target_vol
    )
    if sim is None:
        return -999.0

    net_returns, directions = sim
    if np.std(net_returns) < 1e-10:
        return -999.0

    total_trades = np.sum(np.abs(np.diff(directions)) > 0.5)
    if total_trades == 0:
        return -999.0

    mean_ret = np.mean(net_returns) * 24 * 365
    std_ret = np.std(net_returns) * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)
    return float(sharpe)

def main():
    print("=" * 70)
    print("  OPTUNA BAYESIAN TUNER: COINTEGRATED STAT-ARB (ETH/BTC)")
    print("=" * 70)

    df = load_pairs_data()
    train_df, _ = split_data(df, verbose=False)

    print("\n[1/3] Computing 72h Rolling OLS Cointegration & MS-GARCH-X...")
    coint_train = compute_rolling_cointegration(train_df, window=72)
    train_df = train_df.join(coint_train, how='left')

    msgarch = MSGARCHX()
    msgarch.fit(train_df)
    filtered = msgarch.filter_full_series(train_df)
    volatilities = filtered['ms_garch_volatility'].values

    print("\n[2/3] Extracting 20D Lead-Lag Signatures (4h residual drift)...")
    predictor = SignatureDriftPredictor(lookback_window=24, depth=2, ema_span=3, forward_horizon=4)
    signatures, y_4h_delta_eps, _ = predictor.extract_signatures_from_df(train_df)

    c_eth = train_df['eth_close'].values.astype(np.float64) if 'eth_close' in train_df.columns else train_df['close'].values.astype(np.float64)
    c_btc = train_df['btc_close'].values.astype(np.float64)
    r_eth_1h_all = (c_eth[1:] - c_eth[:-1]) / (c_eth[:-1] + 1e-8)
    r_btc_1h_all = (c_btc[1:] - c_btc[:-1]) / (c_btc[:-1] + 1e-8)

    r_eth_aligned = r_eth_1h_all[23: 23 + len(signatures)]
    r_btc_aligned = r_btc_1h_all[23: 23 + len(signatures)]
    vol_aligned = volatilities[23: 23 + len(signatures)]
    z_aligned = train_df['coint_zscore'].values[24: 24 + len(signatures)]
    beta_aligned = train_df['coint_beta'].values[24: 24 + len(signatures)]

    min_len = min(len(signatures), len(r_eth_aligned), len(r_btc_aligned), len(vol_aligned), len(z_aligned), len(beta_aligned))
    signatures = signatures[:min_len]
    y_4h_delta_eps = y_4h_delta_eps[:min_len]
    r_eth_aligned = r_eth_aligned[:min_len]
    r_btc_aligned = r_btc_aligned[:min_len]
    vol_aligned = vol_aligned[:min_len]
    z_aligned = z_aligned[:min_len]
    beta_aligned = beta_aligned[:min_len]

    valid_mask = ~np.isnan(y_4h_delta_eps) & ~np.isnan(vol_aligned) & ~np.isnan(z_aligned) & ~np.isnan(beta_aligned)
    X_valid = signatures[valid_mask]
    y_valid = y_4h_delta_eps[valid_mask]
    z_valid = z_aligned[valid_mask]
    beta_valid = beta_aligned[valid_mask]
    vol_valid = vol_aligned[valid_mask]
    r_eth_valid = r_eth_aligned[valid_mask]
    r_btc_valid = r_btc_aligned[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    print(f"\n[3/3] Running Optuna Optimization (300 Trials)...")
    study = optuna.create_study(
        study_name="cointegrated_statarb_optimization",
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=False
    )
    
    study.optimize(
        lambda trial: objective(trial, X_scaled, y_valid, z_valid, beta_valid, vol_valid, r_eth_valid, r_btc_valid),
        n_trials=N_TRIALS
    )

    best_trial = study.best_trial
    print("\n" + "=" * 70)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best In-Sample Sharpe: {best_trial.value:.4f}")
    print("Best Parameters:")
    for k, v in best_trial.params.items():
        print(f"  - {k}: {v}")
        
    best_params = best_trial.params
    best_params['lookback_window'] = 24
    best_params['forward_horizon'] = 4
    best_params['coint_window'] = 72
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(best_params, f, indent=2)

if __name__ == "__main__":
    main()
