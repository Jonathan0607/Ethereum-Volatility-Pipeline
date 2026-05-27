#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX                              ║
║  Maximizes simulated Sharpe Ratio over historical ETH data.          ║
║  Auto-writes optimal params → best_params.txt on completion.         ║
║                                                                      ║
║  Usage:  python research/optuna_tuner.py                             ║
║  Resume: Trials persist in sqlite:///research/optuna_study.db        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import gc
import warnings
import numpy as np
import pandas as pd
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from arch import arch_model

import optuna
from optuna.exceptions import TrialPruned

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Resolve project paths — this script runs from the project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)

from data import FEATURE_COLS

PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')
STUDY_DB    = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'optuna_study.db')}"
N_TRIALS    = 60

# ---------------------------------------------------------------------------
# Data loading — uses our standard ingestion pipeline via yfinance
# ---------------------------------------------------------------------------
def load_eth_data() -> pd.DataFrame:
    """Fetch the maximum hourly ETH-USD window via Yahoo Finance."""
    import yfinance as yf

    print("[Sandbox] Fetching ETH-USD hourly data (max window)...")
    df = yf.download("ETH-USD", period="2y", interval="1h", progress=False, auto_adjust=True)
    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.rename(columns={
        "Datetime": "timestamp", "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    print(f"[Sandbox] Loaded {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Feature engineering — mirrors src/data.py exactly
# ---------------------------------------------------------------------------
def get_hurst_exponent(ts, max_lag=20):
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GARCH volatility + stationary features + forward target."""
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    garch = arch_model(df['log_ret'] * 100, vol='Garch', p=1, q=1,
                       mean='Constant', dist='Normal')
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

    # Forward target
    df['fwd_vol_24h'] = df['log_return'].rolling(window=24).std().shift(-24)
    df['hurst'] = df['close'].rolling(window=48).apply(lambda x: get_hurst_exponent(x), raw=True)
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Model architecture — mirrors src/strategy.py ProgressiveModel
# ---------------------------------------------------------------------------
class ProgressiveModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=1,
                 output_dim=1, dropout=0.2):
        super().__init__()
        self.rnn  = nn.RNN(input_dim, hidden_dim, num_layers=1,
                           batch_first=True, nonlinearity='tanh')
        self.gru  = nn.GRU(hidden_dim, hidden_dim, num_layers=1,
                           batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x);   out = self.dropout(out)
        out, _ = self.gru(out); out = self.dropout(out)
        out, _ = self.lstm(out)
        return self.fc(out[:, -1, :])


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------
def create_sequences(features: np.ndarray, targets: np.ndarray, seq_len: int):
    xs, ys = [], []
    for i in range(len(features) - seq_len):
        xs.append(features[i:i + seq_len])
        ys.append(targets[i + seq_len])
    return np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Simulated Sharpe — trains model, runs backtest, returns Sharpe
# ---------------------------------------------------------------------------
def simulate_sharpe(df: pd.DataFrame, params: dict) -> float:
    """
    End-to-end simulation:
      1. Split data 75/25
      2. Train ProgressiveModel on train set
      3. Run vol-scaled strategy on test set
      4. Return annualized Sharpe ratio
    """
    n_lags       = params['n_lags']
    hidden_dim   = params['lstm_hidden_size']
    dropout      = params['dropout_rate']
    lr           = params['learning_rate']
    batch_size   = params['batch_size']
    epochs       = params['epochs']
    z_window        = params['z_window']
    z_buy           = params['z_buy']
    z_sell          = params['z_sell']
    hurst_threshold = params['hurst_threshold']

    # ── Split ──
    split_idx = int(len(df) * 0.75)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    if len(train_df) < n_lags + 100 or len(test_df) < n_lags + 100:
        return -999.0



    # ── Scale features ──
    feat_scaler = StandardScaler()
    train_feats = feat_scaler.fit_transform(train_df[FEATURE_COLS].values.astype(np.float32))
    test_feats  = feat_scaler.transform(test_df[FEATURE_COLS].values.astype(np.float32))

    tgt_scaler = StandardScaler()
    train_targets = tgt_scaler.fit_transform(
        train_df['fwd_vol_24h'].values.astype(np.float32).reshape(-1, 1)
    ).flatten()

    # ── Create sequences ──
    X_train, y_train = create_sequences(train_feats, train_targets, n_lags)
    if len(X_train) == 0:
        return -999.0

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).unsqueeze(-1).to(device)

    # ── Train ──
    model = ProgressiveModel(
        input_dim=len(FEATURE_COLS), hidden_dim=hidden_dim,
        output_dim=1, dropout=dropout
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    # ── Inference on test set (Vectorized for Speed) ──
    model.eval()
    
    # Pre-build the entire test sequence matrix
    X_test, _ = create_sequences(test_feats, np.zeros(len(test_feats)), n_lags)
    
    if len(X_test) == 0:
        return -999.0
        
    X_test_t = torch.from_numpy(X_test).to(device)
    
    with torch.no_grad():
        # Blast the entire matrix through the model at once
        preds = model(X_test_t).cpu().numpy().flatten()

    preds_arr = tgt_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    preds_arr = np.maximum(preds_arr, 1e-8)

    # ── Force MPS memory cleanup ──
    del model, optimizer, X_t, y_t, X_test_t
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Backtest with vol-scaled sizing ──
    bt = test_df.iloc[n_lags:].copy()
    bt = bt.iloc[:len(preds_arr)].copy()
    bt['forecasted_vol'] = preds_arr
    bt['market_returns']  = bt['close'].pct_change()

    # Calculate rolling Z-Score
    roll_mean = bt['close'].rolling(window=z_window).mean()
    roll_std  = bt['close'].rolling(window=z_window).std()
    bt['z_score'] = (bt['close'] - roll_mean) / (roll_std + 1e-8)

    # Risk Filters
    vol_threshold = bt['volatility'].rolling(24).mean() * 1.8
    is_safe_vol = bt['forecasted_vol'] < vol_threshold
    is_mean_reverting = bt['hurst'] < hurst_threshold

    # Execution State Machine
    buy_cond  = is_mean_reverting & is_safe_vol & (bt['z_score'] <= z_buy)
    sell_cond = (bt['z_score'] >= z_sell) | ~is_safe_vol

    # Forward fill signals: 1 when holding, 0 when flat
    bt['signal'] = np.where(buy_cond, 1, np.where(sell_cond, 0, np.nan))
    bt['signal'] = bt['signal'].ffill().fillna(0)

    daily_vol = bt['forecasted_vol'] * np.sqrt(24)
    bt['position_size'] = (0.06 / (daily_vol + 1e-8)).clip(upper=1.0)
    bt.loc[bt['signal'] == 0, 'position_size'] = 0.0

    bt['strategy_returns'] = (
        bt['market_returns'] * bt['signal'].shift(1) * bt['position_size'].shift(1)
    )
    bt.dropna(inplace=True)

    if len(bt) < 100 or bt['strategy_returns'].std() < 1e-10:
        return -999.0

    # ── Annualized Sharpe ──
    mean_ret = bt['strategy_returns'].mean() * 24 * 365
    std_ret  = bt['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe   = mean_ret / (std_ret + 1e-9)
    return float(sharpe)


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def objective(trial, df):
    params = {
        'n_lags':           trial.suggest_int('n_lags', 30, 90),
        'lstm_hidden_size': trial.suggest_categorical('lstm_hidden_size', [64, 128, 256]),
        'dropout_rate':     trial.suggest_float('dropout_rate', 0.2, 0.5),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size':       trial.suggest_categorical('batch_size', [32, 64]),
        'epochs':           trial.suggest_int('epochs', 30, 100),
        'z_window':         trial.suggest_int('z_window', 10, 50),
        'z_buy':            trial.suggest_float('z_buy', -3.5, -1.5),
        'z_sell':           trial.suggest_float('z_sell', 0.0, 1.5),
        'hurst_threshold':  trial.suggest_float('hurst_threshold', 0.40, 0.55),
    }

    sharpe = simulate_sharpe(df, params)
    print(f"  Trial {trial.number:>3d} | Sharpe: {sharpe:+.3f} | "
          f"h={params['lstm_hidden_size']} lr={params['learning_rate']:.5f} "
          f"lag={params['n_lags']} ep={params['epochs']}")
    return sharpe


# ---------------------------------------------------------------------------
# Config auto-update — the Air-Gap bridge to production
# ---------------------------------------------------------------------------
def write_best_params(study):
    """
    Converts best trial params into the format consumed by
    strategy.load_best_params() and overwrites best_params.txt.

    Production reads:  hidden_dim, lr, dropout, input_dim
    Sandbox also persists the full search space for auditability.
    """
    best = study.best_trial.params

    config = {
        # Core model params (consumed by strategy.py / api.py / backtest.py)
        'hidden_dim':  best['lstm_hidden_size'],
        'lr':          best['learning_rate'],
        'dropout':     best['dropout_rate'],
        'input_dim':   len(FEATURE_COLS),
        # Extended search space (metadata for research)
        'n_lags':           best['n_lags'],
        'batch_size':       best['batch_size'],
        'epochs':           best['epochs'],
        'z_window':         best['z_window'],
        'z_buy':            best['z_buy'],
        'z_sell':           best['z_sell'],
        'hurst_threshold':  best['hurst_threshold'],
    }

    with open(PARAMS_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n[Air-Gap] best_params.txt updated → {PARAMS_PATH}")
    print(f"[Air-Gap] Best Sharpe: {study.best_value:+.4f}")
    print(f"[Air-Gap] Config:\n{json.dumps(config, indent=2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX")
    print("  Metric: Maximize Simulated Sharpe Ratio")
    print(f"  Trials: {N_TRIALS}  |  Study DB: {STUDY_DB}")
    print("=" * 60)

    df = load_eth_data()
    df = compute_features(df)
    print(f"[Sandbox] Features computed. {len(df)} usable rows.\n")

    study = optuna.create_study(
        study_name="eth_sharpe_optimization",
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=True,
    )

    try:
        study.optimize(lambda trial: objective(trial, df), n_trials=N_TRIALS)
    except KeyboardInterrupt:
        print("\n[Sandbox] Interrupted — writing best params found so far...")

    try:
        best_trial = study.best_trial
        write_best_params(study)
    except ValueError:
        print("[Sandbox] No completed trials yet. best_params.txt unchanged.")
        
    print("\n[Sandbox] Done.")


if __name__ == "__main__":
    main()
