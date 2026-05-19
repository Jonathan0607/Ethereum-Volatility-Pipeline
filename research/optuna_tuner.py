#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  OPTUNA BAYESIAN HYPERPARAMETER SANDBOX                            ║
║  Maximizes simulated Sharpe Ratio over historical ETH data.        ║
║  Auto-writes optimal params → best_params.txt on completion.       ║
║                                                                    ║
║  Usage:  python research/optuna_tuner.py                           ║
║  Resume: Trials persist in sqlite:///research/optuna_study.db      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
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
    gmm_n        = params['gmm_components']
    fast_span    = params['fast_ema_span']
    slow_span    = params['slow_ema_span']

    # ── Split ──
    split_idx = int(len(df) * 0.75)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()

    if len(train_df) < n_lags + 100 or len(test_df) < n_lags + 100:
        return -999.0

    # ── GMM regime detection (train only) ──
    X_vol = train_df['volatility'].values.reshape(-1, 1)
    low_seed  = np.percentile(X_vol, 10)
    high_seed = np.percentile(X_vol, 90)
    means_init = np.array([[low_seed]] + [[high_seed]] +
                          [[np.percentile(X_vol, 50)]] * (gmm_n - 2)) if gmm_n > 2 \
                 else np.array([[low_seed], [high_seed]])
    gmm = GaussianMixture(n_components=gmm_n, means_init=means_init[:gmm_n],
                          random_state=42)
    gmm.fit(X_vol)

    # Predict regimes
    train_df['regime_gmm'] = gmm.predict(X_vol)
    test_df['regime_gmm']  = gmm.predict(test_df['volatility'].values.reshape(-1, 1))

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

    # ── Inference on test set ──
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(test_feats) - n_lags):
            seq = torch.from_numpy(test_feats[i:i + n_lags]).unsqueeze(0).to(device)
            p = model(seq).cpu().numpy().flatten()[0]
            preds.append(p)

    preds_arr = tgt_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    preds_arr = np.maximum(preds_arr, 1e-8)

    # ── Backtest with vol-scaled sizing ──
    bt = test_df.iloc[n_lags:].copy()
    bt = bt.iloc[:len(preds_arr)].copy()
    bt['forecasted_vol'] = preds_arr
    bt['market_returns']  = bt['close'].pct_change()

    bt['fast_trend'] = bt['close'].ewm(span=fast_span, adjust=False).mean()
    bt['slow_trend'] = bt['close'].ewm(span=slow_span, adjust=False).mean()
    bt['vol_ma']     = bt['volume'].rolling(window=12).mean()

    # Find the lowest regime label as "safe"
    regime_means = {}
    for r in bt['regime_gmm'].unique():
        regime_means[r] = bt.loc[bt['regime_gmm'] == r, 'volatility'].mean()
    safe_regime = min(regime_means, key=regime_means.get) if regime_means else 0

    base_signal = (
        (bt['regime_gmm'] == safe_regime) &
        (bt['close'] > bt['fast_trend']) &
        (bt['volume'] > bt['vol_ma'])
    )
    vol_threshold = bt['volatility'].rolling(24).mean() * 1.8
    lstm_risk_on  = bt['forecasted_vol'] > vol_threshold
    parabolic     = bt['close'] > (bt['slow_trend'] * 1.02)

    bt['signal'] = np.where(base_signal & (~lstm_risk_on | parabolic), 1, 0)

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
        'lstm_num_layers':  trial.suggest_int('lstm_num_layers', 2, 4),
        'dropout_rate':     trial.suggest_float('dropout_rate', 0.2, 0.5),
        'learning_rate':    trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size':       trial.suggest_categorical('batch_size', [32, 64]),
        'epochs':           trial.suggest_int('epochs', 30, 100),
        'gmm_components':   trial.suggest_int('gmm_components', 2, 4),
        'fast_ema_span':    trial.suggest_int('fast_ema_span', 10, 30),
        'slow_ema_span':    trial.suggest_int('slow_ema_span', 40, 100),
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
        'lstm_num_layers':  best['lstm_num_layers'],
        'batch_size':       best['batch_size'],
        'epochs':           best['epochs'],
        'gmm_components':   best['gmm_components'],
        'fast_ema_span':    best['fast_ema_span'],
        'slow_ema_span':    best['slow_ema_span'],
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

    if len(study.trials) > 0 and study.best_trial is not None:
        write_best_params(study)
    else:
        print("[Sandbox] No completed trials. best_params.txt unchanged.")

    print("\n[Sandbox] Done.")


if __name__ == "__main__":
    main()
