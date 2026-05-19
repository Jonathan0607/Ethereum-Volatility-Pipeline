import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import pandas as pd
import numpy as np
import pickle
import os
import ast
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from data import calculate_features, split_data, FEATURE_COLS

# --- CONFIG ---
SEQ_LENGTH = 60
TARGET_COLUMN = 'fwd_vol_24h'
BATCH_SIZE = 1024

# =====================================================================
# 1. MODEL ARCHITECTURE — Progressive Deep Learning
#    Data Flow: Input → RNN → GRU → LSTM → Linear → Output
# =====================================================================

class ProgressiveModel(nn.Module):
    """
    Progressive Deep Learning architecture for volatility forecasting.
    Enforces strict architectural progression: RNN → GRU → LSTM → Linear.

    Each recurrent layer processes the full sequence and passes its output
    sequence to the next layer, progressively refining temporal representations.
    """
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, dropout=0.2):
        super(ProgressiveModel, self).__init__()
        self.hidden_dim = hidden_dim

        # Stage 1: Vanilla RNN — captures basic temporal patterns
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh',
            dropout=0.0  # single layer, no internal dropout
        )

        # Stage 2: GRU — learns gating over RNN representations
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Stage 3: LSTM — full memory cell for long-range dependencies
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0
        )

        # Inter-layer dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]

        # Stage 1: RNN
        rnn_out, _ = self.rnn(x)           # [batch, seq_len, hidden_dim]
        rnn_out = self.dropout(rnn_out)

        # Stage 2: GRU
        gru_out, _ = self.gru(rnn_out)     # [batch, seq_len, hidden_dim]
        gru_out = self.dropout(gru_out)

        # Stage 3: LSTM
        lstm_out, _ = self.lstm(gru_out)   # [batch, seq_len, hidden_dim]

        # Take the final time step and project to output
        out = self.fc(lstm_out[:, -1, :])  # [batch, output_dim]
        return out


# Backward-compatible alias for imports in api.py / backtest.py
LSTMModel = ProgressiveModel


# =====================================================================
# 2. REGIME DETECTION
# =====================================================================

def fit_and_save_regimes(train_df: pd.DataFrame, vol_col: str = 'volatility', n_components: int = 2) -> pd.DataFrame:
    """Fits GMM on TRAINING DATA ONLY and saves it."""
    train_df = train_df.copy()
    if vol_col not in train_df.columns:
        raise ValueError(f"Column '{vol_col}' not found.")

    X = train_df[vol_col].values.reshape(-1, 1)

    low_seed  = np.percentile(X, 10)
    high_seed = np.percentile(X, 90)
    means_init = np.array([[low_seed], [high_seed]])
    
    gmm = GaussianMixture(n_components=n_components, means_init=means_init, random_state=42)
    gmm.fit(X)
    gmm = _enforce_label_order_gmm(gmm)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, '..', 'gmm_model.pkl'), 'wb') as f:
        pickle.dump(gmm, f)
        
    print("[Regimes] Fitted GMM saved successfully.")

    train_df['regime_gmm'] = gmm.predict(X)
    train_df['regime'] = train_df['regime_gmm'] 

    return train_df

def predict_regimes(df: pd.DataFrame, vol_col: str = 'volatility') -> pd.DataFrame:
    """Loads saved models and predicts regimes on new data."""
    df = df.copy()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gmm_path = os.path.join(current_dir, '..', 'gmm_model.pkl')

    with open(gmm_path, 'rb') as f: gmm = pickle.load(f)

    X = df[vol_col].values.reshape(-1, 1)
    df['regime_gmm'] = gmm.predict(X)
    df['regime'] = df['regime_gmm']

    return df

def _enforce_label_order_gmm(gmm: GaussianMixture) -> GaussianMixture:
    means = gmm.means_.flatten()
    if means[0] <= means[1]: return gmm
    gmm.means_ = gmm.means_[[1, 0]]
    gmm.covariances_ = gmm.covariances_[[1, 0]]
    gmm.weights_ = gmm.weights_[[1, 0]]
    gmm.precisions_ = gmm.precisions_[[1, 0]]
    gmm.precisions_cholesky_ = gmm.precisions_cholesky_[[1, 0]]
    return gmm


# =====================================================================
# 3. TRAINING & HYPERPARAMETER TUNING
# =====================================================================

def create_sequences(features: np.ndarray, targets: np.ndarray, seq_length: int):
    """
    Creates input/output sequences for multi-feature time series.
    
    Args:
        features: np.ndarray of shape [N, num_features] — the input feature matrix
        targets:  np.ndarray of shape [N] — the forward target (fwd_vol_24h)
        seq_length: int — lookback window
        
    Returns:
        X: np.ndarray of shape [num_sequences, seq_length, num_features]
        y: np.ndarray of shape [num_sequences]
    """
    xs, ys = [], []
    for i in range(len(features) - seq_length):
        xs.append(features[i:i + seq_length])
        ys.append(targets[i + seq_length])
    return np.array(xs), np.array(ys)

def load_best_params():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    if not os.path.exists(params_path):
        return {'hidden_dim': 64, 'lr': 0.001, 'dropout': 0.2, 'input_dim': len(FEATURE_COLS)}
    with open(params_path, 'r') as f:
        params = ast.literal_eval(f.read())
    # Ensure input_dim is always present
    params['input_dim'] = len(FEATURE_COLS)
    return params

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    learning_rate = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df = calculate_features(df) 
    
    train_df, _ = split_data(df, verbose=False)
    if len(train_df) > 3000:
        train_df = train_df.tail(3000)
    
    # Multi-feature input
    feature_data = train_df[FEATURE_COLS].values.astype(np.float32)
    target_data = train_df[TARGET_COLUMN].values.astype(np.float32)

    # Fit scalers on this trial's data
    feat_scaler = StandardScaler()
    feature_data = feat_scaler.fit_transform(feature_data)
    tgt_scaler = StandardScaler()
    target_scaled = tgt_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

    X, y = create_sequences(feature_data, target_scaled, SEQ_LENGTH)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_train = torch.from_numpy(X_train).to(device)
    X_val = torch.from_numpy(X_val).to(device)
    y_train = torch.from_numpy(y_train).unsqueeze(-1).to(device)
    y_val = torch.from_numpy(y_val).unsqueeze(-1).to(device)
    
    model = ProgressiveModel(
        input_dim=len(FEATURE_COLS),
        hidden_dim=hidden_dim,
        output_dim=1,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        
    return val_loss.item()

def run_optimization():
    print("Starting Bayesian Hyperparameter Optimization (ProgressiveModel)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    
    print(f"Best Trial Loss: {study.best_value:.5f}")
    print(f"Best Params: {study.best_params}")
    
    best = study.best_params
    best['input_dim'] = len(FEATURE_COLS)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(save_path, "w") as f:
        f.write(str(best))
    
    return best

def train_model(hyperparams=None):
    print("Starting Production Training (STRICT SPLIT — ProgressiveModel)...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

    train_raw, _ = split_data(df, verbose=True)
    train_df = calculate_features(train_raw, train_df=train_raw)

    if len(train_df) < SEQ_LENGTH:
        raise ValueError("Training data is too small for the sequence length!")

    train_df = fit_and_save_regimes(train_df)

    # ── Multi-feature extraction ──
    feature_data = train_df[FEATURE_COLS].values.astype(np.float32)
    target_data  = train_df[TARGET_COLUMN].values.astype(np.float32)

    # ── Fit & save StandardScaler for features ──
    feat_scaler = StandardScaler()
    feature_scaled = feat_scaler.fit_transform(feature_data)

    scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(feat_scaler, f)
    print(f"Feature scaler saved → {scaler_path}")

    # ── Fit & save StandardScaler for target ──
    tgt_scaler = StandardScaler()
    target_scaled = tgt_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()

    tgt_scaler_path = os.path.join(current_dir, '..', 'target_scaler.pkl')
    with open(tgt_scaler_path, 'wb') as f:
        pickle.dump(tgt_scaler, f)
    print(f"Target scaler saved → {tgt_scaler_path}")

    # ── Create sequences ──
    X, y = create_sequences(feature_scaled, target_scaled, SEQ_LENGTH)

    X_tensor = torch.from_numpy(X)            # [N, seq_len, 5]
    y_tensor = torch.from_numpy(y).unsqueeze(-1)  # [N, 1]
    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = hyperparams if hyperparams else load_best_params()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Input features: {FEATURE_COLS} (dim={len(FEATURE_COLS)})")
    print(f"Target: {TARGET_COLUMN}")

    model = ProgressiveModel(
        input_dim=len(FEATURE_COLS),
        hidden_dim=params['hidden_dim'],
        output_dim=1,
        dropout=params['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    model.train()
    for epoch in range(200):
        batch_losses = []
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss   = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        avg_loss = np.mean(batch_losses)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/200 | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    print("\n[Train] Artifacts saved:")
    print("  lstm_model.pth       — ProgressiveModel weights (RNN→GRU→LSTM)")
    print("  scaler.pkl           — Feature StandardScaler (5 features)")
    print("  target_scaler.pkl    — Target StandardScaler (fwd_vol_24h)")
    print("  garch_params.npy     — GARCH(1,1) coefficients")
    print("  gmm_model.pkl        — Fitted GMM regime detector")
