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
from hmmlearn.hmm import GaussianHMM
from data import calculate_features, split_data

# --- CONFIG ---
SEQ_LENGTH = 60
Target_Column = 'volatility'
BATCH_SIZE = 1024

# =====================================================================
# 1. MODEL ARCHITECTURE
# =====================================================================

class LSTMModel(nn.Module):
    """
    Standard LSTM architecture for Time Series Forecasting.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# =====================================================================
# 2. REGIME DETECTION
# =====================================================================

def fit_and_save_regimes(train_df: pd.DataFrame, vol_col: str = 'volatility', n_components: int = 2) -> pd.DataFrame:
    """Fits both GMM and HMM on TRAINING DATA ONLY and saves them."""
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

    print("[Regimes] Fitting HMM (This may take a few seconds)...")
    hmm = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(X)
    hmm = _enforce_label_order_hmm(hmm)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, '..', 'gmm_model.pkl'), 'wb') as f:
        pickle.dump(gmm, f)
    with open(os.path.join(current_dir, '..', 'hmm_model.pkl'), 'wb') as f:
        pickle.dump(hmm, f)
        
    print("[Regimes] Fitted GMM & HMM saved successfully.")

    train_df['regime_gmm'] = gmm.predict(X)
    train_df['regime_hmm'] = hmm.predict(X)
    train_df['regime'] = train_df['regime_hmm'] 

    return train_df

def predict_regimes(df: pd.DataFrame, vol_col: str = 'volatility') -> pd.DataFrame:
    """Loads saved models and predicts regimes on new data."""
    df = df.copy()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gmm_path = os.path.join(current_dir, '..', 'gmm_model.pkl')
    hmm_path = os.path.join(current_dir, '..', 'hmm_model.pkl')

    with open(gmm_path, 'rb') as f: gmm = pickle.load(f)
    with open(hmm_path, 'rb') as f: hmm = pickle.load(f)

    X = df[vol_col].values.reshape(-1, 1)
    df['regime_gmm'] = gmm.predict(X)
    df['regime_hmm'] = hmm.predict(X)
    df['regime'] = df['regime_hmm']

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

def _enforce_label_order_hmm(hmm: GaussianHMM) -> GaussianHMM:
    means = hmm.means_.flatten()
    if means[0] <= means[1]: return hmm
    hmm.means_ = hmm.means_[[1, 0]]
    hmm.covars_ = hmm.covars_[[1, 0]]
    hmm.startprob_ = hmm.startprob_[[1, 0]]
    transmat = hmm.transmat_
    transmat = transmat[[1, 0], :]
    transmat = transmat[:, [1, 0]]
    hmm.transmat_ = transmat
    return hmm


# =====================================================================
# 3. TRAINING & HYPERPARAMETER TUNING
# =====================================================================

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

def load_best_params():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    if not os.path.exists(params_path):
        return {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'dropout': 0.2}
    with open(params_path, 'r') as f:
        return ast.literal_eval(f.read())

def objective(trial):
    hidden_dim = trial.suggest_int("hidden_dim", 32, 64)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    learning_rate = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df = calculate_features(df) 
    
    train_df, _ = split_data(df, verbose=False)
    if len(train_df) > 3000:
        train_df = train_df.tail(3000)
    
    data = train_df[Target_Column].values.astype(np.float32)
    mean, std = np.mean(data), np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_train = torch.from_numpy(X_train).unsqueeze(-1).to(device)
    X_val = torch.from_numpy(X_val).unsqueeze(-1).to(device)
    y_train = torch.from_numpy(y_train).unsqueeze(-1).to(device)
    y_val = torch.from_numpy(y_val).unsqueeze(-1).to(device)
    
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout).to(device)
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
    print("Starting Bayesian Hyperparameter Optimization (Strict Safety)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)
    
    print(f"Best Trial Loss: {study.best_value:.5f}")
    print(f"Best Params: {study.best_params}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(save_path, "w") as f:
        f.write(str(study.best_params))
    
    return study.best_params

def train_model(hyperparams=None):
    print("Starting Production Training (STRICT SPLIT)...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

    train_raw, _ = split_data(df, verbose=True)
    train_df = calculate_features(train_raw, train_df=train_raw)

    if len(train_df) < SEQ_LENGTH:
        raise ValueError("Training data is too small for the sequence length!")

    train_df = fit_and_save_regimes(train_df)

    data = train_df[Target_Column].values.astype(np.float32)
    mean, std = np.mean(data), np.std(data)

    np.save(os.path.join(current_dir, '..', 'scaler_mean.npy'), mean)
    np.save(os.path.join(current_dir, '..', 'scaler_std.npy'),  std)
    print(f"Scaler saved — mean: {mean:.6f} | std: {std:.6f}")

    data_scaled = (data - mean) / (std + 1e-8)
    X, y = create_sequences(data_scaled, SEQ_LENGTH)

    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = hyperparams if hyperparams else load_best_params()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    model = LSTMModel(
        input_dim=1,
        hidden_dim=params['hidden_dim'],
        num_layers=params['num_layers'],
        output_dim=1,
        dropout=params['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    model.train()
    for epoch in range(100):
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
            print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    print("\n[Train] Artifacts saved:")
    print("  lstm_model.pth   — LSTM weights")
    print("  scaler_mean.npy  — training volatility mean")
    print("  scaler_std.npy   — training volatility std")
    print("  garch_params.npy — GARCH(1,1) coefficients")
    print("  gmm_model.pkl    — fitted regime detector")
