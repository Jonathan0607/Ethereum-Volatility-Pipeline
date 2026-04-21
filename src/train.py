import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features          # fits GARCH on train, saves garch_params.npy
from regimes import fit_and_save_regimes         # fits GMM on train, saves gmm_model.pkl
from data_split import split_data

# --- CONFIG ---
SEQ_LENGTH    = 60
Target_Column = 'volatility'
BATCH_SIZE    = 1024


def load_best_params():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    if not os.path.exists(params_path):
        return {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.001, 'dropout': 0.2}
    with open(params_path, 'r') as f:
        return ast.literal_eval(f.read())


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def train_model(hyperparams=None):
    print("Starting Production Training (STRICT SPLIT)...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path   = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')

    # ------------------------------------------------------------------ #
    # 1. Split BEFORE feature engineering.
    #
    #    Fix: Previously calculate_features() was called on the full df,
    #    causing GARCH to see test-period returns during fitting. Now we
    #    split first and pass train_raw into calculate_features() so GARCH
    #    is fitted exclusively on historical data. The saved garch_params.npy
    #    is then used by backtest.py for the test slice.
    # ------------------------------------------------------------------ #
    train_raw, _ = split_data(df, verbose=True)

    # Fit GARCH on train only; saves garch_params.npy as a side-effect
    train_df = calculate_features(train_raw, train_df=train_raw)

    if len(train_df) < SEQ_LENGTH:
        raise ValueError("Training data is too small for the sequence length!")

    # ------------------------------------------------------------------ #
    # 2. Fit GMM regime detector on training data and save it.
    #
    #    Fix: Previously detect_regimes() was called inside backtest.py on
    #    the test slice, leaking future regime structure into the signal.
    #    Now the GMM is trained here and saved as gmm_model.pkl; backtest.py
    #    calls predict_regimes() which loads it without refitting.
    # ------------------------------------------------------------------ #
    train_df = fit_and_save_regimes(train_df)

    # ------------------------------------------------------------------ #
    # 3. Fit and save scaler on training volatility only.
    # ------------------------------------------------------------------ #
    data = train_df[Target_Column].values.astype(np.float32)

    mean = np.mean(data)
    std  = np.std(data)

    np.save(os.path.join(current_dir, '..', 'scaler_mean.npy'), mean)
    np.save(os.path.join(current_dir, '..', 'scaler_std.npy'),  std)
    print(f"Scaler saved — mean: {mean:.6f} | std: {std:.6f}")

    data_scaled = (data - mean) / (std + 1e-8)

    # 4. Prepare sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)

    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Load hyperparams
    params = hyperparams if hyperparams else load_best_params()

    # 6. Initialize model
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

    # 7. Training loop
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

        avg_loss   = np.mean(batch_losses)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")

    # 8. Save model
    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    print("\n[Train] Artifacts saved:")
    print("  lstm_model.pth   — LSTM weights")
    print("  scaler_mean.npy  — training volatility mean")
    print("  scaler_std.npy   — training volatility std")
    print("  garch_params.npy — GARCH(1,1) coefficients")
    print("  gmm_model.pkl    — fitted regime detector")


if __name__ == "__main__":
    train_model()