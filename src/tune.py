import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features

# --- CONFIG ---
SEQ_LENGTH = 60
Target_Column = 'volatility'
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def objective(trial):
    # --- UPDATED SEARCH SPACE FOR GARCH ---
    # GARCH is cleaner, so we don't need massive models. 
    # We reduce max hidden_dim to 64 to prevent overfitting.
    hidden_dim = trial.suggest_int("hidden_dim", 32, 64)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    
    # GARCH is smoother, so we can try slightly more aggressive learning rates.
    learning_rate = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    
    # Lower Dropout: GARCH has less "random noise" than rolling std.
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    # 2. Load Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file not found.")
        
    df = pd.read_csv(data_path)
    df = calculate_features(df) 
    
    # Speed Patch for Tuning
    if len(df) > 3000:
        df = df.tail(3000)
    
    data = df[Target_Column].values.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # Train/Val Split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    X_train = torch.from_numpy(X_train).unsqueeze(-1)
    X_val = torch.from_numpy(X_val).unsqueeze(-1)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)
    y_val = torch.from_numpy(y_val).unsqueeze(-1)
    
    # 3. Setup Model
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Increased epochs slightly for GARCH to settle
    n_epochs = 8
    model.train()
    
    for epoch in range(n_epochs):
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
    print("Starting Bayesian Hyperparameter Optimization (GARCH Tuned)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15) # Increased trials slightly
    
    print(f"Best Trial Loss: {study.best_value:.5f}")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(save_path, "w") as f:
        f.write(str(study.best_params))
    
    return study.best_params

if __name__ == "__main__":
    run_optimization()