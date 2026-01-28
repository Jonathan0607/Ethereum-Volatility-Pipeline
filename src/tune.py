import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features
from data_split import split_data  # <--- CRITICAL IMPORT

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
    # Hyperparameters
    hidden_dim = trial.suggest_int("hidden_dim", 32, 64)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    learning_rate = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    # 1. Load Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file not found.")
        
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df = calculate_features(df) 
    
    # 2. STRICT SPLIT: We ONLY optimize on Historical Training Data
    # The 'Future' (Test Set) is completely discarded here.
    train_df, _ = split_data(df, verbose=False)
    
    # Speed Patch: Use last 3000 rows of the TRAINING set (not the future!)
    if len(train_df) > 3000:
        train_df = train_df.tail(3000)
    
    data = train_df[Target_Column].values.astype(np.float32)
    
    # Scale (Fit only on this tuning fold)
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # 3. Internal Train/Val Split
    # We split the History into 80% Train / 20% Validation for the trial
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to Tensors
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    X_train = torch.from_numpy(X_train).unsqueeze(-1).to(device)
    X_val = torch.from_numpy(X_val).unsqueeze(-1).to(device)
    y_train = torch.from_numpy(y_train).unsqueeze(-1).to(device)
    y_val = torch.from_numpy(y_val).unsqueeze(-1).to(device)
    
    # 4. Setup Model
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train (Fast 5 epochs for tuning)
    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Validate
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

if __name__ == "__main__":
    run_optimization()