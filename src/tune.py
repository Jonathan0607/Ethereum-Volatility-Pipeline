import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import sys

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# IMPORT from your actual files
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
    # 1. Hyperparameter Search Space
    hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
    num_layers = trial.suggest_int("num_layers", 1, 2)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    
    # 2. Load and Process Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError("Data file not found. Run fetch_data.py first.")
        
    df = pd.read_csv(data_path)
    
    # Use the features logic
    df = calculate_features(df) 
    
    # --- SPEED PATCH: Limit to last 3000 rows for Tuning ---
    # This prevents the "hanging" issue by reducing data volume during search
    if len(df) > 3000:
        df = df.tail(3000)
    # -------------------------------------------------------
    
    # Use Volatility column as data
    data = df[Target_Column].values.astype(np.float32)
    
    # Normalize (Manual Z-Score)
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    # Create Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # Train/Val Split
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # To Tensor (and add feature dimension)
    X_train = torch.from_numpy(X_train).unsqueeze(-1)
    X_val = torch.from_numpy(X_val).unsqueeze(-1)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)
    y_val = torch.from_numpy(y_val).unsqueeze(-1)
    
    # 3. Setup Model
    # Fixed the call to include output_dim=1
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 4. Fast Training Loop (5 Epochs)
    n_epochs = 5
    model.train()
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # 5. Validate
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        
    return val_loss.item()

def run_optimization():
    print("Starting Bayesian Hyperparameter Optimization...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    
    print("\n--- Optimization Complete ---")
    print(f"Best Trial Loss: {study.best_value:.5f}")
    
    # Save Best Params
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(save_path, "w") as f:
        f.write(str(study.best_params))
    
    return study.best_params

if __name__ == "__main__":
    run_optimization()