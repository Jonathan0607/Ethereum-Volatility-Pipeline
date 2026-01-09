import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
import ast

# Ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features

# --- CONFIG ---
SEQ_LENGTH = 60
Target_Column = 'volatility'

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
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# CHANGED: Added 'hyperparams=None' to accept arguments from pipeline.py
def train_model(hyperparams=None):
    print("Starting Production Training...")
    
    # 1. Load Data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path)
    
    # 2. Features
    df = calculate_features(df)
    data = df[Target_Column].values.astype(np.float32)
    
    # 3. Scale
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    # 4. Prepare Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # Feature dim is 1 (Volatility only)
    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)
    
    # 5. Load Params (Use passed args OR load from file)
    if hyperparams:
        params = hyperparams
    else:
        params = load_best_params()
        
    print(f"[INFO] Training with params: {params}")
    
    # 6. Initialize Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    
    model = LSTMModel(
        input_dim=1,  # Ensuring this is 1 (Fixed)
        hidden_dim=params['hidden_dim'], 
        num_layers=params['num_layers'], 
        output_dim=1, 
        dropout=params['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    # 7. Training Loop
    model.train()
    X_tensor, y_tensor = X_tensor.to(device), y_tensor.to(device)
    
    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/50 | Loss: {loss.item():.5f}")
            
    # 8. Save Model
    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()