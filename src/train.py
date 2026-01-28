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
from features import calculate_features
from data_split import split_data 

# --- CONFIG ---
SEQ_LENGTH = 60
Target_Column = 'volatility'
BATCH_SIZE = 1024

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

def train_model(hyperparams=None):
    print("Starting Production Training (STRICT SPLIT)...")
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    
    df = calculate_features(df)
    
    train_df, _ = split_data(df)
    
    if len(train_df) < SEQ_LENGTH:
        raise ValueError("Training data is too small for the sequence length!")

    data = train_df[Target_Column].values.astype(np.float32)
    
    # 4. Scale (Fit scaler ONLY on training data)
    mean = np.mean(data)
    std = np.std(data)
    
    # SAVE SCALER STATS (Critical for valid Backtesting)
    # We need to use these exact numbers to scale the test data later.
    np.save(os.path.join(current_dir, '..', 'scaler_mean.npy'), mean)
    np.save(os.path.join(current_dir, '..', 'scaler_std.npy'), std)
    print(f"Saved training mean ({mean:.4f}) and std ({std:.4f})")
    
    data_scaled = (data - mean) / (std + 1e-8)
    
    # 5. Prepare Sequences
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 6. Load Params
    if hyperparams:
        params = hyperparams
    else:
        params = load_best_params()
    
    # 7. Initialize Model
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
    
    # 8. Training Loop
    model.train()
    for epoch in range(100):
        batch_losses = []
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        avg_loss = np.mean(batch_losses)
        scheduler.step(avg_loss)
        
        if (epoch+1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
            
    # 9. Save Model
    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()