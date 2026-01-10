import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset # <--- Added this back
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
import os
import sys
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features

# --- CONFIG ---
SEQ_LENGTH = 60
Target_Column = 'volatility'
BATCH_SIZE = 1024  # <--- Process data in chunks of 512 rows

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
    
    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)
    
    # --- NEW: Create DataLoader (The Fix) ---
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Load Params
    if hyperparams:
        params = hyperparams
    else:
        params = load_best_params()
        
    print(f"[INFO] Training with params: {params}")
    
    # 6. Initialize Model
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
    
    # 7. Training Loop (Now using Mini-Batches)
    model.train()
    
    for epoch in range(100):
        batch_losses = []
        
        # Iterate through the loader (Chunks of 512)
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        # Calculate average loss for the epoch
        avg_loss = np.mean(batch_losses)
        
        # Step the Scheduler
        scheduler.step(avg_loss)
        
        if (epoch+1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/100 | Loss: {avg_loss:.5f} | LR: {current_lr:.6f}")
            
    # 8. Save Model
    save_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_model()