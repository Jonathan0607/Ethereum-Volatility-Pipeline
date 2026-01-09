import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import sys
import ast  # Safely evaluate strings to dicts

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import LSTMModel
from features import calculate_features
import tune  # We import this to access create_sequences

# --- CONFIG ---
SEQ_LENGTH = 60
# SPEED FIX: Increased from 64 to 512 for large datasets (2 Years Data)
BATCH_SIZE = 512  
EPOCHS = 100

def load_best_params():
    """
    Loads hyperparameters from best_params.txt so we don't have to re-tune every time.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = ast.literal_eval(f.read())
        print(f"\n[INFO] Loaded Optimized Hyperparameters: {params}")
        return params
    else:
        print("\n[INFO] No params found. Running Optimization first...")
        return None

def run_pipeline():
    print("==========================================")
    print("   ETHEREUM VOLATILITY PIPELINE ENGINE    ")
    print("==========================================\n")
    
    # 1. Load Data
    print("[1/3] Loading 2-Year Dataset...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        print("Error: Data not found. Run 'python src/fetch_data.py'")
        return

    df = pd.read_csv(data_path)
    df = calculate_features(df)
    
    # 2. Hyperparameters (Skip Tuning if done)
    best_params = load_best_params()
    
    if best_params is None:
        print("[2/3] Starting Hyperparameter Optimization...")
        best_params = tune.run_optimization()
    else:
        print("[2/3] Skipping Tuning (Using saved params).")

    # 3. Production Training
    print("\n[3/3] Starting Production Training (Batch Size 512)...")
    
    # Prepare Data
    data = df['volatility'].values.astype(np.float32)
    mean = np.mean(data)
    std = np.std(data)
    data_scaled = (data - mean) / (std + 1e-8)
    
    X, y = tune.create_sequences(data_scaled, SEQ_LENGTH)
    
    # Convert to Tensor
    X_tensor = torch.from_numpy(X).unsqueeze(-1)
    y_tensor = torch.from_numpy(y).unsqueeze(-1)
    
    # Create Loader (Fast Batching)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model
    model = LSTMModel(
        input_dim=1, 
        hidden_dim=best_params['hidden_dim'], 
        num_layers=best_params['num_layers'], 
        output_dim=1, 
        dropout=best_params['dropout']
    )
    
    # Setup Training
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    # Training Loop
    best_loss = float('inf')
    patience = 10
    trigger_times = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        avg_loss = np.mean(train_losses)
        
        # Simple Early Stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(current_dir, '..', 'lstm_model.pth'))
        else:
            trigger_times += 1
            
        if epoch % 10 == 0:
             print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.5f}")
             
        if trigger_times >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\n==========================================")
    print("   PIPELINE COMPLETE. MODEL SAVED.        ")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()