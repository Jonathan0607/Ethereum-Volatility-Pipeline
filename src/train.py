import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- 1. The Optimized Model Architecture ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_layers=1, output_dim=1, dropout=0.25):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Note: batch_first=True requires input shape (batch, seq, features)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# --- 2. Data Loader ---
def get_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Adjust this path if your data is elsewhere
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
        
    df = pd.read_csv(data_path)
    
    # Capitalization fix
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
        
    features = ['close', 'high', 'low', 'volume']
    data = df[features].values
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    window_size = 60
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i+window_size])
        y.append(data_scaled[i+window_size, 0])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)
    
    # Time-based split (Train on past, validate on future)
    train_split = int(len(X) * 0.8)
    return X[:train_split], y[:train_split], X[train_split:], y[train_split:]

# --- 3. The "Smart" Training Loop ---
def train_model(hyperparams=None):
    """
    Main training function.
    Args:
        hyperparams (dict): Optional dictionary of optimized parameters (lr, hidden_dim, etc.)
                            If None, defaults will be used.
    """
    
    # Default Configuration (Baseline)
    config = {
        'hidden_dim': 128,
        'num_layers': 1,
        'dropout': 0.25,
        'lr': 0.005
    }

    # Override defaults if optimizations are passed
    if hyperparams:
        print(f"\n[INFO] Updating config with optimized parameters: {hyperparams}")
        config.update(hyperparams)

    # Setup Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    
    X_train, y_train, X_val, y_val = get_data()
    
    # Use larger batch size for stability
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=False)
    
    # Initialize Model with DYNAMIC Config
    model = LSTMModel(
        hidden_dim=config['hidden_dim'], 
        num_layers=config['num_layers'], 
        dropout=config['dropout']
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    # Scheduler: Reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    epochs = 100
    patience = 15          
    best_val_loss = float('inf')
    counter = 0            
    
    print(f"Starting Training with LR={config['lr']}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
        
        # Update Scheduler
        scheduler.step(val_loss)
        
        # Print Status (Manual LR check)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss:.5f} | LR: {current_lr:.6f}")
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save absolute path relative to script
            save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lstm_model.pth")
            torch.save(model.state_dict(), save_path)
            print("  >>> Best model saved!")
        else:
            counter += 1
            print(f"  >>> No improvement for {counter} epochs.")
            if counter >= patience:
                print("Early Stopping triggered. Training complete.")
                break

if __name__ == "__main__":
    train_model()