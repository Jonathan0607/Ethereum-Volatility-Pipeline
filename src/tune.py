import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# --- 1. Define the Model ---
# This class defines the LSTM architecture we want to optimize.
# We pass parameters like 'hidden_dim' and 'dropout' dynamically from Optuna.
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take the last time step
        return out

# --- 2. Data Helper (Robust for Mac & CSV Variations) ---
def get_data():
    # Get absolute path to the data file relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"ERROR: Could not find file at: {data_path}")
        print("Please run 'python3 src/fetch_data.py' to generate it.")
        raise FileNotFoundError(f"File not found: {data_path}")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Handle Column Capitalization (Close vs close)
    # This ensures code works even if CSV headers are different
    if 'Close' in df.columns:
        df.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Volume': 'volume'}, inplace=True)
    
    # Feature Selection
    features = ['close', 'high', 'low', 'volume'] 
    
    # Verify columns exist
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found columns: {df.columns}")

    # Scale Data
    data = df[features].values
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create Sequences (Lookback Window)
    X, y = [], []
    window_size = 60 # Look back 60 hours
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i+window_size])
        y.append(data_scaled[i+window_size, 0]) # Predict next 'Close' price
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)
    
    # Split into Train/Validation (80% Train, 20% Val)
    train_size = int(len(X) * 0.8)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]

# --- 3. The Optuna Objective Function ---
def objective(trial):
    # Hyperparameters to tune
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 2) # Keep layers low for speed
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    
    # Load Data
    X_train, y_train, X_val, y_val = get_data()
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True) # Larger batch for speed
    
    # Setup Device (Use Mac GPU 'mps' if available)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    model = LSTMModel(input_dim=4, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop (Short epochs for tuning)
    epochs = 3 
    model.train()
    
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
        
        # Pruning (Stop bad trials early)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        model.train()
        
    return val_loss

# --- 4. Run the Optimization ---
if __name__ == "__main__":
    # Create a "Study" to find the best params
    study = optuna.create_study(direction="minimize")
    print("Starting Hyperparameter Tuning...")
    
    # Run 10 trials
    study.optimize(objective, n_trials=10)
    
    print("\nBest Results Found:")
    trial = study.best_trial
    print(f"  Validation Loss: {trial.value}")
    print("  Best Parameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Optional: Save best params to a file (to use later)
    with open("best_params.txt", "w") as f:
        f.write(str(trial.params))