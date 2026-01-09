import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from model import LSTMModel, prepare_data
from features import calculate_features

def train_model(epochs=100, lr=0.01):
    print("Loading data...")
    try:
        df = pd.read_csv('data/eth_hourly.csv')
    except FileNotFoundError:
        print("Error: data/eth_hourly.csv not found. Run fetch_data.py first.")
        return

    # Feature Engineering
    df = calculate_features(df)
    
    # Prepare Data for LSTM
    print("Preparing sequences...")
    X, y = prepare_data(df)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(X).float() # Add feature dimension
    y_tensor = torch.from_numpy(y).float()
    
    # Initialize Model
    model = LSTMModel(input_dim=1, hidden_dim=50, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    print(f"Starting training for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            
    # Save the model
    torch.save(model.state_dict(), 'lstm_model.pth')
    print("Model saved as lstm_model.pth")

if __name__ == "__main__":
    train_model()