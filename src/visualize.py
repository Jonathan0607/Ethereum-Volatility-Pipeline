import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import torch
import ast

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from features import calculate_features
from regimes import detect_regimes
from model import LSTMModel

# --- CONFIG ---
SEQ_LENGTH = 60

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        print("Error: Data file not found.")
        sys.exit(1)
        
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def get_lstm_predictions(df):
    print("   [AI] Generating LSTM Forecasts for Dashboard...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Params
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hidden_dim, num_layers, dropout = 64, 2, 0.2
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = ast.literal_eval(f.read())
            hidden_dim = params.get('hidden_dim', 64)
            num_layers = params.get('num_layers', 2)
            dropout = params.get('dropout', 0.2)
            
    # 2. Load Model
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    model_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        print("   [WARNING] Model not found. Returning flat line.")
        return pd.Series(0, index=df.index)

    # 3. LOAD SAVED SCALER (CRITICAL FIX)
    # We use the training set's mean/std. If missing, we fallback (but warn).
    mean_path = os.path.join(current_dir, '..', 'scaler_mean.npy')
    std_path = os.path.join(current_dir, '..', 'scaler_std.npy')
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        print("   [WARNING] Scaler stats not found! Dashboard may look skewed.")
        data = df['volatility'].values
        mean, std = np.mean(data), np.std(data)

    # 4. Prepare Data
    data = df['volatility'].values.astype(np.float32)
    data_scaled = (data - mean) / (std + 1e-8)
    
    predictions = [np.nan] * SEQ_LENGTH 
    inputs = []
    
    for i in range(len(data_scaled) - SEQ_LENGTH):
        inputs.append(data_scaled[i:i+SEQ_LENGTH])
        
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).unsqueeze(-1).to(device)
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            output = model(batch)
            preds = output.cpu().numpy().flatten() * (std + 1e-8) + mean
            predictions.extend(preds)
            
    return pd.Series(predictions, index=df.index)

def plot_dashboard(df):
    print("   [Visual] Rendering Final Portfolio Dashboard...")
    
    # Filter to last 2000 hours for clarity (approx 3 months)
    subset = df.tail(2000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # --- PLOT 1: Regimes & Price ---
    ax1.plot(subset.index, subset['close'], color='gray', alpha=0.3, label='Price')
    
    # Plot Regimes
    safe = subset[subset['regime'] == 0]
    risk = subset[subset['regime'] == 1]
    
    ax1.scatter(safe.index, safe['close'], color='green', s=10, alpha=0.6, label='Safe Regime')
    ax1.scatter(risk.index, risk['close'], color='red', s=10, alpha=0.6, label='Risk Regime')
    
    ax1.set_title('Market Regimes & AI Strategy', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.15)
    
    # --- PLOT 2: LSTM Brain vs Reality ---
    ax2.plot(subset.index, subset['volatility'], color='blue', alpha=0.5, label='Actual Volatility')
    ax2.plot(subset.index, subset['lstm_pred'], color='magenta', linestyle='--', linewidth=1.5, label='AI Prediction')
    
    # VISUALIZE THE THRESHOLD
    veto_line = subset['volatility'].rolling(24).mean() * 1.8
    ax2.plot(subset.index, veto_line, color='black', linestyle=':', alpha=0.6, label='Risk Threshold (1.8σ)')
    
    # Fill area where AI was triggered
    ax2.fill_between(subset.index, 0, subset['volatility'].max(), 
                     where=(subset['lstm_pred'] > veto_line), 
                     color='red', alpha=0.1, label='AI Signal: CASH')

    ax2.set_title('AI Risk Detection (LSTM)', fontsize=12)
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.15)
    
    # Format Date Axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_dashboard.png')
    plt.savefig(output_path, dpi=300)
    print(f"   [Success] Dashboard saved to {output_path}")

def run_visualizer():
    df = load_data()
    df = calculate_features(df)
    df = detect_regimes(df)
    df['lstm_pred'] = get_lstm_predictions(df)
    plot_dashboard(df)

if __name__ == "__main__":
    run_visualizer()