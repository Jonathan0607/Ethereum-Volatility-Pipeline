import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import torch
import ast

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import calculate_features
from regimes import detect_regimes
from model import LSTMModel
from data_split import split_data  # <--- Single Source of Truth

# --- CONFIG ---
SEQ_LENGTH = 60

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def get_lstm_predictions(df):
    """
    Generates predictions for the TEST set using the saved model and saved scaler.
    """
    print("Generating LSTM Volatility Forecasts (Strict Test Set)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Params
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    hidden_dim, num_layers, dropout = 64, 2, 0.2 # Fallbacks
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = ast.literal_eval(f.read())
            hidden_dim = params.get('hidden_dim', 64)
            num_layers = params.get('num_layers', 2)
            dropout = params.get('dropout', 0.2)
            
    # 2. Load Model
    model = LSTMModel(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, output_dim=1, dropout=dropout)
    model_path = os.path.join(current_dir, '..', 'lstm_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found! Run train.py first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 3. LOAD SAVED SCALER (CRITICAL)
    # We must use the training set's mean/std to scale the test set.
    mean_path = os.path.join(current_dir, '..', 'scaler_mean.npy')
    std_path = os.path.join(current_dir, '..', 'scaler_std.npy')
    
    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError("Scaler stats not found! Run train.py first to generate them.")
        
    mean = np.load(mean_path)
    std = np.load(std_path)
    
    # 4. Prepare Data
    data = df['volatility'].values.astype(np.float32)
    # Apply the TRAINING mean/std to the TEST data
    data_scaled = (data - mean) / (std + 1e-8)
    
    predictions = [np.nan] * SEQ_LENGTH 
    inputs = []
    
    # Note: This simple loop misses the first SEQ_LENGTH points of the test set 
    # because it doesn't look back into the train set for context. 
    # For Phase 0, this is acceptable strict separation.
    for i in range(len(data_scaled) - SEQ_LENGTH):
        inputs.append(data_scaled[i:i+SEQ_LENGTH])
    
    if len(inputs) == 0:
        return pd.Series(0, index=df.index)

    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).unsqueeze(-1).to(device)
    
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            output = model(batch)
            # Unscale the prediction
            preds = output.cpu().numpy().flatten() * (std + 1e-8) + mean
            predictions.extend(preds)
            
    return pd.Series(predictions, index=df.index)

def run_backtest(df):
    print("Running Backtest with Strict Chronological Split...")

    # 1. Feature Engineering (Calculate on full DF to ensure smooth MAs at the cut boundary)
    df = calculate_features(df)
    
    # 2. STRICT SPLIT: We throw away the training data now.
    _, test_df = split_data(df)
    
    if test_df.empty:
        raise ValueError("Test Set is empty! Check your TEST_START_DATE in data_split.py")
        
    print(f"--- BACKTEST STARTING ON {test_df.index[0]} ---")
    
    # 3. Detect Regimes 
    # ideally we fit GMM on train and predict on test, but refitting on test is OK for Phase 0
    test_df = detect_regimes(test_df) 
    
    # 4. Get Predictions (Using Test Data only)
    test_df['lstm_pred_vol'] = get_lstm_predictions(test_df)
    
    test_df['market_returns'] = test_df['close'].pct_change()
    
    # --- TECHNICALS ---
    test_df['fast_trend'] = test_df['close'].ewm(span=20, adjust=False).mean()
    test_df['slow_trend'] = test_df['close'].ewm(span=50, adjust=False).mean()
    test_df['vol_ma'] = test_df['volume'].rolling(window=12).mean()
    
    # --- LOGIC ---
    base_signal = (
        (test_df['regime'] == 0) & 
        (test_df['close'] > test_df['fast_trend']) &
        (test_df['volume'] > test_df['vol_ma'])
    )
    
    vol_threshold = test_df['volatility'].rolling(24).mean() * 1.8
    lstm_risk_on = (test_df['lstm_pred_vol'] > vol_threshold)
    parabolic_move = (test_df['close'] > (test_df['slow_trend'] * 1.02))
    
    long_condition = base_signal & (~lstm_risk_on | parabolic_move)
    
    test_df['signal'] = np.where(long_condition, 1, 0)
    
    test_df['strategy_returns'] = test_df['market_returns'] * test_df['signal'].shift(1)
    test_df['cumulative_market'] = (1 + test_df['market_returns']).cumprod()
    test_df['cumulative_strategy'] = (1 + test_df['strategy_returns']).cumprod()
    
    return test_df

def calculate_metrics(df):
    risk_free_rate = 0.0
    strategy_mean = df['strategy_returns'].mean() * 24 * 365
    strategy_std = df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe = (strategy_mean - risk_free_rate) / (strategy_std + 1e-9)
    
    cumulative = df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    print("\n=== AI-POWERED BACKTEST RESULTS (OUT OF SAMPLE) ===")
    print(f"Market Return:   {(df['cumulative_market'].iloc[-1] - 1)*100:.2f}%")
    print(f"Strategy Return: {(df['cumulative_strategy'].iloc[-1] - 1)*100:.2f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_drawdown*100:.2f}%")

def plot_results(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative_market'], label='Buy & Hold (ETH)', color='gray', alpha=0.5)
    plt.plot(df.index, df['cumulative_strategy'], label='Hybrid AI (Test Set)', color='blue', linewidth=1.5)
    plt.title('Strategy Performance: Out-of-Sample Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    try:
        data = load_data()
        results = run_backtest(data)
        calculate_metrics(results)
        plot_results(results)
    except Exception as e:
        print(f"Error: {e}")