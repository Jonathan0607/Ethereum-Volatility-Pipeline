import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import torch
import ast
import yfinance as yf  # <-- Replaced ccxt with yfinance
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime

# Sibling imports for your models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import calculate_features_test
from strategy import predict_regimes, LSTMModel

app = FastAPI(title="Ethereum Quant Execution Engine")

def initialize_database():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', 'trades.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS paper_trades (
        trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        asset TEXT NOT NULL,
        action TEXT NOT NULL,
        execution_price REAL NOT NULL,
        predicted_volatility REAL NOT NULL,
        regime INTEGER NOT NULL,
        status TEXT NOT NULL,
        realized_pnl_pct REAL DEFAULT 0.0
    )
    ''')
    conn.commit()
    conn.close()
    print(f"[Database] SQLite execution database initialized at {db_path}")

# Add CORS Middleware to allow Next.js port 3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL STATE ---
MODEL_STATE = {
    'lstm': None,
    'mean': None,
    'std': None,
    'device': None
}

@app.on_event("startup")
def load_artifacts():
    print("[API] Booting up Execution Engine...")
    initialize_database()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_STATE['device'] = device
    
    # 1. Load Scaler
    MODEL_STATE['mean'] = np.load(os.path.join(current_dir, '..', 'scaler_mean.npy'))
    MODEL_STATE['std'] = np.load(os.path.join(current_dir, '..', 'scaler_std.npy'))
    
    # 2. Load LSTM
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(params_path, 'r') as f:
        params = ast.literal_eval(f.read())
        
    model = LSTMModel(
        input_dim=1, 
        hidden_dim=params.get('hidden_dim', 64), 
        num_layers=params.get('num_layers', 2), 
        output_dim=1, 
        dropout=params.get('dropout', 0.2)
    )
    model.load_state_dict(torch.load(os.path.join(current_dir, '..', 'lstm_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    MODEL_STATE['lstm'] = model
    
    print("[API] AI Models loaded into memory successfully.")
    
    # 3. Start Scheduler (The Shadow Mode Daemon)
    scheduler = BackgroundScheduler()
    # Runs exactly at the top of every hour (e.g., 10:00:00, 11:00:00)
    scheduler.add_job(execute_trade_cycle, 'cron', minute=0, second=0)
    scheduler.start()
    
    print("[API] Shadow Mode Daemon active. Waiting for next hourly candle...")
    
    # Uncomment the line below if you want it to run immediately upon booting (for testing)
    execute_trade_cycle()

def execute_trade_cycle():
    """
    The Core Loop: Fetch Data -> AI Inference -> Execute Trade -> Log State
    """
    print(f"\n[{datetime.now()}] Waking up for hourly execution cycle...")
    
    try:
        # 1. Fetch live data using Yahoo Finance
        raw_df = yf.download("ETH-USD", period="7d", interval="1h", progress=False, auto_adjust=True)
        
        # Format columns exactly like fetch_data.py
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [col[0] for col in raw_df.columns]
            
        raw_df.rename(columns={
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        }, inplace=True)
        
        # Standardize the index name
        raw_df.index.name = 'timestamp'
        
        # Slice to the last 100 hours
        df = raw_df.tail(100).copy()
        
        # 2. Engineer Features & Detect Regime
        df = calculate_features_test(df)
        df = predict_regimes(df)
        
        # 3. Predict Volatility
        data = df['volatility'].values.astype(np.float32)
        data_scaled = (data - MODEL_STATE['mean']) / (MODEL_STATE['std'] + 1e-8)
        
        seq = data_scaled[-60:]
        seq_tensor = torch.from_numpy(seq).unsqueeze(0).unsqueeze(-1).to(MODEL_STATE['device'])
        
        with torch.no_grad():
            pred_scaled = MODEL_STATE['lstm'](seq_tensor).cpu().numpy().flatten()[0]
            
        pred_vol = pred_scaled * (MODEL_STATE['std'] + 1e-8) + MODEL_STATE['mean']
        
# 4. Evaluate Trading Logic (The Alpha)
        current_row = df.iloc[-1]
        fast_trend = df['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        slow_trend = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        vol_ma = df['volume'].rolling(window=12).mean().iloc[-1]
        
        regime_gmm = current_row['regime_gmm']
        regime_hmm = current_row['regime_hmm']
        close_price = current_row['close']
        
        # GMM ONLY: Winning logic from the Sharpe ablation study (2.47 Sharpe)
        market_safe = (regime_gmm == 0)
        
        base_signal = market_safe and (close_price > fast_trend) and (current_row['volume'] > vol_ma)        
        vol_threshold = df['volatility'].rolling(24).mean().iloc[-1] * 1.8
        lstm_risk_on = pred_vol > vol_threshold
        parabolic_move = close_price > (slow_trend * 1.02)
        
        # The Decision Tree
        action = "CASH"
        if base_signal and (not lstm_risk_on or parabolic_move):
            action = "BUY"
            
        # 5. Log to SQLite Engine (Using GMM as the primary logged regime)
        log_trade(action, close_price, float(pred_vol), int(regime_gmm))        
    except Exception as e:
        print(f"[API] CRITICAL ERROR during execution cycle: {e}")

def log_trade(action, price, pred_vol, regime):
    """Writes the execution state to the local database"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', 'trades.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', ('ETH/USDT', action, price, pred_vol, regime, 'OPEN'))
    
    conn.commit()
    conn.close()
    
    print(f"   [EXECUTED] {action} | Price: ${price:.2f} | AI Vol: {pred_vol:.4f} | Regime: {regime}")

# --- DASHBOARD ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "Execution Engine Online", "engine": "FastAPI + ccxt + PyTorch"}

@app.get("/latest-state")
def get_latest_state():
    """Endpoint for the Next.js Command Center to poll the live SQLite database"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, '..', 'trades.db')
    
    try:
        conn = sqlite3.connect(db_path)
        # Fetch the last 10 execution states
        df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT 10", conn)
        conn.close()
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

@app.get("/backtest-data")
def get_backtest_data():
    """Serves the pre-generated backtest results JSON for the frontend chart."""
    from starlette.responses import Response
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '..', 'backtest_results.json')
    try:
        with open(json_path, 'r') as f:
            content = f.read()
        return Response(content=content, media_type="application/json")
    except FileNotFoundError:
        return {"error": "Backtest results not found. Run backtest.py first."}