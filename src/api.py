import requests
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import torch
import pickle
import ast
import yfinance as yf
import json
from arch import arch_model
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import calculate_features_test, FEATURE_COLS
from strategy import ProgressiveModel
import monte_carlo

app = FastAPI(title="Ethereum Quant Execution Engine")

# Volatility-Scaled Sizing
TARGET_VOLATILITY = 0.06
SEQ_LENGTH = 60

def get_db_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'trades.db')

def initialize_database():
    db_path = get_db_path()
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
        realized_pnl_pct REAL DEFAULT 0.0,
        position_size REAL DEFAULT 1.0
    )
    ''')
    try:
        cursor.execute("ALTER TABLE paper_trades ADD COLUMN position_size REAL DEFAULT 1.0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()
    print(f"[Database] SQLite initialized at {db_path}")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MODEL_STATE = {'model': None, 'feat_scaler': None, 'tgt_scaler': None, 'device': None}

@app.on_event("startup")
def load_artifacts():
    print("[API] Booting Execution Engine (Volatility-Scaled Sizing)...")
    initialize_database()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    MODEL_STATE['device'] = device

    # 1. Load Feature Scaler
    scaler_path = os.path.join(current_dir, '..', 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            MODEL_STATE['feat_scaler'] = pickle.load(f)
    else:
        mean_path = os.path.join(current_dir, '..', 'scaler_mean.npy')
        std_path = os.path.join(current_dir, '..', 'scaler_std.npy')
        if os.path.exists(mean_path) and os.path.exists(std_path):
            print("   [Scaler] WARNING: Legacy scaler. Retrain to generate scaler.pkl.")
            MODEL_STATE['feat_scaler'] = {'legacy': True, 'mean': np.load(mean_path), 'std': np.load(std_path)}

    # 2. Load Target Scaler
    tgt_path = os.path.join(current_dir, '..', 'target_scaler.pkl')
    if os.path.exists(tgt_path):
        with open(tgt_path, 'rb') as f:
            MODEL_STATE['tgt_scaler'] = pickle.load(f)

    # 3. Load ProgressiveModel
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    with open(params_path, 'r') as f:
        params = ast.literal_eval(f.read())

    input_dim = params.get('input_dim', len(FEATURE_COLS))
    model = ProgressiveModel(
        input_dim=input_dim, hidden_dim=params.get('hidden_dim', 64),
        output_dim=1, dropout=params.get('dropout', 0.2)
    )
    model.load_state_dict(torch.load(os.path.join(current_dir, '..', 'lstm_model.pth'), map_location=device))
    model.to(device)
    model.eval()
    MODEL_STATE['model'] = model

    print(f"[API] ProgressiveModel loaded (input_dim={input_dim})")
    print(f"[API] TARGET_VOLATILITY = {TARGET_VOLATILITY}")

    print("[API] Shadow Mode Daemon active.")

def log_trade(action, price, pred_vol, regime, lower_bound=0.0, upper_bound=0.0, position_size=0.0):
    """Writes execution state to DB and fires Discord alert."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT execution_price FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
    open_trade = cursor.fetchone()

    actual_action = action
    status = 'CLOSED'

    if action == "BUY":
        if open_trade is None:
            status = 'OPEN'
        else:
            actual_action = 'HOLDING'
            status = 'OPEN'
    elif action == "CASH":
        if open_trade is not None:
            entry_price = open_trade[0]
            realized_pnl = ((price - entry_price) / entry_price) * 100
            cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl_pct = ? WHERE status = 'OPEN'", (realized_pnl,))
            status = 'CLOSED'
        else:
            actual_action = 'FLAT'
            status = 'CLOSED'
    elif action == "HOLDING":
        status = 'OPEN'
    elif action == "FLAT":
        status = 'CLOSED'

    cursor.execute('''
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status, position_size)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', ('ETH/USDT', actual_action, price, pred_vol, regime, status, position_size))
    conn.commit()
    conn.close()

    size_pct = position_size * 100
    print(f"   [EXECUTED] {actual_action} | ${price:.2f} | Vol: {pred_vol:.4f} | Hurst: {regime:.4f} | Size: {size_pct:.1f}%")

    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if WEBHOOK_URL is None:
        print("[WARN] DISCORD_WEBHOOK_URL not set in environment variables")
        return
    embed_color = 5763719 if actual_action == "BUY" else (15548997 if actual_action == "CASH" else 9807270)
    payload = {
        "username": "Execution Engine",
        "embeds": [{
            "title": f"\U0001f6a8 TRADE EXECUTED: {actual_action}",
            "color": embed_color,
            "fields": [
                {"name": "Asset", "value": "ETH/USD", "inline": True},
                {"name": "Execution Price", "value": f"${price:.2f}", "inline": True},
                {"name": "Hurst Exponent", "value": f"{regime:.4f}", "inline": True},
                {"name": "AI Volatility", "value": f"{pred_vol:.5f}", "inline": True},
                {"name": "Position Size", "value": f"{size_pct:.1f}%", "inline": True},
                {"name": "Target Vol", "value": f"{TARGET_VOLATILITY*100:.1f}%", "inline": True},
                {"name": "24h Risk Bottom", "value": f"${lower_bound:.2f}", "inline": True},
                {"name": "24h Risk Top", "value": f"${upper_bound:.2f}", "inline": True}
            ],
            "footer": {"text": "Chronos • Volatility-Scaled Sizing"}
        }]
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[Alert Error] Discord webhook failed: {e}")

class LiveExecutionPayload(BaseModel):
    current_price: float
    forecasted_vol: float
    z_score: float
    hurst: float

@app.post("/execution/live-stream")
def execute_live_stream_trade(payload: LiveExecutionPayload):
    """Real-Time Gateway: Triggered instantly by the WebSocket stream aggregator"""
    print(f"\n[{datetime.now()}] Live WebSocket signal received...")
    try:
        close_price = payload.current_price
        forecasted_vol = max(payload.forecasted_vol, 1e-8)
        
        # 1. Fetch historical data for Monte Carlo predicted GARCH volatility
        raw_df = yf.download("ETH-USD", period="5d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [col[0] for col in raw_df.columns]
        raw_df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        
        returns = (np.log(raw_df['close'] / raw_df['close'].shift(1)).dropna()) * 100
        garch = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
        garch_fit = garch.fit(update_freq=0, disp='off')
        live_garch_vol = float(garch_fit.conditional_volatility.iloc[-1] / 100)

        # Load best_params config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(current_dir, '..', 'best_params.txt')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                best_params = ast.literal_eval(f.read())
        else:
            best_params = {}
            
        z_buy = best_params.get('z_buy', -2.0)
        z_sell = best_params.get('z_sell', 0.5)
        hurst_threshold = best_params.get('hurst_threshold', 0.45)

        # Get current position from database
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_price FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
        open_trade = cursor.fetchone()
        conn.close()
        current_position = "LONG" if open_trade is not None else "FLAT"

        # Gate 1 & 2: Risk Filters
        vol_threshold = TARGET_VOLATILITY * 1.8 
        is_safe_vol = payload.forecasted_vol < vol_threshold
        is_mean_reverting = payload.hurst < hurst_threshold

        # Gate 3: Execution Logic
        if is_mean_reverting and is_safe_vol and payload.z_score <= z_buy:
            action = "BUY"
        elif payload.z_score >= z_sell or not is_safe_vol:
            action = "CASH"
        else:
            action = "HOLDING" if current_position == "LONG" else "FLAT"

        daily_forecasted_vol = forecasted_vol * np.sqrt(24)
        if action in ["BUY", "HOLDING"]:
            position_size = min(TARGET_VOLATILITY / (daily_forecasted_vol + 1e-8), 1.0)
        else:
            position_size = 0.0

        # 3. Fire custom C++ Monte Carlo
        mc_results = monte_carlo.simulate_gbm(
            current_price=close_price, predicted_vol=live_garch_vol,
            num_sims=10000, steps=24
        )
        
        # 4. Write to DB and push Discord notification (logging hurst in regime column)
        log_trade(action, close_price, float(forecasted_vol), float(payload.hurst),
                  mc_results['lower_bound'], mc_results['upper_bound'], position_size)
                  
        return {"status": "Success", "action": action, "position_size": position_size}
    except Exception as e:
        print(f"[API Webhook Error] Execution failed: {e}")
        return {"status": "Error", "message": str(e)}

# --- DASHBOARD ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "Online", "engine": "ProgressiveModel", "sizing": "Volatility-Scaled"}

@app.get("/latest-state")
def get_latest_state():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM paper_trades ORDER BY timestamp DESC LIMIT 10", conn)
        conn.close()
        return df.to_dict(orient='records')
    except Exception as e:
        return {"error": str(e)}

@app.get("/backtest-data")
def get_backtest_data():
    from starlette.responses import Response
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, '..', 'backtest_results.json')
    try:
        with open(json_path, 'r') as f:
            content = f.read()
        return Response(content=content, media_type="application/json")
    except FileNotFoundError:
        return {"error": "Backtest results not found. Run backtest.py first."}

@app.get("/portfolio-stats")
def get_portfolio_stats():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(realized_pnl_pct) FROM paper_trades WHERE action = 'BUY' AND status = 'CLOSED'")
        total_realized = cursor.fetchone()[0] or 0.0
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action = 'BUY' AND status = 'CLOSED'")
        total_closed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action = 'BUY' AND status = 'CLOSED' AND realized_pnl_pct > 0")
        winning_trades = cursor.fetchone()[0]
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
        unrealized_pnl = 0.0
        cursor.execute("SELECT execution_price FROM paper_trades WHERE status = 'OPEN' AND action = 'BUY' ORDER BY timestamp ASC LIMIT 1")
        open_trade = cursor.fetchone()
        if open_trade:
            entry_price = open_trade[0]
            cursor.execute("SELECT execution_price FROM paper_trades ORDER BY timestamp DESC LIMIT 1")
            latest_row = cursor.fetchone()
            if latest_row:
                unrealized_pnl = ((latest_row[0] - entry_price) / entry_price) * 100
        conn.close()
        return {
            "total_realized_pnl_pct": round(total_realized, 2),
            "win_rate": round(win_rate, 2),
            "total_closed_trades": total_closed,
            "current_unrealized_pnl_pct": round(unrealized_pnl, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/monte-carlo-visual")
def get_monte_carlo_visual():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_price FROM paper_trades ORDER BY timestamp DESC LIMIT 1")
        latest_trade = cursor.fetchone()
        conn.close()
        current_price = latest_trade[0] if latest_trade else 2300.0
    except Exception:
        current_price = 2300.0

    volatility, steps, num_paths = 0.05, 24, 50
    dt = 1.0 / steps
    shocks = np.random.normal(0.0, 1.0, (num_paths, steps))
    paths = np.zeros((num_paths, steps + 1))
    paths[:, 0] = current_price
    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t-1] * np.exp(-0.5 * (volatility**2) * dt + volatility * np.sqrt(dt) * shocks[:, t-1])
    median_path = np.median(paths, axis=0)
    result = []
    for t in range(steps + 1):
        dp = {"hour": t}
        for p in range(num_paths):
            dp[f"path_{p}"] = round(paths[p, t], 2)
        dp["median"] = round(median_path[t], 2)
        result.append(dp)
    return result