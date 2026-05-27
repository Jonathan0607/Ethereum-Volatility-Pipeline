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
from datetime import datetime, timedelta
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
        position_size REAL DEFAULT 1.0,
        gmm_z_score REAL DEFAULT 0.0,
        gmm_cluster INTEGER DEFAULT 1
    )
    ''')
    try:
        cursor.execute("ALTER TABLE paper_trades ADD COLUMN position_size REAL DEFAULT 1.0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE paper_trades ADD COLUMN gmm_z_score REAL DEFAULT 0.0")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE paper_trades ADD COLUMN gmm_cluster INTEGER DEFAULT 1")
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

class TradingState:
    def __init__(self):
        self.cooldown_until = datetime.min

current_state = TradingState()

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
        content = f.read()
        try:
            params = json.loads(content)
        except Exception:
            params = ast.literal_eval(content)

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

def log_trade(action, price, pred_vol, regime, lower_bound=0.0, upper_bound=0.0, position_size=0.0, gmm_z_score=0.0, gmm_cluster=1):
    """Writes execution state to DB and fires Discord alert."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT execution_price, action FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
    open_trade = cursor.fetchone()

    actual_action = action
    status = 'CLOSED'
    realized_pnl = 0.0

    if action == "BUY":
        # Close any existing SHORT position first
        if open_trade is not None and open_trade[1] == 'SELL_SHORT':
            entry_price = open_trade[0]
            realized_pnl = ((entry_price - price) / entry_price) * 100  # Short PnL: entry - exit
            cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl_pct = ? WHERE status = 'OPEN'", (realized_pnl,))
        elif open_trade is not None:
            actual_action = 'HOLDING'
            status = 'OPEN'
        if actual_action == 'BUY':
            status = 'OPEN'
    elif action == "SELL_SHORT":
        # Close any existing LONG position first
        if open_trade is not None and open_trade[1] == 'BUY':
            entry_price = open_trade[0]
            realized_pnl = ((price - entry_price) / entry_price) * 100  # Long PnL: exit - entry
            cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl_pct = ? WHERE status = 'OPEN'", (realized_pnl,))
        elif open_trade is not None:
            actual_action = 'HOLDING'
            status = 'OPEN'
        if actual_action == 'SELL_SHORT':
            status = 'OPEN'
    elif action == "CASH":
        if open_trade is not None:
            entry_price = open_trade[0]
            if open_trade[1] == 'SELL_SHORT':
                realized_pnl = ((entry_price - price) / entry_price) * 100  # Short PnL
            else:
                realized_pnl = ((price - entry_price) / entry_price) * 100  # Long PnL
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
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status, position_size, realized_pnl_pct, gmm_z_score, gmm_cluster)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ETH/USDT', actual_action, price, pred_vol, regime, status, position_size, realized_pnl, gmm_z_score, gmm_cluster))
    conn.commit()
    conn.close()

    size_pct = abs(position_size) * 100
    print(f"   [EXECUTED] {actual_action} | ${price:.2f} | Vol: {pred_vol:.4f} | HMM: {regime:.4f} | GMM Z: {gmm_z_score:.4f} (C:{gmm_cluster}) | Size: {size_pct:.1f}%")

    # Set webhook URL
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if WEBHOOK_URL is None:
        print("[WARN] DISCORD_WEBHOOK_URL not set in environment variables")
        return

    # Update Discord notification colors and titles
    if actual_action == "BUY":
        title = "🟢 LONG ENTRY: BUY"
        embed_color = 5763719  # Green
    elif actual_action == "SELL_SHORT":
        title = "🔴 SHORT ENTRY: SELL_SHORT"
        embed_color = 10038562  # Purple
    elif actual_action == "CASH":
        title = "🚨 FULL LIQUIDATION: CASH"
        embed_color = 15548997  # Red
    else:
        title = f"⚙️ SYSTEM STATE: {actual_action}"
        embed_color = 9807270  # Gray

    fields = [
        {"name": "Asset", "value": "ETH/USD", "inline": True},
        {"name": "Execution Price", "value": f"${price:.2f}", "inline": True},
        {"name": "HMM High-Vol Prob", "value": f"{regime:.4f}", "inline": True},
        {"name": "GMM Z-Score", "value": f"{gmm_z_score:.4f}", "inline": True},
        {"name": "GMM Cluster", "value": str(gmm_cluster), "inline": True},
        {"name": "AI Volatility", "value": f"{pred_vol:.5f}", "inline": True},
        {"name": "Position Size", "value": f"{size_pct:.1f}%", "inline": True},
        {"name": "Target Vol", "value": f"{TARGET_VOLATILITY*100:.1f}%", "inline": True},
    ]
    if actual_action in ["CASH", "BUY", "SELL_SHORT"] and open_trade is not None:
        fields.append({"name": "Realized PnL", "value": f"{realized_pnl:+.4f}%", "inline": True})

    fields.extend([
        {"name": "24h Risk Bottom", "value": f"${lower_bound:.2f}", "inline": True},
        {"name": "24h Risk Top", "value": f"${upper_bound:.2f}", "inline": True}
    ])

    payload = {
        "username": "Execution Engine",
        "embeds": [{
            "title": title,
            "color": embed_color,
            "fields": fields,
            "footer": {"text": "Chronos • Bi-Directional Vol-Scaled"}
        }]
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        print(f"[Alert Error] Discord webhook failed: {e}")

class LiveExecutionPayload(BaseModel):
    current_price: float
    forecasted_vol: float
    prob_high_vol: float
    rolling_max: float
    rolling_min: float
    gmm_z_score: float
    gmm_cluster: int
    closes: list[float]

@app.post("/execution/live-stream")
def execute_live_stream_trade(payload: LiveExecutionPayload):
    """Real-Time Gateway: Triggered instantly by the WebSocket stream aggregator"""
    print(f"\n[{datetime.now()}] Live WebSocket signal received...")
    
    # Check Cooldown (if we just exited a trade with a loss, wait 3 hours)
    if datetime.now() < current_state.cooldown_until:
        print(f"   [COOLDOWN VETO] Strategy is in cooldown state until {current_state.cooldown_until}. Skipping execution.")
        return {"status": "Success", "action": "FLAT", "position_size": 0.0, "reason": "cooldown"}

    try:
        close_price = payload.current_price
        forecasted_vol = max(payload.forecasted_vol, 1e-8)
        
        # 1. Fetch historical data for Monte Carlo predicted GARCH volatility and EMA-200
        raw_df = yf.download("ETH-USD", period="15d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [col[0] for col in raw_df.columns]
        raw_df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}, inplace=True)
        
        returns = (np.log(raw_df['close'] / raw_df['close'].shift(1)).dropna()) * 100
        garch = arch_model(returns, vol='Garch', p=1, q=1, dist='Normal')
        garch_fit = garch.fit(update_freq=0, disp='off')
        live_garch_vol = float(garch_fit.conditional_volatility.iloc[-1] / 100)
        rolling_vol_mean = float(garch_fit.conditional_volatility.tail(24).mean() / 100)
        ema_200 = float(raw_df['close'].ewm(span=200, adjust=False).mean().iloc[-1])

        # Load best_params config
        current_dir = os.path.dirname(os.path.abspath(__file__))
        params_path = os.path.join(current_dir, '..', 'best_params.txt')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                content = f.read()
                try:
                    best_params = json.loads(content)
                except Exception:
                    best_params = ast.literal_eval(content)
        else:
            best_params = {}
            
        hmm_chop_max    = best_params.get('hmm_chop_max', 0.40)
        hmm_trend_min   = best_params.get('hmm_trend_min', 0.60)
        gmm_z_buy       = best_params.get('gmm_z_buy', -1.5)
        gmm_z_sell      = best_params.get('gmm_z_sell', 0.5)
        cooldown_hours  = best_params.get('cooldown_hours', 3)

        # Get current position and size from database
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_price, position_size, timestamp, action FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
        open_trade = cursor.fetchone()
        
        if open_trade is not None:
            current_position = "SHORT" if open_trade[3] == 'SELL_SHORT' else "LONG"
            current_size = open_trade[1]
        else:
            current_position = "FLAT"
            current_size = 0.0
        
        conn.close()

        # Gate 3: Hierarchical Execution Routing Logic
        rolling_mean = sum(payload.closes[-24:]) / 24

        if payload.prob_high_vol > hmm_trend_min:
            # Route to Breakout Agent
            if payload.current_price > payload.rolling_max:
                action = "BUY"
            elif payload.current_price < payload.rolling_min:
                action = "SELL_SHORT"
            else:
                action = "HOLDING" if current_position in ["LONG", "SHORT"] else "FLAT"
        elif payload.prob_high_vol < hmm_chop_max:
            # Route to GMM Agent
            if payload.gmm_z_score < gmm_z_buy or payload.gmm_cluster == 0:
                action = "BUY"
            elif payload.gmm_z_score > gmm_z_sell or payload.gmm_cluster == 2:
                action = "CASH"
            else:
                if current_position == "LONG":
                    action = "HOLDING"
                elif current_position == "SHORT":
                    action = "CASH"  # Cover short in mean reversion mode
                else:
                    action = "FLAT"
        else:
            # Transition Zone - Force Cash
            action = "CASH"

        daily_forecasted_vol = forecasted_vol * np.sqrt(24)
        if action in ["BUY", "SELL_SHORT", "HOLDING"]:
            position_size = min(TARGET_VOLATILITY / (daily_forecasted_vol + 1e-8), 1.0)
        else:
            position_size = 0.0

        # Check Cooldown condition: if we exit and have a realized loss
        if action == "CASH" and open_trade is not None:
            entry_price = open_trade[0]
            if current_position == "LONG" and close_price < entry_price:
                current_state.cooldown_until = datetime.now() + timedelta(hours=cooldown_hours)
                print(f"   [COOLDOWN TRIGGERED] Exited LONG at a loss. Cooldown for {cooldown_hours}h.")
            elif current_position == "SHORT" and close_price > entry_price:
                current_state.cooldown_until = datetime.now() + timedelta(hours=cooldown_hours)
                print(f"   [COOLDOWN TRIGGERED] Exited SHORT at a loss. Cooldown for {cooldown_hours}h.")

        # 3. Fire custom C++ Monte Carlo
        mc_results = monte_carlo.simulate_gbm(
            current_price=close_price, predicted_vol=live_garch_vol,
            num_sims=10000, steps=24
        )
        
        # 4. Write to DB and push Discord notification
        log_trade(action, close_price, float(forecasted_vol), float(payload.prob_high_vol),
                  mc_results['lower_bound'], mc_results['upper_bound'], position_size,
                  payload.gmm_z_score, payload.gmm_cluster)
                  
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