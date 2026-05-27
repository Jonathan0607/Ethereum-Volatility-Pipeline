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

def log_trade(action, price, pred_vol, regime, lower_bound=0.0, upper_bound=0.0, position_size=0.0):
    """Writes execution state to DB and fires Discord alert."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT execution_price FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
    open_trade = cursor.fetchone()

    actual_action = action
    status = 'CLOSED'
    realized_pnl = 0.0

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
    elif action == "PARTIAL_SELL":
        if open_trade is not None:
            entry_price = open_trade[0]
            realized_pnl = ((price - entry_price) / entry_price) * 100
            cursor.execute("UPDATE paper_trades SET position_size = position_size * 0.5 WHERE status = 'OPEN'")
            status = 'CLOSED'
        else:
            actual_action = 'FLAT'
            status = 'CLOSED'
    elif action == "HOLDING":
        status = 'OPEN'
    elif action == "FLAT":
        status = 'CLOSED'

    cursor.execute('''
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status, position_size, realized_pnl_pct)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ETH/USDT', actual_action, price, pred_vol, regime, status, position_size, realized_pnl))
    conn.commit()
    conn.close()

    size_pct = position_size * 100
    print(f"   [EXECUTED] {actual_action} | ${price:.2f} | Vol: {pred_vol:.4f} | Hurst: {regime:.4f} | Size: {size_pct:.1f}%")

    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if WEBHOOK_URL is None:
        print("[WARN] DISCORD_WEBHOOK_URL not set in environment variables")
        return

    # Update Discord notification colors and titles
    if actual_action == "BUY":
        title = "🚨 TRADE EXECUTED: BUY"
        embed_color = 5763719  # Green
    elif actual_action == "PARTIAL_SELL":
        title = "⚖️ PARTIAL PROFIT TAKE: PARTIAL_SELL"
        embed_color = 16753920  # Orange/Gold
    elif actual_action == "CASH":
        title = "🚨 FULL LIQUIDATION: CASH"
        embed_color = 15548997  # Red
    else:
        title = f"⚙️ SYSTEM STATE: {actual_action}"
        embed_color = 9807270  # Gray

    fields = [
        {"name": "Asset", "value": "ETH/USD", "inline": True},
        {"name": "Execution Price", "value": f"${price:.2f}", "inline": True},
        {"name": "Hurst Exponent", "value": f"{regime:.4f}", "inline": True},
        {"name": "AI Volatility", "value": f"{pred_vol:.5f}", "inline": True},
        {"name": "Position Size", "value": f"{size_pct:.1f}%", "inline": True},
        {"name": "Target Vol", "value": f"{TARGET_VOLATILITY*100:.1f}%", "inline": True},
    ]
    if actual_action in ["CASH", "PARTIAL_SELL"] and open_trade is not None:
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
            
        z_buy = best_params.get('z_buy', -2.0)
        z_sell = best_params.get('z_sell', 0.5)
        hurst_threshold = best_params.get('hurst_threshold', 0.45)
        sl_mult = best_params.get('sl_mult', 1.5)
        cooldown_hours = best_params.get('cooldown_hours', 3)
        vol_mult = best_params.get('vol_mult', 1.8)
        use_partial_sell = best_params.get('use_partial_sell', True)
        partial_sell_ratio = best_params.get('partial_sell_ratio', 0.5)
        use_ema_exit = best_params.get('use_ema_exit', False)

        # Get current position and size from database
        db_path = get_db_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT execution_price, position_size, timestamp FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
        open_trade = cursor.fetchone()
        
        current_position = "LONG" if open_trade is not None else "FLAT"
        current_size = open_trade[1] if open_trade is not None else 0.0
        
        has_partial_sold = False
        if open_trade is not None:
            previous_buy_price, _, entry_timestamp = open_trade
            cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE timestamp >= ? AND action = 'PARTIAL_SELL'", (entry_timestamp,))
            has_partial_sold = cursor.fetchone()[0] > 0
            
        conn.close()

        # Gate 1 & 2: Risk Filters
        vol_threshold = rolling_vol_mean * vol_mult
        is_safe_vol = payload.forecasted_vol < vol_threshold
        is_mean_reverting = payload.hurst < hurst_threshold
        is_macro_uptrend = close_price > ema_200

        # Gate 3: Execution Logic
        is_stop_loss_hit = False
        if current_position == "LONG" and open_trade is not None:
            stop_loss_pct = forecasted_vol * sl_mult
            if close_price < previous_buy_price * (1.0 - stop_loss_pct):
                is_stop_loss_hit = True
                action = "CASH"
                print(f"   [STOP LOSS BREACHED] Price dropped below threshold (${close_price:.2f} < ${previous_buy_price * (1.0 - stop_loss_pct):.2f}). Forcing full liquidation.")
            elif use_ema_exit and close_price < ema_200:
                is_stop_loss_hit = True
                action = "CASH"
                print(f"   [EMA COOLDOWN VETO] Price dropped below EMA-200 (${close_price:.2f} < ${ema_200:.2f}). Forcing full liquidation.")

        if not is_stop_loss_hit:
            # Scale z_buy dynamically based on forecasted volatility
            dynamic_z_buy = z_buy - (forecasted_vol * 10)
            
            if is_mean_reverting and is_safe_vol and is_macro_uptrend and payload.z_score <= dynamic_z_buy:
                action = "BUY"
            elif payload.z_score >= z_sell or not is_safe_vol:
                action = "CASH" # 100% Exit
            elif use_partial_sell and payload.z_score >= 0.0 and current_position == "LONG" and not has_partial_sold:
                action = "PARTIAL_SELL"
            else:
                action = "HOLDING" if current_position == "LONG" else "FLAT"

        daily_forecasted_vol = forecasted_vol * np.sqrt(24)
        if action in ["BUY", "HOLDING"]:
            position_size = min(TARGET_VOLATILITY / (daily_forecasted_vol + 1e-8), 1.0)
        elif action == "PARTIAL_SELL":
            position_size = current_size * partial_sell_ratio
        else:
            position_size = 0.0

        # Check Cooldown condition: if we sell/exit and close_price < previous_buy_price
        if action == "CASH" and open_trade is not None:
            previous_buy_price = open_trade[0]
            if close_price < previous_buy_price:
                current_state.cooldown_until = datetime.now() + timedelta(hours=cooldown_hours)
                print(f"   [COOLDOWN TRIGGERED] Exited trade at a loss (${close_price:.2f} < ${previous_buy_price:.2f}). Cooldown active for {cooldown_hours} hours until {current_state.cooldown_until}")

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