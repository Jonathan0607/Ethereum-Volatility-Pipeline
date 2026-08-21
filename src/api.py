import requests
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kelly_execution import compute_volatility_multiplier, execute_kelly_rebalance
from alpaca_client import AlpacaExecutionClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Ethereum MS-GARCH Quantitative Execution Engine")
alpaca_client = AlpacaExecutionClient()

# Target annualized volatility for Kelly scaling
TARGET_VOLATILITY = 0.20


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
        regime REAL NOT NULL,
        status TEXT NOT NULL,
        realized_pnl_pct REAL DEFAULT 0.0,
        position_size REAL DEFAULT 1.0
    )
    ''')
    conn.commit()
    conn.close()
    logger.info(f"[Database] SQLite initialized at {db_path}")


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class TradingState:
    def __init__(self):
        self.cooldown_until = datetime.min
        self.current_position = "FLAT"  # "FLAT", "LONG", "SHORT"
        self.position_size = 0.0
        self.entry_price = 0.0


current_state = TradingState()


@app.on_event("startup")
def load_artifacts():
    logger.info("[API] Booting 2-State MS-GARCH & Kelly Execution Engine...")
    initialize_database()
    logger.info(f"[API] TARGET_VOLATILITY = {TARGET_VOLATILITY:.2%}")
    
    # State reconciliation from database
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT action, position_size, execution_price FROM paper_trades ORDER BY timestamp DESC LIMIT 1")
        last_row = cursor.fetchone()
        conn.close()
        
        if last_row:
            action, size, price = last_row
            if action == "BUY":
                current_state.current_position = "LONG"
                current_state.position_size = size
                current_state.entry_price = price
                logger.info(f"[Startup] Reconciled state: LONG (size={size}, entry_price={price})")
            elif action == "SELL_SHORT":
                current_state.current_position = "SHORT"
                current_state.position_size = size
                current_state.entry_price = price
                logger.info(f"[Startup] Reconciled state: SHORT (size={size}, entry_price={price})")
            elif action in ["CASH", "FLAT"]:
                current_state.current_position = "FLAT"
                current_state.position_size = 0.0
                current_state.entry_price = 0.0
                logger.info("[Startup] Reconciled state: FLAT")
            elif action == "HOLDING":
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT action, position_size, execution_price FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
                open_trade = cursor.fetchone()
                conn.close()
                if open_trade:
                    o_action, o_size, o_price = open_trade
                    current_state.current_position = "SHORT" if o_action == "SELL_SHORT" else "LONG"
                    current_state.position_size = o_size
                    current_state.entry_price = o_price
                    logger.info(f"[Startup] Reconciled state: Active {current_state.current_position} (holding, size={o_size}, entry_price={o_price})")
                else:
                    current_state.current_position = "FLAT"
                    current_state.position_size = 0.0
                    current_state.entry_price = 0.0
                    logger.info("[Startup] Reconciled state: FLAT")
        else:
            current_state.current_position = "FLAT"
            current_state.position_size = 0.0
            current_state.entry_price = 0.0
            logger.info("[Startup] No prior trades found. System is FLAT")
    except Exception as e:
        logger.error(f"[Startup Error] State reconciliation failed: {e}")


def log_trade(action: str, price: float, pred_vol: float, regime: float, position_size: float = 0.0):
    """Writes execution state to SQLite DB and dispatches streamlined Discord alert."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT execution_price, action FROM paper_trades WHERE status = 'OPEN' ORDER BY timestamp ASC LIMIT 1")
    open_trade = cursor.fetchone()

    actual_action = action
    status = 'CLOSED'
    realized_pnl = 0.0

    if action == "BUY":
        if open_trade is not None and open_trade[1] == 'SELL_SHORT':
            entry_price = open_trade[0]
            realized_pnl = ((entry_price - price) / entry_price) * 100
            cursor.execute("UPDATE paper_trades SET status = 'CLOSED', realized_pnl_pct = ? WHERE status = 'OPEN'", (realized_pnl,))
        elif open_trade is not None:
            actual_action = 'HOLDING'
            status = 'OPEN'
        if actual_action == 'BUY':
            status = 'OPEN'
    elif action == "SELL_SHORT":
        if open_trade is not None and open_trade[1] == 'BUY':
            entry_price = open_trade[0]
            realized_pnl = ((price - entry_price) / entry_price) * 100
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
                realized_pnl = ((entry_price - price) / entry_price) * 100
            else:
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
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status, position_size, realized_pnl_pct)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ETH/USD', actual_action, price, pred_vol, regime, status, position_size, realized_pnl))
    conn.commit()
    conn.close()

    size_pct = abs(position_size) * 100
    logger.info(f"[EXECUTED] {actual_action} | ${price:.2f} | AI Vol: {pred_vol:.4f} | High-Vol Prob: {regime:.4f} | Size: {size_pct:.1f}%")

    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if WEBHOOK_URL is None:
        return

    if actual_action == "BUY":
        title = "🟢 LONG ENTRY: BUY"
        embed_color = 5763719  # Green
    elif actual_action == "SELL_SHORT":
        title = "🔴 SHORT ENTRY: SELL_SHORT"
        embed_color = 10038562  # Purple
    elif actual_action == "CASH":
        title = "🚨 RISK-OFF LIQUIDATION: CASH"
        embed_color = 15548997  # Red
    else:
        title = f"⚙️ SYSTEM STATE: {actual_action}"
        embed_color = 9807270  # Gray

    fields = [
        {"name": "Execution Price", "value": f"${price:,.2f}", "inline": True},
        {"name": "HMM High-Vol Prob", "value": f"{regime:.4f}", "inline": True},
        {"name": "AI Forecasted Volatility", "value": f"{pred_vol:.5f}", "inline": True},
        {"name": "Target Vol", "value": f"{TARGET_VOLATILITY * 100:.1f}%", "inline": True},
        {"name": "Target Size %", "value": f"{size_pct:.1f}%", "inline": True},
    ]
    if actual_action in ["CASH", "BUY", "SELL_SHORT"] and open_trade is not None:
        fields.append({"name": "Realized PnL", "value": f"{realized_pnl:+.2f}%", "inline": True})

    payload = {
        "username": "Ethereum MS-GARCH Engine",
        "embeds": [{
            "title": title,
            "color": embed_color,
            "fields": fields,
            "footer": {"text": "MS-GARCH(1,1)-X • Kelly Sizing"}
        }]
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"[Discord Error] Webhook delivery failed: {e}")


class LiveExecutionPayload(BaseModel):
    current_price: float
    forecasted_vol: float
    prob_high_vol: float
    rolling_max: float = 0.0
    rolling_min: float = 0.0
    vol_24h: float = 0.0
    vol_168h: float = 0.0
    closes: list[float] = []


@app.post("/execution/live-stream")
def execute_live_stream_trade(payload: LiveExecutionPayload, background_tasks: BackgroundTasks):
    """
    Live Execution Gateway:
    Routes trades purely based on 2-State MS-GARCH regime probability and forecasted volatility.
    Applies volatility-scaled Kelly sizing with 15% deadband and strict Risk-Off 0.0 liquidation override.
    """
    logger.info(f"Live WebSocket tick received: ETH=${payload.current_price:,.2f} | Forecast Vol={payload.forecasted_vol:.4f} | P(High-Vol)={payload.prob_high_vol:.4f}")

    # Check Cooldown
    if datetime.now() < current_state.cooldown_until:
        logger.info(f"[COOLDOWN VETO] Active cooldown until {current_state.cooldown_until}. Holding state.")
        return {
            "status": "Success",
            "action": "HOLDING" if current_state.current_position != "FLAT" else "FLAT",
            "position_size": current_state.position_size,
            "reason": "cooldown"
        }

    try:
        close_price = float(payload.current_price)
        forecasted_vol = max(float(payload.forecasted_vol), 1e-8)
        prob_high_vol = float(payload.prob_high_vol)

        # 1. Volatility Multiplier from Kelly Module
        vol_multiplier = compute_volatility_multiplier(forecasted_vol, target_vol=TARGET_VOLATILITY)

        # 2. Pure MS-GARCH Regime Direction
        if prob_high_vol > 0.5:
            # High-Volatility Regime -> Risk-Off (FLAT)
            ideal_target = 0.0
        else:
            # Low-Volatility Regime -> Long exposure scaled by Kelly multiplier
            ideal_target = float(np.clip(1.0 * vol_multiplier, 0.0, 1.0))

        # 3. Current Signed Position Size
        if current_state.current_position == "LONG":
            current_signed = current_state.position_size
        elif current_state.current_position == "SHORT":
            current_signed = -current_state.position_size
        else:
            current_signed = 0.0

        # 4. Deadband & Risk-Off Override
        if ideal_target == 0.0:
            # Safeguard 1: Regime-Shift / Risk-Off liquidation ALWAYS bypasses deadband
            if current_state.current_position != "FLAT":
                action = "CASH"
                target_position_size = 0.0
                position_size = 0.0
            else:
                action = "FLAT"
                target_position_size = 0.0
                position_size = 0.0
        else:
            # Active Position Rebalancing: enforce strict 15% (0.15) deadband
            new_target, rebalanced = execute_kelly_rebalance(current_signed, ideal_target, rebalance_deadband=0.15)
            if not rebalanced:
                action = "HOLDING"
                position_size = current_state.position_size
                target_position_size = current_signed
            else:
                if new_target > 0:
                    action = "BUY"
                    position_size = new_target
                elif new_target < 0:
                    action = "SELL_SHORT"
                    position_size = abs(new_target)
                else:
                    action = "CASH" if current_state.current_position != "FLAT" else "FLAT"
                    position_size = 0.0
                target_position_size = new_target

        # 5. Update In-Memory Trading State
        if action == "BUY":
            current_state.current_position = "LONG"
            current_state.position_size = position_size
            current_state.entry_price = close_price
        elif action == "SELL_SHORT":
            current_state.current_position = "SHORT"
            current_state.position_size = position_size
            current_state.entry_price = close_price
        elif action in ["CASH", "FLAT"]:
            # Trigger 3-hour cooldown on loss exit
            if current_state.current_position == "LONG" and close_price < current_state.entry_price:
                current_state.cooldown_until = datetime.now() + timedelta(hours=3)
                logger.info("[COOLDOWN TRIGGERED] Exited LONG at a loss. 3h cooldown activated.")
            elif current_state.current_position == "SHORT" and close_price > current_state.entry_price:
                current_state.cooldown_until = datetime.now() + timedelta(hours=3)
                logger.info("[COOLDOWN TRIGGERED] Exited SHORT at a loss. 3h cooldown activated.")
            current_state.current_position = "FLAT"
            current_state.position_size = 0.0
            current_state.entry_price = 0.0

        # 6. Execute Order via Alpaca Client
        safe_size = max(0.0, min(float(position_size), 1.0))
        logger.info(f"ROUTING DECISION: Action={action}, Size={safe_size:.2%}, Price=${close_price:,.2f}")
        alpaca_client.execute_trade(action=action, target_weight=safe_size, current_price=close_price)

        # 7. Asynchronous Logging and Discord Notification
        background_tasks.add_task(
            log_trade,
            action,
            close_price,
            float(forecasted_vol),
            float(prob_high_vol),
            position_size
        )

        return {
            "status": "success",
            "action": action,
            "target_size": safe_size
        }

    except Exception as e:
        logger.error(f"[Execution Engine Error] Execution failed: {e}")
        return {"status": "Error", "message": str(e)}


# --- DASHBOARD ENDPOINTS ---
@app.get("/")
def health_check():
    return {
        "status": "Online",
        "engine": "2-State MS-GARCH Volatility Engine",
        "sizing": "Volatility-Scaled Kelly (0.15 Deadband)"
    }


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
        cursor.execute("SELECT SUM(realized_pnl_pct) FROM paper_trades WHERE action IN ('BUY', 'SELL_SHORT') AND status = 'CLOSED'")
        total_realized = cursor.fetchone()[0] or 0.0
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action IN ('BUY', 'SELL_SHORT') AND status = 'CLOSED'")
        total_closed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action IN ('BUY', 'SELL_SHORT') AND status = 'CLOSED' AND realized_pnl_pct > 0")
        winning_trades = cursor.fetchone()[0]
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
        unrealized_pnl = 0.0
        cursor.execute("SELECT execution_price, action FROM paper_trades WHERE status = 'OPEN' AND action IN ('BUY', 'SELL_SHORT') ORDER BY timestamp ASC LIMIT 1")
        open_trade = cursor.fetchone()
        if open_trade:
            entry_price, o_action = open_trade
            cursor.execute("SELECT execution_price FROM paper_trades ORDER BY timestamp DESC LIMIT 1")
            latest_row = cursor.fetchone()
            if latest_row:
                latest_p = latest_row[0]
                if o_action == "SELL_SHORT":
                    unrealized_pnl = ((entry_price - latest_p) / entry_price) * 100
                else:
                    unrealized_pnl = ((latest_p - entry_price) / entry_price) * 100
        conn.close()
        return {
            "total_realized_pnl_pct": round(total_realized, 2),
            "win_rate": round(win_rate, 2),
            "total_closed_trades": total_closed,
            "current_unrealized_pnl_pct": round(unrealized_pnl, 2)
        }
    except Exception as e:
        return {"error": str(e)}