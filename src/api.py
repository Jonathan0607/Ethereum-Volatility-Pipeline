import requests
import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kelly_execution import compute_volatility_multiplier, execute_kelly_rebalance
from alpaca_client import AlpacaExecutionClient
from strategy import compute_unified_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="Ethereum Perpetual Funding Squeeze & Quantitative Execution Engine")
alpaca_client = AlpacaExecutionClient()

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
    logger.info("[API] Booting Perpetual Funding Squeeze Execution Engine...")
    initialize_database()


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
    elif action in ["SELL", "CASH", "FLAT"]:
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

    cursor.execute('''
    INSERT INTO paper_trades (asset, action, execution_price, predicted_volatility, regime, status, position_size, realized_pnl_pct)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', ('ETH/USD', actual_action, price, pred_vol, regime, status, position_size, realized_pnl))
    conn.commit()
    conn.close()

    logger.info(f"[EXECUTED] {actual_action} | ${price:.2f} | Size: {position_size:.1f}")

    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if WEBHOOK_URL is None:
        return

    embed_color = 5763719 if actual_action == "BUY" else (15548997 if actual_action in ["SELL", "CASH"] else 9807270)
    payload = {
        "username": "Ethereum Funding Squeeze Harvester",
        "embeds": [{
            "title": f"⚡ STRATEGY ACTION: {actual_action}",
            "color": embed_color,
            "fields": [
                {"name": "Execution Price", "value": f"${price:,.2f}", "inline": True},
                {"name": "Action", "value": actual_action, "inline": True},
                {"name": "Position Size", "value": f"{position_size:.1f}x", "inline": True},
            ],
            "footer": {"text": "Perpetual Funding Rate Squeeze Harvester • 7D Donchian"}
        }]
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"[Discord Error] Webhook delivery failed: {e}")


# --- LIVE ORCHESTRATION ROUTE ---
@app.post("/execute/squeeze_harvester")
def execute_squeeze_harvester(background_tasks: BackgroundTasks):
    """
    Live Production Orchestration Endpoint:
    1. Fetches latest 168 hours of ETH spot candle data.
    2. Fetches past 90 epochs (30 days) of 8-hour Binance funding rates.
    3. Queries Alpaca for current position (1.0 or 0.0).
    4. Computes target position via compute_unified_signal().
    5. Dispatches BUY/SELL orders through alpaca_client if state changes.
    6. Returns execution telemetry.
    """
    try:
        # 1. Spot Data Acquisition (Last 168+ hours)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly_with_funding.csv')

        if os.path.exists(data_path):
            spot_df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
            spot_df.sort_index(inplace=True)
            if 'close' not in spot_df.columns and 'eth_close' in spot_df.columns:
                spot_df['close'] = spot_df['eth_close']
            spot_df = spot_df.tail(200)
        else:
            # Fallback mock dataframe
            dates = pd.date_range(end=datetime.utcnow(), periods=170, freq='h')
            spot_df = pd.DataFrame({'close': np.linspace(2500, 2600, 170)}, index=dates)

        # 2. Funding Data Acquisition (90 8-hour epochs = 30 days)
        funding_df = None
        try:
            url = 'https://fapi.binance.com/fapi/v1/fundingRate'
            r = requests.get(url, params={'symbol': 'ETHUSDT', 'limit': 90}, timeout=5)
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) >= 90:
                    funding_df = pd.DataFrame(data)
                    funding_df['fundingRate'] = funding_df['fundingRate'].astype(float)
        except Exception as e:
            logger.warning(f"Binance API query failed: {e}. Using local derivatives cache...")

        if funding_df is None or len(funding_df) < 90:
            # Use local historical funding rates
            cache_path = os.path.join(current_dir, '..', 'data', 'eth_hourly_with_funding.csv')
            if os.path.exists(cache_path):
                df_c = pd.read_csv(cache_path)
                fr_col = 'funding_rate' if 'funding_rate' in df_c.columns else 'fundingRate'
                funding_df = pd.DataFrame({'fundingRate': df_c[fr_col].tail(90).values})
            else:
                funding_df = pd.DataFrame({'fundingRate': np.full(90, 0.0001)})

        # 3. State Check
        current_pos = alpaca_client.get_position(symbol="ETH")

        # 4. Signal Computation
        target_pos = compute_unified_signal(
            spot_df=spot_df,
            funding_df=funding_df,
            current_position=current_pos
        )

        # Calculate Telemetry for Response
        close_series = spot_df['close'].astype(float)
        upper_168 = float(close_series.iloc[-168:].max())
        lower_168 = float(close_series.iloc[-168:].min())
        current_close = float(close_series.iloc[-1])

        funding_series = funding_df['fundingRate'].astype(float)
        roll_mean = float(funding_series.rolling(90, min_periods=30).mean().iloc[-1])
        roll_std = float(funding_series.rolling(90, min_periods=30).std().iloc[-1])
        roll_std = max(roll_std, 1e-6) if not np.isnan(roll_std) else 1e-6
        z_f = float((funding_series.iloc[-1] - roll_mean) / roll_std)

        # 5. Execution Routing
        action = "HOLDING"
        if target_pos == 1.0 and current_pos == 0.0:
            action = "BUY"
            alpaca_client.execute_trade(action="BUY", target_weight=1.0, current_price=current_close)
            current_state.current_position = "LONG"
            current_state.position_size = 1.0
            background_tasks.add_task(log_trade, "BUY", current_close, 0.0, z_f, 1.0)
        elif target_pos == 0.0 and current_pos == 1.0:
            action = "SELL"
            alpaca_client.execute_trade(action="SELL", target_weight=0.0, current_price=current_close)
            current_state.current_position = "FLAT"
            current_state.position_size = 0.0
            background_tasks.add_task(log_trade, "SELL", current_close, 0.0, z_f, 0.0)

        return {
            "status": "success",
            "action": action,
            "current_position": current_pos,
            "target_position": target_pos,
            "funding_zscore": round(z_f, 4),
            "donchian_bounds": {
                "upper_168": round(upper_168, 2),
                "lower_168": round(lower_168, 2),
                "current_close": round(current_close, 2)
            }
        }

    except Exception as e:
        logger.error(f"Squeeze Harvester Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- DASHBOARD & MONITORING ENDPOINTS ---
@app.get("/")
def health_check():
    return {
        "status": "Online",
        "engine": "Perpetual Funding Rate Squeeze Harvester",
        "channel": "168h (7-Day) Donchian",
        "funding_window": "30-Day (90 Epochs) Z-Score"
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
    json_path = os.path.join(current_dir, '..', 'research', 'results', 'funding_squeeze_results.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(current_dir, '..', 'research', 'funding_squeeze_results.json')
    try:
        with open(json_path, 'r') as f:
            content = f.read()
        return Response(content=content, media_type="application/json")
    except FileNotFoundError:
        return {"error": "Funding squeeze backtest results not found."}


@app.get("/portfolio-stats")
def get_portfolio_stats():
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT SUM(realized_pnl_pct) FROM paper_trades WHERE action IN ('BUY', 'SELL') AND status = 'CLOSED'")
        total_realized = cursor.fetchone()[0] or 0.0
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action IN ('BUY', 'SELL') AND status = 'CLOSED'")
        total_closed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM paper_trades WHERE action IN ('BUY', 'SELL') AND status = 'CLOSED' AND realized_pnl_pct > 0")
        winning_trades = cursor.fetchone()[0]
        win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
        conn.close()
        return {
            "total_realized_pnl_pct": round(total_realized, 2),
            "win_rate": round(win_rate, 2),
            "total_closed_trades": total_closed
        }
    except Exception as e:
        return {"error": str(e)}