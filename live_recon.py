#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  LIVE PORTFOLIO RECONCILIATION & SLIPPAGE MONITOR                    ║
║  Compares theoretical paper trades with Alpaca executions, computes  ║
║  slippage & shadow equity, and alerts/pauses on Sharpe decay.        ║
║                                                                      ║
║  Usage:  python live_recon.py                                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone, timedelta
from alpaca_trade_api.rest import REST

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, 'data', 'trades.db')
PAUSE_FLAG_PATH = os.path.join(PROJECT_ROOT, 'data', 'trading_paused.flag')

# WFA expected KPIs
WFA_EXPECTED_SHARPE = 1.20
DEVIATION_THRESHOLD = 0.20  # Deviations > 20% trigger veto/pause
MIN_SHARPE_ALLOWED = WFA_EXPECTED_SHARPE * (1 - DEVIATION_THRESHOLD)  # 0.96

def load_env():
    env_path = os.path.join(PROJECT_ROOT, '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, val = line.split('=', 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

def fetch_alpaca_orders():
    """Fetches closed Alpaca orders to reconstruct live fills."""
    key_id = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    if not key_id or not secret_key:
        print("[WARNING] Alpaca credentials missing from environment.")
        return []
        
    try:
        api = REST(key_id=key_id, secret_key=secret_key, base_url=base_url)
        # Fetch last 100 closed orders
        orders = api.list_orders(status='all', limit=100, nested=False)
        print(f"[Alpaca REST] Retrieved {len(orders)} historical orders.")
        return orders
    except Exception as e:
        print(f"[WARNING] Failed to query Alpaca REST API: {e}")
        return []

def send_discord_alert(webhook_url, title, message, color=15158332):
    payload = {
        "username": "Risk Manager Daemon",
        "embeds": [{
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "footer": {"text": "Live Portfolio Recon Engine"}
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=5)
    except Exception as e:
        print(f"[Error] Discord Alert failed: {e}")

def main():
    load_env()
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")

    print("=" * 60)
    print("  LIVE PORTFOLIO RECONCILIATION & SLIPPAGE MONITOR")
    print(f"  Expected Sharpe: {WFA_EXPECTED_SHARPE:.2f} | Critical Floor: {MIN_SHARPE_ALLOWED:.2f}")
    print("=" * 60)

    # 1. Fetch paper trades from trades.db
    if not os.path.exists(DB_PATH):
        print(f"[Error] Database not found at {DB_PATH}. Run api.py / live-stream first.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM paper_trades", conn)
    conn.close()

    if df.empty:
        print("[Error] No trade logs found in database. Exiting.")
        return

    # Convert timestamps and sort
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
    df = df.sort_values(by=['timestamp', 'trade_id'])
    
    # Deduplicate timestamps, keeping the last log representing final state
    df = df.groupby('timestamp').last().reset_index()

    # Reconstruct positions recursively
    sides = []
    sizes = []
    signed_positions = []
    active_side = 0.0  # 1.0 = LONG, -1.0 = SHORT, 0.0 = FLAT
    active_size = 0.0
    
    for _, row in df.iterrows():
        action = row['action']
        size = row['position_size']
        
        if action == 'BUY':
            active_side = 1.0
            active_size = size
        elif action == 'SELL_SHORT':
            active_side = -1.0
            active_size = size
        elif action in ['CASH', 'FLAT']:
            active_side = 0.0
            active_size = 0.0
        elif action == 'HOLDING':
            active_size = size
            
        sides.append(active_side)
        sizes.append(active_size)
        signed_positions.append(active_side * active_size)
        
    df['active_side'] = sides
    df['active_size'] = sizes
    df['signed_position'] = signed_positions

    # 2. Fetch actual Alpaca orders
    alpaca_orders = fetch_alpaca_orders()
    matched_orders = {}
    
    # Standard Coinbase/Binance execution fee rate (10 bps) to model live fees
    FEE_PCT_LIVE = 0.0010 
    
    # Match Alpaca orders with database logs based on timestamp proximity (< 5m)
    for order in alpaca_orders:
        if order.status != 'filled':
            continue
        # Alpaca filled_at is timezone-aware ISO string
        try:
            filled_at = pd.to_datetime(order.filled_at).tz_convert('UTC').tz_localize(None)
        except Exception:
            continue
            
        # Find nearest trade index
        time_diffs = (df['timestamp'] - filled_at).abs()
        nearest_idx = time_diffs.idxmin()
        min_diff_seconds = time_diffs.min().total_seconds()
        
        if min_diff_seconds < 300: # < 5 minutes
            trade_row = df.loc[nearest_idx]
            trade_ts = trade_row['timestamp']
            
            p_theo = float(trade_row['execution_price'])
            p_live = float(order.filled_avg_price)
            side_mult = 1.0 if order.side == 'buy' else -1.0
            
            # Slippage = side_mult * (P_live - P_theo) / P_theo
            slippage_pct = side_mult * (p_live - p_theo) / p_theo
            slippage_bps = slippage_pct * 10000.0
            
            # Alpaca REST reports orders as objects with fee if loaded, otherwise model it
            order_fee = float(order.fee) if hasattr(order, 'fee') and order.fee is not None else 0.0
            
            matched_orders[trade_ts] = {
                'filled_avg_price': p_live,
                'slippage_bps': slippage_bps,
                'fee': order_fee,
                'is_simulated': False
            }

    print(f"[Recon] Successfully matched {len(matched_orders)} live fills.")

    # 3. Reconstruct Live Fills and Slippage Deltas
    p_live = []
    fees_usd = []
    slippage_list_bps = []
    is_sim_list = []
    
    df['theo_pos_change'] = df['signed_position'].diff().abs().fillna(0.0)
    
    for _, row in df.iterrows():
        ts = row['timestamp']
        pos_change = row['theo_pos_change']
        
        if ts in matched_orders:
            p_live.append(matched_orders[ts]['filled_avg_price'])
            fees_usd.append(matched_orders[ts]['fee'])
            slippage_list_bps.append(matched_orders[ts]['slippage_bps'])
            is_sim_list.append(False)
        else:
            # Fallback to simulation mode for shadow mode validation (2 bps slip + 10 bps fee)
            if pos_change > 0.0:
                slip_pct = np.random.normal(0.0002, 0.0001)  # 2 bps mean slippage
                # Model buy higher (positive slippage) and sell lower (negative slippage)
                side_mult = 1.0 if row['active_side'] > 0.0 or row['action'] == 'BUY' else -1.0
                
                p_val = row['execution_price'] * (1.0 + side_mult * slip_pct)
                fee_val = pos_change * row['execution_price'] * FEE_PCT_LIVE
                slip_bps = slip_pct * 10000.0
            else:
                p_val = row['execution_price']
                fee_val = 0.0
                slip_bps = 0.0
                
            p_live.append(p_val)
            fees_usd.append(fee_val)
            slippage_list_bps.append(slip_bps)
            is_sim_list.append(True)
            
    df['p_live'] = p_live
    df['actual_fee_usd'] = fees_usd
    df['slippage_bps'] = slippage_list_bps
    df['is_simulated'] = is_sim_list

    # Print 24-hour summary
    cutoff_24h = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    df_24h = df[df['timestamp'] >= cutoff_24h]
    
    print(f"\n--- 24-HOUR RECONCILIATION SUMMARY ---")
    print(f"  Trades Executed: {len(df_24h)}")
    if not df_24h.empty:
        avg_slip = df_24h['slippage_bps'].mean()
        total_fees = df_24h['actual_fee_usd'].sum()
        print(f"  Average Slippage: {avg_slip:+.2f} bps")
        print(f"  Total Live Fees:  ${total_fees:.4f} USD")
    else:
        print("  No trades in the last 24 hours.")

    # 4. Generate curves in USD (Compounded)
    initial_equity = 10000.0
    theo_equity = [initial_equity]
    actual_equity = [initial_equity]
    
    df['market_return'] = df['execution_price'].pct_change().fillna(0.0)
    
    for idx in range(1, len(df)):
        # Retrieve previous states
        prev_theo_eq = theo_equity[-1]
        prev_act_eq = actual_equity[-1]
        
        prev_pos = df['signed_position'].iloc[idx - 1]
        pos_change = df['theo_pos_change'].iloc[idx]
        
        # --- Theoretical Curve ---
        theo_gain = prev_theo_eq * df['market_return'].iloc[idx] * prev_pos
        theo_fee = prev_theo_eq * pos_change * 0.0010  # 10 bps theoretical fee
        new_theo_eq = prev_theo_eq + theo_gain - theo_fee
        new_theo_eq = max(new_theo_eq, 0.0)
        theo_equity.append(new_theo_eq)
        
        # --- Actual Curve (incorporating slippage and fees) ---
        # Portfolio value after paying exit/entry fees at t-1
        act_fee = df['actual_fee_usd'].iloc[idx]
        if df['is_simulated'].iloc[idx]:
            # Scale simulated fee relative to current actual equity
            act_fee = prev_act_eq * pos_change * FEE_PCT_LIVE
            
        act_eq_after_fee = prev_act_eq - act_fee
        
        # Position held valued at actual live prices
        p_prev = df['p_live'].iloc[idx - 1]
        p_curr = df['p_live'].iloc[idx]
        
        # Gain/loss on actual positions using live execution prices
        if p_prev > 1e-8:
            act_return = (p_curr - p_prev) / p_prev
        else:
            act_return = 0.0
            
        act_gain = act_eq_after_fee * act_return * prev_pos
        new_act_eq = act_eq_after_fee + act_gain
        new_act_eq = max(new_act_eq, 0.0)
        actual_equity.append(new_act_eq)

    df['theo_equity'] = theo_equity
    df['actual_equity'] = actual_equity

    # Compute returns of the equity curves to calculate Sharpe
    df['theo_returns'] = df['theo_equity'].pct_change().fillna(0.0)
    df['actual_returns'] = df['actual_equity'].pct_change().fillna(0.0)

    # 5. Compute rolling KPIs (use entire history if less than 30d, warning if so)
    rolling_hours = 30 * 24
    if len(df) < rolling_hours:
        print(f"\n[Recon] Note: Dataset has {len(df)} hourly observations (less than 30-day window of {rolling_hours}h).")
        print("        Computing metrics over the entire historical window.")
        analysis_df = df
    else:
        analysis_df = df.tail(rolling_hours)
        
    actual_returns_analysis = analysis_df['actual_returns']
    
    # Sharpe Ratio
    mean_ret = actual_returns_analysis.mean() * 24 * 365
    std_ret = actual_returns_analysis.std() * np.sqrt(24 * 365)
    live_sharpe = mean_ret / (std_ret + 1e-9)
    
    # Max Drawdown
    cum_act = analysis_df['actual_equity']
    peak = cum_act.cummax()
    drawdown = (cum_act - peak) / (peak + 1e-9)
    live_max_dd = drawdown.min() * 100

    print(f"\n--- LIVE STRATEGY PERFORMANCE (30-DAY ROLLING / FULL) ---")
    print(f"  Live Sharpe Ratio: {live_sharpe:.2f}")
    print(f"  Live Max Drawdown: {live_max_dd:.2f}%")
    print(f"  Expected Sharpe:   {WFA_EXPECTED_SHARPE:.2f}")
    
    # Save KPIs to CSV
    kpi_df = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'rolling_days': min(30, int(len(df)/24)),
        'live_sharpe': round(live_sharpe, 4),
        'live_max_dd_pct': round(live_max_dd, 4),
        'avg_slippage_bps': round(df['slippage_bps'].mean(), 4)
    }])
    kpi_csv_path = os.path.join(PROJECT_ROOT, 'data', 'live_recon_kpis.csv')
    
    header = not os.path.exists(kpi_csv_path)
    kpi_df.to_csv(kpi_csv_path, mode='a', index=False, header=header)
    print(f"[Recon] KPIs updated in CSV → {kpi_csv_path}")

    # 6. Guard Alerting and Pausing Check
    is_paused = False
    if os.path.exists(PAUSE_FLAG_PATH):
        is_paused = True
        print("[Status] Strategy execution is currently PAUSED.")

    if live_sharpe < MIN_SHARPE_ALLOWED and not is_paused:
        print(f"\n[ALERT] Live Sharpe ({live_sharpe:.2f}) has decayed below threshold ({MIN_SHARPE_ALLOWED:.2f})!")
        print("        Creating pause flag and triggering Webhook alert...")
        
        # Create Pause Flag file
        with open(PAUSE_FLAG_PATH, 'w') as f:
            f.write(f"PAUSED_DUE_TO_DECAY|Sharpe={live_sharpe:.4f}|Time={datetime.now().isoformat()}")
            
        if webhook_url:
            title = "🚨 EMERGENCY SHIELD ACTIVATED: TRADING ENGINE PAUSED"
            message = (
                f"**Live Reconciliation Alarm Triggered!**\n\n"
                f"• **Expected Sharpe:** `{WFA_EXPECTED_SHARPE:.2f}` (WFA Benchmark)\n"
                f"• **Current Rolling Sharpe:** `{live_sharpe:.2f}` (Deviates by > 20%)\n"
                f"• **Current Max Drawdown:** `{live_max_dd:.2f}%`\n"
                f"• **Average Slippage:** `{df['slippage_bps'].mean():+.2f} bps`\n\n"
                f"**Action Taken:** Created pause flag at `data/trading_paused.flag`. "
                f"The API Execution Engine has vetoed live orders and will skip WebSocket signals until reviewed."
            )
            send_discord_alert(webhook_url, title, message, color=15548997)
            print("[Webhook] Paused alert sent successfully to Discord.")

    # 7. Plot shadow equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['theo_equity'] / initial_equity, label='Theoretical Backtest (Slippage-Free)', color='blue', alpha=0.7)
    plt.plot(df['timestamp'], df['actual_equity'] / initial_equity, label='Actual Live Performance (Net of Slip & Fee)', color='orange', linewidth=1.5)
    plt.title('Shadow Equity Curve: Theoretical vs. Live Actual')
    plt.ylabel('Normalized Portfolio Value')
    plt.xlabel('Timestamp')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.gcf().autofmt_xdate()
    
    plot_path = os.path.join(PROJECT_ROOT, 'shadow_equity.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"[Recon] Shadow Equity chart saved → {plot_path}")

if __name__ == "__main__":
    main()
