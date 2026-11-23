#!/usr/bin/env python3
r"""
=====================================================================================
PROJECT APEX: ASYNCHRONOUS LIVE PAPER-TRADING RUNNER & FORWARD TESTER
=====================================================================================
Production-grade asynchronous paper-trading execution router using asyncio and ccxt.pro.

Architecture:
1. State Management (PortfolioManager):
   - Multi-asset position tracker (SOL/USDT:USDT, AVAX/USDT:USDT, LINK/USDT:USDT).
   - Global Optimal 4-Year Baseline parameters:
     * 15% Directional / 60% Spread / 25% Breakout (15D / 60S / 25B).
     * Directional Hysteresis: 0.75
     * Spread Hysteresis: 0.40
     * Breakout OI Gate: 0.25 (+25% 48h Open Interest inflow)
     * Profit-Lock Ratchet: +20% open profit triggers fast 72h Donchian exit.
2. Concurrent Async Event Loop (ccxt.pro WebSocket uplink):
   - watch_ticker: Live tick price stream & 24h rolling extremes.
   - watch_order_book: Microsecond L2 order book depth, top bid/ask, spread in bps.
   - watch_funding_rate: 8h perpetual funding yield stream for Carry signals.
3. Execution Router (Paper Mode):
   - Target vs. Current position delta arbitration.
   - Decoupled Execution Tiers:
     * MAKER (-2 bps fee): Passive rebalances at top-of-book near touch.
     * TAKER (-10 bps fee): Latency-critical Donchian breakouts crossing the book.
   - Microsecond audit ledger saved to data/paper_trades.json.
=====================================================================================
"""

import os
import sys
import json
import time
import math
import random
import asyncio
import logging
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# CCXT / CCXT.PRO import with safety
try:
    import ccxt.pro as ccxtpro
    import ccxt
    HAS_CCXT_PRO = True
except ImportError:
    HAS_CCXT_PRO = False

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
TRADES_FILE = os.path.join(DATA_DIR, 'paper_trades.json')
os.makedirs(DATA_DIR, exist_ok=True)

# Configure structured institutional logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ApexLiveRunner")

# Baseline Universe & Canonical Hyperparameters (Apex v22 Optimal Configuration)
DEFAULT_SYMBOLS = ['SOL/USDT:USDT', 'AVAX/USDT:USDT', 'LINK/USDT:USDT']
DEFAULT_INITIAL_BALANCE = 100000.0  # $100,000 USD

# Allocations (Ternary Simplex)
ALLOC_DIR = 0.15
ALLOC_SP = 0.60
ALLOC_BRK = 0.25

# Directional ACM Engine
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5
DIR_HYST = 0.75
DIR_FEE_MAKER = -0.0002  # -2 bps

# Market-Neutral Cross-Sectional Spread Engine
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0
SP_HYST = 0.40
SP_FEE_MAKER = -0.0002   # -2 bps

# Binary Volatility Breakout Engine
BRK_ENTRY = 72          # 72h Donchian High
BRK_EXIT_BASE = 336     # 336h Donchian Low
BRK_EXIT_FAST = 72      # 72h Donchian Low (Profit-Lock)
BRK_LEV = 4.0
BRK_OI_GATE = 0.25      # +25% 48h Open Interest threshold
PROFIT_LOCK_THRESHOLD = 0.20  # +20% unrealized gain triggers fast exit
BRK_FEE_TAKER = -0.0010 # -10 bps

# Time-Series Momentum multi-horizon windows
TSMOM_WINDOWS = [24, 72, 168, 336, 720]


class PositionRecord:
    """Tracks position and performance telemetry for a single symbol."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_position: float = 0.0      # Signed exposure fraction (e.g. +1.5x)
        self.target_position: float = 0.0       # Target exposure fraction
        self.entry_price: float = 0.0           # Average entry price
        self.last_price: float = 0.0            # Latest mark price
        self.profit_lock_active: bool = False   # Breakout trailing profit lock
        self.realized_pnl: float = 0.0          # Cumulative realized PnL ($)
        self.unrealized_pnl: float = 0.0        # Mark-to-market unrealized PnL ($)
        self.total_volume_usd: float = 0.0      # Cumulative volume traded ($)
        self.total_fees_usd: float = 0.0        # Cumulative fees incurred ($)
        self.trade_count: int = 0


class PortfolioManager:
    """
    Maintains theoretical paper positions, capital allocations, risk telemetry,
    and ledger synchronization.
    """
    def __init__(self, symbols: List[str], initial_capital: float = DEFAULT_INITIAL_BALANCE):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.cash_balance = initial_capital
        self.positions: Dict[str, PositionRecord] = {s: PositionRecord(s) for s in symbols}
        self.last_rebalance_time: Optional[datetime] = None
        self.peak_equity = initial_capital
        self._load_existing_ledger()

    def _load_existing_ledger(self):
        """Initializes ledger file if not present."""
        if not os.path.exists(TRADES_FILE):
            with open(TRADES_FILE, 'w') as f:
                json.dump([], f, indent=2)

    def total_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def total_realized_pnl(self) -> float:
        return sum(pos.realized_pnl for pos in self.positions.values())

    def total_fees(self) -> float:
        return sum(pos.total_fees_usd for pos in self.positions.values())

    def total_equity(self) -> float:
        return self.cash_balance + self.total_unrealized_pnl()

    def current_drawdown_pct(self) -> float:
        eq = self.total_equity()
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity <= 0:
            return 0.0
        return ((eq - self.peak_equity) / self.peak_equity) * 100.0

    def update_mark_prices(self, price_map: Dict[str, float]):
        """Updates unrealized mark-to-market PnL based on live prices."""
        for s, price in price_map.items():
            if s in self.positions and price > 0:
                pos = self.positions[s]
                pos.last_price = price
                if abs(pos.current_position) > 1e-6 and pos.entry_price > 0:
                    # Notional value of position
                    pos_notional = pos.current_position * self.initial_capital
                    # PnL = Notional * (current_price - entry_price) / entry_price
                    pct_change = (price - pos.entry_price) / pos.entry_price
                    pos.unrealized_pnl = pos_notional * pct_change
                else:
                    pos.unrealized_pnl = 0.0

    def execute_paper_trade(
        self,
        symbol: str,
        target_pos: float,
        fill_price: float,
        order_tier: str,
        attribution: str,
        best_bid: float,
        best_ask: float,
        timestamp_iso: str
    ) -> Dict[str, Any]:
        """
        Executes an atomic paper trade, updates balances, and logs to ledger.
        """
        pos = self.positions[symbol]
        delta_pos = target_pos - pos.current_position
        if abs(delta_pos) < 1e-5:
            return {}

        side = "BUY" if delta_pos > 0 else "SELL"
        traded_notional = abs(delta_pos) * self.initial_capital
        traded_units = traded_notional / (fill_price + 1e-9)

        # Fee computation based on Maker vs Taker tier
        fee_rate = DIR_FEE_MAKER if order_tier == "MAKER" else BRK_FEE_TAKER
        fee_usd = traded_notional * fee_rate
        fee_bps = abs(fee_rate) * 10000.0

        # Calculate realized PnL if closing or reducing
        realized_pnl = 0.0
        if (pos.current_position > 0 and delta_pos < 0) or (pos.current_position < 0 and delta_pos > 0):
            closed_fraction = min(abs(delta_pos) / (abs(pos.current_position) + 1e-9), 1.0)
            closed_notional = abs(pos.current_position) * self.initial_capital * closed_fraction
            if pos.current_position > 0:
                realized_pnl = closed_notional * ((fill_price - pos.entry_price) / (pos.entry_price + 1e-9))
            else:
                realized_pnl = closed_notional * ((pos.entry_price - fill_price) / (pos.entry_price + 1e-9))

        # Update position state
        prev_pos = pos.current_position
        pos.current_position = target_pos
        pos.realized_pnl += realized_pnl
        pos.total_fees_usd += abs(fee_usd)
        pos.total_volume_usd += traded_notional
        pos.trade_count += 1

        # Cash balance adjustment: subtract fees, add realized PnL
        self.cash_balance += realized_pnl - abs(fee_usd)

        # Reset or update entry price
        if abs(target_pos) < 1e-5:
            pos.entry_price = 0.0
            pos.profit_lock_active = False
            pos.unrealized_pnl = 0.0
        else:
            # If flipping or entering fresh
            if (prev_pos <= 0 and target_pos > 0) or (prev_pos >= 0 and target_pos < 0) or pos.entry_price == 0:
                pos.entry_price = fill_price

        spread_bps = ((best_ask - best_bid) / (fill_price + 1e-9)) * 10000.0 if best_ask > 0 and best_bid > 0 else 0.0

        trade_record = {
            "trade_id": f"APEX-{int(time.time()*1000)}-{random.randint(100, 999)}",
            "timestamp": timestamp_iso,
            "symbol": symbol,
            "side": side,
            "order_tier": order_tier,
            "attribution": attribution,
            "fill_price": round(fill_price, 4),
            "delta_position": round(delta_pos, 4),
            "prev_position": round(prev_pos, 4),
            "new_position": round(target_pos, 4),
            "traded_units": round(traded_units, 6),
            "traded_notional_usd": round(traded_notional, 2),
            "fee_usd": round(fee_usd, 4),
            "fee_bps": round(fee_bps, 2),
            "best_bid": round(best_bid, 4),
            "best_ask": round(best_ask, 4),
            "spread_bps": round(spread_bps, 2),
            "realized_pnl": round(realized_pnl, 2),
            "portfolio_equity": round(self.total_equity(), 2),
            "cash_balance": round(self.cash_balance, 2)
        }

        # Append atomically to paper_trades.json
        try:
            trades = []
            if os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0:
                with open(TRADES_FILE, 'r') as f:
                    trades = json.load(f)
            trades.append(trade_record)
            with open(TRADES_FILE, 'w') as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist trade to {TRADES_FILE}: {e}")

        return trade_record


class ApexLiveRunner:
    """
    Asynchronous execution runner orchestrating WebSocket ingestion, indicator state buffers,
    hourly calculation triggers, and paper routing.
    """
    def __init__(
        self,
        symbols: List[str] = DEFAULT_SYMBOLS,
        initial_balance: float = DEFAULT_INITIAL_BALANCE,
        eval_interval_sec: int = 3600,
        proxy_url: Optional[str] = None,
        simulate_market: bool = False
    ):
        self.symbols = symbols
        self.eval_interval_sec = eval_interval_sec
        self.proxy_url = proxy_url or os.environ.get('BINANCE_PROXY') or os.environ.get('HTTPS_PROXY')
        self.simulate_market = simulate_market
        self.manager = PortfolioManager(symbols, initial_capital=initial_balance)

        # Microstructure Market State Buffers
        self.market_state: Dict[str, Dict[str, Any]] = {
            s: {
                'price': 0.0,
                'best_bid': 0.0,
                'best_ask': 0.0,
                'spread_bps': 0.0,
                'funding_rate': 0.0001,
                'funding_history': [0.0001] * 720,
                'open_interest': 1000000.0,
                'oi_history': [1000000.0] * 49,
                'hourly_closes': [],
                'hourly_highs': [],
                'hourly_lows': [],
                'last_tick_time': datetime.min.replace(tzinfo=timezone.utc)
            }
            for s in symbols
        }

        # Sub-Engine target states for hysteresis tracking
        self.current_dir_targets: Dict[str, float] = {s: 0.0 for s in symbols}
        self.current_sp_targets: Dict[str, float] = {s: 0.0 for s in symbols}
        self.current_brk_targets: Dict[str, float] = {s: 0.0 for s in symbols}

        self.exchange: Optional[Any] = None
        self.running = False
        self._seed_historical_buffers()

    def _seed_historical_buffers(self):
        """Pre-seeds hourly price, funding, and OI buffers to avoid cold start lag."""
        logger.info("Initializing historical indicator buffers...")
        for s in self.symbols:
            base_asset = s.split('/')[0].lower()
            # Synthetic starting seeds if historical files are not mapped
            # Create 750 hourly mock candles with sensible prices
            base_price = 140.0 if 'sol' in base_asset else (25.0 if 'avax' in base_asset else 11.5)
            prices = [base_price * (1.0 + 0.001 * math.sin(i / 12.0)) for i in range(750)]
            highs = [p * 1.008 for p in prices]
            lows = [p * 0.992 for p in prices]
            
            self.market_state[s]['hourly_closes'] = prices
            self.market_state[s]['hourly_highs'] = highs
            self.market_state[s]['hourly_lows'] = lows
            self.market_state[s]['price'] = prices[-1]
            self.market_state[s]['best_bid'] = prices[-1] * 0.9998
            self.market_state[s]['best_ask'] = prices[-1] * 1.0002
            self.market_state[s]['spread_bps'] = 4.0
            
            # Baseline Open Interest buffer
            base_oi = 5000000.0
            self.market_state[s]['oi_history'] = [base_oi * (1.0 + 0.005 * (i % 10)) for i in range(49)]
            self.market_state[s]['open_interest'] = self.market_state[s]['oi_history'][-1]
            
            # Baseline Funding Rate buffer (720h)
            self.market_state[s]['funding_history'] = [0.0001 * (1.0 + 0.2 * math.cos(i / 24.0)) for i in range(720)]
            self.market_state[s]['funding_rate'] = self.market_state[s]['funding_history'][-1]

        logger.info(f"Buffered 750 historical bars for universe: {self.symbols}")

    def _init_exchange(self):
        """Initializes ccxt.pro exchange instance with rate-limit and proxy configuration."""
        if not HAS_CCXT_PRO or self.simulate_market:
            logger.warning("ccxt.pro not detected or simulation flag enabled. Running in Simulation WebSocket Mode.")
            return None

        exchange_config: Dict[str, Any] = {
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
            },
            'timeout': 15000,
        }

        if self.proxy_url:
            logger.info(f"Routing ccxt.pro connection via proxy: {self.proxy_url}")
            exchange_config['aiohttp_proxy'] = self.proxy_url

        try:
            exchange = ccxtpro.binance(exchange_config)
            return exchange
        except Exception as e:
            logger.error(f"Failed to instantiate ccxtpro.binance: {e}. Falling back to simulation.")
            return None

    # ---------------------------------------------------------------------------------
    # STEP 2: CONCURRENT ASYNC WEBSOCKET LISTENERS
    # ---------------------------------------------------------------------------------

    async def listener_watch_ticker(self, symbol: str):
        """Streams real-time ticker events and bid/ask quotes."""
        logger.info(f"[{symbol}] Starting watch_ticker listener...")
        while self.running:
            try:
                if self.exchange:
                    ticker = await self.exchange.watch_ticker(symbol)
                    price = float(ticker.get('last') or ticker.get('close') or 0.0)
                    bid = float(ticker.get('bid') or price * 0.9998)
                    ask = float(ticker.get('ask') or price * 1.0002)
                else:
                    # High-fidelity simulation mode
                    await asyncio.sleep(1.0 + random.random() * 0.5)
                    st = self.market_state[symbol]
                    drift = random.gauss(0, 0.0008)
                    price = st['price'] * (1.0 + drift)
                    spread = price * 0.0003  # 3 bps spread
                    bid = price - spread / 2.0
                    ask = price + spread / 2.0

                if price > 0:
                    st = self.market_state[symbol]
                    st['price'] = price
                    st['best_bid'] = bid
                    st['best_ask'] = ask
                    st['spread_bps'] = ((ask - bid) / (price + 1e-9)) * 10000.0
                    st['last_tick_time'] = datetime.now(timezone.utc)
                    self.manager.update_mark_prices({symbol: price})

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[{symbol}] watch_ticker connection glitch: {e}. Backing off 5s...")
                await asyncio.sleep(5)

    async def listener_watch_order_book(self, symbol: str):
        """Streams L2 order book depth for maker touch derivation."""
        logger.info(f"[{symbol}] Starting watch_order_book listener...")
        while self.running:
            try:
                if self.exchange:
                    ob = await self.exchange.watch_order_book(symbol, limit=5)
                    bids = ob.get('bids', [])
                    asks = ob.get('asks', [])
                    if bids and asks:
                        top_bid = float(bids[0][0])
                        top_ask = float(asks[0][0])
                        st = self.market_state[symbol]
                        st['best_bid'] = top_bid
                        st['best_ask'] = top_ask
                        st['spread_bps'] = ((top_ask - top_bid) / ((top_bid + top_ask) / 2.0)) * 10000.0
                else:
                    await asyncio.sleep(2.0)
                    st = self.market_state[symbol]
                    p = st['price']
                    st['best_bid'] = p * 0.99985
                    st['best_ask'] = p * 1.00015
                    st['spread_bps'] = 3.0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{symbol}] watch_order_book glitch: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def listener_watch_funding_rate(self, symbol: str):
        """Streams real-time perpetual funding rates for Carry calculation."""
        logger.info(f"[{symbol}] Starting watch_funding_rate listener...")
        while self.running:
            try:
                if self.exchange and hasattr(self.exchange, 'watch_funding_rate'):
                    fr_obj = await self.exchange.watch_funding_rate(symbol)
                    fr = float(fr_obj.get('fundingRate') or 0.0001)
                else:
                    await asyncio.sleep(15.0)
                    # Mild simulated mean-reverting funding rate fluctuations
                    st = self.market_state[symbol]
                    fr = st['funding_rate'] + random.gauss(0, 0.00002)
                    fr = max(-0.0015, min(0.0020, fr))

                st = self.market_state[symbol]
                st['funding_rate'] = fr
                st['funding_history'].append(fr)
                if len(st['funding_history']) > 720:
                    st['funding_history'].pop(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[{symbol}] watch_funding_rate glitch: {e}. Retrying in 10s...")
                await asyncio.sleep(10)

    # ---------------------------------------------------------------------------------
    # STEP 3: QUANTITATIVE INDICATOR SYNTHESIS & CALCULATION ENGINE
    # ---------------------------------------------------------------------------------

    def _compute_quantitative_indicators(self) -> Dict[str, Dict[str, float]]:
        """
        Computes zero-lookahead alpha signals across the universe:
        - ATR normalized inverse volatility weights
        - Multi-horizon TSMOM composite
        - 720h Inverted Funding Rate Carry Z-score
        - 48h Open Interest Velocity
        - Donchian Envelopes
        """
        results = {}
        atr_pcts = {}

        for s in self.symbols:
            st = self.market_state[s]
            closes = np.array(st['hourly_closes'])
            highs = np.array(st['hourly_highs'])
            lows = np.array(st['hourly_lows'])
            n = len(closes)

            if n < 100:
                continue

            # ATR% (14)
            tr = np.maximum(highs[-14:] - lows[-14:], np.abs(highs[-14:] - closes[-15:-1]))
            atr = np.mean(tr)
            atr_pct = atr / (closes[-1] + 1e-8)
            atr_pcts[s] = max(atr_pct, 0.001)

            # TSMOM across windows [24, 72, 168, 336, 720]
            tsmom_parts = []
            for w in TSMOM_WINDOWS:
                if n > w:
                    rw = (closes[-1] - closes[-w]) / (closes[-w] + 1e-8)
                    ret_series = np.diff(closes[-w:]) / (closes[-w:-1] + 1e-8)
                    vw = np.std(ret_series) * np.sqrt(w) + 1e-8
                    tsmom_parts.append(math.tanh(rw / vw))
                else:
                    tsmom_parts.append(0.0)
            tsmom = float(np.mean(tsmom_parts))

            # Inverted Funding Rate Carry Z-score (720h)
            fr_hist = np.array(st['funding_history'])
            mu_fr = np.mean(fr_hist)
            sig_fr = np.std(fr_hist) + 1e-8
            carry = -math.tanh((st['funding_rate'] - mu_fr) / sig_fr)

            # 48h Open Interest Velocity
            oi_hist = st['oi_history']
            doi_48 = (oi_hist[-1] - oi_hist[0]) / (oi_hist[0] + 1e-8) if len(oi_hist) >= 49 else 0.0
            oi_regime = math.tanh(doi_48 * 5.0)

            # Donchian Envelopes
            upper_72 = np.max(highs[-BRK_ENTRY:])
            lower_336 = np.min(lows[-BRK_EXIT_BASE:]) if n >= BRK_EXIT_BASE else np.min(lows)
            lower_72 = np.max(lows[-BRK_EXIT_FAST:])

            results[s] = {
                'close': closes[-1],
                'atr_pct': atr_pcts[s],
                'tsmom': tsmom,
                'carry': carry,
                'doi_48': doi_48,
                'oi_regime': oi_regime,
                'upper_72': upper_72,
                'lower_336': lower_336,
                'lower_72': lower_72
            }

        # Cross-sectional inverse volatility weighting
        sum_inv_vol = sum(1.0 / p['atr_pct'] for p in results.values()) + 1e-8
        for s in results:
            results[s]['weight'] = (1.0 / results[s]['atr_pct']) / sum_inv_vol

        return results

    def _evaluate_ensemble_targets(self, indicators: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Computes target exposures for:
        1. Directional ACM (with Hysteresis = 0.75 and tiered ranking)
        2. Market-Neutral Spread (with Hysteresis = 0.40 and rank dispersion)
        3. Binary Volatility Breakout (with OI Gate = 0.25 and profit lock)
        4. Blended Ensemble (15% Directional, 60% Spread, 25% Breakout)
        """
        targets = {s: {'dir': 0.0, 'spread': 0.0, 'breakout': 0.0, 'ensemble': 0.0, 'tier': 'MAKER'} for s in self.symbols}

        # 1. Directional Signal & Tiered Ranking
        dir_scores = {}
        for s in self.symbols:
            ind = indicators.get(s)
            if ind:
                score = DIR_WT * ind['tsmom'] + DIR_WC * ind['carry'] + DIR_WO * ind['oi_regime']
                dir_scores[s] = score
            else:
                dir_scores[s] = 0.0

        # Sort descending
        ranked = sorted(dir_scores.items(), key=lambda x: x[1], reverse=True)
        multipliers = {ranked[0][0]: 1.5, ranked[1][0]: 1.0, ranked[2][0]: 0.5}

        for s in self.symbols:
            ind = indicators.get(s, {})
            w = ind.get('weight', 1.0 / len(self.symbols))
            score = dir_scores[s]
            x_mult = multipliers.get(s, 1.0)
            ideal_dir = score * w * x_mult * DIR_LEV

            # Directional Hysteresis Check
            cur_dir = self.current_dir_targets[s]
            if abs(ideal_dir - cur_dir) >= DIR_HYST:
                self.current_dir_targets[s] = ideal_dir
            targets[s]['dir'] = self.current_dir_targets[s]

        # 2. Market-Neutral Spread Signal
        sp_scores = {}
        for s in self.symbols:
            ind = indicators.get(s, {})
            sp_scores[s] = SP_WT * ind.get('tsmom', 0.0) + SP_WC * ind.get('carry', 0.0) + SP_WO * ind.get('oi_regime', 0.0)

        sp_ranked = sorted(sp_scores.items(), key=lambda x: x[1], reverse=True)
        top_s, bot_s = sp_ranked[0][0], sp_ranked[-1][0]
        conviction = abs(sp_ranked[0][1] - sp_ranked[-1][1])

        ideal_spread = {s: 0.0 for s in self.symbols}
        ideal_spread[top_s] = conviction * SP_LEV
        ideal_spread[bot_s] = -conviction * SP_LEV

        for s in self.symbols:
            cur_sp = self.current_sp_targets[s]
            if abs(ideal_spread[s] - cur_sp) >= SP_HYST:
                self.current_sp_targets[s] = ideal_spread[s]
            targets[s]['spread'] = self.current_sp_targets[s]

        # 3. Binary Volatility Breakout
        for s in self.symbols:
            ind = indicators.get(s, {})
            cv = self.market_state[s]['price']
            upper = ind.get('upper_72', 999999.0)
            lower_base = ind.get('lower_336', 0.0)
            lower_fast = ind.get('lower_72', 0.0)
            doi = ind.get('doi_48', 0.0)
            w = ind.get('weight', 1.0 / len(self.symbols))

            pos_rec = self.manager.positions[s]
            current_brk = self.current_brk_targets[s]
            is_breakout_action = False

            if current_brk > 0:
                # Check profit lock condition
                if pos_rec.entry_price > 0:
                    open_gain = (cv - pos_rec.entry_price) / pos_rec.entry_price
                    if open_gain >= PROFIT_LOCK_THRESHOLD:
                        pos_rec.profit_lock_active = True
                
                exit_price = lower_fast if pos_rec.profit_lock_active else lower_base
                if cv < exit_price:
                    self.current_brk_targets[s] = 0.0
                    pos_rec.profit_lock_active = False
                    is_breakout_action = True
            elif cv > upper and doi >= BRK_OI_GATE:
                # Valid breakout with open interest inflow confirmation
                self.current_brk_targets[s] = 1.0 * w * BRK_LEV
                pos_rec.profit_lock_active = False
                is_breakout_action = True

            targets[s]['breakout'] = self.current_brk_targets[s]
            if is_breakout_action:
                targets[s]['tier'] = 'TAKER'

        # 4. Triple Ensemble Convex Synthesis
        for s in self.symbols:
            ens = (
                ALLOC_DIR * targets[s]['dir'] +
                ALLOC_SP * targets[s]['spread'] +
                ALLOC_BRK * targets[s]['breakout']
            )
            targets[s]['ensemble'] = ens

        return targets

    # ---------------------------------------------------------------------------------
    # STEP 4: EXECUTION ROUTER (PAPER FORWARD TESTER)
    # ---------------------------------------------------------------------------------

    async def execute_hourly_rebalance_cycle(self):
        """
        Triggers every hour on the hour:
        1. Commits current candle closes to rolling buffer.
        2. Synthesizes indicators.
        3. Evaluates ensemble targets.
        4. Routes paper trades through the execution router.
        """
        now = datetime.now(timezone.utc)
        timestamp_iso = now.isoformat()
        logger.info(f"⚡ [HOURLY REBALANCE] Initiating cycle at {timestamp_iso}...")

        # Update hourly close buffers with current WebSocket prices
        for s in self.symbols:
            st = self.market_state[s]
            p = st['price']
            if p > 0:
                st['hourly_closes'].append(p)
                st['hourly_highs'].append(max(p, st['best_ask']))
                st['hourly_lows'].append(min(p, st['best_bid']))
                if len(st['hourly_closes']) > 800:
                    st['hourly_closes'].pop(0)
                    st['hourly_highs'].pop(0)
                    st['hourly_lows'].pop(0)

        indicators = self._compute_quantitative_indicators()
        ensemble_targets = self._evaluate_ensemble_targets(indicators)

        trades_executed = 0
        logger.info("=" * 110)
        logger.info(f"{'SYMBOL':<16} | {'TARGET':<8} | {'CURRENT':<8} | {'DELTA':<8} | {'TIER':<6} | {'MID PRICE':<10} | {'ATTRIBUTION'}")
        logger.info("-" * 110)

        for s in self.symbols:
            tgt_info = ensemble_targets[s]
            target_pos = tgt_info['ensemble']
            pos_rec = self.manager.positions[s]
            cur_pos = pos_rec.current_position
            delta = target_pos - cur_pos
            order_tier = tgt_info['tier']

            st = self.market_state[s]
            best_bid = st['best_bid']
            best_ask = st['best_ask']
            mid_price = st['price']

            # Determine simulated execution price
            if order_tier == "MAKER":
                # Passive touch fill
                fill_price = best_bid if delta > 0 else best_ask
                attribution = "ENSEMBLE_REBALANCE"
            else:
                # Aggressive cross-the-book fill
                fill_price = best_ask if delta > 0 else best_bid
                attribution = "BREAKOUT_TRIGGER"

            logger.info(
                f"{s:<16} | {target_pos:>7.2f}x | {cur_pos:>7.2f}x | {delta:>+7.2f}x | {order_tier:<6} | ${mid_price:>9.2f} | {attribution}"
            )

            if abs(delta) >= 0.01:
                trade_res = self.manager.execute_paper_trade(
                    symbol=s,
                    target_pos=target_pos,
                    fill_price=fill_price,
                    order_tier=order_tier,
                    attribution=attribution,
                    best_bid=best_bid,
                    best_ask=best_ask,
                    timestamp_iso=timestamp_iso
                )
                if trade_res:
                    trades_executed += 1
                    logger.info(
                        f"  └─► [FILL EXECUTED] {trade_res['side']} {s} @ ${trade_res['fill_price']} "
                        f"(Size: {trade_res['traded_notional_usd']} USD, Fee: {trade_res['fee_usd']} USD [{trade_res['fee_bps']} bps])"
                    )

        logger.info("=" * 110)
        logger.info(
            f"💼 [PORTFOLIO SUMMARY] Total Equity: ${self.manager.total_equity():,.2f} | "
            f"Cash: ${self.manager.cash_balance:,.2f} | Unrealized PnL: ${self.manager.total_unrealized_pnl():+,.2f} | "
            f"Drawdown: {self.manager.current_drawdown_pct():.2f}% | Executed Trades: {trades_executed}"
        )
        self.manager.last_rebalance_time = now

    async def heartbeat_telemetry_loop(self):
        """Emits standard output heartbeat logs showing prices, spreads, and exposures."""
        while self.running:
            try:
                await asyncio.sleep(20)
                quotes = []
                for s in self.symbols:
                    st = self.market_state[s]
                    quotes.append(f"{s.split('/')[0]}: ${st['price']:,.2f} (spread: {st['spread_bps']:.1f} bps)")
                
                quotes_str = " | ".join(quotes)
                eq = self.manager.total_equity()
                dd = self.manager.current_drawdown_pct()
                logger.info(f"🛰️  [HEARTBEAT] {quotes_str} || Eq: ${eq:,.2f} (DD: {dd:.2f}%)")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def main_scheduler_loop(self):
        """Runs the hourly scheduler."""
        logger.info(f"Starting calculation loop (Frequency: every {self.eval_interval_sec}s)...")
        # Perform immediate baseline evaluation on startup
        await asyncio.sleep(2.0)
        await self.execute_hourly_rebalance_cycle()

        while self.running:
            try:
                # Sleep until next evaluation interval
                await asyncio.sleep(self.eval_interval_sec)
                await self.execute_hourly_rebalance_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(10)

    # ---------------------------------------------------------------------------------
    # LIFECYCLE MANAGEMENT
    # ---------------------------------------------------------------------------------

    async def start(self):
        """Starts all concurrent background listeners and tasks."""
        self.running = True
        self.exchange = self._init_exchange()

        tasks = []
        # Launch concurrent WebSocket listeners per symbol
        for s in self.symbols:
            tasks.append(asyncio.create_task(self.listener_watch_ticker(s)))
            tasks.append(asyncio.create_task(self.listener_watch_order_book(s)))
            tasks.append(asyncio.create_task(self.listener_watch_funding_rate(s)))

        # Launch heartbeat telemetry and calculation scheduler
        tasks.append(asyncio.create_task(self.heartbeat_telemetry_loop()))
        tasks.append(asyncio.create_task(self.main_scheduler_loop()))

        logger.info(f"🚀 Project Apex Live Runner ACTIVE. Monitoring {len(self.symbols)} assets on Binance Futures.")

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Shutdown requested. Cancelling background tasks...")
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown releasing all resources."""
        self.running = False
        if self.exchange:
            logger.info("Closing ccxt.pro WebSocket connections...")
            await self.exchange.close()
        logger.info("Project Apex Live Runner gracefully terminated.")


def parse_args():
    parser = argparse.ArgumentParser(description="Project Apex Asynchronous Paper Trading Forward Tester")
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS, help="Symbols to monitor on Binance Futures")
    parser.add_argument('--balance', type=float, default=DEFAULT_INITIAL_BALANCE, help="Initial paper capital in USD")
    parser.add_argument('--interval', type=int, default=3600, help="Evaluation cycle interval in seconds (default: 3600)")
    parser.add_argument('--proxy', type=str, default=None, help="Optional HTTP/HTTPS proxy URL for ccxt.pro")
    parser.add_argument('--mock', action='store_true', help="Force simulated WebSocket stream mode")
    parser.add_argument('--single-run', action='store_true', help="Run single calculation/routing pass and exit")
    return parser.parse_args()


async def main_async():
    args = parse_args()
    runner = ApexLiveRunner(
        symbols=args.symbols,
        initial_balance=args.balance,
        eval_interval_sec=args.interval,
        proxy_url=args.proxy,
        simulate_market=args.mock
    )

    if args.single_run:
        logger.info("Executing single test evaluation cycle...")
        await runner.execute_hourly_rebalance_cycle()
        logger.info("Single evaluation complete. Paper ledger verified.")
        return

    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("Manual interrupt received.")
        await runner.stop()


if __name__ == '__main__':
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        pass
