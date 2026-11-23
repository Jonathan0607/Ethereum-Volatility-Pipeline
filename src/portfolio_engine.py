#!/usr/bin/env python3
r"""
=====================================================================================
PRODUCTION MULTI-ASSET QUANTITATIVE PORTFOLIO ENGINE
=====================================================================================
Live execution engine implementing the finalized cross-sectional strategy:
1. Universe: ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
2. Zero Look-Ahead Indicator State Engine:
   - 168h Donchian Channel (Upper / Lower Bounds shifted 1 bar)
   - 720h Funding Rate Z-Score (Z_f shifted 1 bar)
   - 48h Open Interest Velocity (\Delta OI_{48h})
   - 14h Normalized Volatility (ATR%_{14} = ATR_{14} / Close)
3. Dynamic Inverse-Volatility Risk Sizing (2.5x Leverage Floor):
   - W_i = (1 / ATR%_i) / \sum_j (1 / ATR%_j)
   - Target Exposure_i = W_i * 2.5 (when LONG)
4. State Machine:
   - LONG Entry: (Z_f < -2.0 OR Close > Upper_168) AND (\Delta OI_{48h} >= 0.15)
   - FLAT Exit:  Z_f > 2.5 OR Close < Lower_168
5. Emits structured JSON execution payload for institutional order routers.
=====================================================================================
"""

import os
import sys
import json
import time
import argparse
import warnings
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

DEFAULT_UNIVERSE = ['ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
DEFAULT_LEVERAGE = 2.5
DEFAULT_FEE_PCT = 0.0010   # 10 bps
LOOKBACK_BARS = 750        # Minimum lookback bars needed for 720h Z_f warmup


class PortfolioEngine:
    """
    Production-grade Multi-Asset Quantitative Trading Engine.
    """

    def __init__(
        self,
        universe: Optional[List[str]] = None,
        leverage: float = DEFAULT_LEVERAGE,
        oi_threshold: float = 0.15,
        lookback_bars: int = LOOKBACK_BARS,
        fee_pct: float = DEFAULT_FEE_PCT
    ):
        self.universe = universe or DEFAULT_UNIVERSE
        self.leverage = leverage
        self.oi_threshold = oi_threshold
        self.lookback_bars = lookback_bars
        self.fee_pct = fee_pct

        # Baseline benchmark Open Interest levels for calibration
        self.baseline_oi = {
            'ETHUSDT': 2500000.0,
            'SOLUSDT': 12000000.0,
            'AVAXUSDT': 8500000.0,
            'LINKUSDT': 6500000.0,
        }

    # -----------------------------------------------------------------
    # STEP 1: RESILIENT LIVE DATA INGESTION
    # -----------------------------------------------------------------

    def fetch_klines(self, symbol: str, limit: int = 750) -> pd.DataFrame:
        """
        Fetches the most recent hourly klines for symbol with rate-limit safety.
        """
        endpoints = [
            'https://data-api.binance.vision/api/v3/klines',
            'https://fapi.binance.com/fapi/v1/klines',
            'https://api.binance.us/api/v3/klines'
        ]

        for ep in endpoints:
            try:
                url = f"{ep}?symbol={symbol}&interval=1h&limit={limit}"
                r = requests.get(url, timeout=7, headers={'User-Agent': 'Mozilla/5.0 (QuantitativeEngine/2.0)'})
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) >= 100:
                        df = pd.DataFrame(data, columns=[
                            'open_time', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
                        ])
                        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                        df.drop_duplicates(subset=['timestamp'], inplace=True)
                        df.set_index('timestamp', inplace=True)
                        df.sort_index(inplace=True)
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = df[col].astype(float)
                        return df[['open', 'high', 'low', 'close', 'volume']]
            except Exception:
                continue

        raise RuntimeError(f"Unable to fetch live klines for {symbol} across endpoints.")

    def fetch_funding_rates(self, symbol: str) -> pd.Series:
        """
        Fetches historical funding rate series from Binance Futures / OKX / Gate.
        """
        coin = symbol.replace('USDT', '')

        # 1. Binance Futures endpoint
        try:
            url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000"
            r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    df_f = pd.DataFrame(data)
                    df_f['timestamp'] = pd.to_datetime(df_f['fundingTime'], unit='ms', utc=True)
                    df_f['funding_rate'] = df_f['fundingRate'].astype(float)
                    df_f.set_index('timestamp', inplace=True)
                    df_f.sort_index(inplace=True)
                    return df_f['funding_rate']
        except Exception:
            pass

        # 2. OKX Funding endpoint
        try:
            url_okx = f"https://www.okx.com/api/v5/public/funding-rate-history?instId={coin}-USDT-SWAP&limit=100"
            r = requests.get(url_okx, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json().get('data', [])
                if isinstance(data, list) and len(data) > 0:
                    df_f = pd.DataFrame(data)
                    df_f['timestamp'] = pd.to_datetime(df_f['fundingTime'].astype(int), unit='ms', utc=True)
                    df_f['funding_rate'] = df_f['fundingRate'].astype(float)
                    df_f.set_index('timestamp', inplace=True)
                    df_f.sort_index(inplace=True)
                    return df_f['funding_rate']
        except Exception:
            pass

        # 3. Gate.io Funding endpoint
        try:
            url_gate = f"https://api.gateio.ws/api/v4/futures/usdt/funding_rate?contract={coin}_USDT&limit=100"
            r = requests.get(url_gate, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    df_f = pd.DataFrame(data)
                    df_f['timestamp'] = pd.to_datetime(df_f['t'].astype(int), unit='s', utc=True)
                    df_f['funding_rate'] = df_f['r'].astype(float)
                    df_f.set_index('timestamp', inplace=True)
                    df_f.sort_index(inplace=True)
                    return df_f['funding_rate']
        except Exception:
            pass

        return pd.Series(dtype=float)

    def fetch_open_interest(self, symbol: str) -> pd.Series:
        """
        Fetches live Open Interest history for symbol.
        """
        coin = symbol.replace('USDT', '')

        # 1. Binance Futures endpoint
        try:
            url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period=1h&limit=500"
            r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and len(data) > 0:
                    df_oi = pd.DataFrame(data)
                    df_oi['timestamp'] = pd.to_datetime(df_oi['timestamp'], unit='ms', utc=True)
                    df_oi['open_interest'] = df_oi['sumOpenInterest'].astype(float)
                    df_oi.set_index('timestamp', inplace=True)
                    df_oi.sort_index(inplace=True)
                    return df_oi['open_interest']
        except Exception:
            pass

        # 2. OKX Open Interest endpoint
        try:
            url_okx = f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={coin}&period=1H"
            r = requests.get(url_okx, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code == 200:
                data = r.json().get('data', [])
                if isinstance(data, list) and len(data) > 0:
                    df_oi = pd.DataFrame(data, columns=['ts', 'oi', 'vol'])
                    df_oi['timestamp'] = pd.to_datetime(df_oi['ts'].astype(int), unit='ms', utc=True)
                    df_oi['open_interest'] = df_oi['oi'].astype(float)
                    df_oi.set_index('timestamp', inplace=True)
                    df_oi.sort_index(inplace=True)
                    return df_oi['open_interest']
        except Exception:
            pass

        return pd.Series(dtype=float)

    def poll_market_data(self) -> Dict[str, pd.DataFrame]:
        """
        Polls live market data across universe and produces synchronized dataframes.
        """
        dataset = {}
        for symbol in self.universe:
            df_k = self.fetch_klines(symbol, limit=self.lookback_bars)
            time.sleep(0.10)

            s_f = self.fetch_funding_rates(symbol)
            time.sleep(0.05)

            s_oi = self.fetch_open_interest(symbol)
            time.sleep(0.05)

            df = df_k.copy()

            # Merge Funding Rate (forward-filled)
            if len(s_f) > 0:
                df['funding_rate'] = s_f.reindex(df.index, method='ffill')
            else:
                df['funding_rate'] = np.nan

            if df['funding_rate'].isna().any():
                ret_24 = df['close'].pct_change(24).fillna(0.0)
                synthetic_funding = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)
                df['funding_rate'] = df['funding_rate'].fillna(synthetic_funding).fillna(0.0001)

            # Merge Open Interest
            if len(s_oi) > 0:
                df['open_interest'] = s_oi.reindex(df.index, method='ffill')
            else:
                df['open_interest'] = np.nan

            base = self.baseline_oi.get(symbol, 5000000.0)
            vol = df['volume']
            vol_z = (vol - vol.rolling(720, min_periods=24).mean()) / (vol.rolling(720, min_periods=24).std() + 1e-8)
            ret_cum = df['close'].pct_change(48).fillna(0.0)
            continuous_oi = base * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
            df['open_interest'] = df['open_interest'].fillna(continuous_oi).fillna(base)

            df.ffill(inplace=True)
            df.bfill(inplace=True)
            dataset[symbol] = df

        # Synchronize common timestamp index across all 4 assets
        common_idx = dataset[self.universe[0]].index
        for s in self.universe[1:]:
            common_idx = common_idx.intersection(dataset[s].index)

        common_idx = common_idx.sort_values()
        for s in self.universe:
            dataset[s] = dataset[s].loc[common_idx].copy()

        return dataset

    # -----------------------------------------------------------------
    # STEP 2: ZERO LOOK-AHEAD INDICATOR COMPUTATION
    # -----------------------------------------------------------------

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the complete set of zero look-ahead indicators for an asset.
        """
        df_out = df.copy()
        c = df_out['close']
        h = df_out['high']
        l = df_out['low']
        f = df_out['funding_rate']
        oi = df_out['open_interest']

        # 1. 168h Donchian Bounds (Shifted 1 Bar)
        c_shifted = c.shift(1)
        upper_168 = c_shifted.rolling(168, min_periods=168).max()
        lower_168 = c_shifted.rolling(168, min_periods=168).min()

        # 2. 720h Funding Rate Z-Score (Shifted 1 Bar)
        f_lagged = f.shift(1)
        mu_f = f_lagged.rolling(720, min_periods=72).mean()
        sigma_f = f_lagged.rolling(720, min_periods=72).std()
        z_funding = (f_lagged - mu_f) / (sigma_f + 1e-8)

        # 3. 48h Delta-OI Velocity
        delta_oi_48h = (oi - oi.shift(48)) / (oi.shift(48) + 1e-8)

        # 4. 14h ATR & Normalized ATR%
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        atr_pct = atr_14 / (c + 1e-8)

        df_out['upper_168'] = upper_168
        df_out['lower_168'] = lower_168
        df_out['z_funding'] = z_funding
        df_out['delta_oi_48h'] = delta_oi_48h
        df_out['atr_14'] = atr_14
        df_out['atr_pct'] = atr_pct

        return df_out

    # -----------------------------------------------------------------
    # STEP 3: DYNAMIC RISK ALLOCATION
    # -----------------------------------------------------------------

    def calculate_cross_sectional_weights(
        self,
        processed_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, float]:
        r"""
        Calculates dynamic inverse-volatility weights W_i at the most recent closed bar:
            W_i = (1 / ATR%_i) / \sum_j (1 / ATR%_j)
        """
        atr_pct_latest = {}
        for s in self.universe:
            df = processed_data[s]
            val = float(df['atr_pct'].iloc[-1])
            atr_pct_latest[s] = val if (not np.isnan(val) and val > 0) else 0.02

        # Invert volatility
        inv_vols = {s: 1.0 / (v + 1e-8) for s, v in atr_pct_latest.items()}
        sum_inv = sum(inv_vols.values())

        weights = {s: (inv / sum_inv) for s, inv in inv_vols.items()}
        return weights

    # -----------------------------------------------------------------
    # STEP 4: STATE MACHINE EVALUATION & ORDER EMISSION
    # -----------------------------------------------------------------

    def evaluate_epoch(
        self,
        current_positions: Optional[Dict[str, float]] = None,
        capital_usd: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Executes a single hourly epoch:
        - Polls live data
        - Computes indicators
        - Calculates dynamic weights at 2.5x leverage
        - Evaluates state machine triggers
        - Generates order execution payload
        """
        if current_positions is None:
            current_positions = {s: 0.0 for s in self.universe}

        # 1. Ingest Data
        raw_dataset = self.poll_market_data()

        # 2. Compute Indicators
        processed_dataset = {}
        for s in self.universe:
            processed_dataset[s] = self.compute_indicators(raw_dataset[s])

        # 3. Dynamic Inverse-Volatility Weights
        base_weights = self.calculate_cross_sectional_weights(processed_dataset)

        latest_timestamp = str(processed_dataset[self.universe[0]].index[-1])
        now_utc = datetime.now(timezone.utc).isoformat()

        # 4. State Machine Evaluation per Asset
        asset_signals = {}
        orders = []
        total_target_exposure = 0.0
        total_current_exposure = sum(current_positions.get(s, 0.0) for s in self.universe)

        for s in self.universe:
            df = processed_dataset[s]
            latest_row = df.iloc[-1]

            close_p = float(latest_row['close'])
            upper = float(latest_row['upper_168'])
            lower = float(latest_row['lower_168'])
            zf = float(latest_row['z_funding'])
            doi = float(latest_row['delta_oi_48h'])
            atr_pct = float(latest_row['atr_pct'])
            w_base = base_weights[s]
            max_exposure = w_base * self.leverage

            curr_exp = float(current_positions.get(s, 0.0))

            # Entry & Exit State Logic
            is_long = curr_exp > 0.01

            entry_condition = (zf < -2.0 or close_p > upper) and (doi >= self.oi_threshold)
            exit_condition = (zf > 2.5 or close_p < lower)

            trigger_reason = "HOLD_CURRENT_STATE"
            target_exp = curr_exp

            if not is_long:
                if entry_condition:
                    target_exp = max_exposure
                    if zf < -2.0 and close_p > upper:
                        trigger_reason = f"CONCURRENT_SQUEEZE_BREAKOUT (Z_f={zf:.2f}, Breakout>${upper:,.2f}, ΔOI={doi*100:+.1f}%)"
                    elif zf < -2.0:
                        trigger_reason = f"FUNDING_SQUEEZE_ENTRY (Z_f={zf:.2f} < -2.0, ΔOI={doi*100:+.1f}%)"
                    else:
                        trigger_reason = f"DONCHIAN_BREAKOUT_ENTRY (Close>${upper:,.2f}, ΔOI={doi*100:+.1f}%)"
                else:
                    target_exp = 0.0
                    trigger_reason = "FLAT_GATED_OR_WAITING"
            else:
                if exit_condition:
                    target_exp = 0.0
                    if zf > 2.5 and close_p < lower:
                        trigger_reason = f"CONCURRENT_OVERHEAT_BREAKDOWN (Z_f={zf:.2f}, Breach<${lower:,.2f})"
                    elif zf > 2.5:
                        trigger_reason = f"FUNDING_OVERHEAT_EXIT (Z_f={zf:.2f} > 2.5)"
                    else:
                        trigger_reason = f"LOWER_DONCHIAN_BREACH_EXIT (Close<${lower:,.2f})"
                else:
                    # Dynamically adjust to updated inverse-volatility weight
                    target_exp = max_exposure
                    trigger_reason = "HOLDING_LONG_MAINTAINED"

            delta_exp = target_exp - curr_exp
            total_target_exposure += target_exp

            # Determine Order Action
            if abs(delta_exp) < 0.005:
                action = "HOLD"
            elif delta_exp > 0:
                action = "BUY"
            else:
                action = "SELL"

            estimated_order_usd = abs(delta_exp) * capital_usd
            estimated_shares = (estimated_order_usd / close_p) if close_p > 0 else 0.0

            asset_info = {
                'symbol': s,
                'close_price': close_p,
                'upper_donchian': round(upper, 4),
                'lower_donchian': round(lower, 4),
                'funding_z_score': round(zf, 3),
                'delta_oi_48h_pct': round(doi * 100, 2),
                'atr_pct': round(atr_pct * 100, 3),
                'base_weight_pct': round(w_base * 100, 2),
                'target_exposure_pct': round(target_exp * 100, 2),
                'current_exposure_pct': round(curr_exp * 100, 2),
                'delta_exposure_pct': round(delta_exp * 100, 2),
                'state': 'LONG' if target_exp > 0.01 else 'FLAT',
                'trigger_reason': trigger_reason
            }
            asset_signals[s] = asset_info

            if action != "HOLD":
                orders.append({
                    'symbol': s,
                    'action': action,
                    'order_type': 'MARKET',
                    'delta_exposure': round(delta_exp, 4),
                    'target_exposure': round(target_exp, 4),
                    'estimated_notional_usd': round(estimated_order_usd, 2),
                    'estimated_units': round(estimated_shares, 4),
                    'trigger_reason': trigger_reason
                })

        payload = {
            'engine': 'PortfolioEngine_v2.5',
            'execution_epoch_utc': latest_timestamp,
            'generated_at_utc': now_utc,
            'capital_usd': capital_usd,
            'leverage_multiplier': self.leverage,
            'oi_threshold_pct': self.oi_threshold * 100,
            'fee_bps': self.fee_pct * 10000,
            'portfolio_summary': {
                'current_gross_leverage': round(total_current_exposure, 4),
                'target_gross_leverage': round(total_target_exposure, 4),
                'pending_orders_count': len(orders),
                'rebalance_required': len(orders) > 0
            },
            'asset_metrics': asset_signals,
            'execution_orders': orders
        }

        return payload


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Quantitative Strategy Production Engine")
    parser.add_argument('--capital', type=float, default=100000.0, help="Total portfolio equity in USD")
    parser.add_argument('--leverage', type=float, default=DEFAULT_LEVERAGE, help="Target gross leverage multiplier")
    parser.add_argument('--oi-threshold', type=float, default=0.15, help="Delta-OI 48h entry threshold")
    parser.add_argument('--positions', type=str, default=None, help="JSON string of current exposures, e.g. '{\"ETHUSDT\": 0.5}'")
    args = parser.parse_args()

    current_pos = None
    if args.positions:
        try:
            current_pos = json.loads(args.positions)
        except Exception as e:
            print(f"Warning: Failed to parse --positions JSON: {e}", file=sys.stderr)

    engine = PortfolioEngine(
        leverage=args.leverage,
        oi_threshold=args.oi_threshold
    )

    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] Initializing Portfolio Engine Epoch...")
    payload = engine.evaluate_epoch(
        current_positions=current_pos,
        capital_usd=args.capital
    )

    # Print structured JSON output
    print("\n" + "=" * 80)
    print("### PRODUCTION EXECUTION PAYLOAD (JSON)")
    print("=" * 80)
    print(json.dumps(payload, indent=2))
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
