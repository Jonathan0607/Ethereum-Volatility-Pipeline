"""
Fetch Free Causal Macro & Liquidity Regimes for APEX v45.

Sources:
1. TradFi Daily: QQQ, ^VIX, UUP, ^TNX via yfinance.
2. DeFiLlama: Global total stablecoin circulating market cap.
3. APEX Internal: Trailing crypto vol, cross-asset dispersion, sleeve trailing metrics, carry spread.

STRICT UTC CAUSALITY:
All daily TradFi series are shifted by 1 full day before forward-filling onto UTC hourly bars.
Output: apex_multimodal_risk/data/processed/macro_regimes_hourly.parquet
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.apex_v45_champion import (
    UNIVERSE,
    load_data_and_features,
    run_directional_engine,
    run_funding_arb_engine,
    run_spread_engine,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "apex_multimodal_risk" / "data" / "processed"


def fetch_tradfi_daily() -> pd.DataFrame:
    """Download daily QQQ, ^VIX, UUP, ^TNX from yfinance (2022-01-01 to 2026-09-01)."""
    logger.info("Downloading daily TradFi data from yfinance...")
    tickers = ["QQQ", "^VIX", "UUP", "^TNX"]
    data = yf.download(tickers, start="2022-01-01", end="2026-09-01", interval="1d", progress=False)

    df_close = data["Close"].copy()
    # Normalize column names
    col_map = {"QQQ": "qqq", "^VIX": "vix", "UUP": "uup", "^TNX": "tnx"}
    df_close = df_close.rename(columns=col_map)
    df_close.index = pd.to_datetime(df_close.index).tz_localize(None)

    # Compute daily features on close prices
    df_feat = pd.DataFrame(index=df_close.index)

    # QQQ
    qqq = df_close["qqq"]
    df_feat["macro_qqq_ret_5d"] = qqq.pct_change(5)
    df_feat["macro_qqq_ret_20d"] = qqq.pct_change(20)
    qqq_daily_ret = qqq.pct_change()
    df_feat["macro_qqq_vol_5d"] = qqq_daily_ret.rolling(5).std() * np.sqrt(252.0)
    df_feat["macro_qqq_vol_20d"] = qqq_daily_ret.rolling(20).std() * np.sqrt(252.0)

    # VIX
    vix = df_close["vix"]
    df_feat["macro_vix_level"] = vix
    df_feat["macro_vix_chg_5d"] = vix - vix.shift(5)

    # UUP (Dollar index momentum)
    uup = df_close["uup"]
    df_feat["macro_uup_mom_20d"] = uup.pct_change(20)

    # TNX (10Y yield)
    tnx = df_close["tnx"]
    df_feat["macro_tnx_level"] = tnx
    df_feat["macro_tnx_chg_20d"] = tnx - tnx.shift(20)

    # STRICT CAUSALITY: Shift all daily features by 1 full day
    # So on UTC calendar day D, we only know data up to D-1 close!
    df_feat_lagged = df_feat.shift(1)
    df_feat_lagged["date"] = df_feat_lagged.index.strftime("%Y-%m-%d")

    logger.info(f"TradFi daily features computed: {len(df_feat_lagged)} days.")
    return df_feat_lagged


def fetch_defillama_stablecoins() -> pd.DataFrame:
    """Fetch global stablecoin total circulating supply from DeFiLlama API."""
    logger.info("Fetching DeFiLlama stablecoin supply history...")
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            records = resp.json()
            rows = []
            for r in records:
                ts = pd.to_datetime(int(r["date"]), unit="s")
                # peggedUSD total circulating
                supply = float(r.get("totalCirculating", {}).get("peggedUSD", 0.0))
                rows.append({"timestamp": ts, "stablecoin_supply": supply})
            df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")

            df["macro_stbl_growth_7d"] = df["stablecoin_supply"].pct_change(7)
            df["macro_stbl_growth_30d"] = df["stablecoin_supply"].pct_change(30)

            # Strictly lag by 1 day
            df["macro_stbl_growth_7d"] = df["macro_stbl_growth_7d"].shift(1)
            df["macro_stbl_growth_30d"] = df["macro_stbl_growth_30d"].shift(1)

            logger.info(f"DeFiLlama stablecoin features computed: {len(df)} days.")
            return df[["date", "macro_stbl_growth_7d", "macro_stbl_growth_30d"]]
        else:
            logger.warning(f"DeFiLlama API returned status {resp.status_code}. Skipping gracefully.")
    except Exception as e:
        logger.warning(f"DeFiLlama fetch failed: {e}. Skipping gracefully.")

    # Return empty fallback
    return pd.DataFrame(columns=["date", "macro_stbl_growth_7d", "macro_stbl_growth_30d"])


def build_macro_regimes_hourly() -> pd.DataFrame:
    """Compute and combine TradFi, DeFiLlama, and APEX internal microstructure features."""
    logger.info("Loading APEX v45 champion features and return streams...")
    dataset, processed, garch_wts, eval_index, n, _split_idx, macro_regime, mtf_skew_mult, arb_mag_scale = (
        load_data_and_features()
    )

    # Run component engines to obtain the 3 untouched sleeve streams and active gross exposures
    dir_stream, dir_decomp = run_directional_engine(processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult)
    spread_stream, sp_decomp = run_spread_engine(processed, dataset, UNIVERSE, eval_index, n)
    arb_stream, arb_decomp = run_funding_arb_engine(dataset, UNIVERSE, eval_index, n, arb_mag_scale)

    r_dir = dir_stream * macro_regime
    r_spread = spread_stream * macro_regime
    r_fund = arb_stream

    # Extract active gross exposures for true incremental turnover accounting
    from apex_multimodal_risk.src.train.walkforward_equity_curve import (
        run_custom_directional_engine,
        run_custom_spread_engine,
    )
    _, _, pos_dir = run_custom_directional_engine(processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult, gate_mask=None)
    _, _, pos_sp = run_custom_spread_engine(processed, dataset, UNIVERSE, eval_index, n, gate_mask=None)

    gross_dir = np.sum([np.abs(pos_dir[s]) for s in UNIVERSE], axis=0) * macro_regime
    gross_spread = np.sum([np.abs(pos_sp[s]) for s in UNIVERSE], axis=0) * macro_regime
    # Funding arb positions are active across UNIVERSE
    gross_fund = np.ones(n, dtype=np.float64) * 0.8

    # Create base hourly dataframe
    df_hourly = pd.DataFrame({"timestamp": eval_index})
    df_hourly["r_dir"] = r_dir
    df_hourly["r_spread"] = r_spread
    df_hourly["r_fund"] = r_fund
    df_hourly["r_apex_champion"] = 0.25 * r_dir + 0.55 * r_spread + 0.20 * r_fund
    df_hourly["gross_dir"] = gross_dir
    df_hourly["gross_spread"] = gross_spread
    df_hourly["gross_fund"] = gross_fund

    # 1. APEX Internal Microstructure Features (Hourly Causal)
    logger.info("Computing causal APEX internal crypto microstructure features...")
    # Hourly returns per coin
    coin_rets = {}
    coin_frs = {}
    for s in UNIVERSE:
        coin_rets[s] = processed[s].iloc[len(processed[s]) - n:]["hourly_return"].values
        coin_frs[s] = dataset[s].iloc[len(dataset[s]) - n:]["funding_rate"].values

    mat_ret = np.column_stack([coin_rets[s] for s in UNIVERSE])  # (n, 3)
    mat_fr = np.column_stack([coin_frs[s] for s in UNIVERSE])  # (n, 3)

    # Average realized return across 3 coins
    mean_crypto_ret = np.mean(mat_ret, axis=1)

    # Trailing 72h and 168h crypto realized vol (annualized, strictly shifted by 1 bar)
    s_ret = pd.Series(mean_crypto_ret)
    df_hourly["crypto_vol_72h"] = s_ret.shift(1).rolling(72, min_periods=24).std().fillna(0.02) * np.sqrt(8760.0)
    df_hourly["crypto_vol_168h"] = s_ret.shift(1).rolling(168, min_periods=48).std().fillna(0.02) * np.sqrt(8760.0)

    # Cross-asset return dispersion (std across 3 assets of 24h rolling return, shifted 1)
    df_coin_rets = pd.DataFrame(mat_ret)
    coin_24h_rets = df_coin_rets.shift(1).rolling(24, min_periods=12).sum()
    df_hourly["crypto_dispersion_24h"] = coin_24h_rets.std(axis=1).fillna(0.01)

    # Funding rate carry spread (max - min funding rate across 3 coins, trailing 24h mean, shifted 1)
    df_coin_fr = pd.DataFrame(mat_fr)
    fr_spread = (df_coin_fr.max(axis=1) - df_coin_fr.min(axis=1)).abs()
    df_hourly["crypto_fr_carry_spread_24h"] = fr_spread.shift(1).rolling(24, min_periods=6).mean().fillna(0.0)

    # Trailing 7d (168h) sleeve Sharpe ratios (strictly shifted 1 bar)
    for name, s_arr in [("dir", r_dir), ("spread", r_spread), ("fund", r_fund)]:
        ser = pd.Series(s_arr)
        r_mean = ser.shift(1).rolling(168, min_periods=48).mean().fillna(0.0)
        r_std = ser.shift(1).rolling(168, min_periods=48).std().fillna(1e-4)
        r_std = np.where(r_std < 1e-6, 1e-4, r_std)
        df_hourly[f"sleeve_sharpe_7d_{name}"] = (r_mean / r_std) * np.sqrt(8760.0)

        # Trailing 7d drawdown
        cum = (1.0 + ser.shift(1)).cumprod()
        cummax = cum.cummax()
        df_hourly[f"sleeve_dd_7d_{name}"] = ((cum - cummax) / (cummax + 1e-8)).fillna(0.0)

    # 2. Merge TradFi Daily and DeFiLlama (strictly lagged by 1 UTC day)
    df_tradfi = fetch_tradfi_daily()
    df_stbl = fetch_defillama_stablecoins()

    df_hourly["date"] = df_hourly["timestamp"].dt.strftime("%Y-%m-%d")

    # Merge TradFi features on date
    df_hourly = df_hourly.merge(df_tradfi, on="date", how="left")

    # Merge DeFiLlama on date
    if len(df_stbl) > 0:
        df_hourly = df_hourly.merge(df_stbl, on="date", how="left")
    else:
        df_hourly["macro_stbl_growth_7d"] = 0.0
        df_hourly["macro_stbl_growth_30d"] = 0.0

    # Forward fill weekend/holiday gaps in daily TradFi
    tradfi_cols = [c for c in df_hourly.columns if c.startswith("macro_")]
    df_hourly[tradfi_cols] = df_hourly[tradfi_cols].ffill().bfill().fillna(0.0)

    df_hourly = df_hourly.drop(columns=["date"]).sort_values("timestamp").reset_index(drop=True)
    logger.info(f"Final macro regimes hourly dataset shape: {df_hourly.shape}")

    # Save to parquet (gitignored)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "macro_regimes_hourly.parquet"
    df_hourly.to_parquet(out_path, index=False)
    logger.info(f"Saved to {out_path}")
    return df_hourly


if __name__ == "__main__":
    build_macro_regimes_hourly()
