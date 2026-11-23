"""
Causal Alignment & Target Construction Module.
Aligns hourly market OHLCV + funding + OI data with deduplicated news clusters.
Enforces zero lookahead: T - 24h < ingested_at <= T.
Calculates 7-day rolling trailing news centroids and embedding novelty scores.
Constructs forward 6h tail crash and worst-case drawdown targets.
Persists data/processed/aligned_hourly.parquet via DuckDB.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import duckdb

from apex_multimodal_risk.src.utils import setup_logger, load_config, PROJECT_ROOT

logger = setup_logger("align_targets")


def load_market_data(pkl_path: str) -> Dict[str, pd.DataFrame]:
    """Loads read-only market data dictionary (SOL, AVAX, LINK)."""
    if not os.path.isabs(pkl_path):
        pkl_path = str(PROJECT_ROOT / pkl_path)
    logger.info(f"Loading audited market data from '{pkl_path}' (read-only)...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def compute_causal_market_targets(
    df: pd.DataFrame,
    trailing_vol_lookback: int = 24,
    forward_horizon: int = 6,
    tail_vol_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    Computes causal trailing 24h realized volatility and forward 6h drawdown targets.
    Strictly causal: vol at bar T uses only prices <= T.
    Forward targets use prices strictly in [T+1, T+forward_horizon].
    """
    df = df.copy()
    close = df["close"]
    ret_1h = close.pct_change()
    
    # Trailing 24h realized volatility: sample std of 1h returns over bars [T-23h, T]
    trailing_vol = ret_1h.rolling(window=trailing_vol_lookback, min_periods=trailing_vol_lookback).std()
    df["trailing_24h_realized_vol"] = trailing_vol
    
    # Forward minimum cumulative return over the next forward_horizon hours:
    # R_{T, T+h} = (Close_{T+h} - Close_T) / Close_T
    fwd_rets = []
    for h in range(1, forward_horizon + 1):
        ret_h = (close.shift(-h) - close) / close
        fwd_rets.append(ret_h)
        
    fwd_rets_df = pd.concat(fwd_rets, axis=1)
    fwd_min_ret = fwd_rets_df.min(axis=1)
    df["fwd_min_ret_6h"] = fwd_min_ret
    
    # Tail crash: minimum forward return is worse than -tail_vol_multiplier * trailing_vol
    crash_condition = fwd_min_ret < (-tail_vol_multiplier * trailing_vol)
    df["tail_crash_6h"] = np.where(crash_condition, 1, 0)
    
    # Target validity mask (False for initial warm-up and last forward_horizon bars)
    df["target_valid"] = (~trailing_vol.isna()) & (~fwd_min_ret.isna())
    
    return df


def align_news_to_bars(
    market_df: pd.DataFrame,
    news_df: pd.DataFrame,
    asset_col: str,
    lookback_hours: int = 24,
    trailing_centroid_days: int = 7,
) -> pd.DataFrame:
    """
    Vectorized causal alignment of news clusters to hourly market bars.
    Maps news strictly satisfying: T - 24h < ingested_at <= T.
    Computes trailing 7-day centroid strictly from clusters <= T and novelty score.
    """
    n_bars = len(market_df)
    bar_timestamps = market_df.index
    bar_secs = (bar_timestamps.astype("int64") // 10**9).to_numpy()
    
    # News timestamps and embeddings
    ingested_dt = pd.to_datetime(news_df["ingested_at"], utc=True)
    news_secs = (ingested_dt.astype("int64") // 10**9).to_numpy()
    embs = np.array(news_df["embedding"].to_list(), dtype=np.float32)
    
    # Entity masks
    is_asset = news_df[asset_col].to_numpy()
    is_macro = news_df["entity_macro"].to_numpy()
    confirm_counts = news_df["confirmation_count"].to_numpy()
    
    # Risk flags
    risk_exploit = news_df["risk_exploit"].to_numpy()
    risk_outage = news_df["risk_outage"].to_numpy()
    risk_reg = news_df["risk_regulatory"].to_numpy()
    risk_insolv = news_df["risk_insolvency"].to_numpy()
    risk_macro = news_df["risk_macro_shock"].to_numpy()
    
    # Searchsorted boundaries
    sec_24h = lookback_hours * 3600
    sec_7d = trailing_centroid_days * 24 * 3600
    
    idx_24h_start = np.searchsorted(news_secs, bar_secs - sec_24h, side="right")
    idx_now = np.searchsorted(news_secs, bar_secs, side="right")
    idx_7d_start = np.searchsorted(news_secs, bar_secs - sec_7d, side="right")
    
    news_count_24h = np.zeros(n_bars, dtype=np.int32)
    news_count_24h_asset = np.zeros(n_bars, dtype=np.int32)
    news_count_24h_macro = np.zeros(n_bars, dtype=np.int32)
    confirm_count_24h = np.zeros(n_bars, dtype=np.int32)
    novelty_scores = np.zeros(n_bars, dtype=np.float32)
    
    has_exploit_24h = np.zeros(n_bars, dtype=np.int8)
    has_outage_24h = np.zeros(n_bars, dtype=np.int8)
    has_reg_24h = np.zeros(n_bars, dtype=np.int8)
    has_insolv_24h = np.zeros(n_bars, dtype=np.int8)
    has_macro_24h = np.zeros(n_bars, dtype=np.int8)
    
    for i in range(n_bars):
        s24 = idx_24h_start[i]
        e_now = idx_now[i]
        s7d = idx_7d_start[i]
        
        if e_now > s24:
            # Active news clusters in (T - 24h, T]
            n_24 = e_now - s24
            news_count_24h[i] = n_24
            news_count_24h_asset[i] = int(np.sum(is_asset[s24:e_now]))
            news_count_24h_macro[i] = int(np.sum(is_macro[s24:e_now]))
            confirm_count_24h[i] = int(np.sum(confirm_counts[s24:e_now]))
            
            has_exploit_24h[i] = int(np.any(risk_exploit[s24:e_now]))
            has_outage_24h[i] = int(np.any(risk_outage[s24:e_now]))
            has_reg_24h[i] = int(np.any(risk_reg[s24:e_now]))
            has_insolv_24h[i] = int(np.any(risk_insolv[s24:e_now]))
            has_macro_24h[i] = int(np.any(risk_macro[s24:e_now]))
            
            # Trailing 7-day centroid strictly from past clusters <= T
            if e_now > s7d:
                c7d = np.mean(embs[s7d:e_now], axis=0)
                norm = np.linalg.norm(c7d)
                if norm > 1e-6:
                    c7d /= norm
                    newest_emb = embs[e_now - 1]
                    cos_sim = float(np.dot(newest_emb, c7d))
                    # Cosine distance = 1 - cosine similarity
                    novelty_scores[i] = float(np.clip(1.0 - cos_sim, 0.0, 2.0))
                    
    res_df = market_df.copy()
    res_df["news_count_24h"] = news_count_24h
    res_df["news_count_24h_asset"] = news_count_24h_asset
    res_df["news_count_24h_macro"] = news_count_24h_macro
    res_df["confirm_count_24h"] = confirm_count_24h
    res_df["embedding_novelty_score"] = novelty_scores
    res_df["has_exploit_24h"] = has_exploit_24h
    res_df["has_outage_24h"] = has_outage_24h
    res_df["has_reg_24h"] = has_reg_24h
    res_df["has_insolv_24h"] = has_insolv_24h
    res_df["has_macro_24h"] = has_macro_24h
    return res_df


def align_causal_targets(config_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main causal alignment entry point.
    Loads deduplicated news and audited market data, aligns features and targets causally,
    and writes aligned panel dataset to data/processed/aligned_hourly.parquet via DuckDB.
    """
    config = load_config(config_path)
    align_cfg = config["causal_alignment"]
    
    market_path = align_cfg["market_data_path"]
    market_data = load_market_data(market_path)
    
    news_parquet_path = config["noise_filter"]["deduplication"]["processed_parquet_path"]
    if not os.path.isabs(news_parquet_path):
        news_parquet_path = str(PROJECT_ROOT / news_parquet_path)
        
    aligned_parquet_path = align_cfg["aligned_parquet_path"]
    if not os.path.isabs(aligned_parquet_path):
        aligned_parquet_path = str(PROJECT_ROOT / aligned_parquet_path)
    os.makedirs(os.path.dirname(aligned_parquet_path), exist_ok=True)
    
    logger.info(f"Loading deduplicated news from '{news_parquet_path}' via DuckDB...")
    con = duckdb.connect()
    news_df = con.execute(f"SELECT * FROM '{news_parquet_path}' ORDER BY ingested_at ASC").df()
    con.close()
    
    # Overlap range determination
    min_market_date = min(df.index.min() for df in market_data.values())
    max_market_date = max(df.index.max() for df in market_data.values())
    min_news_date = news_df["ingested_at"].min()
    max_news_date = news_df["ingested_at"].max()
    
    overlap_start = max(min_market_date, min_news_date).floor("h")
    overlap_end = min(max_market_date, max_news_date).floor("h")
    
    logger.info(f"Market data date range: {min_market_date} to {max_market_date}")
    logger.info(f"News dataset date range: {min_news_date} to {max_news_date}")
    logger.info(f"Overlap date range for alignment: {overlap_start} to {overlap_end}")
    
    asset_entity_map = {
        "SOLUSDT": "entity_sol",
        "AVAXUSDT": "entity_avax",
        "LINKUSDT": "entity_link",
    }
    
    aligned_dfs = []
    base_rates = {}
    
    for sym in align_cfg["assets"]:
        if sym not in market_data:
            logger.warning(f"Symbol {sym} not found in market data. Skipping.")
            continue
            
        logger.info(f"Processing causal market targets and news alignment for {sym}...")
        df_sym = market_data[sym].copy()
        
        # Filter to overlap range
        df_sym = df_sym.loc[(df_sym.index >= overlap_start) & (df_sym.index <= overlap_end)].copy()
        
        # Compute targets
        df_sym = compute_causal_market_targets(
            df_sym,
            trailing_vol_lookback=align_cfg.get("trailing_vol_lookback_hours", 24),
            forward_horizon=align_cfg.get("forward_horizon_hours", 6),
            tail_vol_multiplier=align_cfg.get("tail_vol_multiplier", 2.0),
        )
        
        # Align news
        entity_col = asset_entity_map.get(sym, "entity_macro")
        df_aligned = align_news_to_bars(
            market_df=df_sym,
            news_df=news_df,
            asset_col=entity_col,
            lookback_hours=align_cfg.get("news_lookback_hours", 24),
            trailing_centroid_days=align_cfg.get("trailing_centroid_days", 7),
        )
        
        df_aligned["asset"] = sym
        df_aligned["timestamp"] = df_aligned.index
        aligned_dfs.append(df_aligned)
        
        valid_targets = df_aligned.loc[df_aligned["target_valid"], "tail_crash_6h"]
        rate = float(valid_targets.mean()) if len(valid_targets) > 0 else 0.0
        base_rates[sym] = {
            "base_rate": rate,
            "positive_count": int(valid_targets.sum()),
            "total_valid": len(valid_targets),
        }
        logger.info(f"{sym} Tail Crash (6h) Base Rate: {rate:.2%} ({int(valid_targets.sum()):,} / {len(valid_targets):,} bars)")
        
    panel_df = pd.concat(aligned_dfs, axis=0).reset_index(drop=True)
    panel_df = panel_df.sort_values(by=["timestamp", "asset"]).reset_index(drop=True)
    
    # Persist aligned panel dataset via DuckDB
    logger.info(f"Persisting {len(panel_df):,} aligned rows to '{aligned_parquet_path}' via DuckDB...")
    con = duckdb.connect()
    con.register("aligned_hourly_df", panel_df)
    con.execute(f"COPY aligned_hourly_df TO '{aligned_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    
    audit_summary = {
        "overlap_start": str(overlap_start),
        "overlap_end": str(overlap_end),
        "total_aligned_rows": len(panel_df),
        "base_rates": base_rates,
    }
    logger.info("Causal alignment and target generation complete.")
    return panel_df, audit_summary


if __name__ == "__main__":
    align_causal_targets()
