"""
Shared Causal Market Feature Module.
Computes strictly causal per-asset features across SOLUSDT, AVAXUSDT, LINKUSDT:
- Log returns (1h, 6h, 24h, 72h)
- Trailing realized volatility (24h, 72h, 168h)
- GARCH(1,1) / EWMA conditional volatility proxy aligned to APEX
- Multi-timeframe return skewness (72h, 168h, 336h)
- Funding rate 720h rolling z-score
- Open Interest 48h delta and 720h rolling z-score
- Cross-asset return dispersion and cross-sectional rank
Includes CPCVStandardScaler for leak-free standardization fit strictly on train folds.
"""

import json
import os
import pickle
from typing import Any

import numpy as np
import pandas as pd

from apex_multimodal_risk.src.utils import (
    PROJECT_ROOT,
    enforce_utc_columns,
    read_parquet_utc,
    setup_logger,
)

logger = setup_logger("shared_features")

# Default feature column names
MARKET_NUMERICAL_COLS = [
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "ret_72h",
    "realized_vol_24h",
    "realized_vol_72h",
    "realized_vol_168h",
    "garch_cond_vol",
    "skew_72h",
    "skew_168h",
    "skew_336h",
    "funding_zscore",
    "oi_delta_48h",
    "oi_zscore",
    "cross_dispersion_24h",
    "cross_rank_24h",
]

NEWS_AGG_COLS = [
    "news_count_24h",
    "news_count_24h_asset",
    "news_count_24h_macro",
    "confirm_count_24h",
    "embedding_novelty_score",
    "has_exploit_24h",
    "has_outage_24h",
    "has_reg_24h",
    "has_insolv_24h",
    "has_macro_24h",
]


def compute_ewma_vol(returns: pd.Series, span: int = 48) -> pd.Series:
    """Computes strictly causal EWMA conditional volatility."""
    variance = (returns ** 2).ewm(span=span, min_periods=span).mean()
    return np.sqrt(variance)


def build_per_asset_causal_features(
    df: pd.DataFrame,
    garch_weights_series: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Computes single-asset causal features strictly using historical information <= bar T.
    """
    df = df.copy()
    close = df["close"]
    fr = df["funding_rate"]
    oi = df["open_interest"] if "open_interest" in df.columns else df["close"] * 1000.0

    # 1. Log returns
    log_c = np.log(close)
    df["ret_1h"] = (log_c - log_c.shift(1)).fillna(0.0)
    df["ret_6h"] = (log_c - log_c.shift(6)).fillna(0.0)
    df["ret_24h"] = (log_c - log_c.shift(24)).fillna(0.0)
    df["ret_72h"] = (log_c - log_c.shift(72)).fillna(0.0)

    # 2. Trailing realized volatility of 1h returns
    ret_1h = df["ret_1h"]
    df["realized_vol_24h"] = ret_1h.rolling(24, min_periods=12).std().fillna(0.0) * np.sqrt(24)
    df["realized_vol_72h"] = ret_1h.rolling(72, min_periods=24).std().fillna(0.0) * np.sqrt(72)
    df["realized_vol_168h"] = ret_1h.rolling(168, min_periods=48).std().fillna(0.0) * np.sqrt(168)

    # 3. GARCH / EWMA conditional volatility proxy
    if garch_weights_series is not None and len(garch_weights_series) == len(df):
        # In APEX v45, garch_wts is inverse conditional volatility
        # Convert to conditional vol scale
        inv_w = garch_weights_series.replace(0.0, np.nan).ffill().fillna(0.3)
        df["garch_cond_vol"] = (1.0 / (inv_w + 1e-4)).clip(0.5, 20.0)
    else:
        df["garch_cond_vol"] = compute_ewma_vol(ret_1h, span=72).fillna(0.0)

    # 4. Multi-timeframe return skewness (causally shifted by 1 bar as in APEX v45)
    df["skew_72h"] = ret_1h.rolling(72, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
    df["skew_168h"] = ret_1h.rolling(168, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
    df["skew_336h"] = ret_1h.rolling(336, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)

    # 5. Funding rate z-score (strictly causal: shift 1 before rolling stats)
    fr_shifted = fr.shift(1).fillna(0.0)
    fr_mean = fr_shifted.rolling(720, min_periods=72).mean()
    fr_std = fr_shifted.rolling(720, min_periods=72).std().replace(0.0, 1e-8)
    df["funding_zscore"] = ((fr_shifted - fr_mean) / fr_std).fillna(0.0).clip(-5.0, 5.0)

    # 6. Open Interest delta (48h) and 720h rolling z-score
    oi_shifted = oi.shift(1).bfill()
    oi_ret_48 = (oi_shifted.pct_change(48)).fillna(0.0)
    df["oi_delta_48h"] = oi_ret_48.clip(-1.0, 2.0)
    oi_mean = oi_shifted.rolling(720, min_periods=72).mean()
    oi_std = oi_shifted.rolling(720, min_periods=72).std().replace(0.0, 1e-8)
    df["oi_zscore"] = ((oi_shifted - oi_mean) / oi_std).fillna(0.0).clip(-5.0, 5.0)

    return df


def build_market_features(
    market_data: dict[str, pd.DataFrame],
    garch_weights_dict: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Computes causal per-asset features and cross-asset dispersion across all assets.
    Returns a unified long DataFrame with columns:
      ['timestamp', 'asset', *MARKET_NUMERICAL_COLS]
    """
    universe = list(market_data.keys())
    asset_dfs = {}

    for sym in universe:
        raw_df = market_data[sym].copy()
        raw_df = enforce_utc_columns(raw_df)
        gw_series = None
        if garch_weights_dict is not None and sym in garch_weights_dict:
            gw_arr = garch_weights_dict[sym]
            # Match lengths: if gw_arr corresponds to iloc[720:], align to index
            if len(gw_arr) == len(raw_df):
                gw_series = pd.Series(gw_arr, index=raw_df.index)
            elif len(gw_arr) == len(raw_df) - 720:
                gw_series = pd.Series(np.nan, index=raw_df.index)
                gw_series.iloc[720:] = gw_arr
                gw_series = gw_series.bfill()
        asset_dfs[sym] = build_per_asset_causal_features(raw_df, gw_series)

    # Cross-asset metrics at each timestamp T
    # Cross dispersion: standard deviation of 24h log returns across assets
    ret_24h_matrix = pd.DataFrame(
        {sym: asset_dfs[sym]["ret_24h"] for sym in universe}
    )
    cross_disp = ret_24h_matrix.std(axis=1).fillna(0.0)
    cross_ranks = ret_24h_matrix.rank(axis=1, pct=True)

    long_dfs = []
    for sym in universe:
        df = asset_dfs[sym].copy()
        df["cross_dispersion_24h"] = cross_disp
        df["cross_rank_24h"] = cross_ranks[sym].fillna(0.5)
        df["asset"] = sym
        df = df.reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "timestamp"})
        long_dfs.append(df)

    combined = pd.concat(long_dfs, axis=0).sort_values(["timestamp", "asset"]).reset_index(drop=True)
    return combined


def load_and_merge_features_and_targets(
    aligned_parquet_path: str = "apex_multimodal_risk/data/processed/aligned_hourly.parquet",
    market_data_path: str = "data/true_historical_dataset_4yr.pkl",
    garch_cache_path: str = "data/garch_weights_cache.pkl",
) -> pd.DataFrame:
    """
    Loads aligned targets and merges causal market features with aligned news targets.
    Guarantees strict UTC timestamps (+00:00) and no future lookahead.
    """
    if not os.path.isabs(aligned_parquet_path):
        aligned_parquet_path = str(PROJECT_ROOT / aligned_parquet_path)
    if not os.path.isabs(market_data_path):
        market_data_path = str(PROJECT_ROOT / market_data_path)
    if not os.path.isabs(garch_cache_path):
        garch_cache_path = str(PROJECT_ROOT / garch_cache_path)

    logger.info("Loading aligned targets parquet...")
    aligned_df = read_parquet_utc(aligned_parquet_path)

    logger.info("Loading audited market data pickle (read-only)...")
    with open(market_data_path, "rb") as f:
        market_data = pickle.load(f)

    garch_weights_dict = None
    if os.path.exists(garch_cache_path):
        logger.info("Loading cached GARCH weights (read-only)...")
        with open(garch_cache_path, "rb") as f:
            garch_weights_dict = pickle.load(f)

    logger.info("Computing causal market features...")
    market_feats_df = build_market_features(market_data, garch_weights_dict)

    logger.info("Merging market features with aligned targets...")
    # Keep only the timestamps that exist in aligned_df
    merged = pd.merge(
        aligned_df,
        market_feats_df[["timestamp", "asset"] + MARKET_NUMERICAL_COLS],
        on=["timestamp", "asset"],
        how="inner",
    )
    merged = enforce_utc_columns(merged)
    logger.info(f"Merged panel shape: {merged.shape}, unique timestamps: {merged['timestamp'].nunique()}")
    return merged


class CPCVStandardScaler:
    """
    StandardScaler strictly fitted on a CPCV training split.
    Guarantees no cross-fold contamination or future data leakage.
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols
        self.means_: np.ndarray | None = None
        self.stds_: np.ndarray | None = None

    def fit(self, df_train: pd.DataFrame) -> "CPCVStandardScaler":
        values = df_train[self.feature_cols].to_numpy(dtype=np.float32)
        self.means_ = np.nanmean(values, axis=0)
        self.stds_ = np.nanstd(values, axis=0)
        # Avoid division by zero
        self.stds_[self.stds_ < 1e-6] = 1.0
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.means_ is None or self.stds_ is None:
            raise ValueError("Scaler must be fitted before transforming.")
        values = df[self.feature_cols].to_numpy(dtype=np.float32)
        # Impute any NaNs with training mean
        values = np.where(np.isnan(values), self.means_, values)
        return (values - self.means_) / self.stds_

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        return self.fit(df_train).transform(df_train)


class OnlineFeatureState:
    """
    Stateful incremental feature calculator maintaining causal rolling windows for:
    - 16 Market Features:
      * Log returns: ret_1h, ret_6h, ret_24h, ret_72h
      * Realized volatility: realized_vol_24h, realized_vol_72h, realized_vol_168h
      * GARCH conditional volatility proxy (or EWMA fallback)
      * Skewness: skew_72h, skew_168h, skew_336h (causally shifted 1 bar)
      * Funding rate 720h rolling z-score (causally shifted 1 bar)
      * Open interest 48h delta and 720h rolling z-score (causally shifted 1 bar)
      * Cross-asset return dispersion and percentile rank (24h)
    - 4 NLP Gated Features:
      * 7-day trailing news embedding centroid and cosine distance novelty
      * 5-prototype cosine similarities (24h max)
      * Sparse gating with tau_nov and tau_proto
      * Alert flag and news burst ratio
    """

    FEATURE_NAMES = [
        "ret_1h",
        "ret_6h",
        "ret_24h",
        "ret_72h",
        "realized_vol_24h",
        "realized_vol_72h",
        "realized_vol_168h",
        "garch_cond_vol",
        "skew_72h",
        "skew_168h",
        "skew_336h",
        "funding_zscore",
        "oi_delta_48h",
        "oi_zscore",
        "cross_dispersion_24h",
        "cross_rank_24h",
        "gated_novelty",
        "gated_proto_sim",
        "exogenous_alert_flag",
        "news_burst_ratio",
    ]

    def __init__(
        self,
        universe: list[str] | None = None,
        max_lookback: int = 750,
        proto_matrix: np.ndarray | None = None,
        tau_nov: float = 0.2880023717880249,
        tau_proto: float = 0.7178921699523926,
    ):
        self.universe = universe or ["SOLUSDT", "AVAXUSDT", "LINKUSDT"]
        self.max_lookback = max_lookback
        self.tau_nov = tau_nov
        self.tau_proto = tau_proto

        if proto_matrix is None:
            proto_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/risk_prototypes.json"
            if proto_path.exists():
                with open(proto_path) as f:
                    pdata = json.load(f)
                pkeys = list(pdata.get("prototypes", {}).keys())
                self.proto_matrix = np.array(
                    [pdata["prototypes"][k]["vector"] for k in pkeys],
                    dtype=np.float32,
                )
            else:
                self.proto_matrix = None
        else:
            self.proto_matrix = np.asarray(proto_matrix, dtype=np.float32)

        self.reset()

    def reset(self):
        """Resets all internal history buffers."""
        self.close_history: dict[str, list[float]] = {s: [] for s in self.universe}
        self.ret1h_history: dict[str, list[float]] = {s: [] for s in self.universe}
        self.fr_history: dict[str, list[float]] = {s: [] for s in self.universe}
        self.oi_history: dict[str, list[float]] = {s: [] for s in self.universe}
        self.garch_history: dict[str, list[float]] = {s: [] for s in self.universe}
        self.news_buffer: list[dict[str, Any]] = []
        self.timestamps: list[pd.Timestamp] = []
        self.last_features: dict[str, dict[str, float]] = {}

    def warmup(
        self,
        df_history: pd.DataFrame,
        news_df: pd.DataFrame | None = None,
    ):
        """
        Seeds rolling state from historical dataframes up to T.
        df_history must contain timestamp, asset, close, funding_rate, open_interest.
        """
        self.reset()
        unique_ts = sorted(df_history["timestamp"].unique())
        # Keep trailing max_lookback bars
        warm_ts = unique_ts[-self.max_lookback:] if len(unique_ts) > self.max_lookback else unique_ts

        # Feed market bars through step
        for ts in warm_ts:
            ts_dt = pd.to_datetime(ts, utc=True)
            sub_bar = df_history[df_history["timestamp"] == ts]
            bdict = {}
            for _, r in sub_bar.iterrows():
                bdict[r["asset"]] = {
                    "close": float(r["close"]),
                    "funding_rate": float(r.get("funding_rate", 0.0)),
                    "open_interest": float(r.get("open_interest", r["close"] * 1000.0)),
                    "garch_cond_vol": float(r["garch_cond_vol"]) if "garch_cond_vol" in r else None,
                    "garch_wts": float(r["garch_wts"]) if "garch_wts" in r else None,
                }
            self.step(ts_dt, bdict, news_items=None)

        # Seed news buffer at the end of warmup with exactly the trailing 7 days of news up to warm_ts[-1]
        if news_df is not None and len(news_df) > 0 and len(warm_ts) > 0:
            warmup_end_sec = int(pd.to_datetime(warm_ts[-1], utc=True).timestamp())
            min_news_sec = warmup_end_sec - 7 * 86400
            news_secs = news_df["ingested_at"].to_numpy(dtype="datetime64[s]").astype("int64")
            mask_news = (news_secs > min_news_sec) & (news_secs <= warmup_end_sec)
            sub_news = news_df[mask_news].sort_values("ingested_at")

            self.news_buffer = []
            for _, nrow in sub_news.iterrows():
                sec_val = int(pd.to_datetime(nrow["ingested_at"], utc=True).timestamp())
                self.news_buffer.append(
                    {
                        "timestamp_sec": sec_val,
                        "embedding": np.asarray(nrow["embedding"], dtype=np.float32),
                        "entity_sol": int(nrow.get("entity_sol", 0)),
                        "entity_avax": int(nrow.get("entity_avax", 0)),
                        "entity_link": int(nrow.get("entity_link", 0)),
                        "entity_macro": int(nrow.get("entity_macro", 0)),
                    }
                )

    def step(
        self,
        timestamp: pd.Timestamp | str,
        ohlcv_bar_dict: dict[str, dict[str, float]],
        news_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Feeds a single bar T across universe assets and any arriving news items.
        Returns dict mapping each asset to 20-dim feature vector (np.float32).
        """
        ts = pd.to_datetime(timestamp, utc=True)
        ts_sec = int(ts.timestamp())
        self.timestamps.append(ts)
        if len(self.timestamps) > self.max_lookback:
            self.timestamps.pop(0)

        # 1. Ingest arriving news items (must have timestamp_sec <= ts_sec)
        if news_items:
            for item in news_items:
                if "timestamp_sec" in item:
                    sec_val = int(item["timestamp_sec"])
                elif "timestamp" in item:
                    sec_val = int(pd.to_datetime(item["timestamp"], utc=True).timestamp())
                else:
                    sec_val = ts_sec

                if sec_val <= ts_sec:
                    self.news_buffer.append(
                        {
                            "timestamp_sec": sec_val,
                            "embedding": np.asarray(item["embedding"], dtype=np.float32),
                            "entity_sol": int(item.get("entity_sol", 0)),
                            "entity_avax": int(item.get("entity_avax", 0)),
                            "entity_link": int(item.get("entity_link", 0)),
                            "entity_macro": int(item.get("entity_macro", 0)),
                        }
                    )

        # Prune news older than 7 days from current bar T (strictly <= ts_sec)
        cutoff_7d_sec = ts_sec - 7 * 86400
        self.news_buffer = [
            item for item in self.news_buffer
            if cutoff_7d_sec < item["timestamp_sec"] <= ts_sec
        ]

        # 2. Compute returns and market metrics per asset
        raw_mkt_features: dict[str, dict[str, float]] = {}
        ret_24h_map: dict[str, float] = {}

        for sym in self.universe:
            bdata = ohlcv_bar_dict.get(sym, {})
            close = float(bdata.get("close", 0.0))
            fr = float(bdata.get("funding_rate", 0.0))
            oi = float(bdata.get("open_interest", close * 1000.0))
            gw = bdata.get("garch_wts", None)
            gcond = bdata.get("garch_cond_vol", None)

            c_hist = self.close_history[sym]

            # Log returns
            if len(c_hist) >= 1:
                r1 = float(np.log(close) - np.log(c_hist[-1]))
            else:
                r1 = 0.0

            if len(c_hist) >= 6:
                r6 = float(np.log(close) - np.log(c_hist[-6]))
            else:
                r6 = 0.0

            if len(c_hist) >= 24:
                r24 = float(np.log(close) - np.log(c_hist[-24]))
            else:
                r24 = 0.0

            if len(c_hist) >= 72:
                r72 = float(np.log(close) - np.log(c_hist[-72]))
            else:
                r72 = 0.0

            ret_24h_map[sym] = r24

            # Update histories
            c_hist.append(close)
            if len(c_hist) > self.max_lookback:
                c_hist.pop(0)

            r_hist = self.ret1h_history[sym]
            r_hist.append(r1)
            if len(r_hist) > self.max_lookback:
                r_hist.pop(0)

            # Realized vol of ret_1h
            if len(r_hist) >= 12:
                v24 = float(pd.Series(r_hist[-24:]).std(ddof=1) * np.sqrt(24))
            else:
                v24 = 0.0

            if len(r_hist) >= 24:
                v72 = float(pd.Series(r_hist[-72:]).std(ddof=1) * np.sqrt(72))
            else:
                v72 = 0.0

            if len(r_hist) >= 48:
                v168 = float(pd.Series(r_hist[-168:]).std(ddof=1) * np.sqrt(168))
            else:
                v168 = 0.0

            # GARCH conditional volatility
            if gcond is not None:
                garch_val = float(gcond)
            elif gw is not None:
                inv_w = 0.3 if (gw == 0.0 or np.isnan(gw)) else float(gw)
                garch_val = float(np.clip(1.0 / (inv_w + 1e-4), 0.5, 20.0))
            else:
                s_ret = pd.Series(r_hist)
                var = (s_ret ** 2).ewm(span=72, min_periods=72).mean().iloc[-1]
                garch_val = float(np.sqrt(var)) if not np.isnan(var) else 0.0

            # Return skewness: shifted causally by 1 bar (exclude current r1)
            p_rets = r_hist[:-1]
            if len(p_rets) >= 24:
                sk72_val = float(pd.Series(p_rets[-72:]).skew())
                sk72 = 0.0 if np.isnan(sk72_val) else sk72_val
                sk168_val = float(pd.Series(p_rets[-168:]).skew())
                sk168 = 0.0 if np.isnan(sk168_val) else sk168_val
                sk336_val = float(pd.Series(p_rets[-336:]).skew())
                sk336 = 0.0 if np.isnan(sk336_val) else sk336_val
            else:
                sk72, sk168, sk336 = 0.0, 0.0, 0.0

            # Funding rate z-score: shifted causally by 1 bar
            p_fr = self.fr_history[sym]
            if len(p_fr) >= 72:
                w_fr = p_fr[-720:]
                std_f = float(pd.Series(w_fr).std(ddof=1))
                std_f = 1e-8 if (std_f == 0.0 or np.isnan(std_f)) else std_f
                fz = float(np.clip((p_fr[-1] - np.mean(w_fr)) / std_f, -5.0, 5.0))
            else:
                fz = 0.0

            p_fr.append(fr)
            if len(p_fr) > self.max_lookback:
                p_fr.pop(0)

            # Open interest: shifted causally by 1 bar
            p_oi = self.oi_history[sym]
            if len(p_oi) >= 49:
                oi_curr = p_oi[-1]
                oi_base = p_oi[-49]
                oi_d = float(np.clip((oi_curr - oi_base) / (oi_base + 1e-12), -1.0, 2.0))
            else:
                oi_d = 0.0

            if len(p_oi) >= 72:
                w_oi = p_oi[-720:]
                std_o = float(pd.Series(w_oi).std(ddof=1))
                std_o = 1e-8 if (std_o == 0.0 or np.isnan(std_o)) else std_o
                oiz = float(np.clip((p_oi[-1] - np.mean(w_oi)) / std_o, -5.0, 5.0))
            else:
                oiz = 0.0

            p_oi.append(oi)
            if len(p_oi) > self.max_lookback:
                p_oi.pop(0)

            raw_mkt_features[sym] = {
                "ret_1h": r1,
                "ret_6h": r6,
                "ret_24h": r24,
                "ret_72h": r72,
                "realized_vol_24h": v24,
                "realized_vol_72h": v72,
                "realized_vol_168h": v168,
                "garch_cond_vol": garch_val,
                "skew_72h": sk72,
                "skew_168h": sk168,
                "skew_336h": sk336,
                "funding_zscore": fz,
                "oi_delta_48h": oi_d,
                "oi_zscore": oiz,
            }

        # 3. Cross-asset dispersion and percentile rank
        s_ret24 = pd.Series(ret_24h_map)
        cross_disp = float(s_ret24.std(ddof=1)) if len(s_ret24) > 1 else 0.0
        ranks = s_ret24.rank(pct=True)

        for sym in self.universe:
            raw_mkt_features[sym]["cross_dispersion_24h"] = cross_disp
            raw_mkt_features[sym]["cross_rank_24h"] = float(ranks[sym])

        # 4. Causal NLP Features & Gating
        cutoff_24h_sec = ts_sec - 24 * 3600
        cutoff_6h_sec = ts_sec - 6 * 3600

        # Trailing 7-day novelty across all news <= ts_sec (matching align_targets.py)
        all_news_7d = [item for item in self.news_buffer if item["timestamp_sec"] > cutoff_7d_sec]
        all_news_24h = [item for item in all_news_7d if item["timestamp_sec"] > cutoff_24h_sec]

        if len(all_news_24h) > 0 and len(all_news_7d) > 0:
            embs_7d = np.array([item["embedding"] for item in all_news_7d], dtype=np.float32)
            c7d = np.mean(embs_7d, axis=0)
            norm = np.linalg.norm(c7d)
            if norm > 1e-6:
                c7d /= norm
                newest = all_news_24h[-1]["embedding"]
                novelty = float(np.clip(1.0 - np.dot(newest, c7d), 0.0, 2.0))
            else:
                novelty = 0.0
        else:
            novelty = 0.0

        out_vectors: dict[str, np.ndarray] = {}
        for sym in self.universe:
            prefix = sym.replace("USDT", "").lower()
            rel_news = [
                n for n in all_news_7d
                if (n.get(f"entity_{prefix}", 0) == 1 or n.get("entity_macro", 0) == 1)
            ]

            news_24h = [n for n in rel_news if n["timestamp_sec"] > cutoff_24h_sec]
            news_6h = [n for n in rel_news if n["timestamp_sec"] > cutoff_6h_sec]

            if len(news_24h) > 0:
                # 5 risk prototypes max similarity in past 24h
                if self.proto_matrix is not None:
                    embs_24 = np.array([n["embedding"] for n in news_24h], dtype=np.float32)
                    sims_24 = np.dot(embs_24, self.proto_matrix.T)
                    max_proto = float(np.max(sims_24))
                else:
                    max_proto = 0.0
            else:
                max_proto = 0.0

            # News burst ratio: (c6 / 6) / (c168 / 168 + 1e-3)
            c6 = float(len(news_6h))
            c168 = float(len(rel_news))
            burst = float((c6 / 6.0) / ((c168 / 168.0) + 1e-3))

            # Sparse gating
            g_nov = float(max(0.0, novelty - self.tau_nov))
            g_proto = float(max(0.0, max_proto - self.tau_proto))
            alert = 1.0 if (novelty > self.tau_nov or max_proto > self.tau_proto) else 0.0

            mkt = raw_mkt_features[sym]
            feat_dict = {
                **mkt,
                "gated_novelty": g_nov,
                "gated_proto_sim": g_proto,
                "exogenous_alert_flag": alert,
                "news_burst_ratio": burst,
                "embedding_novelty_score": novelty,
                "max_proto_sim_24h": max_proto,
            }
            self.last_features[sym] = feat_dict

            vec = np.array([feat_dict[k] for k in self.FEATURE_NAMES], dtype=np.float32)
            out_vectors[sym] = vec

        return out_vectors

    def get_named_features(self, asset: str) -> dict[str, float]:
        """Returns the last computed dictionary of named features for asset."""
        return self.last_features.get(asset, {})

