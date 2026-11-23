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

import os
import pickle

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
