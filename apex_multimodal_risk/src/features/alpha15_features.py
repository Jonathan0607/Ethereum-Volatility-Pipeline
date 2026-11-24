"""
Pure NumPy & Pandas Cross-Sectional Alpha (21 Features) & Dynamic Risk (13 Features) Pipeline.
STRICTLY zero PyTorch, zero LightGBM, and zero DuckDB imports for ultra-lean serving and offline parity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE_15: list[str] = [
    "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "DOTUSDT", "ETCUSDT", "ETHUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT",
]

ALPHA_FEATURE_COLS: list[str] = [
    "rev_res_6h",
    "rev_res_24h",
    "mom_72h",
    "mom_168h",
    "mom_336h",
    "mom_720h",
    "ram_72h",
    "ram_168h",
    "ram_336h",
    "ram_720h",
    "dist_168h_hl",
    "fr_level",
    "fr_carry_72h",
    "fr_zscore_30d",
    "cum_funding_168h",
    "vol_ratio_24_168",
    "vol_surge",
    "amihud_24h",
    "skew_72h",
    "skew_168h",
    "downside_ratio_72h",
]

RISK_FEATURE_COLS: list[str] = [
    "beta_24h",
    "beta_72h",
    "beta_168h",
    "beta_720h",
    "vol_ratio_btc",
    "corr_btc_72h",
    "corr_btc_168h",
    "vol_expansion_24_168",
    "fr_zscore_30d",
    "skew_168h",
    "downside_ratio_72h",
    "cross_dispersion",
    "pc1_absorption",
]


def compute_alpha15_and_risk_features(
    df_panel: pd.DataFrame,
    btc_symbol: str = "BTCUSDT",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Computes strictly causal 21 cross-sectional alpha features and 13 dynamic risk features
    on a panel of 15 perpetual assets.

    Guarantees:
      1. Zero lookahead: all rolling statistics, min/max, skew, and covariances use strictly trailing data.
      2. Forward-only warmup handling: uses forward-filling or neutral defaults (never bfill).
      3. Cross-sectional rank normalization [-1.0, +1.0] across assets for each hour for alpha features.
      4. Explicit btc_symbol lookup rather than assuming hardcoded asset positions.
    """
    df = df_panel.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    assets = sorted(df["asset"].unique())
    if btc_symbol not in assets:
        raise ValueError(f"Required benchmark asset {btc_symbol} not found in panel assets: {assets}")
    btc_idx = assets.index(btc_symbol)

    feature_dfs = []
    for asset, g in df.groupby("asset"):
        g = g.sort_values("timestamp").copy()
        c = g["close"].values
        v = g["volume"].values
        fr = g["funding_rate"].values
        hours = g["timestamp"].dt.hour.values
        eff_fr = np.where(np.isin(hours, [0, 8, 16]), fr, 0.0)

        ret_1h = pd.Series(c).pct_change().fillna(0.0).values
        g["ret_1h"] = ret_1h

        g["raw_ret_6h"] = pd.Series(c).pct_change(6).fillna(0.0).values
        g["raw_ret_24h"] = pd.Series(c).pct_change(24).fillna(0.0).values

        for w in [72, 168, 336, 720]:
            rw = pd.Series(c).pct_change(w).fillna(0.0).values
            vw = pd.Series(ret_1h).rolling(w, min_periods=max(12, w // 4)).std().fillna(0.0).values * np.sqrt(w)
            g[f"mom_{w}h"] = rw
            g[f"ram_{w}h"] = np.tanh(rw / (vw + 1e-6))

        high_vals = g["high"].values
        low_vals = g["low"].values
        high_168 = pd.Series(high_vals).rolling(168, min_periods=24).max().ffill().fillna(high_vals[0]).values
        low_168 = pd.Series(low_vals).rolling(168, min_periods=24).min().ffill().fillna(low_vals[0]).values
        g["dist_168h_hl"] = (c - low_168) / (high_168 - low_168 + 1e-6)

        g["fr_level"] = fr
        g["fr_carry_72h"] = pd.Series(fr).rolling(72, min_periods=12).sum().fillna(0.0).values
        fr_mu = pd.Series(fr).rolling(720, min_periods=72).mean().ffill().fillna(0.0).values
        fr_sig = pd.Series(fr).rolling(720, min_periods=72).std().ffill().fillna(1.0).values
        g["fr_zscore_30d"] = np.clip((fr - fr_mu) / (fr_sig + 1e-8), -4.0, 4.0)
        g["cum_funding_168h"] = pd.Series(eff_fr).rolling(168, min_periods=24).sum().fillna(0.0).values

        vol_24 = pd.Series(ret_1h).rolling(24, min_periods=12).std().ffill().fillna(1e-4).values
        vol_168 = pd.Series(ret_1h).rolling(168, min_periods=24).std().ffill().fillna(1e-4).values
        g["vol_ratio_24_168"] = vol_24 / (vol_168 + 1e-6)
        g["trailing_vol_168h"] = vol_168 * np.sqrt(168)

        v_24 = pd.Series(v).rolling(24, min_periods=12).mean().ffill().fillna(1.0).values
        v_168 = pd.Series(v).rolling(168, min_periods=24).mean().ffill().fillna(1.0).values
        g["vol_surge"] = np.log((v_24 + 1.0) / (v_168 + 1.0))

        dollar_vol = v * c
        amihud = np.abs(ret_1h) / (dollar_vol + 1e-4)
        g["amihud_24h"] = pd.Series(amihud).rolling(24, min_periods=12).mean().ffill().fillna(0.0).values

        g["skew_72h"] = pd.Series(ret_1h).rolling(72, min_periods=24).skew().fillna(0.0).values
        g["skew_168h"] = pd.Series(ret_1h).rolling(168, min_periods=24).skew().fillna(0.0).values
        neg_ret = np.where(ret_1h < 0.0, ret_1h, 0.0)
        semi_var_72 = pd.Series(neg_ret**2).rolling(72, min_periods=24).mean().fillna(0.0).values
        tot_var_72 = pd.Series(ret_1h**2).rolling(72, min_periods=24).mean().fillna(1e-8).values
        g["downside_ratio_72h"] = semi_var_72 / (tot_var_72 + 1e-8)

        feature_dfs.append(g)

    df_panel_out = pd.concat(feature_dfs, axis=0).sort_values(["timestamp", "asset"]).reset_index(drop=True)

    # Cross-sectional market returns
    mkt_ret_6h = df_panel_out.groupby("timestamp")["raw_ret_6h"].transform("mean")
    mkt_ret_24h = df_panel_out.groupby("timestamp")["raw_ret_24h"].transform("mean")
    df_panel_out["rev_res_6h"] = df_panel_out["raw_ret_6h"] - mkt_ret_6h
    df_panel_out["rev_res_24h"] = df_panel_out["raw_ret_24h"] - mkt_ret_24h

    # Rank normalize alpha features to [-1, +1] per hour
    n_assets = len(assets)
    denom = max(1.0, float(n_assets - 1))
    for col in ALPHA_FEATURE_COLS:
        ranks = df_panel_out.groupby("timestamp")[col].rank(method="average")
        df_panel_out[col] = (2.0 * (ranks - 1.0) / denom) - 1.0

    # -----------------------------------------------------------------------------------
    # Dynamic Risk Features (13 Features)
    # -----------------------------------------------------------------------------------
    timestamps = sorted(df_panel_out["timestamp"].unique())
    n_ts = len(timestamps)

    ret_df = (
        df_panel_out.pivot(index="timestamp", columns="asset", values="ret_1h")
        .reindex(index=timestamps, columns=assets)
        .fillna(0.0)
    )

    beta_mats: dict[int, np.ndarray] = {}
    corr_mats: dict[int, np.ndarray] = {}
    for w in [24, 72, 168, 720]:
        min_p = 72 if w >= 72 else max(12, w // 4)
        cov_btc = ret_df.rolling(w, min_periods=min_p).cov(ret_df[btc_symbol])
        var_btc = ret_df[btc_symbol].rolling(w, min_periods=min_p).var()
        b_mat = (cov_btc.div(var_btc + 1e-8, axis=0)).clip(0.3, 2.5).fillna(1.0).to_numpy(copy=True)
        b_mat[:, btc_idx] = 1.0
        beta_mats[w] = b_mat

        if w in [72, 168]:
            c_mat = ret_df.rolling(w, min_periods=min_p).corr(ret_df[btc_symbol]).clip(-1.0, 1.0).fillna(0.0).values
            corr_mats[w] = c_mat

    vol_168_df = ret_df.rolling(168, min_periods=24).std().fillna(1e-4)
    vol_btc_168 = vol_168_df[btc_symbol].values[:, None]
    vol_ratio_btc_mat = np.clip(vol_168_df.values / (vol_btc_168 + 1e-6), 0.2, 5.0)

    vol_24_df = ret_df.rolling(24, min_periods=12).std().fillna(1e-4)
    vol_expansion_mat = np.clip(vol_24_df.values / (vol_168_df.values + 1e-6), 0.2, 5.0)

    cross_disp_series = ret_df.std(axis=1).values[:, None]
    cross_disp_mat = np.repeat(cross_disp_series, n_assets, axis=1)

    pc1_absorption_arr = np.full(n_ts, 0.50)
    step = 24
    for t_i in range(0, n_ts, step):
        if t_i < 168:
            continue
        sub_r = ret_df.values[t_i - 168 : t_i]
        corr_168 = np.corrcoef(sub_r, rowvar=False)
        corr_168 = np.nan_to_num(corr_168, nan=0.0)
        eigvals = np.linalg.eigvalsh(corr_168)
        eigvals = np.sort(eigvals)[::-1]
        abs_ratio = float(eigvals[0] / np.sum(eigvals)) if np.sum(eigvals) > 0 else 0.50
        pc1_absorption_arr[t_i : min(t_i + step, n_ts)] = abs_ratio
    pc1_absorption_mat = np.repeat(pc1_absorption_arr[:, None], n_assets, axis=1)

    ts_map = {t: i for i, t in enumerate(timestamps)}
    asset_map = {a: i for i, a in enumerate(assets)}
    t_indices = [ts_map[t] for t in df_panel_out["timestamp"]]
    a_indices = [asset_map[a] for a in df_panel_out["asset"]]

    df_panel_out["beta_24h"] = beta_mats[24][t_indices, a_indices]
    df_panel_out["beta_72h"] = beta_mats[72][t_indices, a_indices]
    df_panel_out["beta_168h"] = beta_mats[168][t_indices, a_indices]
    df_panel_out["beta_720h"] = beta_mats[720][t_indices, a_indices]
    df_panel_out["vol_ratio_btc"] = vol_ratio_btc_mat[t_indices, a_indices]
    df_panel_out["corr_btc_72h"] = corr_mats[72][t_indices, a_indices]
    df_panel_out["corr_btc_168h"] = corr_mats[168][t_indices, a_indices]
    df_panel_out["vol_expansion_24_168"] = vol_expansion_mat[t_indices, a_indices]
    df_panel_out["cross_dispersion"] = cross_disp_mat[t_indices, a_indices]
    df_panel_out["pc1_absorption"] = pc1_absorption_mat[t_indices, a_indices]

    return df_panel_out, ALPHA_FEATURE_COLS, RISK_FEATURE_COLS


def extract_bar_features_for_serving(
    df_panel_with_features: pd.DataFrame,
    target_timestamp: pd.Timestamp | str,
    assets: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extracts the (15, 21) alpha feature matrix and (15, 13) risk feature matrix
    for a specific timestamp ready for POST /v1/alpha submission.
    """
    ts = pd.Timestamp(target_timestamp)
    sub = df_panel_with_features[df_panel_with_features["timestamp"] == ts].copy()
    if len(sub) == 0:
        raise ValueError(f"No rows found for timestamp {target_timestamp}")

    target_assets = assets or sorted(sub["asset"].unique())
    sub = sub.set_index("asset").reindex(target_assets)

    X_alpha = sub[ALPHA_FEATURE_COLS].values.astype(np.float32)
    X_risk = sub[RISK_FEATURE_COLS].values.astype(np.float32)
    return X_alpha, X_risk, target_assets
