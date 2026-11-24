#!/usr/bin/env python3
"""
src/train/exp015_tc_adaptive.py
===================================================================================
EXP-015: Transfer-Coefficient Diagnostics, Exponential Time-Decay Adaptation, and
Continuous Dispersion-Gated Portfolio Construction across the 4.17-year 15-coin panel
(2022-06-01 to 2026-08-31 UTC, 33,600 OOS hourly bars).

Stages:
  STAGE 0: Mandatory Baseline Reconciliation (locking EXP-013 / EXP-008d A0 + R2 math).
  STAGE 1: Alpha Decay Curve (12h-168h), 48h Quintile Monotonicity, & Transfer Coefficient.
  STAGE 2: Exponential Time-Decay Adaptation (W_unif / W_365d / W_180d) + Regime A1_reg.
  STAGE 3: Continuous EMA-Smoothed Nullspace (K1) & Causal Dispersion Gating (K2).
  STAGE 4: Apples-to-Apples Honest Vol-Matched Evaluation, 15-Split Purged CPCV,
           DSR (M=15, M=30), 1,000-Sample Circular Block Bootstrap, & Hard Kill Switch.
===================================================================================
"""

from __future__ import annotations

import logging
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "archive") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "archive"))

import apex_v45_champion as v45

from apex_multimodal_risk.src.serving.portfolio_math import project_beta_neutral
from apex_multimodal_risk.src.serving.tc_adaptive_math import (
    compute_causal_dispersion_scale,
    compute_exponential_time_decay_weights,
    compute_rank_gaussian_scores,
    simulate_continuous_ema_sleeve,
)
from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("exp015_tc_adaptive")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

np.random.seed(42)

HOURS_PER_YEAR = 8760.0
EULER_MASCHERONI = 0.5772156649015329

DATA_PATH = PROJECT_ROOT / "apex_multimodal_risk/data/processed/universe_15_microstructure_4yr.parquet"
CHART_PATH = PROJECT_ROOT / "apex_multimodal_risk/docs/exp015_tc_adaptive_analysis.png"

OOS_START = "2022-11-01 00:00:00+00:00"
WINDOW_A_END = "2025-05-23 23:00:00+00:00"
WINDOW_B_START = "2025-05-24 00:00:00+00:00"
OOS_END = "2026-08-31 23:00:00+00:00"

# 16 expanding quarterly refit boundaries; Window A uses the exact 10 EXP-013 quarters,
# and Window B aligns strictly at the 2025-05-24 untouched holdout boundary.
QUARTERLY_TEST_STARTS = [
    "2022-11-01 00:00:00+00:00",
    "2023-02-01 00:00:00+00:00",
    "2023-05-01 00:00:00+00:00",
    "2023-08-01 00:00:00+00:00",
    "2023-11-01 00:00:00+00:00",
    "2024-02-01 00:00:00+00:00",
    "2024-05-01 00:00:00+00:00",
    "2024-08-01 00:00:00+00:00",
    "2024-11-01 00:00:00+00:00",
    "2025-02-01 00:00:00+00:00",
    "2025-05-24 00:00:00+00:00",
    "2025-08-24 00:00:00+00:00",
    "2025-11-24 00:00:00+00:00",
    "2026-02-24 00:00:00+00:00",
    "2026-05-24 00:00:00+00:00",
    "2026-08-01 00:00:00+00:00",
]

# Exact EXP-013 / EXP-008d Baseline 20 Alpha Features
BASELINE_A0_FEATURES = [
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
    "vol_ratio_24_168",
    "vol_surge",
    "amihud_24h",
    "skew_72h",
    "skew_168h",
    "downside_ratio_72h",
]

# Pure Orthogonal Crypto Derivatives Microstructure Features (16)
ORTHO_A1_FEATURES = [
    "oi_usd_chg_24h",
    "oi_usd_chg_168h",
    "oi_to_vol_24h",
    "oi_price_divergence",
    "top_pos_ls_z7d",
    "global_acc_ls_z7d",
    "smart_retail_spread",
    "ofi_1h",
    "ofi_6h",
    "ofi_24h",
    "ofi_72h",
    "ofi_price_absorption",
    "whale_trade_size_surge",
    "basis_24h",
    "basis_z7d",
    "basis_funding_spread",
]

# Strictly 3 Slow Causal Market-Regime Features (broadcasted cross-sectionally at bar t)
REGIME_FEATURES = [
    "regime_btc_ret_30d_z",
    "regime_cs_disp_7d",
    "regime_btc_vol_30d_pct",
]

A1_REG_FEATURES = ORTHO_A1_FEATURES + REGIME_FEATURES

# Exact EXP-012 / EXP-013 13 Risk Features for R2 Dynamic Beta
RISK_R2_FEATURES = [
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

# Frozen EXP-008d / EXP-013 Huber LightGBM Alpha Hyperparameters (NO hyperparameter search)
LGB_ALPHA_PARAMS = {
    "objective": "huber",
    "max_depth": 3,
    "num_leaves": 7,
    "min_child_samples": 300,
    "colsample_bytree": 0.6,
    "reg_lambda": 10.0,
    "n_estimators": 60,
    "learning_rate": 0.03,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1,
}

# Frozen EXP-012 / EXP-013 R2 Dynamic Beta Hyperparameters
LGB_BETA_PARAMS = {
    "objective": "huber",
    "n_estimators": 75,
    "learning_rate": 0.04,
    "num_leaves": 12,
    "max_depth": 4,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 5.0,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1,
}


# =================================================================================
# 1. Statistical, Metric & Feature Engineering Helpers
# =================================================================================

def compute_hac_stats(x: np.ndarray, lags: int = 48) -> dict[str, float]:
    """Computes sample mean, naive SE, and Newey-West (Bartlett) HAC SE and t-stat."""
    clean_x = np.asarray(x, dtype=np.float64)
    clean_x = clean_x[~np.isnan(clean_x)]
    n = len(clean_x)
    if n <= lags + 2:
        return {"mean": 0.0, "naive_se": 0.0, "hac_se": 0.0, "naive_t": 0.0, "hac_t": 0.0}
    mean_val = float(np.mean(clean_x))
    e = clean_x - mean_val
    gamma_0 = float(np.mean(e**2))
    omega = gamma_0
    for j in range(1, lags + 1):
        weight = 1.0 - (j / (lags + 1.0))
        gamma_j = float(np.mean(e[j:] * e[:-j]))
        omega += 2.0 * weight * gamma_j
    omega = max(omega, 1e-12)
    naive_se = float(np.std(clean_x, ddof=1) / np.sqrt(n))
    hac_se = float(np.sqrt(omega / n))
    naive_t = mean_val / (naive_se + 1e-12)
    hac_t = mean_val / (hac_se + 1e-12)
    return {
        "mean": mean_val,
        "naive_se": naive_se,
        "hac_se": hac_se,
        "naive_t": naive_t,
        "hac_t": hac_t,
    }


def vectorized_cs_spearman(pred_mat: np.ndarray, targ_mat: np.ndarray) -> np.ndarray:
    """Computes cross-sectional Spearman rank correlation across axis=1 vectorially."""
    valid = ~np.isnan(targ_mat) & ~np.isnan(pred_mat)
    full_row = np.all(valid, axis=1)
    out = np.full(pred_mat.shape[0], np.nan, dtype=np.float64)
    if not np.any(full_row):
        return out

    rp = scipy_stats.rankdata(pred_mat[full_row], axis=1)
    rt = scipy_stats.rankdata(targ_mat[full_row], axis=1)
    rp_c = rp - np.mean(rp, axis=1, keepdims=True)
    rt_c = rt - np.mean(rt, axis=1, keepdims=True)
    denom = np.sqrt(np.sum(rp_c**2, axis=1) * np.sum(rt_c**2, axis=1))
    corr = np.sum(rp_c * rt_c, axis=1) / np.maximum(denom, 1e-12)
    out[full_row] = corr
    return out


def cs_zscore_matrix(mat: np.ndarray) -> np.ndarray:
    """Cross-sectionally z-scores predictions at every timestamp along axis=1."""
    mu = np.mean(mat, axis=1, keepdims=True)
    sig = np.std(mat, axis=1, keepdims=True)
    return (mat - mu) / np.maximum(sig, 1e-8)


def fast_metrics(
    h_rets: np.ndarray,
    turnover: np.ndarray | None = None,
    net_beta: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes annualized performance, risk, turnover, and beta leakage metrics."""
    n_hrs = len(h_rets)
    if n_hrs == 0:
        return {
            "cagr": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "max_dd": 0.0,
            "turnover": 0.0,
            "p95_leak": 0.0,
        }
    mu = float(np.mean(h_rets)) * HOURS_PER_YEAR
    vol = float(np.std(h_rets)) * np.sqrt(HOURS_PER_YEAR)
    sharpe = mu / (vol + 1e-12) if vol > 1e-10 else 0.0
    cum = np.cumprod(1.0 + h_rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / np.maximum(peak, 1e-12)
    max_dd = float(np.min(dd) * 100.0)
    n_years = n_hrs / HOURS_PER_YEAR
    cagr = float(((cum[-1] ** (1.0 / max(n_years, 0.05))) - 1.0) * 100.0) if cum[-1] > 0 else -100.0
    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-6 else 0.0
    ann_turn = float(np.sum(turnover) / max(n_years, 0.05)) if turnover is not None else 0.0

    p95_leak = 0.0
    if net_beta is not None and len(net_beta) > 24:
        roll_7d = pd.Series(net_beta).rolling(168, min_periods=24).mean().abs().dropna().values
        if len(roll_7d) > 0:
            p95_leak = float(np.percentile(roll_7d, 95))

    return {
        "cagr": cagr,
        "vol": vol * 100.0,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_dd": max_dd,
        "turnover": ann_turn,
        "p95_leak": p95_leak,
    }


def apply_honest_vol_targeting(
    r_raw: np.ndarray,
    slippage_bps: float,
    underlying_turnover: np.ndarray,
    target_vol: float = 0.1844,
    min_lev: float = 0.50,
    max_lev: float = 1.25,
    lookback_hours: int = 2160,
    deadband: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal 90-day realized volatility targeting using O(1) prefix sums (algebraically identical
    to rolling slice std(ddof=0) within < 1e-14), with 0.05 deadband and daily leverage turnover drag.
    """
    n_ts = len(r_raw)
    kappa = np.ones(n_ts, dtype=np.float64)
    cur_kappa = 1.0
    lev_turnover = np.zeros(n_ts, dtype=np.float64)

    cs1 = np.concatenate([[0.0], np.cumsum(r_raw)])
    cs2 = np.concatenate([[0.0], np.cumsum(r_raw**2)])
    sqrt_hyr = np.sqrt(HOURS_PER_YEAR)

    for t in range(168, n_ts, 24):
        t_start = max(0, t - lookback_hours)
        m = t - t_start
        mean_x = (cs1[t] - cs1[t_start]) / m
        var_x = max(0.0, (cs2[t] - cs2[t_start]) / m - mean_x * mean_x)
        hist_vol = float(np.sqrt(var_x) * sqrt_hyr)
        target_k = float(np.clip(target_vol / (hist_vol + 1e-6), min_lev, max_lev))
        if abs(target_k - cur_kappa) > deadband:
            lev_turnover[t] = abs(target_k - cur_kappa)
            cur_kappa = target_k
        kappa[t : min(t + 24, n_ts)] = cur_kappa

    underlying_drag = kappa * underlying_turnover * (slippage_bps / 10000.0)
    leverage_drag = lev_turnover * (slippage_bps / 10000.0)
    r_scaled = kappa * r_raw - underlying_drag - leverage_drag
    total_turnover = kappa * underlying_turnover + lev_turnover
    return r_scaled, total_turnover, kappa


def simulate_k0_7tranche(
    pred_mat: np.ndarray,
    ret_1h_mat: np.ndarray,
    fr_mat: np.ndarray,
    vol_mat: np.ndarray,
    beta_mat: np.ndarray,
    ts_index: pd.DatetimeIndex,
    max_weight: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """K0 Legacy Constructor: exact EXP-013 / EXP-008d 7-tranche weekly binary top-4/bottom-4 P1."""
    n_ts, n_assets = pred_mat.shape
    sub_weights = [np.zeros((n_ts, n_assets), dtype=np.float64) for _ in range(7)]

    for k in range(7):
        current_longs: set[int] = set()
        current_shorts: set[int] = set()
        current_w = np.zeros(n_assets, dtype=np.float64)

        for t in range(n_ts):
            if (t - 24 * k) >= 0 and (t - 24 * k) % 168 == 0:
                scores = pred_mat[t]
                vols = np.maximum(vol_mat[t], 1e-4)
                ranked_indices = np.argsort(scores)

                top_6 = set(ranked_indices[-6:])
                bot_6 = set(ranked_indices[:6])

                if len(current_longs) > 0 and len(current_shorts) > 0:
                    new_longs = {i for i in current_longs if i in top_6}
                    new_shorts = {i for i in current_shorts if i in bot_6}
                    for idx in reversed(ranked_indices):
                        if len(new_longs) == 4:
                            break
                        if idx not in new_shorts:
                            new_longs.add(idx)
                    for idx in ranked_indices:
                        if len(new_shorts) == 4:
                            break
                        if idx not in new_longs:
                            new_shorts.add(idx)
                else:
                    new_shorts = set(ranked_indices[:4])
                    new_longs = set(ranked_indices[-4:])

                current_longs = new_longs
                current_shorts = new_shorts

                w0 = np.zeros(n_assets, dtype=np.float64)
                inv_vols = 1.0 / vols
                long_sum = sum(inv_vols[i] for i in current_longs)
                short_sum = sum(inv_vols[i] for i in current_shorts)
                for idx in current_longs:
                    w0[idx] = +0.50 * (inv_vols[idx] / long_sum)
                for idx in current_shorts:
                    w0[idx] = -0.50 * (inv_vols[idx] / short_sum)

                current_w = project_beta_neutral(w0, beta_mat[t], gross_cap=1.0, max_weight=max_weight)

            sub_weights[k][t] = current_w

    weights = np.mean(sub_weights, axis=0)
    pos_shifted = np.zeros_like(weights)
    pos_shifted[1:, :] = weights[:-1, :]
    turnover = np.sum(
        np.abs(np.diff(np.concatenate([np.zeros((1, n_assets)), pos_shifted], axis=0), axis=0)),
        axis=1,
    )

    funding_hours = ts_index.hour.isin([0, 8, 16])
    fund_pnl = np.where(funding_hours, np.sum(-pos_shifted * fr_mat, axis=1), 0.0)
    price_pnl = np.sum(pos_shifted * ret_1h_mat, axis=1)
    gross_pnl = price_pnl + fund_pnl
    return gross_pnl, turnover, pos_shifted, weights


def engineer_panel_features_and_matrices() -> dict[str, Any]:
    """
    Loads universe_15_microstructure_4yr.parquet and constructs:
      - Exact EXP-013 / EXP-008d A0 features & volatility-standardized target_total_48h
      - Exact 16 Pure Orthogonal Microstructure A1 features
      - Strictly 3 slow causal market-regime state variables (not cross-sectionally ranked)
      - Exact EXP-012 / EXP-013 13 R2 dynamic beta features & strictly forward 48h OLS beta target
      - Multi-horizon forward total returns (12h, 24h, 48h, 72h, 120h, 168h) for Stage 1 decay
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing required microstructure dataset at {DATA_PATH}")

    logger.info(f"Loading 4.17-year microstructure panel from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    logger.info("Computing baseline A0, orthogonal A1, and multi-horizon targets...")
    horizons_stage1 = [12, 24, 48, 72, 120, 168]
    feature_dfs = []

    for _asset, group in df.groupby("asset"):
        g = group.copy().sort_values("timestamp").reset_index(drop=True)
        c = g["close"].values
        v = g["volume"].values
        qv = g["quote_volume"].values
        tc = g["trade_count"].values
        tbv = g["taker_buy_base_vol"].values
        fr = g["funding_rate"].values
        prem = g["premium_close"].values
        oi_val = g["sum_open_interest_value"].values
        top_ls = g["sum_toptrader_long_short_ratio"].values
        glob_ls = g["count_long_short_ratio"].values

        h = g["timestamp"].dt.hour.values
        high_vals = g["high"].values
        low_vals = g["low"].values
        eff_fr = np.where(np.isin(h, [0, 8, 16]), fr, 0.0)

        ret_1h = np.concatenate([[0.0], (c[1:] / c[:-1]) - 1.0])
        g["ret_1h"] = ret_1h
        g["raw_ret_6h"] = pd.Series(c).pct_change(6).fillna(0.0).values
        g["raw_ret_24h"] = pd.Series(c).pct_change(24).fillna(0.0).values

        for w in [72, 168, 336, 720]:
            rw = pd.Series(c).pct_change(w).fillna(0.0).values
            vw = pd.Series(ret_1h).rolling(w, min_periods=max(12, w // 4)).std().fillna(0.0).values * np.sqrt(w)
            g[f"mom_{w}h"] = rw
            g[f"ram_{w}h"] = np.tanh(rw / (vw + 1e-6))

        high_168 = pd.Series(high_vals).rolling(168, min_periods=24).max().ffill().fillna(high_vals[0]).values
        low_168 = pd.Series(low_vals).rolling(168, min_periods=24).min().ffill().fillna(low_vals[0]).values
        g["dist_168h_hl"] = (c - low_168) / (high_168 - low_168 + 1e-6)

        g["fr_level"] = fr
        g["fr_carry_72h"] = pd.Series(fr).rolling(72, min_periods=12).sum().fillna(0.0).values
        fr_mu = pd.Series(fr).rolling(720, min_periods=72).mean().ffill().fillna(0.0).values
        fr_sig = pd.Series(fr).rolling(720, min_periods=72).std().ffill().fillna(1.0).values
        g["fr_zscore_30d"] = np.clip((fr - fr_mu) / (fr_sig + 1e-8), -4.0, 4.0)

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

        # Orthogonal Derivatives Microstructure (16 features)
        s_oi = pd.Series(oi_val)
        g["oi_usd_chg_24h"] = (s_oi / s_oi.shift(24) - 1.0).fillna(0.0).values
        g["oi_usd_chg_168h"] = (s_oi / s_oi.shift(168) - 1.0).fillna(0.0).values
        s_qv_24 = pd.Series(qv).rolling(24, min_periods=12).sum().fillna(1e5)
        g["oi_to_vol_24h"] = (s_oi / (s_qv_24 + 1e-5)).fillna(0.0).values
        g["oi_price_divergence"] = np.sign(g["raw_ret_24h"].values) * g["oi_usd_chg_24h"].values

        s_top = pd.Series(top_ls)
        top_mu = s_top.rolling(168, min_periods=24).mean().ffill().fillna(1.0)
        top_sig = s_top.rolling(168, min_periods=24).std().ffill().fillna(0.1)
        g["top_pos_ls_z7d"] = np.clip((s_top - top_mu) / (top_sig + 1e-5), -3.0, 3.0).values

        s_glob = pd.Series(glob_ls)
        glob_mu = s_glob.rolling(168, min_periods=24).mean().ffill().fillna(1.0)
        glob_sig = s_glob.rolling(168, min_periods=24).std().ffill().fillna(0.1)
        g["global_acc_ls_z7d"] = np.clip((s_glob - glob_mu) / (glob_sig + 1e-5), -3.0, 3.0).values
        g["raw_top_ls"] = top_ls
        g["raw_glob_ls"] = glob_ls

        ofi_raw = np.clip(2.0 * (tbv / (v + 1e-5)) - 1.0, -1.0, 1.0)
        g["ofi_1h"] = ofi_raw
        g["ofi_6h"] = pd.Series(ofi_raw).rolling(6, min_periods=2).mean().fillna(0.0).values
        g["ofi_24h"] = pd.Series(ofi_raw).rolling(24, min_periods=6).mean().fillna(0.0).values
        g["ofi_72h"] = pd.Series(ofi_raw).rolling(72, min_periods=12).mean().fillna(0.0).values

        s_tc_24 = pd.Series(tc).rolling(24, min_periods=12).sum().fillna(1.0)
        s_qv_168 = pd.Series(qv).rolling(168, min_periods=24).sum().fillna(1e5)
        s_tc_168 = pd.Series(tc).rolling(168, min_periods=24).sum().fillna(1.0)
        size_24 = s_qv_24 / (s_tc_24 + 1.0)
        size_168 = s_qv_168 / (s_tc_168 + 1.0)
        g["whale_trade_size_surge"] = np.clip(size_24 / (size_168 + 1e-5), 0.1, 10.0).values

        s_prem = pd.Series(prem)
        g["basis_24h"] = s_prem.rolling(24, min_periods=6).mean().fillna(0.0).values
        b_mu = s_prem.rolling(168, min_periods=24).mean().ffill().fillna(0.0)
        b_sig = s_prem.rolling(168, min_periods=24).std().ffill().fillna(0.001)
        g["basis_z7d"] = np.clip((s_prem - b_mu) / (b_sig + 1e-5), -3.0, 3.0).values
        fr_mean_24 = pd.Series(fr).rolling(24, min_periods=6).mean().fillna(0.0)
        g["basis_funding_spread"] = (s_prem - fr_mean_24).values

        # Forward Multi-Horizon Total Returns (12h, 24h, 48h, 72h, 120h, 168h)
        cs = np.cumsum(np.concatenate([[0.0], eff_fr]))
        n_pts = len(eff_fr)
        for H in horizons_stage1:
            fwd_p = (pd.Series(c).shift(-H) / c - 1.0).values
            idx_end = np.minimum(np.arange(n_pts) + H + 1, len(cs) - 1)
            idx_start = np.minimum(np.arange(n_pts) + 1, len(cs) - 1)
            fwd_fund = cs[idx_end] - cs[idx_start]
            fwd_fund[np.arange(n_pts) + H >= n_pts] = np.nan
            g[f"fwd_total_{H}h"] = fwd_p - fwd_fund

        feature_dfs.append(g)

    df_panel = pd.concat(feature_dfs, axis=0).sort_values(["timestamp", "asset"]).reset_index(drop=True)

    # Cross-sectional baseline residuals (exact EXP-008d / EXP-013 math)
    mkt_ret_6h = df_panel.groupby("timestamp")["raw_ret_6h"].transform("mean")
    mkt_ret_24h = df_panel.groupby("timestamp")["raw_ret_24h"].transform("mean")
    df_panel["rev_res_6h"] = df_panel["raw_ret_6h"] - mkt_ret_6h
    df_panel["rev_res_24h"] = df_panel["raw_ret_24h"] - mkt_ret_24h

    # Cross-sectional orthogonal divergences
    rk_top = df_panel.groupby("timestamp")["raw_top_ls"].rank(method="average")
    rk_glob = df_panel.groupby("timestamp")["raw_glob_ls"].rank(method="average")
    df_panel["smart_retail_spread"] = (2.0 * (rk_top - 1.0) / 14.0 - 1.0) - (2.0 * (rk_glob - 1.0) / 14.0 - 1.0)

    rk_ofi = df_panel.groupby("timestamp")["ofi_24h"].rank(method="average")
    rk_ret = df_panel.groupby("timestamp")["raw_ret_24h"].rank(method="average")
    df_panel["ofi_price_absorption"] = (2.0 * (rk_ofi - 1.0) / 14.0 - 1.0) - (2.0 * (rk_ret - 1.0) / 14.0 - 1.0)

    # Volatility-standardized, median-demeaned 48h training target (EXACT EXP-013 math!)
    med_val = df_panel.groupby("timestamp")["fwd_total_48h"].transform("median")
    df_panel["target_total_48h"] = (df_panel["fwd_total_48h"] - med_val) / (df_panel["trailing_vol_168h"] + 1e-6)

    # Rank-normalize cross-sectional alpha features [-1, +1] (excluding broadcasted regime vars)
    cs_rank_cols = sorted(list(set(BASELINE_A0_FEATURES + ORTHO_A1_FEATURES)))
    ranks_cs = df_panel.groupby("timestamp")[cs_rank_cols].rank(method="average")
    df_panel[cs_rank_cols] = 2.0 * (ranks_cs - 1.0) / 14.0 - 1.0

    # Matrix dimensions for risk features, causal dispersion, regime features & backtests
    timestamps = sorted(df_panel["timestamp"].unique())
    assets = sorted(df_panel["asset"].unique())
    ts_index = pd.DatetimeIndex(timestamps)
    n_ts = len(timestamps)
    n_assets = len(assets)
    btc_idx = assets.index("BTCUSDT")

    ret_df = df_panel.pivot(index="timestamp", columns="asset", values="ret_1h").reindex(
        index=timestamps, columns=assets
    ).fillna(0.0)
    ret_1h_mat = ret_df.values
    fr_mat = df_panel.pivot(index="timestamp", columns="asset", values="funding_rate").reindex(
        index=timestamps, columns=assets
    ).fillna(0.0001).values
    vol_mat = df_panel.pivot(index="timestamp", columns="asset", values="trailing_vol_168h").reindex(
        index=timestamps, columns=assets
    ).fillna(1e-4).values

    # Exact EXP-012 / EXP-013 causal R2 risk features
    logger.info("Computing causal EXP-013 R2 dynamic beta features & strictly forward 48h OLS beta target...")
    beta_mats: dict[int, np.ndarray] = {}
    corr_mats: dict[int, np.ndarray] = {}
    for w in [24, 72, 168, 720]:
        min_p = 72 if w >= 72 else max(12, w // 4)
        cov_b = ret_df.rolling(w, min_periods=min_p).cov(ret_df["BTCUSDT"])
        var_b = ret_df["BTCUSDT"].rolling(w, min_periods=min_p).var()
        bm = (cov_b.div(var_b + 1e-8, axis=0)).clip(0.3, 2.5).fillna(1.0).values
        bm[:, btc_idx] = 1.0
        beta_mats[w] = bm
        if w in [72, 168]:
            cm = ret_df.rolling(w, min_periods=min_p).corr(ret_df["BTCUSDT"]).clip(-1.0, 1.0).fillna(0.0).values
            corr_mats[w] = cm

    vol_168_df = ret_df.rolling(168, min_periods=24).std().fillna(1e-4)
    vol_btc_168 = vol_168_df["BTCUSDT"].values[:, None]
    vol_ratio_btc_mat = np.clip(vol_168_df.values / (vol_btc_168 + 1e-6), 0.2, 5.0)
    vol_24_df = ret_df.rolling(24, min_periods=12).std().fillna(1e-4)
    vol_expansion_mat = np.clip(vol_24_df.values / (vol_168_df.values + 1e-6), 0.2, 5.0)

    cross_disp = np.std(ret_1h_mat, axis=1)
    pc1_abs = np.full(n_ts, 0.50, dtype=np.float64)
    for ti in range(720, n_ts, 24):
        cov = np.cov(ret_1h_mat[ti - 720 : ti], rowvar=False)
        evals = np.maximum(np.linalg.eigvalsh(cov), 0.0)
        tot = np.sum(evals)
        if tot > 1e-12:
            pc1_abs[ti : min(ti + 24, n_ts)] = float(evals[-1] / tot)

    # Strictly forward realized 48h OLS beta: ret_1h_mat[ti + 1 : ti + 49]
    fwd_beta_48 = np.full((n_ts, n_assets), np.nan, dtype=np.float64)
    for ti in range(n_ts - 48):
        fwd_w = ret_1h_mat[ti + 1 : ti + 49]
        b_col = fwd_w[:, btc_idx]
        var_b = float(np.var(b_col))
        if var_b > 1e-8:
            b_cent = b_col - np.mean(b_col)
            w_cent = fwd_w - np.mean(fwd_w, axis=0)
            covs = np.mean(w_cent * b_cent[:, None], axis=0)
            betas = np.clip(covs / var_b, 0.3, 2.5)
            betas[btc_idx] = 1.0
            fwd_beta_48[ti] = betas
        else:
            fwd_beta_48[ti] = 1.0

    # Strictly 3 Slow Causal Market-Regime State Variables for A1_reg
    btc_r = pd.Series(ret_1h_mat[:, btc_idx])
    btc_ret_30d = btc_r.rolling(720, min_periods=72).sum()
    reg_ret_mu = btc_ret_30d.expanding(min_periods=168).mean()
    reg_ret_sig = btc_ret_30d.expanding(min_periods=168).std().fillna(0.1)
    reg_btc_ret_30d_z = np.clip(((btc_ret_30d - reg_ret_mu) / (reg_ret_sig + 1e-6)).fillna(0.0).values, -4.0, 4.0)

    # Trailing 7d CS dispersion + causal K2 scale
    gross_scale_k2, trailing_7d_disp, expanding_med_disp = compute_causal_dispersion_scale(ret_1h_mat)

    # Expanding-past BTC 30d realized volatility percentile [0, 1]
    btc_vol_30d = btc_r.rolling(720, min_periods=72).std().fillna(1e-4).values * np.sqrt(HOURS_PER_YEAR)
    btc_vol_pct = np.full(n_ts, 0.5, dtype=np.float64)
    for ti in range(168, n_ts):
        past = btc_vol_30d[: ti + 1]
        btc_vol_pct[ti] = float(np.mean(past <= btc_vol_30d[ti]))

    ts_map = {ts: idx for idx, ts in enumerate(timestamps)}
    asset_map = {a: idx for idx, a in enumerate(assets)}
    row_ts = df_panel["timestamp"].map(ts_map).values
    row_asset = df_panel["asset"].map(asset_map).values

    df_panel["beta_24h"] = beta_mats[24][row_ts, row_asset]
    df_panel["beta_72h"] = beta_mats[72][row_ts, row_asset]
    df_panel["beta_168h"] = beta_mats[168][row_ts, row_asset]
    df_panel["beta_720h"] = beta_mats[720][row_ts, row_asset]
    df_panel["vol_ratio_btc"] = vol_ratio_btc_mat[row_ts, row_asset]
    df_panel["corr_btc_72h"] = corr_mats[72][row_ts, row_asset]
    df_panel["corr_btc_168h"] = corr_mats[168][row_ts, row_asset]
    df_panel["vol_expansion_24_168"] = vol_expansion_mat[row_ts, row_asset]
    df_panel["cross_dispersion"] = cross_disp[row_ts]
    df_panel["pc1_absorption"] = pc1_abs[row_ts]
    df_panel["beta_fwd_48h"] = fwd_beta_48[row_ts, row_asset]

    df_panel["regime_btc_ret_30d_z"] = reg_btc_ret_30d_z[row_ts]
    df_panel["regime_cs_disp_7d"] = trailing_7d_disp[row_ts]
    df_panel["regime_btc_vol_30d_pct"] = btc_vol_pct[row_ts]

    # Forward multi-horizon total return matrices (n_ts, n_assets)
    fwd_mats = {
        H: df_panel.pivot(index="timestamp", columns="asset", values=f"fwd_total_{H}h")
        .reindex(index=timestamps, columns=assets)
        .values
        for H in horizons_stage1
    }

    return {
        "df_panel": df_panel,
        "timestamps": timestamps,
        "ts_index": ts_index,
        "assets": assets,
        "ts_map": ts_map,
        "asset_map": asset_map,
        "btc_idx": btc_idx,
        "ret_1h_mat": ret_1h_mat,
        "fr_mat": fr_mat,
        "vol_mat": vol_mat,
        "fwd_beta_48": fwd_beta_48,
        "fwd_mats": fwd_mats,
        "gross_scale_k2": gross_scale_k2,
        "trailing_7d_disp": trailing_7d_disp,
        "expanding_med_disp": expanding_med_disp,
    }


# =================================================================================
# 2. Walk-Forward Training Engine (16 Quarterly Refits, Strictly Causal Embargo)
# =================================================================================

def run_walkforward_models(data_bundle: dict[str, Any]) -> dict[str, Any]:
    """
    Trains frozen R2 Dynamic Beta AND all alpha priors across 16 quarterly expanding refits
    with 168h embargo and n_jobs=1.
    """
    logger.info("Executing 16-Quarter Expanding Walk-Forward Training (168h Embargo, n_jobs=1)...")
    t0 = time.time()

    df_panel = data_bundle["df_panel"]
    ts_index = data_bundle["ts_index"]
    assets = data_bundle["assets"]
    ts_map = data_bundle["ts_map"]
    asset_map = data_bundle["asset_map"]
    btc_idx = data_bundle["btc_idx"]
    n_ts, n_assets = len(ts_index), len(assets)

    model_specs = {
        "A0_W_unif": (BASELINE_A0_FEATURES, None),
        "A0_W_365d": (BASELINE_A0_FEATURES, 365.0),
        "A0_W_180d": (BASELINE_A0_FEATURES, 180.0),
        "A1_W_unif": (ORTHO_A1_FEATURES, None),
        "A1_W_365d": (ORTHO_A1_FEATURES, 365.0),
        "A1_W_180d": (ORTHO_A1_FEATURES, 180.0),
        "A1_reg_W_unif": (A1_REG_FEATURES, None),
        "A1_reg_W_365d": (A1_REG_FEATURES, 365.0),
        "A1_reg_W_180d": (A1_REG_FEATURES, 180.0),
    }

    preds = {name: np.zeros((n_ts, n_assets), dtype=np.float64) for name in model_specs}
    preds["placebo_shuffle"] = np.zeros((n_ts, n_assets), dtype=np.float64)
    pred_beta_r2 = np.ones((n_ts, n_assets), dtype=np.float64)

    rng_placebo = np.random.RandomState(42)
    oos_end_ts = pd.Timestamp(OOS_END)

    for q_idx, q_start_str in enumerate(QUARTERLY_TEST_STARTS):
        t_start = pd.Timestamp(q_start_str)
        t_end = pd.Timestamp(QUARTERLY_TEST_STARTS[q_idx + 1]) if q_idx + 1 < len(QUARTERLY_TEST_STARTS) else oos_end_ts
        t_train_end = t_start - pd.Timedelta(hours=168)  # 168h embargo

        train_mask = (
            (df_panel["timestamp"] >= pd.Timestamp("2022-07-01 00:00:00+00:00"))
            & (df_panel["timestamp"] <= t_train_end)
            & ~df_panel["target_total_48h"].isnull()
        )
        if q_idx == len(QUARTERLY_TEST_STARTS) - 1:
            test_mask = (df_panel["timestamp"] >= t_start) & (df_panel["timestamp"] <= t_end)
        else:
            test_mask = (df_panel["timestamp"] >= t_start) & (df_panel["timestamp"] < t_end)

        df_tr = df_panel.loc[train_mask]
        df_te = df_panel.loc[test_mask]
        y_tr_alpha = df_tr["target_total_48h"].values

        age_days = (t_train_end - df_tr["timestamp"]).dt.total_seconds().values / 86400.0
        sub_ts = df_te["timestamp"].map(ts_map).values
        sub_as = df_te["asset"].map(asset_map).values

        # Train each alpha configuration
        for name, (feats, hl_days) in model_specs.items():
            sw = compute_exponential_time_decay_weights(age_days, hl_days)
            mdl = lgb.LGBMRegressor(**LGB_ALPHA_PARAMS)
            mdl.fit(df_tr[feats].values, y_tr_alpha, sample_weight=sw)
            preds[name][sub_ts, sub_as] = mdl.predict(df_te[feats].values)

        # Feature-shuffle null placebo
        X_shuf = df_tr[BASELINE_A0_FEATURES].values.copy()
        perm_idx = rng_placebo.permutation(len(X_shuf))
        mdl_pl = lgb.LGBMRegressor(**LGB_ALPHA_PARAMS)
        mdl_pl.fit(X_shuf[perm_idx], y_tr_alpha)
        preds["placebo_shuffle"][sub_ts, sub_as] = mdl_pl.predict(df_te[BASELINE_A0_FEATURES].values)

        # Train Frozen R2 Dynamic Beta
        beta_mask = train_mask & ~df_panel["beta_fwd_48h"].isnull()
        X_tr_b = df_panel.loc[beta_mask, RISK_R2_FEATURES].values
        y_tr_b = df_panel.loc[beta_mask, "beta_fwd_48h"].values
        m_beta = lgb.LGBMRegressor(**LGB_BETA_PARAMS)
        m_beta.fit(X_tr_b, y_tr_b)
        p_beta = m_beta.predict(df_te[RISK_R2_FEATURES].values)
        p_beta_clamped = np.where(sub_as == btc_idx, 1.0, np.clip(p_beta, 0.3, 2.5))
        pred_beta_r2[sub_ts, sub_as] = p_beta_clamped

    logger.info(f"16 quarterly refits completed in {time.time() - t0:.1f}s")
    return {"preds": preds, "pred_beta_r2": pred_beta_r2}


# =================================================================================
# 3. Stage 1 Diagnostics, Anti-Overfitting Tests, & Chart Generation
# =================================================================================

def print_stage1_diagnostics(
    preds: dict[str, np.ndarray],
    pred_beta_r2: np.ndarray,
    data_bundle: dict[str, Any],
    wA_idx: np.ndarray,
    wB_idx: np.ndarray,
) -> None:
    """Computes and prints Alpha Decay Curve (12h-168h), Quintile Monotonicity, & Transfer Coefficient."""
    print("\n" + "=" * 120)
    print("STAGE 1: ALPHA DECAY CURVE (GROSS 0 BPS), QUINTILE MONOTONICITY & MEASURED TRANSFER COEFFICIENT")
    print("=" * 120)

    fwd_mats = data_bundle["fwd_mats"]
    horizons = [12, 24, 48, 72, 120, 168]

    for sig_label, pred_mat in [("A0_W_unif (20 OHLCV)", preds["A0_W_unif"]), ("A1_W_unif (16 Ortho)", preds["A1_W_unif"])]:
        print(f"\n--- Alpha Decay Curve: {sig_label} ---")
        print(f"{'Horizon':<8} | {'Win A IC':>9} | {'Win A HAC-t':>11} | {'Win A T4-B4':>12} || {'Win B IC':>9} | {'Win B HAC-t':>11} | {'Win B T4-B4':>12}")
        print("-" * 100)
        for H in horizons:
            targ = fwd_mats[H]
            ic_vec = vectorized_cs_spearman(pred_mat, targ)
            rk_pred = scipy_stats.rankdata(pred_mat, axis=1)  # 1..15
            top4_mask = rk_pred >= 12.0
            bot4_mask = rk_pred <= 4.0
            spread_vec = np.where(
                np.any(np.isnan(targ), axis=1),
                np.nan,
                np.mean(np.where(top4_mask, targ, 0.0), axis=1) * (15.0 / 4.0)
                - np.mean(np.where(bot4_mask, targ, 0.0), axis=1) * (15.0 / 4.0),
            )
            ann_mult = (HOURS_PER_YEAR / float(H)) * 100.0
            nw_a = compute_hac_stats(ic_vec[wA_idx], lags=max(24, H))
            nw_b = compute_hac_stats(ic_vec[wB_idx], lags=max(24, H))
            sp_a = float(np.nanmean(spread_vec[wA_idx]) * ann_mult)
            sp_b = float(np.nanmean(spread_vec[wB_idx]) * ann_mult)
            print(
                f"{f'{H}h':<8} | {nw_a['mean']:>+9.4f} | {nw_a['hac_t']:>+11.2f} | {sp_a:>+11.2f}% || "
                f"{nw_b['mean']:>+9.4f} | {nw_b['hac_t']:>+11.2f} | {sp_b:>+11.2f}%"
            )

    # 48h Gross Total Return Quintile Monotonicity
    print("\n--- 48h Gross Total Return Quintile Monotonicity (Annualized %) ---")
    targ_48 = fwd_mats[48]
    ann_48 = (HOURS_PER_YEAR / 48.0) * 100.0
    print(f"{'Model / Window':<28} | {'Q1 (Low)':>10} | {'Q2':>10} | {'Q3':>10} | {'Q4':>10} | {'Q5 (High)':>10} | {'Q5 - Q1':>10}")
    print("-" * 104)
    for sig_label, pred_mat in [("A0_W_unif", preds["A0_W_unif"]), ("A1_W_unif", preds["A1_W_unif"])]:
        rk_pred = scipy_stats.rankdata(pred_mat, axis=1)
        valid_rows = ~np.any(np.isnan(targ_48), axis=1)
        for win_name, idx in [("Window A", wA_idx), ("Window B", wB_idx)]:
            sub_idx = idx[valid_rows[idx]]
            r_sub = rk_pred[sub_idx]
            t_sub = targ_48[sub_idx]
            qs = []
            for q in range(5):
                lo, hi = 1.0 + q * 3.0, 3.0 + q * 3.0
                mask_q = (r_sub >= lo) & (r_sub <= hi)
                q_ret = float(np.mean(np.sum(np.where(mask_q, t_sub, 0.0), axis=1) / 3.0) * ann_48)
                qs.append(q_ret)
            print(
                f"{f'{sig_label} ({win_name})':<28} | {qs[0]:>+9.2f}% | {qs[1]:>+9.2f}% | {qs[2]:>+9.2f}% | "
                f"{qs[3]:>+9.2f}% | {qs[4]:>+9.2f}% | {qs[4]-qs[0]:>+9.2f}%"
            )

    # Measured Transfer Coefficient (TC = mean cs corr(w_projected, alpha_hat))
    print("\n--- Measured Transfer Coefficient TC = corr(w_projected, alpha_hat) ---")
    oos_idx = np.concatenate([wA_idx, wB_idx])
    for sig_label, pred_mat in [("A0_W_unif", preds["A0_W_unif"]), ("A1_W_unif", preds["A1_W_unif"])]:
        _, _, _, w_k0 = simulate_k0_7tranche(
            pred_mat[oos_idx],
            data_bundle["ret_1h_mat"][oos_idx],
            data_bundle["fr_mat"][oos_idx],
            data_bundle["vol_mat"][oos_idx],
            pred_beta_r2[oos_idx],
            data_bundle["ts_index"][oos_idx],
        )
        z_mat = compute_rank_gaussian_scores(pred_mat[oos_idx])
        w_star_mat = np.array(
            [
                project_beta_neutral(z_mat[t], pred_beta_r2[oos_idx[t]], gross_cap=1.0, max_weight=0.15)
                for t in range(len(oos_idx))
            ]
        )
        _, _, _, w_k1 = simulate_continuous_ema_sleeve(
            pred_mat[oos_idx],
            data_bundle["ret_1h_mat"][oos_idx],
            data_bundle["fr_mat"][oos_idx],
            pred_beta_r2[oos_idx],
            data_bundle["ts_index"][oos_idx],
            rebal_hours=12,
            ema_halflife=48.0,
            deadband=0.015,
        )

        def _mean_tc(w_m: np.ndarray, a_m: np.ndarray) -> float:
            corrs = []
            for t in range(168, len(w_m)):
                if np.std(w_m[t]) > 1e-8 and np.std(a_m[t]) > 1e-8:
                    corrs.append(float(np.corrcoef(w_m[t], a_m[t])[0, 1]))
            return float(np.mean(corrs))

        tc_k0 = _mean_tc(w_k0, pred_mat[oos_idx])
        tc_star = _mean_tc(w_star_mat, pred_mat[oos_idx])
        tc_k1 = _mean_tc(w_k1, pred_mat[oos_idx])
        print(
            f"  {sig_label:<22}: K0 Binary 7-Tranche TC = {tc_k0:+.4f} | "
            f"Continuous Nullspace Target w* TC = {tc_star:+.4f} | K1 EMA(48h) Realized TC = {tc_k1:+.4f}"
        )


def compute_dsr_and_psr(
    ann_sharpe: float,
    h_rets: np.ndarray,
    n_trials: int = 15,
    var_trials: float = 0.08,
) -> tuple[float, float, float]:
    """Bailey & Lopez de Prado PSR & DSR adjusted for non-normal skew/kurtosis and trial count M."""
    clean_rets = h_rets[~np.isnan(h_rets)]
    n_obs = len(clean_rets)
    sk = float(scipy_stats.skew(clean_rets)) if n_obs > 2 else 0.0
    kt = float(scipy_stats.kurtosis(clean_rets, fisher=False)) if n_obs > 3 else 3.0
    term1 = HOURS_PER_YEAR
    term2 = sk * ann_sharpe * np.sqrt(HOURS_PER_YEAR)
    term3 = ((kt - 1.0) / 4.0) * (ann_sharpe**2)
    var_sr = max(1e-8, (term1 - term2 + term3) / max(n_obs - 1.0, 1.0))
    se = float(np.sqrt(var_sr))

    psr = float(scipy_stats.norm.cdf(ann_sharpe / se))
    t1 = (1.0 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    t2 = EULER_MASCHERONI * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_bench = float(np.sqrt(max(1e-6, var_trials)) * (t1 + t2))
    dsr = float(scipy_stats.norm.cdf((ann_sharpe - sr_bench) / se))
    return psr, dsr, sr_bench


def run_cpcv_best_candidate(
    data_bundle: dict[str, Any],
    best_cfg: dict[str, Any],
    champ_oos: np.ndarray,
    oos_idx: np.ndarray,
) -> dict[str, Any]:
    """
    Runs 15-Split Purged CPCV (N=6, k=2 -> 5 full stitched 4.17y paths) with strict
    bidirectional 168h purge + embargo on the promoted Stage 3/4 candidate.
    """
    logger.info(f"Running 15-Split Purged CPCV on promoted candidate ({best_cfg['label']})...")
    df_panel = data_bundle["df_panel"]
    ts_index = data_bundle["ts_index"]
    assets = data_bundle["assets"]
    ts_map = data_bundle["ts_map"]
    asset_map = data_bundle["asset_map"]
    btc_idx = data_bundle["btc_idx"]
    n_oos, n_assets = len(oos_idx), len(assets)

    n_blocks = 6
    block_size = n_oos // n_blocks
    blocks = [oos_idx[i * block_size : (i + 1) * block_size] for i in range(n_blocks)]
    split_combos = list(combinations(range(n_blocks), 2))

    # Map local OOS timestamp indices
    split_a0 = np.zeros((15, n_oos, n_assets), dtype=np.float64)
    split_a1r = np.zeros((15, n_oos, n_assets), dtype=np.float64)
    split_beta = np.ones((15, n_oos, n_assets), dtype=np.float64)

    a0_hl = best_cfg["a0_hl"]
    a1r_hl = best_cfg["a1r_hl"]

    for s_idx, (b1, b2) in enumerate(split_combos):
        test_block_ids = {b1, b2}
        test_hours = np.concatenate([blocks[b1], blocks[b2]])
        test_ts_set = set(ts_index[test_hours])

        # Bidirectional 168h purging + embargo relative to each test block
        valid_train_ts = []
        for ti in oos_idx:
            t = ts_index[ti]
            ok = True
            for b_id in test_block_ids:
                b_start = ts_index[blocks[b_id][0]]
                b_end = ts_index[blocks[b_id][-1]]
                if (t >= b_start - pd.Timedelta(hours=168)) and (t <= b_end + pd.Timedelta(hours=168)):
                    ok = False
                    break
            if ok:
                valid_train_ts.append(t)

        train_mask = df_panel["timestamp"].isin(set(valid_train_ts)) & ~df_panel["target_total_48h"].isnull()
        test_mask = df_panel["timestamp"].isin(test_ts_set)

        df_tr = df_panel.loc[train_mask]
        df_te = df_panel.loc[test_mask]
        y_tr_a = df_tr["target_total_48h"].values
        t_max_tr = df_tr["timestamp"].max()
        age_days = (t_max_tr - df_tr["timestamp"]).dt.total_seconds().values / 86400.0

        m_a0 = lgb.LGBMRegressor(**LGB_ALPHA_PARAMS)
        m_a0.fit(df_tr[BASELINE_A0_FEATURES].values, y_tr_a, sample_weight=compute_exponential_time_decay_weights(age_days, a0_hl))
        p_a0 = m_a0.predict(df_te[BASELINE_A0_FEATURES].values)

        m_a1r = lgb.LGBMRegressor(**LGB_ALPHA_PARAMS)
        m_a1r.fit(df_tr[A1_REG_FEATURES].values, y_tr_a, sample_weight=compute_exponential_time_decay_weights(age_days, a1r_hl))
        p_a1r = m_a1r.predict(df_te[A1_REG_FEATURES].values)

        beta_mask = train_mask & ~df_panel["beta_fwd_48h"].isnull()
        m_b = lgb.LGBMRegressor(**LGB_BETA_PARAMS)
        m_b.fit(df_panel.loc[beta_mask, RISK_R2_FEATURES].values, df_panel.loc[beta_mask, "beta_fwd_48h"].values)
        p_b = m_b.predict(df_te[RISK_R2_FEATURES].values)

        sub_t = df_te["timestamp"].map(ts_map).values - oos_idx[0]
        sub_a = df_te["asset"].map(asset_map).values
        split_a0[s_idx, sub_t, sub_a] = p_a0
        split_a1r[s_idx, sub_t, sub_a] = p_a1r
        split_beta[s_idx, sub_t, sub_a] = np.where(sub_a == btc_idx, 1.0, np.clip(p_b, 0.3, 2.5))

    path_pairings = [
        [(0, 1), (2, 3), (4, 5)],
        [(0, 2), (1, 4), (3, 5)],
        [(0, 3), (1, 5), (2, 4)],
        [(0, 4), (1, 3), (2, 5)],
        [(0, 5), (1, 2), (3, 4)],
    ]

    sub_ret = data_bundle["ret_1h_mat"][oos_idx]
    sub_fr = data_bundle["fr_mat"][oos_idx]
    sub_vol = data_bundle["vol_mat"][oos_idx]
    sub_ts = ts_index[oos_idx]
    g_scale = data_bundle["gross_scale_k2"][oos_idx] if best_cfg["constructor"] == "K2" else None
    hl_ema = best_cfg["ema_hl"]

    path_results = []
    for p_id, pairs in enumerate(path_pairings):
        stitched_a0 = np.zeros((n_oos, n_assets), dtype=np.float64)
        stitched_a1r = np.zeros((n_oos, n_assets), dtype=np.float64)
        stitched_beta = np.ones((n_oos, n_assets), dtype=np.float64)

        for b_a, b_b in pairs:
            s_i = split_combos.index((min(b_a, b_b), max(b_a, b_b)))
            for b_id in (b_a, b_b):
                s0, s1 = b_id * block_size, (b_id + 1) * block_size
                stitched_a0[s0:s1] = split_a0[s_i, s0:s1]
                stitched_a1r[s0:s1] = split_a1r[s_i, s0:s1]
                stitched_beta[s0:s1] = split_beta[s_i, s0:s1]

        if best_cfg["signal"] == "Stack_50_50":
            stitched_pred = 0.5 * cs_zscore_matrix(stitched_a0) + 0.5 * cs_zscore_matrix(stitched_a1r)
        elif best_cfg["signal"] == "A1_reg":
            stitched_pred = stitched_a1r
        else:
            stitched_pred = stitched_a0

        if best_cfg["constructor"] == "K0":
            gross, turn, _, _ = simulate_k0_7tranche(stitched_pred, sub_ret, sub_fr, sub_vol, stitched_beta, sub_ts)
        else:
            gross, turn, _, _ = simulate_continuous_ema_sleeve(
                stitched_pred,
                sub_ret,
                sub_fr,
                stitched_beta,
                sub_ts,
                rebal_hours=12,
                ema_halflife=hl_ema,
                deadband=0.015,
                gross_scale_arr=g_scale,
            )

        r10 = gross - turn * (10.0 / 10000.0)
        raw_b90 = 0.90 * champ_oos + 0.10 * gross
        r10_b90, _, _ = apply_honest_vol_targeting(raw_b90, 10.0, 0.10 * turn)
        path_results.append(
            {
                "path_id": p_id + 1,
                "cum_r10": np.cumprod(1.0 + r10),
                "sr_stand": fast_metrics(r10, turn)["sharpe"],
                "sr_blend90": fast_metrics(r10_b90)["sharpe"],
            }
        )

    return {"paths": path_results}


def run_circular_bootstrap(
    r_stand_10: np.ndarray,
    gross_stand: np.ndarray,
    turn_stand: np.ndarray,
    champ_oos: np.ndarray,
    n_boot: int = 1000,
    block_len: int = 168,
) -> dict[str, Any]:
    """Paired 1,000-sample circular block bootstrap comparing 90/10 Honest VM vs 100% APEX Honest VM."""
    logger.info("Running 1,000-sample Paired Circular Block Bootstrap (168h blocks)...")
    n = len(r_stand_10)
    n_blocks = n // block_len
    rng = np.random.RandomState(42)

    boot_stand_sr = []
    boot_delta_sr = []
    sqrt_hyr = np.sqrt(HOURS_PER_YEAR)

    for _ in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = np.concatenate([(s + np.arange(block_len)) % n for s in starts])[:n]

        sub_st = r_stand_10[idx]
        sr_st = float((np.mean(sub_st) * HOURS_PER_YEAR) / (np.std(sub_st) * sqrt_hyr + 1e-12))
        boot_stand_sr.append(sr_st)

        sub_ch = champ_oos[idx]
        sub_gr = gross_stand[idx]
        sub_tu = turn_stand[idx]

        r10_ap_vm, _, _ = apply_honest_vol_targeting(sub_ch, 10.0, np.zeros(n))
        r10_b90_vm, _, _ = apply_honest_vol_targeting(0.90 * sub_ch + 0.10 * sub_gr, 10.0, 0.10 * sub_tu)

        sr_ap = float((np.mean(r10_ap_vm) * HOURS_PER_YEAR) / (np.std(r10_ap_vm) * sqrt_hyr + 1e-12))
        sr_b90 = float((np.mean(r10_b90_vm) * HOURS_PER_YEAR) / (np.std(r10_b90_vm) * sqrt_hyr + 1e-12))
        boot_delta_sr.append(sr_b90 - sr_ap)

    stand_arr = np.array(boot_stand_sr)
    delta_arr = np.array(boot_delta_sr)
    return {
        "stand_mean": float(np.mean(stand_arr)),
        "stand_ci": (float(np.percentile(stand_arr, 2.5)), float(np.percentile(stand_arr, 97.5))),
        "p_stand_gt_0": float(np.mean(stand_arr <= 0.0)),
        "delta_mean": float(np.mean(delta_arr)),
        "delta_ci": (float(np.percentile(delta_arr, 2.5)), float(np.percentile(delta_arr, 97.5))),
        "p_delta_gt_0": float(np.mean(delta_arr <= 0.0)),
    }


def plot_exp015_chart(
    data_bundle: dict[str, Any],
    preds: dict[str, np.ndarray],
    sleeves: dict[str, dict[str, Any]],
    best_key: str,
    cpcv_paths: list[dict[str, Any]],
    r10_apex_vm: np.ndarray,
    r10_blend90_vm: np.ndarray,
    oos_idx: np.ndarray,
    wA_idx: np.ndarray,
    wB_idx: np.ndarray,
) -> None:
    """Generates clean 4-panel EXP-015 diagnostic & validation chart."""
    logger.info(f"Generating 4-panel diagnostic chart at {CHART_PATH}...")
    ts_oos = data_bundle["ts_index"][oos_idx]
    fwd_mats = data_bundle["fwd_mats"]
    horizons = [12, 24, 48, 72, 120, 168]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=200)
    plt.subplots_adjust(hspace=0.28, wspace=0.20)

    # Panel 1: 12h-168h Alpha Decay Curve (Window A vs Window B)
    ax1 = axes[0, 0]
    for label, p_mat, color, ls in [
        ("A0 Baseline (Window A)", preds["A0_W_unif"], "#1f77b4", "-"),
        ("A0 Baseline (Window B Holdout)", preds["A0_W_unif"], "#1f77b4", "--"),
        ("A1 Pure Ortho (Window A)", preds["A1_W_unif"], "#2ca02c", "-"),
        ("A1 Pure Ortho (Window B Holdout)", preds["A1_W_unif"], "#2ca02c", "--"),
    ]:
        use_idx = wA_idx if "Window A" in label else wB_idx
        ics = [float(np.nanmean(vectorized_cs_spearman(p_mat, fwd_mats[h])[use_idx])) for h in horizons]
        ax1.plot(horizons, ics, marker="o", color=color, ls=ls, lw=1.8, label=label)
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax1.set_title("Panel 1: Cross-Sectional Rank IC Alpha Decay (12h to 168h)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Forward Horizon (Hours)")
    ax1.set_ylabel("Mean Spearman Rank IC")
    ax1.set_xticks(horizons)
    ax1.legend(loc="best", fontsize=8.5)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Standalone 4.17Y Equity Curves (10 bps) with Window B Shaded
    ax2 = axes[0, 1]
    plot_keys = [
        ("K0_A0_base", "K0 Legacy (A0)", "#7f7f7f", "--"),
        ("K1_48_A0_base", "K1 EMA48 (A0)", "#1f77b4", "-."),
        ("K0_Stack_50_50", "K0 Legacy (Stack 50/50)", "#ff7f0e", "--"),
        ("K1_48_Stack_50_50", "K1 EMA48 (Stack 50/50)", "#9467bd", "-"),
        ("K2_48_Stack_50_50", "K2 Disp-Gated EMA48 (Stack 50/50)", "#2ca02c", "-"),
    ]
    for k, lbl, col, ls in plot_keys:
        if k in sleeves:
            eq = np.cumprod(1.0 + sleeves[k]["r10"])
            sr = fast_metrics(sleeves[k]["r10"])["sharpe"]
            ax2.plot(ts_oos, eq, label=f"{lbl} (SR {sr:+.2f})", color=col, ls=ls, lw=1.7)
    holdout_dt = pd.Timestamp(WINDOW_B_START)
    ax2.axvspan(holdout_dt, ts_oos[-1], color="gray", alpha=0.18, label="Holdout Window B")
    ax2.axvline(holdout_dt, color="red", ls=":", lw=1.4)
    ax2.set_title("Panel 2: Standalone 4.17Y Cumulative Equity (10 bps Slippage)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Growth of $1.00")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: 5 Stitched CPCV Combinatorial Paths
    ax3 = axes[1, 0]
    c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, p in enumerate(cpcv_paths):
        ax3.plot(ts_oos, p["cum_r10"], color=c_list[i % len(c_list)], lw=1.6, label=f"Path {p['path_id']} (SR {p['sr_stand']:+.2f})")
    ax3.axvspan(holdout_dt, ts_oos[-1], color="gray", alpha=0.15)
    ax3.set_title(f"Panel 3: 5 Stitched Purged CPCV Paths ({best_key}, 10 bps)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Path Growth of $1.00")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Causal Dispersion Gross-Scale Timeline + Honest VM Drawdown Comparison
    ax4 = axes[1, 1]
    g_scale = data_bundle["gross_scale_k2"][oos_idx]
    eq_ap = np.cumprod(1.0 + r10_apex_vm)
    dd_ap = (eq_ap - np.maximum.accumulate(eq_ap)) / np.maximum(np.maximum.accumulate(eq_ap), 1e-12) * 100.0
    eq_bl = np.cumprod(1.0 + r10_blend90_vm)
    dd_bl = (eq_bl - np.maximum.accumulate(eq_bl)) / np.maximum(np.maximum.accumulate(eq_bl), 1e-12) * 100.0

    ax4.plot(ts_oos, dd_ap, color="#ff7f0e", lw=1.4, label=f"100% APEX Honest VM MaxDD ({np.min(dd_ap):.1f}%)")
    ax4.plot(ts_oos, dd_bl, color="#1f77b4", lw=1.6, label=f"90/10 {best_key} Honest VM MaxDD ({np.min(dd_bl):.1f}%)")
    ax4.axvspan(holdout_dt, ts_oos[-1], color="gray", alpha=0.15)
    ax4.set_ylabel("Portfolio Drawdown (%)", fontsize=10)
    ax4_r = ax4.twinx()
    ax4_r.plot(ts_oos, g_scale, color="#2ca02c", alpha=0.65, lw=1.2, label="Causal K2 Gross Scale [0.25, 1.0]")
    ax4_r.set_ylabel("K2 Dispersion Gross Scale", color="#2ca02c", fontsize=10)
    ax4_r.set_ylim(0.15, 1.10)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    ax4.set_title("Panel 4: Honest VM Drawdown Compression & K2 Dispersion Scale", fontsize=12, fontweight="bold")
    ax4.grid(True, alpha=0.3)

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CHART_PATH, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved EXP-015 diagnostic chart to {CHART_PATH}")


# =================================================================================
# 4. Main Orchestration (Stages 0 through 4)
# =================================================================================

def main() -> None:
    t_all = time.time()
    print("=" * 125)
    print("EXP-015: TRANSFER-COEFFICIENT DIAGNOSTICS, TIME-DECAY ADAPTATION & CONTINUOUS DISPERSION-GATED PORTFOLIOS")
    print("=" * 125)

    data_bundle = engineer_panel_features_and_matrices()
    wf = run_walkforward_models(data_bundle)
    preds = wf["preds"]
    beta_r2 = wf["pred_beta_r2"]

    ts_index = data_bundle["ts_index"]
    ret_1h = data_bundle["ret_1h_mat"]
    fr_mat = data_bundle["fr_mat"]
    vol_mat = data_bundle["vol_mat"]
    fwd_beta = data_bundle["fwd_beta_48"]
    g_scale = data_bundle["gross_scale_k2"]

    oos_mask = (ts_index >= pd.Timestamp(OOS_START)) & (ts_index <= pd.Timestamp(OOS_END))
    wA_mask = (ts_index >= pd.Timestamp(OOS_START)) & (ts_index <= pd.Timestamp(WINDOW_A_END))
    wB_mask = (ts_index >= pd.Timestamp(WINDOW_B_START)) & (ts_index <= pd.Timestamp(OOS_END))

    oos_idx = np.where(oos_mask)[0]
    wA_idx = np.where(wA_mask)[0]
    wB_idx = np.where(wB_mask)[0]

    # Local masks inside OOS slice (len = 33,600)
    sub_ts = ts_index[oos_idx]
    m_A_rel = sub_ts <= pd.Timestamp(WINDOW_A_END)
    m_B_rel = sub_ts >= pd.Timestamp(WINDOW_B_START)
    m_C_rel = np.ones(len(oos_idx), dtype=bool)

    # Frozen APEX v45 Baseline stream over OOS (unlevered and honest vol-matched)
    ds_v45, pr_v45, gw_v45, ev_v45, n_v45, _, mr_v45, sk_v45, arb_v45 = v45.load_data_and_features()
    dir_s, _ = v45.run_directional_engine(pr_v45, ds_v45, v45.UNIVERSE, ev_v45, n_v45, gw_v45, sk_v45)
    sp_s, _ = v45.run_spread_engine(pr_v45, ds_v45, v45.UNIVERSE, ev_v45, n_v45)
    arb_s, _ = v45.run_funding_arb_engine(ds_v45, v45.UNIVERSE, ev_v45, n_v45, arb_v45)
    champ_full = 0.25 * dir_s * mr_v45 + 0.55 * sp_s * mr_v45 + 0.20 * arb_s
    champ_oos = pd.Series(champ_full, index=ev_v45).reindex(sub_ts).fillna(0.0).values
    r10_apex_vm, _, _ = apply_honest_vol_targeting(champ_oos, 10.0, np.zeros(len(champ_oos)))

    # -------------------------------------------------------------------------
    # STAGE 0: MANDATORY BASELINE RECONCILIATION
    # -------------------------------------------------------------------------
    sub_ret = ret_1h[oos_idx]
    sub_fr = fr_mat[oos_idx]
    sub_vol = vol_mat[oos_idx]
    sub_b_r2 = beta_r2[oos_idx]
    sub_b_true = np.nan_to_num(fwd_beta[oos_idx], nan=1.0)
    sub_g_scale = g_scale[oos_idx]

    g_s0, t_s0, pos_s0, _ = simulate_k0_7tranche(preds["A0_W_unif"][oos_idx], sub_ret, sub_fr, sub_vol, sub_b_r2, sub_ts)
    r10_s0 = g_s0 - t_s0 * (10.0 / 10000.0)
    m_s0_wA = fast_metrics(r10_s0[m_A_rel], t_s0[m_A_rel], np.sum(pos_s0[m_A_rel] * sub_b_true[m_A_rel], axis=1))

    print("\n" + "=" * 125)
    print("STAGE 0: MANDATORY BASELINE RECONCILIATION (EXP-014 0.36 vs EXP-013 0.84 IN WINDOW A)")
    print("=" * 125)
    print("1. Target Scaling: EXP-014 removed volatility scaling (CS_median + /trailing_vol_168h), skewing Huber gradients to high-vol alts.")
    print("2. Alpha Features/Hypers: EXP-014 changed Huber n_estimators(100 vs 60), colsample(0.5 vs 0.6), lambda(15 vs 10) & Holdout boundary.")
    print("3. R2 Beta Target Bug: EXP-014 fit beta on trailing 720h rolling cov of 48h returns & noisy features vs EXP-013 causal forward 48h OLS beta.")
    print(f"--> VERIFIED LOCKED BASELINE A0 + R2 (Window A, 10 bps): Sharpe = {m_s0_wA['sharpe']:.2f} | CAGR = {m_s0_wA['cagr']:.2f}% | MaxDD = {m_s0_wA['max_dd']:.1f}% | Turn = {m_s0_wA['turnover']:.1f}x")
    if abs(m_s0_wA["sharpe"] - 0.84) > 0.10:
        logger.warning(f"Stage 0 Window A Sharpe ({m_s0_wA['sharpe']:.2f}) drifted beyond tolerance from 0.84 benchmark!")

    # -------------------------------------------------------------------------
    # STAGE 1: TRANSFER-COEFFICIENT & ALPHA DECAY DIAGNOSTICS
    # -------------------------------------------------------------------------
    print_stage1_diagnostics(preds, beta_r2, data_bundle, wA_idx, wB_idx)

    # -------------------------------------------------------------------------
    # STAGE 2: ADAPTIVE TIME-DECAY & REGIME-CONDITIONED ORTHOGONAL ALPHA
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print("STAGE 2: ADAPTIVE TIME-DECAY & REGIME-CONDITIONED ORTHOGONAL ALPHA (NEWEY-WEST HAC IC, L=48)")
    print("=" * 125)
    targ48 = data_bundle["fwd_mats"][48]

    # Pick causal best priors strictly on Window A OOS IC (zero Window B snooping!)
    a0_priors = ["A0_W_unif", "A0_W_365d", "A0_W_180d"]
    a1r_priors = ["A1_reg_W_unif", "A1_reg_W_365d", "A1_reg_W_180d"]

    ic_stats_store = {}
    for name, p_mat in preds.items():
        ic_vec = vectorized_cs_spearman(p_mat, targ48)
        ic_stats_store[name] = {
            "wA": compute_hac_stats(ic_vec[wA_idx], lags=48),
            "wB": compute_hac_stats(ic_vec[wB_idx], lags=48),
            "wC": compute_hac_stats(ic_vec[oos_idx], lags=48),
        }

    best_a0_key = max(a0_priors, key=lambda k: ic_stats_store[k]["wA"]["mean"])
    best_a1r_key = max(a1r_priors, key=lambda k: ic_stats_store[k]["wA"]["mean"])

    # Form Stack_50_50 from Window-A-selected priors
    preds["Stack_50_50"] = 0.5 * cs_zscore_matrix(preds[best_a0_key]) + 0.5 * cs_zscore_matrix(preds[best_a1r_key])
    ic_stack = vectorized_cs_spearman(preds["Stack_50_50"], targ48)
    ic_stats_store["Stack_50_50"] = {
        "wA": compute_hac_stats(ic_stack[wA_idx], lags=48),
        "wB": compute_hac_stats(ic_stack[wB_idx], lags=48),
        "wC": compute_hac_stats(ic_stack[oos_idx], lags=48),
    }

    print(f"Causal Window-A Selected Priors: A0 -> {best_a0_key} | A1_reg -> {best_a1r_key}")
    print(f"{'Model / Prior':<22} | {'Win A IC (HAC-t)':>18} | {'Win B IC (HAC-t)':>18} | {'Full C IC (HAC-t)':>18}")
    print("-" * 86)
    print_order = [
        "A0_W_unif", "A0_W_365d", "A0_W_180d",
        "A1_W_unif", "A1_W_365d", "A1_W_180d",
        "A1_reg_W_unif", "A1_reg_W_365d", "A1_reg_W_180d",
        "Stack_50_50", "placebo_shuffle",
    ]
    for name in print_order:
        st = ic_stats_store[name]
        sA = f"{st['wA']['mean']:+.4f} ({st['wA']['hac_t']:+.2f})"
        sB = f"{st['wB']['mean']:+.4f} ({st['wB']['hac_t']:+.2f})"
        sC = f"{st['wC']['mean']:+.4f} ({st['wC']['hac_t']:+.2f})"
        print(f"{name:<22} | {sA:>18} | {sB:>18} | {sC:>18}")

    # -------------------------------------------------------------------------
    # STAGE 3 & 4: CONSTRUCTORS K0, K1 (hl=48/72), K2 (hl=48/72) & MULTI-WINDOW TEAR SHEETS
    # -------------------------------------------------------------------------
    hl_map = {"unif": None, "365d": 365.0, "180d": 180.0}
    best_a0_hl = hl_map[best_a0_key.split("_W_")[-1]]
    best_a1r_hl = hl_map[best_a1r_key.split("_W_")[-1]]

    signal_map = {
        "A0_base": ("A0_W_unif", None, best_a1r_hl),
        "A1_reg": (best_a1r_key, best_a0_hl, best_a1r_hl),
        "Stack_50_50": ("Stack_50_50", best_a0_hl, best_a1r_hl),
    }

    constructor_configs = [
        ("K0", None, "K0 (Legacy 7-Tranche P1)"),
        ("K1_48", 48.0, "K1 (EMA 48h Nullspace)"),
        ("K1_72", 72.0, "K1 (EMA 72h Nullspace)"),
        ("K2_48", 48.0, "K2 (Disp-Gated EMA 48h)"),
        ("K2_72", 72.0, "K2 (Disp-Gated EMA 72h)"),
    ]

    sleeves: dict[str, dict[str, Any]] = {}
    for c_key, ema_hl, _c_label in constructor_configs:
        for s_key, (pred_name, _a0_hl, _a1r_hl) in signal_map.items():
            full_key = f"{c_key}_{s_key}"
            p_sub = preds[pred_name][oos_idx]
            if c_key == "K0":
                gross, turn, pos, _ = simulate_k0_7tranche(p_sub, sub_ret, sub_fr, sub_vol, sub_b_r2, sub_ts)
            else:
                g_arr = sub_g_scale if c_key.startswith("K2") else None
                gross, turn, pos, _ = simulate_continuous_ema_sleeve(
                    p_sub, sub_ret, sub_fr, sub_b_r2, sub_ts, rebal_hours=12, ema_halflife=float(ema_hl), deadband=0.015, gross_scale_arr=g_arr
                )

            net_beta = np.sum(pos * sub_b_true, axis=1)
            r0 = gross.copy()
            r10 = gross - turn * (10.0 / 10000.0)
            r25 = gross - turn * (25.0 / 10000.0)
            b90, t90, _ = apply_honest_vol_targeting(0.90 * champ_oos + 0.10 * gross, 10.0, 0.10 * turn)
            b80, t80, _ = apply_honest_vol_targeting(0.80 * champ_oos + 0.20 * gross, 10.0, 0.20 * turn)

            sleeves[full_key] = {
                "c_type": "K0" if c_key == "K0" else ("K2" if c_key.startswith("K2") else "K1"),
                "ema_hl": ema_hl,
                "signal": s_key,
                "a0_hl": best_a0_hl if s_key != "A0_base" else None,
                "a1r_hl": best_a1r_hl,
                "gross": gross,
                "turn": turn,
                "pos": pos,
                "net_beta": net_beta,
                "r0": r0,
                "r10": r10,
                "r25": r25,
                "b90": b90,
                "t90": t90,
                "b80": b80,
                "t80": t80,
                "pred_sub": p_sub,
            }

    print("\n" + "=" * 145)
    print("STAGE 4.1: STRICT APPLES-TO-APPLES MULTI-WINDOW TEAR SHEETS (10 BPS PRINCIPAL, 0 & 25 BPS SENSITIVITY)")
    print("=" * 145)

    windows_eval = [
        ("Window A (Audited: 2022-11-01 to 2025-05-23 | 22,440h)", m_A_rel),
        ("Window B (UNTOUCHED HOLDOUT: 2025-05-24 to 2026-08-31 | 11,160h)", m_B_rel),
        ("Window C (Full 4.17Y OOS: 2022-11-01 to 2026-08-31 | 33,600h)", m_C_rel),
    ]

    for win_title, w_m in windows_eval:
        print(f"\n--- {win_title} ---")
        m_ap_unlev = fast_metrics(champ_oos[w_m])
        m_ap_vm = fast_metrics(r10_apex_vm[w_m])
        print(
            f"{'Strategy / Blend':<26} | {'CAGR':>8} | {'Vol':>7} | {'SR(0)':>6} | {'SR(10)':>7} | {'SR(25)':>7} | "
            f"{'Calmar':>7} | {'MaxDD':>7} | {'Turn':>7} | {'95%Leak':>8} | {'rho':>7} | {'90/10 VM':>8} | {'80/20 VM':>8}"
        )
        print("-" * 145)
        print(
            f"{'100% APEX v45 [Unlevered]':<26} | {m_ap_unlev['cagr']:>+7.2f}% | {m_ap_unlev['vol']:>6.2f}% | {'—':>6} | "
            f"{m_ap_unlev['sharpe']:>+7.2f} | {'—':>7} | {m_ap_unlev['calmar']:>7.2f} | {m_ap_unlev['max_dd']:>6.1f}% | "
            f"{'0.0x':>7} | {'—':>8} | {'1.0000':>7} | {'—':>8} | {'—':>8}"
        )
        print(
            f"{'100% APEX v45 [Honest VM]':<26} | {m_ap_vm['cagr']:>+7.2f}% | {m_ap_vm['vol']:>6.2f}% | {'—':>6} | "
            f"{m_ap_vm['sharpe']:>+7.2f} | {'—':>7} | {m_ap_vm['calmar']:>7.2f} | {m_ap_vm['max_dd']:>6.1f}% | "
            f"{'0.0x':>7} | {'—':>8} | {'1.0000':>7} | {m_ap_vm['sharpe']:>+8.2f} | {m_ap_vm['sharpe']:>+8.2f}"
        )

        for key, sl in sleeves.items():
            m0 = fast_metrics(sl["r0"][w_m], sl["turn"][w_m])
            m10 = fast_metrics(sl["r10"][w_m], sl["turn"][w_m], sl["net_beta"][w_m])
            m25 = fast_metrics(sl["r25"][w_m], sl["turn"][w_m])
            mb90 = fast_metrics(sl["b90"][w_m], sl["t90"][w_m])
            mb80 = fast_metrics(sl["b80"][w_m], sl["t80"][w_m])
            rho = float(np.corrcoef(sl["r10"][w_m], r10_apex_vm[w_m])[0, 1])
            print(
                f"{key:<26} | {m10['cagr']:>+7.2f}% | {m10['vol']:>6.2f}% | {m0['sharpe']:>+6.2f} | {m10['sharpe']:>+7.2f} | "
                f"{m25['sharpe']:>+7.2f} | {m10['calmar']:>7.2f} | {m10['max_dd']:>6.1f}% | {m10['turnover']:>6.1f}x | "
                f"{m10['p95_leak']:>8.4f} | {rho:>+7.4f} | {mb90['sharpe']:>+8.2f} | {mb80['sharpe']:>+8.2f}"
            )

    # Select best candidate among new K1/K2 constructors STRICTLY via Window A Sharpe (no Window B peeking!)
    k12_keys = [k for k in sleeves if k.startswith("K1_") or k.startswith("K2_")]
    best_key = max(k12_keys, key=lambda k: fast_metrics(sleeves[k]["r10"][m_A_rel])["sharpe"])
    best_sl = sleeves[best_key]

    # Year-by-Year Net Sharpe (2022-2026) at 10 bps
    print("\n" + "=" * 115)
    print(f"YEAR-BY-YEAR NET SHARPE COMPARISON (10 BPS) — BEST CAUSAL CANDIDATE: {best_key}")
    print("=" * 115)
    print(f"{'Year':<10} | {'Bars':>6} | {'APEX [Unlev]':>13} | {'APEX [Honest VM]':>17} | {'Stand ' + best_key:>20} | {'90/10 Blend VM':>15} | {'80/20 Blend VM':>15}")
    print("-" * 115)
    for yr in [2022, 2023, 2024, 2025, 2026]:
        ym = sub_ts.year == yr
        if np.any(ym):
            sr_u = fast_metrics(champ_oos[ym])["sharpe"]
            sr_v = fast_metrics(r10_apex_vm[ym])["sharpe"]
            sr_s = fast_metrics(best_sl["r10"][ym])["sharpe"]
            sr_9 = fast_metrics(best_sl["b90"][ym])["sharpe"]
            sr_8 = fast_metrics(best_sl["b80"][ym])["sharpe"]
            print(f"{yr:<10} | {int(np.sum(ym)):>6} | {sr_u:>+13.2f} | {sr_v:>+17.2f} | {sr_s:>+20.2f} | {sr_9:>+15.2f} | {sr_8:>+15.2f}")

    # -------------------------------------------------------------------------
    # STAGE 4.2: FULL ANTI-OVERFITTING TORTURE BATTERY ON BEST CANDIDATE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 125)
    print(f"STAGE 4.2: ANTI-OVERFITTING TORTURE BATTERY — CANDIDATE {best_key}")
    print("=" * 125)

    # 1. 15-Split Purged CPCV
    best_cfg = {
        "label": best_key,
        "constructor": best_sl["c_type"],
        "ema_hl": best_sl["ema_hl"],
        "signal": best_sl["signal"],
        "a0_hl": best_sl["a0_hl"],
        "a1r_hl": best_sl["a1r_hl"],
    }
    cpcv_out = run_cpcv_best_candidate(data_bundle, best_cfg, champ_oos, oos_idx)
    cpcv_paths = cpcv_out["paths"]
    cpcv_stand_srs = [p["sr_stand"] for p in cpcv_paths]
    cpcv_blend_srs = [p["sr_blend90"] for p in cpcv_paths]
    sr_apex_vm_full = fast_metrics(r10_apex_vm)["sharpe"]

    print("1. 15-Split Purged CPCV (N=6, k=2, Bidirectional 168h Purge/Embargo, 5 Stitched 4.17Y Paths):")
    print(f"   Standalone 10 bps Path Sharpes: {[round(float(x), 3) for x in cpcv_stand_srs]}")
    print(f"   -> Min = {np.min(cpcv_stand_srs):+.2f} | Median = {np.median(cpcv_stand_srs):+.2f} | Mean = {np.mean(cpcv_stand_srs):+.2f} | % Positive = {np.mean([x > 0 for x in cpcv_stand_srs]) * 100:.1f}%")
    print(f"   90/10 Honest VM Path Sharpes:   {[round(float(x), 3) for x in cpcv_blend_srs]} vs APEX Honest VM ({sr_apex_vm_full:+.2f})")

    # 2. Bailey & Lopez de Prado DSR (M=15 and M=30)
    all_stand_srs_c = [fast_metrics(sleeves[k]["r10"])["sharpe"] for k in sleeves]
    emp_var_trials = max(0.05, float(np.var(all_stand_srs_c)))
    sr_best_full = fast_metrics(best_sl["r10"])["sharpe"]
    sr_b90_full = fast_metrics(best_sl["b90"])["sharpe"]

    psr_s, dsr_s15, sb15 = compute_dsr_and_psr(sr_best_full, best_sl["r10"], n_trials=15, var_trials=emp_var_trials)
    _, dsr_s30, sb30 = compute_dsr_and_psr(sr_best_full, best_sl["r10"], n_trials=30, var_trials=emp_var_trials)
    psr_b, dsr_b15, bb15 = compute_dsr_and_psr(sr_b90_full, best_sl["b90"], n_trials=15, var_trials=0.06)
    _, dsr_b30, bb30 = compute_dsr_and_psr(sr_b90_full, best_sl["b90"], n_trials=30, var_trials=0.06)

    print("\n2. Bailey & Lopez de Prado DSR / PSR:")
    print(f"   Standalone ({best_key}): PSR = {psr_s*100:.2f}% | DSR(M=15, SR*={sb15:.2f}) = {dsr_s15*100:.2f}% | DSR(M=30, SR*={sb30:.2f}) = {dsr_s30*100:.2f}%")
    print(f"   90/10 Blend Honest VM:   PSR = {psr_b*100:.2f}% | DSR(M=15, SR*={bb15:.2f}) = {dsr_b15*100:.2f}% | DSR(M=30, SR*={bb30:.2f}) = {dsr_b30*100:.2f}%")

    # 3. Circular Block Bootstrap (1,000 resamples, 168h blocks)
    boot = run_circular_bootstrap(best_sl["r10"], best_sl["gross"], best_sl["turn"], champ_oos)
    print("\n3. Paired Circular Block Bootstrap (1,000 Resamples, 168h Blocks):")
    print(f"   (a) Standalone 10 bps Sharpe > 0:      Mean = {boot['stand_mean']:+.2f}, 95% CI = [{boot['stand_ci'][0]:+.2f}, {boot['stand_ci'][1]:+.2f}], one-sided p = {boot['p_stand_gt_0']:.4f}")
    print(f"   (b) Marginal Blend Lift Delta_SR > 0: Mean = {boot['delta_mean']:+.3f}, 95% CI = [{boot['delta_ci'][0]:+.3f}, {boot['delta_ci'][1]:+.3f}], one-sided p = {boot['p_delta_gt_0']:.4f}")

    # 4. Execution Latency (1h/2h/4h), Friction Breakeven (bps), & Shuffle Placebo
    mean_g = float(np.mean(best_sl["gross"]))
    mean_t = float(np.mean(best_sl["turn"]))
    breakeven_bps = float((mean_g / max(mean_t, 1e-12)) * 10000.0)

    lat_srs = {}
    pos_base = best_sl["pos"]
    for extra_lag in [0, 1, 3]:  # total lag 1h, 2h, 4h relative to weights
        pos_l = np.zeros_like(pos_base)
        if extra_lag == 0:
            pos_l = pos_base
        else:
            pos_l[extra_lag:, :] = pos_base[:-extra_lag, :]
        t_l = np.sum(np.abs(np.diff(np.concatenate([np.zeros((1, pos_l.shape[1])), pos_l], axis=0), axis=0)), axis=1)
        f_h = sub_ts.hour.isin([0, 8, 16])
        g_l = np.sum(pos_l * sub_ret, axis=1) + np.where(f_h, np.sum(-pos_l * sub_fr, axis=1), 0.0)
        r10_l = g_l - t_l * (10.0 / 10000.0)
        lat_srs[1 + extra_lag] = fast_metrics(r10_l)["sharpe"]

    # Shuffle placebo under K1/K2 constructor
    g_pl, t_pl, _, _ = simulate_continuous_ema_sleeve(
        preds["placebo_shuffle"][oos_idx], sub_ret, sub_fr, sub_b_r2, sub_ts, rebal_hours=12, ema_halflife=float(best_sl["ema_hl"]), deadband=0.015, gross_scale_arr=(sub_g_scale if best_sl["c_type"] == "K2" else None)
    )
    sr_placebo_10 = fast_metrics(g_pl - t_pl * (10.0 / 10000.0))["sharpe"]

    print("\n4. Execution Latency, Friction Breakeven & Shuffle Placebo:")
    print(f"   Friction Breakeven (One-Way): {breakeven_bps:.2f} bps")
    print(f"   Latency Net Sharpe (10 bps):  1h = {lat_srs[1]:+.2f} | 2h = {lat_srs[2]:+.2f} | 4h = {lat_srs[4]:+.2f}")
    print(f"   Shuffle Placebo Net Sharpe:   {sr_placebo_10:+.2f}")

    # Generate 4-panel chart
    plot_exp015_chart(
        data_bundle,
        preds,
        sleeves,
        best_key,
        cpcv_paths,
        r10_apex_vm,
        best_sl["b90"],
        oos_idx,
        wA_idx,
        wB_idx,
    )

    # -------------------------------------------------------------------------
    # RUTHLESS FINAL VERDICT & HARD KILL SWITCH
    # -------------------------------------------------------------------------
    wB_sr = fast_metrics(best_sl["r10"][m_B_rel])["sharpe"]
    gate1 = (sr_best_full >= 0.75) and (wB_sr > 0.0)
    gate2 = all(x > 0.0 for x in cpcv_stand_srs) and (float(np.median(cpcv_stand_srs)) >= 0.65)
    gate3 = (dsr_s15 * 100.0) >= 80.0
    gate4 = (boot["delta_mean"] > 0.0) and (boot["p_delta_gt_0"] < 0.10)

    print("\n" + "=" * 125)
    print("RUTHLESS HARD KILL SWITCH DIAGNOSTIC")
    print("=" * 125)
    print(f"  Gate 1 [Full 4.17Y SR >= 0.75 AND Window B SR > 0.00]:  {sr_best_full:+.2f} & {wB_sr:+.2f} -> {'PASS' if gate1 else 'FAIL'}")
    print(f"  Gate 2 [100% CPCV Paths > 0 AND Median SR >= 0.65]:    Med {np.median(cpcv_stand_srs):+.2f} (Min {np.min(cpcv_stand_srs):+.2f}) -> {'PASS' if gate2 else 'FAIL'}")
    print(f"  Gate 3 [Standalone DSR (M=15) >= 80%]:                 {dsr_s15*100:.2f}% -> {'PASS' if gate3 else 'FAIL'}")
    print(f"  Gate 4 [Bootstrap Delta_SR > 0 vs APEX VM (p < 0.10)]: {boot['delta_mean']:+.3f} (p={boot['p_delta_gt_0']:.4f}) -> {'PASS' if gate4 else 'FAIL'}")

    verdict = "[GENUINE 4.17Y EDGE UNLOCKED]" if (gate1 and gate2 and gate3 and gate4) else "[SIGNAL REAL BUT MONETIZATION BOUNDED / FREEZE REPO]"
    print(f"\nFINAL SCIENTIFIC VERDICT: {verdict}")
    print(f"Total Runtime: {time.time() - t_all:.1f}s")
    print("=" * 125)


if __name__ == "__main__":
    main()
