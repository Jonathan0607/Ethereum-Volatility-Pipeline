"""
Pure NumPy/SciPy/Pandas Mathematical Primitives for EXP-015:
  - Exponential time-decay sample weights
  - Cross-sectional rank-Gaussian score mapping
  - Causal dispersion-gated gross exposure scaling
  - Continuous EMA-smoothed nullspace portfolio constructor (K1 / K2)

STRICTLY zero PyTorch, zero LightGBM, and zero Matplotlib imports for lean serving CI compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from apex_multimodal_risk.src.serving.portfolio_math import project_beta_neutral


def compute_exponential_time_decay_weights(
    age_days: np.ndarray,
    halflife_days: float | None,
) -> np.ndarray:
    """
    Computes strictly non-negative exponential time-decay sample weights:
      w_i = exp(-ln(2) * max(age_days_i, 0) / halflife_days)
    Normalized to mean 1.0 so LightGBM sample count scaling matches uniform training.
    """
    ages = np.maximum(np.asarray(age_days, dtype=np.float64), 0.0)
    if halflife_days is None or halflife_days <= 0.0:
        return np.ones_like(ages, dtype=np.float64)

    raw_w = np.exp(-np.log(2.0) * ages / float(halflife_days))
    mean_w = float(np.mean(raw_w))
    if mean_w > 1e-12:
        return raw_w / mean_w
    return np.ones_like(ages, dtype=np.float64)


def compute_rank_gaussian_scores(scores: np.ndarray) -> np.ndarray:
    """
    Maps cross-sectional scores to standard normal rank-Gaussian scores:
      z_i = norm.ppf((rank_i - 0.5) / N)
    Supports both 1D (N,) and 2D (T, N) arrays cleanly.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.ndim == 1:
        n = len(arr)
        if n == 0:
            return arr
        ranks = rankdata(arr, method="average")
        return norm.ppf((ranks - 0.5) / float(n))
    if arr.ndim == 2:
        n = arr.shape[1]
        if n == 0:
            return arr
        ranks = rankdata(arr, method="average", axis=1)
        return norm.ppf((ranks - 0.5) / float(n))
    raise ValueError(f"Expected 1D or 2D array, got shape {arr.shape}")


def compute_causal_dispersion_scale(
    ret_1h_mat: np.ndarray,
    smooth_halflife_hours: float = 72.0,
    min_scale: float = 0.25,
    max_scale: float = 1.00,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes strictly causal cross-sectional dispersion throttle:
      disp_ratio_t = trailing_7d_cs_dispersion(t) / expanding_median_cs_dispersion(<= t)
      gross_scale_t = EMA_72h(clip(disp_ratio_t, min_scale, max_scale))

    Zero future lookahead and zero backward fill (.bfill()).
    Returns: (gross_scale_arr, trailing_7d_disp_arr, expanding_med_disp_arr)
    """
    r_mat = np.asarray(ret_1h_mat, dtype=np.float64)
    cs_disp_1h = np.std(r_mat, axis=1)

    s_disp = pd.Series(cs_disp_1h)
    trailing_7d = s_disp.rolling(168, min_periods=24).mean()
    # Strictly causal warmup fallback (expanding min_periods=1 before 24h), never .bfill()
    warmup_trail = s_disp.expanding(min_periods=1).mean()
    trailing_7d = trailing_7d.fillna(warmup_trail).values

    s_trail = pd.Series(trailing_7d)
    expanding_med = s_trail.expanding(min_periods=168).median()
    warmup_med = s_trail.expanding(min_periods=1).median()
    expanding_med = expanding_med.fillna(warmup_med).values

    disp_ratio = trailing_7d / (expanding_med + 1e-12)
    raw_scale = np.clip(disp_ratio, min_scale, max_scale)

    smoothed_scale = (
        pd.Series(raw_scale)
        .ewm(halflife=smooth_halflife_hours, min_periods=1)
        .mean()
        .values
    )
    gross_scale = np.clip(smoothed_scale, min_scale, max_scale)
    return gross_scale, trailing_7d, expanding_med


def simulate_continuous_ema_sleeve(
    pred_mat: np.ndarray,
    ret_1h_mat: np.ndarray,
    fr_mat: np.ndarray,
    beta_mat: np.ndarray,
    ts_index: pd.DatetimeIndex,
    rebal_hours: int = 12,
    ema_halflife: float = 48.0,
    deadband: float = 0.015,
    gross_scale_arr: np.ndarray | None = None,
    max_weight: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates K1 / K2 Continuous EMA-Smoothed Nullspace Portfolio Constructor.

    Every `rebal_hours`:
      1. Maps predictions to cross-sectional rank-Gaussian scores.
      2. Projects continuously onto the beta-neutral nullspace using `project_beta_neutral`.
      3. Applies exponential weight smoothing `w_t = (1 - eta)*w_{t-1} + eta*w_star`.
      4. Enforces `deadband` per-coin threshold to suppress micro-churn while
         re-enforcing strict `sum(|w|) <= gross_cap` and `< 1e-10` dollar/beta neutrality.
      5. Earns strictly lagged 1-bar returns (`pos_shifted[t] = weights[t-1]`).
    """
    n_ts, n_assets = pred_mat.shape
    weights = np.zeros((n_ts, n_assets), dtype=np.float64)
    eta = 1.0 - float(np.exp(-np.log(2.0) * float(rebal_hours) / float(ema_halflife)))

    # Precompute all rank-Gaussian scores vectorized
    z_mat = compute_rank_gaussian_scores(pred_mat)
    w_prev = np.zeros(n_assets, dtype=np.float64)

    for t in range(n_ts):
        g_cap = 1.0 if gross_scale_arr is None else float(np.clip(gross_scale_arr[t], 0.25, 1.00))
        if t % rebal_hours == 0:
            b_t = beta_mat[t]
            w_star = project_beta_neutral(z_mat[t], b_t, gross_cap=g_cap, max_weight=max_weight)
            if t == 0:
                w_cur = w_star.copy()
            else:
                w_cand = (1.0 - eta) * w_prev + eta * w_star
                delta = np.abs(w_cand - w_prev)
                if float(np.max(delta)) < deadband and float(np.sum(np.abs(w_prev))) <= g_cap + 1e-9:
                    w_cur = w_prev.copy()
                else:
                    w_masked = np.where(delta >= deadband, w_cand, w_prev)
                    w_cur = project_beta_neutral(w_masked, b_t, gross_cap=g_cap, max_weight=max_weight)
            w_prev = w_cur
        weights[t] = w_prev

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
