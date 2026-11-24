#!/usr/bin/env python3
"""
===================================================================================
EXP-010: ML-Filtered Rolling PCA / Eigenportfolio Statistical Arbitrage
===================================================================================
End-to-end quantitative research on cross-sectional statistical arbitrage:
  1. Causal Rolling Eigenportfolio Extraction:
     - Trailing 720h (30-day) standardized hourly return matrix across 15 coins.
     - Refit PCA every 24 hours (strictly zero lookahead).
     - Retains top K=2 eigenportfolios (PC1 market beta, PC2 sector/altcoin factor).
     - Records PC1 Absorption Ratio (lambda_1 / sum(lambda_i)).
  2. Idiosyncratic Residual Spread & AR(1) OU Half-Life:
     - Regress trailing 168h returns of each coin onto PC1 & PC2 (causal OLS).
     - Integrate residuals into cumulative spread S_{i,t} and causal z-score z_{i,t}.
     - Fit AR(1) Ornstein-Uhlenbeck process to estimate mean-reversion half-life.
  3. Discrete Divergence Events & Hedged Trade Construction:
     - Trigger: |z_{i,t}| >= 2.0 with a 24-hour cooldown per coin.
     - Analytic nullspace hedge: strictly dollar-neutral & factor-neutral.
     - Exit: z crosses 0 (take-profit), |z| >= 4.0 (stop-loss), or 120h (time-stop).
  4. Walk-Forward ML Regime-Break Filter:
     - 10 expanding quarterly refits with 168h purge/embargo.
     - Regularized LightGBM predicting P(mean-reversion succeeds).
     - Classification metrics: ROC-AUC, PR-AUC, Brier Skill Score (BSS), ECE,
       Top-Half vs. Bottom-Half win rate & avg PnL (bps), Label Shuffle Placebo.
  5. Multi-Coin Hedged Book Simulation & APEX v45 Blends:
     - Vol-normalized equal-risk sizing, capped at max gross 1.0x.
     - Realized 8h perpetual funding + slippage on BOTH coin and hedge legs.
     - S1 (Unfiltered), S2 (ML-Filtered Top-50%), S3 (High-Conviction Top-30%), Placebo.
     - Honest Vol-Matched & Unlevered Blends with Frozen APEX v45 (0 / 10 / 25 bps).
===================================================================================
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import apex_v45_champion as v45
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("exp010_pca_statarb")

HOURS_PER_YEAR = 8760.0

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
]
TEST_END_FINAL = "2025-05-23 23:00:00+00:00"  # Exact 22,440 hours matching EXP-003e

ML_FEATURE_COLS = [
    "abs_z",
    "ou_halflife",
    "pc1_absorption_ratio",
    "cross_sectional_dispersion",
    "vol_ratio_24_168",
    "vol_surge",
    "amihud_24h",
    "fr_zscore_30d",
    "mom_72h",
]


# ===================================================================================
# STATISTICAL METRICS & CALIBRATION HELPERS
# ===================================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE) across n_bins equal partitions."""
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (n_bin / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculates event-level classification and skill metrics."""
    n = len(y_true)
    if n == 0 or len(np.unique(y_true)) < 2:
        return {"auc": 0.5, "prauc": 0.5, "bss": 0.0, "ece": 0.0, "win_rate": 0.0}

    auc = float(roc_auc_score(y_true, y_prob))
    prauc = float(average_precision_score(y_true, y_prob))
    bs = float(brier_score_loss(y_true, y_prob))
    bs_ref = float(np.mean(y_true) * (1.0 - np.mean(y_true)))
    bss = float(1.0 - (bs / (bs_ref + 1e-12))) if bs_ref > 0 else 0.0
    ece = compute_ece(y_true, y_prob, n_bins=10)
    win_rate = float(np.mean(y_true) * 100.0)

    return {
        "auc": auc,
        "prauc": prauc,
        "bss": bss,
        "ece": ece,
        "win_rate": win_rate,
    }


def compute_portfolio_metrics(
    r_net: np.ndarray,
    turnover: np.ndarray,
    slippage_bps: float,
    ts_index: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Calculates standardized annualized quantitative performance metrics."""
    n = len(r_net)
    if n == 0:
        return {}

    years = n / HOURS_PER_YEAR
    cum = np.cumprod(1.0 + r_net)
    cummax = np.maximum.accumulate(cum)
    dd = (cum - cummax) / (cummax + 1e-12)
    max_dd = float(np.min(dd) * 100.0)

    cagr = float((cum[-1] ** (1.0 / years) - 1.0) * 100.0) if cum[-1] > 0 else -100.0
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

    mean_r = float(np.mean(r_net))
    std_r = float(np.std(r_net)) + 1e-12
    sharpe = float((mean_r / std_r) * np.sqrt(HOURS_PER_YEAR))
    vol = float(std_r * np.sqrt(HOURS_PER_YEAR) * 100.0)

    ann_turnover = float(np.mean(turnover) * HOURS_PER_YEAR)
    drag_bps_yr = float(ann_turnover * slippage_bps)

    df_eval = pd.DataFrame({"r": r_net}, index=ts_index)
    sr_years = {}
    for yr in [2022, 2023, 2024, 2025]:
        sub = df_eval[df_eval.index.year == yr]["r"]
        sr_years[yr] = float((sub.mean() / (sub.std() + 1e-12)) * np.sqrt(HOURS_PER_YEAR)) if len(sub) > 0 else 0.0

    return {
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_dd": max_dd,
        "ann_turnover": ann_turnover,
        "drag_bps_yr": drag_bps_yr,
        "sr_years": sr_years,
    }


def apply_honest_vol_targeting(
    r_raw: np.ndarray,
    slippage_bps: float,
    underlying_turnover: np.ndarray,
    target_vol: float = 0.1844,
    min_lev: float = 0.50,
    max_lev: float = 1.25,
    lookback_hours: int = 2160,  # 90 days
    deadband: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Applies the EXACT causal 90-day vol-targeting rule with 0.05 deadband and daily rebalancing:
      1. Evaluates trailing 90-day realized volatility at daily UTC boundary.
      2. If |target_k - cur_k| > 0.05, adjusts leverage and incurs exact incremental turnover:
         lev_turnover_t = |kappa_t - kappa_{t-1}| * 1.0 (active gross notional)
      3. Charges slippage on both underlying strategy turnover (scaled by kappa) and leverage adjustment turnover.
    """
    n_ts = len(r_raw)
    kappa = np.ones(n_ts)
    cur_kappa = 1.0
    lev_turnover = np.zeros(n_ts)

    for t in range(n_ts):
        if t % 24 == 0 and t > 0:
            t_start = max(0, t - lookback_hours)
            hist = r_raw[t_start:t]
            if len(hist) >= 168:
                hist_vol = float(np.std(hist) * np.sqrt(HOURS_PER_YEAR))
                target_k = float(np.clip(target_vol / (hist_vol + 1e-6), min_lev, max_lev))
                if abs(target_k - cur_kappa) > deadband:
                    lev_turnover[t] = abs(target_k - cur_kappa) * 1.0
                    cur_kappa = target_k
        kappa[t] = cur_kappa

    underlying_drag = kappa * underlying_turnover * (slippage_bps / 10000.0)
    leverage_drag = lev_turnover * (slippage_bps / 10000.0)
    r_scaled = kappa * r_raw - underlying_drag - leverage_drag

    total_turnover = kappa * underlying_turnover + lev_turnover
    return r_scaled, total_turnover, kappa


# ===================================================================================
# FACTOR EXTRACTION & RESIDUAL SPREAD ENGINE
# ===================================================================================

def extract_causal_eigenportfolios_and_residuals(
    df: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Performs causal rolling 720h PCA (refit every 24h), extracts PC1 and PC2,
    regresses trailing 168h coin returns onto eigenportfolios (causal OLS),
    integrates residual spreads, and computes strictly causal z-scores and OU half-lives.
    """
    assets = sorted(df["asset"].unique())
    timestamps = sorted(df["timestamp"].unique())
    n_ts = len(timestamps)
    n_assets = len(assets)

    logger.info(f"Extracting rolling eigenportfolios across {n_ts} hourly bars for {n_assets} coins...")
    piv_close = df.pivot(index="timestamp", columns="asset", values="close")
    piv_ret = piv_close.pct_change().fillna(0.0)
    ret_mat = piv_ret.values

    piv_vol = df.pivot(index="timestamp", columns="asset", values="volume").fillna(0.0).values
    piv_fr = df.pivot(index="timestamp", columns="asset", values="funding_rate").fillna(0.0).values

    # Precompute causal features per coin
    logger.info("Computing causal regime and crowding features for ML filter...")
    vol_24 = pd.DataFrame(ret_mat).rolling(24, min_periods=12).std().bfill().values
    vol_168 = pd.DataFrame(ret_mat).rolling(168, min_periods=24).std().bfill().values
    vol_ratio_mat = vol_24 / (vol_168 + 1e-6)

    v_24 = pd.DataFrame(piv_vol).rolling(24, min_periods=12).mean().bfill().values
    v_168 = pd.DataFrame(piv_vol).rolling(168, min_periods=24).mean().bfill().values
    vol_surge_mat = np.log((v_24 + 1.0) / (v_168 + 1.0))

    dollar_vol = piv_vol * piv_close.values
    amihud = np.abs(ret_mat) / (dollar_vol + 1e-4)
    amihud_mat = pd.DataFrame(amihud).rolling(24, min_periods=12).mean().bfill().values

    fr_df = pd.DataFrame(piv_fr)
    fr_mu = fr_df.rolling(720, min_periods=72).mean().bfill().values
    fr_sig = fr_df.rolling(720, min_periods=72).std().bfill().values
    fr_zscore_mat = np.clip((piv_fr - fr_mu) / (fr_sig + 1e-8), -4.0, 4.0)

    mom_72h_mat = pd.DataFrame(piv_close.values).pct_change(72).fillna(0.0).values
    cross_sec_disp = np.std(ret_mat, axis=1)

    # 1. Rolling 720h PCA (Refit every 24h)
    f1 = np.zeros(n_ts)
    f2 = np.zeros(n_ts)
    ar1_series = np.zeros(n_ts)
    pca_v1_mat = np.zeros((n_ts, n_assets))
    pca_v2_mat = np.zeros((n_ts, n_assets))
    pca_sig_mat = np.zeros((n_ts, n_assets))

    cur_v1 = np.zeros(n_assets)
    cur_v2 = np.zeros(n_assets)
    cur_mu = np.zeros(n_assets)
    cur_sig = np.ones(n_assets)
    cur_ar1 = 0.0

    for t in range(720, n_ts):
        if t % 24 == 0:
            window = ret_mat[t - 720 : t]
            cur_mu = np.mean(window, axis=0)
            cur_sig = np.std(window, axis=0) + 1e-8
            std_win = (window - cur_mu) / cur_sig
            pca = PCA(n_components=2)
            pca.fit(std_win)
            cur_v1 = pca.components_[0]
            cur_v2 = pca.components_[1]
            cur_ar1 = pca.explained_variance_ratio_[0]

        r_std = (ret_mat[t] - cur_mu) / cur_sig
        f1[t] = np.dot(cur_v1, r_std)
        f2[t] = np.dot(cur_v2, r_std)
        ar1_series[t] = cur_ar1
        pca_v1_mat[t] = cur_v1
        pca_v2_mat[t] = cur_v2
        pca_sig_mat[t] = cur_sig

    # 2. Rolling 168h OLS Regression onto PC1 & PC2 & OU Half-Life
    logger.info("Computing causal 168h OLS residual spreads, z-scores and OU half-lives...")
    z_mat = np.zeros((n_ts, n_assets))
    halflife_mat = np.zeros((n_ts, n_assets))
    beta_f1 = np.zeros((n_ts, n_assets))
    beta_f2 = np.zeros((n_ts, n_assets))
    ones_168 = np.ones((168, 1))

    for t in range(888, n_ts):
        x_mat = np.column_stack([ones_168, f1[t - 168 : t, None], f2[t - 168 : t, None]])
        y_mat = ret_mat[t - 168 : t]
        xtx = x_mat.T @ x_mat
        inv_xtx = np.linalg.pinv(xtx)
        beta = inv_xtx @ (x_mat.T @ y_mat)
        beta_f1[t] = beta[1]
        beta_f2[t] = beta[2]

        resid = y_mat - x_mat @ beta
        s_spread = np.cumsum(resid, axis=0)
        mu_s = np.mean(s_spread, axis=0)
        sig_s = np.std(s_spread, axis=0) + 1e-8
        z_mat[t] = (s_spread[-1] - mu_s) / sig_s

        s_prev = s_spread[:-1] - np.mean(s_spread[:-1], axis=0)
        delta_s = np.diff(s_spread, axis=0)
        delta_s_dm = delta_s - np.mean(delta_s, axis=0)
        b = np.sum(s_prev * delta_s_dm, axis=0) / (np.sum(s_prev**2, axis=0) + 1e-12)
        kappa = -np.log(1.0 + np.clip(b, -0.999, -0.001))
        halflife_mat[t] = np.where(b < 0, 0.69315 / kappa, 168.0)

    # 3. Detect Discrete Divergence Entry Events & Construct Hedged Trades
    logger.info("Extracting discrete divergence entry events (|z| >= 2.0 with 24h cooldown)...")
    events: list[dict[str, Any]] = []
    last_entry = {i: -100 for i in range(n_assets)}
    active_until = {i: -100 for i in range(n_assets)}

    for t in range(888, n_ts - 120):
        # Build nullspace projection matrix for dollar and PC factor neutrality at bar t
        c_proj = np.column_stack([np.ones(n_assets), beta_f1[t], beta_f2[t]])
        inv_cc = np.linalg.pinv(c_proj.T @ c_proj)
        p_proj = np.eye(n_assets) - c_proj @ inv_cc @ c_proj.T

        for i in range(n_assets):
            if t < active_until[i] or (t - last_entry[i]) < 24:
                continue

            z_val = z_mat[t, i]
            if abs(z_val) >= 2.0:
                direction = -1.0 if z_val >= 2.0 else +1.0
                last_entry[i] = t

                # Forward search for exit condition
                exit_t = t + 120
                exit_reason = "time_stop"
                for tau in range(t + 1, min(t + 121, n_ts)):
                    z_tau = z_mat[tau, i]
                    # Take-profit: z crosses 0.0
                    if (direction == -1.0 and z_tau <= 0.0) or (direction == +1.0 and z_tau >= 0.0):
                        exit_t = tau
                        exit_reason = "take_profit"
                        break
                    # Stop-loss: |z| blows out to 4.0
                    if abs(z_tau) >= 4.0:
                        exit_t = tau
                        exit_reason = "stop_loss"
                        break

                active_until[i] = exit_t

                # Exact analytic nullspace hedged trade weights
                e_i = np.zeros(n_assets)
                e_i[i] = direction
                w_hedged = p_proj @ e_i
                gross = np.sum(np.abs(w_hedged))
                w_trade = w_hedged / (gross + 1e-12)

                # Realized gross holding return of the hedged trade
                trade_slice = slice(t + 1, exit_t + 1)
                r_sub = ret_mat[trade_slice]  # (duration, 15)
                fr_sub = piv_fr[trade_slice]
                ts_sub = [timestamps[idx] for idx in range(t + 1, exit_t + 1)]

                price_pnl = np.sum(r_sub @ w_trade)
                fund_hours = np.array([ts.hour in [0, 8, 16] for ts in ts_sub])
                fund_pnl = np.sum(np.where(fund_hours[:, None], -w_trade * fr_sub, 0.0))
                trade_pnl_gross = float(price_pnl + fund_pnl)
                y_rev = 1 if trade_pnl_gross > 0.0 else 0

                events.append({
                    "event_id": len(events),
                    "entry_idx": t,
                    "exit_idx": exit_t,
                    "duration": exit_t - t,
                    "entry_time": timestamps[t],
                    "exit_time": timestamps[exit_t],
                    "exit_reason": exit_reason,
                    "asset_idx": i,
                    "asset": assets[i],
                    "direction": direction,
                    "z_entry": float(z_val),
                    "abs_z": float(abs(z_val)),
                    "ou_halflife": float(np.clip(halflife_mat[t, i], 1.0, 168.0)),
                    "pc1_absorption_ratio": float(ar1_series[t]),
                    "cross_sectional_dispersion": float(cross_sec_disp[t]),
                    "vol_ratio_24_168": float(vol_ratio_mat[t, i]),
                    "vol_surge": float(vol_surge_mat[t, i]),
                    "amihud_24h": float(amihud_mat[t, i]),
                    "fr_zscore_30d": float(fr_zscore_mat[t, i]),
                    "mom_72h": float(mom_72h_mat[t, i]),
                    "w_trade": w_trade,
                    "trailing_vol": float(vol_168[t, i]),
                    "trade_pnl_gross": trade_pnl_gross,
                    "trade_pnl_bps": trade_pnl_gross * 10000.0,
                    "y_rev": y_rev,
                })

    df_events = pd.DataFrame(events)
    logger.info(f"Detected {len(df_events)} divergence events (Baseline Reversion Win Rate: {df_events['y_rev'].mean()*100:.2f}%)")

    market_matrices = {
        "ret_mat": ret_mat,
        "fr_mat": piv_fr,
        "timestamps": pd.DatetimeIndex(timestamps),
        "assets": assets,
        "piv_close": piv_close,
        "vol_168": vol_168,
    }
    return market_matrices, df_events


# ===================================================================================
# WALK-FORWARD ML CLASSIFIER (REGIME-BREAK FILTER)
# ===================================================================================

def train_walkforward_ml_filter(
    df_events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Trains LightGBM binary classifier strictly on closed past events (exit_time <= train_end - 168h).
    Generates out-of-sample predicted probabilities for S2 (top-50%) and S3 (top-30%),
    plus a label shuffle placebo.
    """
    logger.info("Executing 10 expanding quarterly walk-forward refits with 168h purge/embargo...")
    df_events = df_events.sort_values("entry_time").reset_index(drop=True)
    df_events["p_rev"] = np.nan
    df_events["p_rev_placebo"] = np.nan
    df_events["train_median_threshold"] = np.nan
    df_events["train_q70_threshold"] = np.nan

    for q in range(len(QUARTERLY_TEST_STARTS)):
        t_start = pd.Timestamp(QUARTERLY_TEST_STARTS[q])
        t_end = (
            pd.Timestamp(QUARTERLY_TEST_STARTS[q + 1])
            if q + 1 < len(QUARTERLY_TEST_STARTS)
            else pd.Timestamp(TEST_END_FINAL)
        )
        t_train_end = t_start - timedelta(hours=168)

        # Strictly purged and embargoed training events
        train_mask = df_events["exit_time"] <= t_train_end
        test_mask = (df_events["entry_time"] >= t_start) & (df_events["entry_time"] < t_end)

        df_tr = df_events[train_mask]
        df_te = df_events[test_mask]

        if len(df_tr) < 30 or len(df_te) == 0:
            continue

        x_tr = df_tr[ML_FEATURE_COLS].values
        y_tr = df_tr["y_rev"].values
        x_te = df_te[ML_FEATURE_COLS].values

        # 1. Target LightGBM Model
        clf = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=15,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42 + q,
            verbose=-1,
        )
        clf.fit(x_tr, y_tr)
        p_te = clf.predict_proba(x_te)[:, 1]
        p_tr = clf.predict_proba(x_tr)[:, 1]

        # 2. Label Shuffle Placebo Model
        np.random.seed(42 + q)
        y_tr_shuffled = np.random.permutation(y_tr)
        clf_placebo = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=15,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=42 + q,
            verbose=-1,
        )
        clf_placebo.fit(x_tr, y_tr_shuffled)
        p_te_placebo = clf_placebo.predict_proba(x_te)[:, 1]

        # Record test predictions and causal training thresholds
        df_events.loc[test_mask, "p_rev"] = p_te
        df_events.loc[test_mask, "p_rev_placebo"] = p_te_placebo
        df_events.loc[test_mask, "train_median_threshold"] = float(np.median(p_tr))
        df_events.loc[test_mask, "train_q70_threshold"] = float(np.quantile(p_tr, 0.70))

    # Restrict to evaluated OOS events
    oos_mask = df_events["p_rev"].notnull()
    df_oos = df_events[oos_mask].copy().reset_index(drop=True)

    y_oos = df_oos["y_rev"].values
    p_oos = df_oos["p_rev"].values
    p_pl_oos = df_oos["p_rev_placebo"].values

    m_target = compute_classification_metrics(y_oos, p_oos)
    m_placebo = compute_classification_metrics(y_oos, p_pl_oos)

    # Top-Half vs Bottom-Half predicted win rate & PnL
    median_p = np.median(p_oos)
    top_mask = p_oos >= median_p
    bot_mask = p_oos < median_p

    top_win = float(np.mean(y_oos[top_mask]) * 100.0)
    bot_win = float(np.mean(y_oos[bot_mask]) * 100.0)
    top_pnl_bps = float(np.mean(df_oos.loc[top_mask, "trade_pnl_bps"]))
    bot_pnl_bps = float(np.mean(df_oos.loc[bot_mask, "trade_pnl_bps"]))

    summary = {
        "n_oos_events": len(df_oos),
        "baseline_win_rate": float(np.mean(y_oos) * 100.0),
        "m_target": m_target,
        "m_placebo": m_placebo,
        "top_half_win": top_win,
        "bot_half_win": bot_win,
        "top_half_pnl_bps": top_pnl_bps,
        "bot_half_pnl_bps": bot_pnl_bps,
    }
    return df_events, summary


# ===================================================================================
# MULTI-COIN HEDGED STAT-ARB BOOK SIMULATOR
# ===================================================================================

def simulate_stat_arb_portfolio(
    df_events: pd.DataFrame,
    allowed_event_ids: set[int],
    market_matrices: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Simulates the active multi-coin hedged stat-arb book across the exact 22,440 OOS hours:
      - Vol-normalized equal-risk sizing per active trade.
      - Smooth gross leverage cap at 1.0x.
      - Causal 1-hour lag: target position at t takes effect at hour t+1.
      - Charges realized turnover slippage and 8h funding settlements on BOTH coin & hedge legs.
    """
    timestamps = market_matrices["timestamps"]
    ret_mat = market_matrices["ret_mat"]
    fr_mat = market_matrices["fr_mat"]
    n_assets = len(market_matrices["assets"])

    oos_start_ts = pd.Timestamp(QUARTERLY_TEST_STARTS[0])
    oos_end_ts = pd.Timestamp(TEST_END_FINAL)
    oos_indices = [i for i, ts in enumerate(timestamps) if ts >= oos_start_ts and ts <= oos_end_ts]
    n_oos = len(oos_indices)
    ts_oos = timestamps[oos_indices]

    # Map global timestamps to OOS bar index
    ts_to_oos_idx = {timestamps[idx]: i for i, idx in enumerate(oos_indices)}

    active_weight_mat = np.zeros((n_oos, n_assets))

    for row in df_events.itertuples():
        if row.event_id not in allowed_event_ids:
            continue

        w_trade = row.w_trade
        t_entry = row.entry_time
        t_exit = row.exit_time
        vol_entry = getattr(row, "trailing_vol", 0.02)
        scale_risk = 1.0 / (vol_entry + 1e-4)

        # Trade is active starting at entry_time + 1h until exit_time
        cur_t = t_entry + timedelta(hours=1)
        while cur_t <= t_exit and cur_t in ts_to_oos_idx:
            idx_oos = ts_to_oos_idx[cur_t]
            active_weight_mat[idx_oos] += scale_risk * w_trade
            cur_t += timedelta(hours=1)

    # Smoothly cap gross leverage at 1.0x
    gross_exposure = np.sum(np.abs(active_weight_mat), axis=1)
    cap_factors = np.where(gross_exposure > 1.0, 1.0 / gross_exposure, 1.0)
    w_capped = active_weight_mat * cap_factors[:, None]

    # Causal execution: positions set at t are held during t+1
    pos_shifted = pd.DataFrame(w_capped).shift(1).fillna(0.0).values
    turnover = np.sum(
        np.abs(np.diff(np.concatenate([np.zeros((1, n_assets)), pos_shifted], axis=0), axis=0)),
        axis=1,
    )

    ret_oos = ret_mat[oos_indices]
    fr_oos = fr_mat[oos_indices]

    funding_hours = ts_oos.hour.isin([0, 8, 16])
    fund_pnl = np.where(funding_hours, np.sum(-pos_shifted * fr_oos, axis=1), 0.0)
    price_pnl = np.sum(pos_shifted * ret_oos, axis=1)
    gross_pnl = price_pnl + fund_pnl

    r0 = gross_pnl - turnover * 0.0
    r10 = gross_pnl - turnover * (10.0 / 10000.0)
    r25 = gross_pnl - turnover * (25.0 / 10000.0)

    m = {
        "m0": compute_portfolio_metrics(r0, turnover, 0.0, ts_oos),
        "m10": compute_portfolio_metrics(r10, turnover, 10.0, ts_oos),
        "m25": compute_portfolio_metrics(r25, turnover, 25.0, ts_oos),
        "trade_count": len(allowed_event_ids),
    }
    return r10, turnover, m


# ===================================================================================
# MAIN EXECUTION ORCHESTRATION
# ===================================================================================

def run_experiment() -> None:
    print("=" * 135)
    print("EXP-010: ML-FILTERED ROLLING PCA / EIGENPORTFOLIO STATISTICAL ARBITRAGE")
    print(f"Coverage: {QUARTERLY_TEST_STARTS[0]} to {TEST_END_FINAL} (Exact 22,440 Audited Bars)")
    print("=" * 135)

    # 1. Load Parquet Dataset
    parquet_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/universe_15_hourly.parquet"
    logger.info(f"Loading 15-coin hourly dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["asset", "timestamp"]).reset_index(drop=True)

    # 2. Extract Causal Rolling PCA, Residuals & Divergence Events
    market_matrices, df_events = extract_causal_eigenportfolios_and_residuals(df)

    # 3. Train Walk-Forward ML Regime-Break Filter
    df_events, ml_summary = train_walkforward_ml_filter(df_events)

    # 4. Form Strategy Sleeves (Restricted to OOS Events)
    oos_events = df_events[df_events["p_rev"].notnull()].copy()
    s1_ids = set(oos_events["event_id"])
    s2_ids = set(oos_events[oos_events["p_rev"] >= oos_events["train_median_threshold"]]["event_id"])
    s3_ids = set(oos_events[oos_events["p_rev"] >= oos_events["train_q70_threshold"]]["event_id"])

    np.random.seed(42)
    sample_size = len(s2_ids)
    all_oos_list = list(s1_ids)
    placebo_ids = set(np.random.choice(all_oos_list, size=sample_size, replace=False))

    sleeves = {
        "S1 (Unfiltered PCA Stat-Arb)": s1_ids,
        "S2 (ML-Filtered Top-50%)": s2_ids,
        "S3 (ML-Filtered High-Conviction)": s3_ids,
        "Shuffle Placebo": placebo_ids,
    }

    sleeve_results = {}
    for name, ids in sleeves.items():
        r10, turn, m = simulate_stat_arb_portfolio(df_events, ids, market_matrices)
        sleeve_results[name] = {"r10": r10, "turnover": turn, "metrics": m}

    # 5. Load Frozen APEX v45 Champion Baseline
    logger.info("Loading Frozen APEX v45 Champion Baseline...")
    dataset_v45, processed_v45, garch_wts_v45, eval_index_v45, n_v45, _, macro_regime_v45, mtf_skew_v45, arb_scale_v45 = (
        v45.load_data_and_features()
    )
    dir_s, _ = v45.run_directional_engine(
        processed_v45, dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45, garch_wts_v45, mtf_skew_v45
    )
    sp_s, _ = v45.run_spread_engine(processed_v45, dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45)
    arb_s, _ = v45.run_funding_arb_engine(dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45, arb_scale_v45)
    champ_full = 0.25 * dir_s * macro_regime_v45 + 0.55 * sp_s * macro_regime_v45 + 0.20 * arb_s

    df_champ_full = pd.DataFrame({"r_apex": champ_full}, index=eval_index_v45)
    slice_22440 = df_champ_full.loc[QUARTERLY_TEST_STARTS[0] : TEST_END_FINAL]
    r_apex_base = slice_22440["r_apex"].values
    hourly_index = slice_22440.index

    # Correlation with APEX v45
    for name, res in sleeve_results.items():
        corr = float(np.corrcoef(res["r10"], r_apex_base)[0, 1])
        res["corr_apex"] = corr

    # -----------------------------------------------------------------------------------
    # SECTION 1: OOS EVENT-LEVEL CLASSIFICATION & REGIME-BREAK FILTER REPORT
    # -----------------------------------------------------------------------------------
    print("\n" + "-" * 135)
    print("SECTION 1: OOS EVENT-LEVEL CLASSIFICATION & REGIME-BREAK ML FILTER METRICS (10 WALK-FORWARD REFITS)")
    print("-" * 135)
    print(f"Total Evaluated OOS Divergence Events: {ml_summary['n_oos_events']}")
    print(f"Baseline Reversion Win Rate:          {ml_summary['baseline_win_rate']:.2f}%")
    print(f"LightGBM Target Model ROC-AUC:         {ml_summary['m_target']['auc']:.4f}")
    print(f"LightGBM Target Model PR-AUC:          {ml_summary['m_target']['prauc']:.4f}")
    print(f"Brier Skill Score (BSS):               {ml_summary['m_target']['bss']:+.4f}")
    print(f"Expected Calibration Error (ECE):      {ml_summary['m_target']['ece']:.4f}")
    print(f"Top-Half vs. Bottom-Half Predicted:    Win Rate: {ml_summary['top_half_win']:.2f}% vs. {ml_summary['bot_half_win']:.2f}% ({ml_summary['top_half_win'] - ml_summary['bot_half_win']:+.2f}%)")
    print(f"                                       Avg PnL:  {ml_summary['top_half_pnl_bps']:+.1f} bps vs. {ml_summary['bot_half_pnl_bps']:+.1f} bps ({ml_summary['top_half_pnl_bps'] - ml_summary['bot_half_pnl_bps']:+.1f} bps spread)")
    print(f"Label Shuffle Placebo ROC-AUC:         {ml_summary['m_placebo']['auc']:.4f} (Null ~0.5000)")
    print(f"Label Shuffle Placebo BSS:             {ml_summary['m_placebo']['bss']:+.4f} (Null ~0.0000)")
    print("-" * 135)

    # -----------------------------------------------------------------------------------
    # SECTION 2: STANDALONE STAT-ARB SLEEVE TEAR SHEET
    # -----------------------------------------------------------------------------------
    print("\n" + "-" * 135)
    print("SECTION 2: STANDALONE STAT-ARB SLEEVE TEAR SHEET (22,440 AUDITED HOURLY BARS)")
    print("-" * 135)
    print(
        f"{'Strategy / Sleeve':<34s} | {'Trades':<6s} | {'CAGR (0/10/25)':<19s} | {'Vol(10)':<7s} | {'Sharpe (0/10/25)':<19s} | {'Calmar':<6s} | {'MaxDD':<7s} | {'Turnover':<9s} | {'rho(APEX)':<9s}"
    )
    print("-" * 135)

    for name in ["S1 (Unfiltered PCA Stat-Arb)", "S2 (ML-Filtered Top-50%)", "S3 (ML-Filtered High-Conviction)", "Shuffle Placebo"]:
        res = sleeve_results[name]
        m = res["metrics"]
        m0, m10, m25 = m["m0"], m["m10"], m["m25"]
        cagr_str = f"{m0['cagr']:.1f}/{m10['cagr']:.1f}/{m25['cagr']:.1f}%"
        sr_str = f"{m0['sharpe']:.2f}/{m10['sharpe']:.2f}/{m25['sharpe']:.2f}"
        turn_str = f"{m10['ann_turnover']:.1f}x/yr"
        corr_str = f"{res['corr_apex']:+.3f}"
        print(
            f"{name:<34s} | {m['trade_count']:<6d} | {cagr_str:<19s} | {m10['vol']:<6.1f}% | {sr_str:<19s} | {m10['calmar']:<6.2f} | {m10['max_dd']:<6.1f}% | {turn_str:<9s} | {corr_str:<9s}"
        )
    print("-" * 135)

    # -----------------------------------------------------------------------------------
    # SECTION 3: APEX v45 BLENDS (UNLEVERED & HONEST VOL-MATCHED)
    # -----------------------------------------------------------------------------------
    print("\n" + "-" * 135)
    print("SECTION 3: BLENDED PORTFOLIOS (FROZEN APEX v45 + S2 ML-FILTERED STAT-ARB)")
    print("-" * 135)
    print(
        f"{'Portfolio Configuration':<48s} | {'Slippage':<8s} | {'CAGR':<7s} | {'Sharpe':<6s} | {'Calmar':<6s} | {'MaxDD':<7s} | {'2022':<6s} | {'2023':<6s} | {'2024':<6s} | {'2025':<6s}"
    )
    print("-" * 135)

    m_apex_unlev = compute_portfolio_metrics(r_apex_base, np.zeros(len(r_apex_base)), 0.0, hourly_index)
    for s in [0.0, 10.0, 25.0]:
        name_str = "100% Frozen APEX v45 [Unlevered Baseline]" if s == 0.0 else f"  + {int(s)} bps Slippage"
        print(
            f"{name_str:<48s} | {int(s):<2d} bps   | {m_apex_unlev['cagr']:<7.2f} | {m_apex_unlev['sharpe']:<6.2f} | {m_apex_unlev['calmar']:<6.2f} | {m_apex_unlev['max_dd']:<7.1f} | {m_apex_unlev['sr_years'][2022]:<+6.2f} | {m_apex_unlev['sr_years'][2023]:<+6.2f} | {m_apex_unlev['sr_years'][2024]:<+6.2f} | {m_apex_unlev['sr_years'][2025]:<+6.2f}"
        )
    print("-" * 135)

    apex_vm_results = {}
    for s in [0.0, 10.0, 25.0]:
        r_apex_vm, turn_apex_vm, _ = apply_honest_vol_targeting(
            r_apex_base, s, underlying_turnover=np.zeros(len(r_apex_base)), target_vol=0.1844, deadband=0.05
        )
        m_apex_vm = compute_portfolio_metrics(r_apex_vm, turn_apex_vm, s, hourly_index)
        apex_vm_results[s] = m_apex_vm
        name_str = (
            "100% Frozen APEX v45 [Honest Vol-Matched]" if s == 0.0 else f"  + {int(s)} bps Slippage (Lev Fee Charged)"
        )
        print(
            f"{name_str:<48s} | {int(s):<2d} bps   | {m_apex_vm['cagr']:<7.2f} | {m_apex_vm['sharpe']:<6.2f} | {m_apex_vm['calmar']:<6.2f} | {m_apex_vm['max_dd']:<7.1f} | {m_apex_vm['sr_years'][2022]:<+6.2f} | {m_apex_vm['sr_years'][2023]:<+6.2f} | {m_apex_vm['sr_years'][2024]:<+6.2f} | {m_apex_vm['sr_years'][2025]:<+6.2f}"
        )
    print("-" * 135)

    # Blends with S2
    best_res = sleeve_results["S2 (ML-Filtered Top-50%)"]
    turn_best = best_res["turnover"]
    r_best_raw = best_res["r10"] + turn_best * (10.0 / 10000.0)

    print("SUB-SECTION 3A: UNLEVERED BLENDS (APEX + S2 ML-FILTERED STAT-ARB)")
    print("-" * 135)
    blend_results = {}
    for w_ml in [0.10, 0.15, 0.20]:
        w_apex = 1.0 - w_ml
        r_raw_blend = w_apex * r_apex_base + w_ml * r_best_raw
        for s in [0.0, 10.0, 25.0]:
            drag = w_ml * turn_best * (s / 10000.0)
            r_unlev = r_raw_blend - drag
            m_unlev = compute_portfolio_metrics(r_unlev, np.zeros(len(r_unlev)), 0.0, hourly_index)
            blend_results[(w_ml, "unlevered", s)] = m_unlev
            name_str = (
                f"{int(w_apex*100)}% APEX / {int(w_ml*100)}% ML [Unlevered]" if s == 0.0 else f"  + {int(s)} bps Slippage"
            )
            print(
                f"{name_str:<48s} | {int(s):<2d} bps   | {m_unlev['cagr']:<7.2f} | {m_unlev['sharpe']:<6.2f} | {m_unlev['calmar']:<6.2f} | {m_unlev['max_dd']:<7.1f} | {m_unlev['sr_years'][2022]:<+6.2f} | {m_unlev['sr_years'][2023]:<+6.2f} | {m_unlev['sr_years'][2024]:<+6.2f} | {m_unlev['sr_years'][2025]:<+6.2f}"
            )
    print("-" * 135)

    print("SUB-SECTION 3B: HONEST VOL-MATCHED BLENDS (APEX + S2 ML-FILTERED STAT-ARB) [TARGET VOL = 18.44%]")
    print("-" * 135)
    for w_ml in [0.10, 0.20]:
        w_apex = 1.0 - w_ml
        r_raw_blend = w_apex * r_apex_base + w_ml * r_best_raw
        turn_blend = w_ml * turn_best
        for s in [0.0, 10.0, 25.0]:
            r_vm, turn_vm, kappa = apply_honest_vol_targeting(
                r_raw_blend, s, underlying_turnover=turn_blend, target_vol=0.1844, deadband=0.05
            )
            m_vm = compute_portfolio_metrics(r_vm, turn_vm, s, hourly_index)
            blend_results[(w_ml, "honest_vol_matched", s)] = m_vm
            name_str = (
                f"{int(w_apex*100)}% APEX / {int(w_ml*100)}% ML [Honest VM]" if s == 0.0 else f"  + {int(s)} bps Slippage (Lev Fee Charged)"
            )
            print(
                f"{name_str:<48s} | {int(s):<2d} bps   | {m_vm['cagr']:<7.2f} | {m_vm['sharpe']:<6.2f} | {m_vm['calmar']:<6.2f} | {m_vm['max_dd']:<7.1f} | {m_vm['sr_years'][2022]:<+6.2f} | {m_vm['sr_years'][2023]:<+6.2f} | {m_vm['sr_years'][2024]:<+6.2f} | {m_vm['sr_years'][2025]:<+6.2f}"
            )
    print("-" * 135)

    # -----------------------------------------------------------------------------------
    # SECTION 4: SCIENTIFIC VERDICT & CONCLUSION
    # -----------------------------------------------------------------------------------
    m_base_10 = apex_vm_results[10.0]
    m_s2_10 = blend_results[(0.10, "honest_vol_matched", 10.0)]
    m_s2_25 = blend_results[(0.10, "honest_vol_matched", 25.0)]

    print("\n" + "=" * 135)
    print("SCIENTIFIC VERDICT & PCA STAT-ARB CONCLUSION:")
    print("=" * 135)
    print("  1. ML Regime-Break Filter Discrimination:")
    print(f"     - ROC-AUC: {ml_summary['m_target']['auc']:.4f} (vs Placebo: {ml_summary['m_placebo']['auc']:.4f})")
    print(f"     - Top-Half vs Bottom-Half Win Rate: {ml_summary['top_half_win']:.2f}% vs {ml_summary['bot_half_win']:.2f}% ({ml_summary['top_half_win'] - ml_summary['bot_half_win']:+.2f}%)")
    print(f"     - Top-Half vs Bottom-Half Avg PnL:  {ml_summary['top_half_pnl_bps']:+.1f} bps vs {ml_summary['bot_half_pnl_bps']:+.1f} bps ({ml_summary['top_half_pnl_bps'] - ml_summary['bot_half_pnl_bps']:+.1f} bps spread)")
    print("  2. Standalone Stat-Arb Sleeve Performance (10 bps):")
    s2_m = sleeve_results["S2 (ML-Filtered Top-50%)"]["metrics"]["m10"]
    s1_m = sleeve_results["S1 (Unfiltered PCA Stat-Arb)"]["metrics"]["m10"]
    print(f"     - S1 (Unfiltered):  CAGR={s1_m['cagr']:.2f}%, Sharpe={s1_m['sharpe']:.2f}, MaxDD={s1_m['max_dd']:.1f}%, Turnover={s1_m['ann_turnover']:.1f}x/yr")
    print(f"     - S2 (ML-Filtered): CAGR={s2_m['cagr']:.2f}%, Sharpe={s2_m['sharpe']:.2f}, MaxDD={s2_m['max_dd']:.1f}%, Turnover={s2_m['ann_turnover']:.1f}x/yr")
    print("  3. APEX v45 Blend Comparison (10 bps):")
    print(f"     - Honest Vol-Matched APEX v45 Baseline: CAGR={m_base_10['cagr']:.2f}%, Sharpe={m_base_10['sharpe']:.2f}, Calmar={m_base_10['calmar']:.2f}, MaxDD={m_base_10['max_dd']:.1f}%")
    print(f"     - 90/10 Blend with S2 (10 bps):         CAGR={m_s2_10['cagr']:.2f}%, Sharpe={m_s2_10['sharpe']:.2f}, Calmar={m_s2_10['calmar']:.2f}, MaxDD={m_s2_10['max_dd']:.1f}%")
    print(f"     - 90/10 Blend with S2 (25 bps stress):  CAGR={m_s2_25['cagr']:.2f}%, Sharpe={m_s2_25['sharpe']:.2f}, Calmar={m_s2_25['calmar']:.2f}, MaxDD={m_s2_25['max_dd']:.1f}%")
    print("=" * 135)


if __name__ == "__main__":
    run_experiment()
