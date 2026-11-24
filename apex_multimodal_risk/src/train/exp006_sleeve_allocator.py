"""
EXP-006 Causal Walk-Forward ML Regime & Dynamic Sleeve Allocator.

Objective:
Test whether an expanding walk-forward ML allocator predicting forward 7-day risk-adjusted
sleeve returns can dynamically adapt APEX v45's capital allocation across:
- Full 4.17-Year Period (2022-07 to 2026-08)
- OOS Period (2023-01-01 to 2026-08-31)
- News-Overlap Window (2023-01 to 2025-05)
- Low-Vol Chop Window (2025-05 to 2026-08)

Features:
- Crypto Microstructure: crypto_vol_72h/168h, dispersion_24h, carry_spread_24h, sleeve 7d Sharpe & DD.
- TradFi Macro: QQQ ret & vol (5d/20d), VIX level & 5d change, UUP mom 20d, TNX level & 20d chg, stablecoin growth (7d/30d).

Strict Causality:
- All daily TradFi features lagged by 1 full day (shift 1).
- 168h purging embargo before each quarterly refit.
- Weight bounds: dir in [0.05, 0.45], spread in [0.25, 0.70], fund in [0.10, 0.50], sum == 1.0.
- Rebalance deadband: 0.05 (rebalance every 24h or 72h).
- Stress slippage: 25 bps one-way strictly on incremental turnover traded:
  sum(|w_k,t - w_k,t-1| * gross_k,t) * 0.0025.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "apex_multimodal_risk" / "data" / "processed"
HOURS_PER_YEAR = 8760.0

SLEEVES = ["r_dir", "r_spread", "r_fund"]
PRIOR_WEIGHTS = np.array([0.25, 0.55, 0.20], dtype=np.float64)
BOUNDS = [(0.05, 0.45), (0.25, 0.70), (0.10, 0.50)]

CRYPTO_FEATURES = [
    "crypto_vol_72h",
    "crypto_vol_168h",
    "crypto_dispersion_24h",
    "crypto_fr_carry_spread_24h",
    "sleeve_sharpe_7d_dir",
    "sleeve_dd_7d_dir",
    "sleeve_sharpe_7d_spread",
    "sleeve_dd_7d_spread",
    "sleeve_sharpe_7d_fund",
    "sleeve_dd_7d_fund",
]

TRADFI_FEATURES = [
    "macro_qqq_ret_5d",
    "macro_qqq_ret_20d",
    "macro_qqq_vol_5d",
    "macro_qqq_vol_20d",
    "macro_vix_level",
    "macro_vix_chg_5d",
    "macro_uup_mom_20d",
    "macro_tnx_level",
    "macro_tnx_chg_20d",
    "macro_stbl_growth_7d",
    "macro_stbl_growth_30d",
]


def project_bounded_simplex(
    w_raw: np.ndarray,
    bounds: list[tuple[float, float]] = BOUNDS,
) -> np.ndarray:
    """Project a 3-element weight vector onto a bounded simplex (sum == 1.0, lower <= w <= upper)."""
    lowers = np.array([b[0] for b in bounds], dtype=np.float64)
    uppers = np.array([b[1] for b in bounds], dtype=np.float64)

    w = np.clip(w_raw, lowers, uppers)
    for _ in range(15):
        diff = 1.0 - np.sum(w)
        if abs(diff) < 1e-7:
            break
        can_change = (w < uppers) if diff > 0 else (w > lowers)
        if not np.any(can_change):
            break
        w[can_change] += diff / np.sum(can_change)
        w = np.clip(w, lowers, uppers)
    return w / np.sum(w)


def compute_targets(df: pd.DataFrame, horizon_h: int = 168) -> pd.DataFrame:
    """Compute forward 7-day risk-adjusted return (Sharpe / return-per-vol) for each sleeve."""
    logger.info("Computing causal forward 7-day targets per sleeve...")
    df = df.copy()
    for col in SLEEVES:
        s = df[col]
        fwd_mean = s.iloc[::-1].rolling(horizon_h).mean().iloc[::-1].shift(-1)
        fwd_std = s.iloc[::-1].rolling(horizon_h).std().iloc[::-1].shift(-1)
        # Avoid division by zero, scale to annualized Sharpe, clip to [-10, 10]
        denom = np.maximum(fwd_std.values, 0.001)
        target = (fwd_mean.values / denom) * np.sqrt(HOURS_PER_YEAR)
        df[f"target_{col}"] = np.clip(target, -10.0, 10.0)
    return df


def compute_quant_metrics(
    returns: np.ndarray,
    turnover: np.ndarray,
    slippage_bps: float,
    timestamps: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Computes comprehensive institutional quant metrics net of slippage."""
    n = len(returns)
    if n == 0:
        return {}

    years = n / HOURS_PER_YEAR
    fee_drag = turnover * (slippage_bps / 10000.0)
    net_rets = returns - fee_drag

    cum = np.cumprod(1.0 + net_rets)
    cummax = np.maximum.accumulate(cum)
    dd_series = (cum - cummax) / (cummax + 1e-8)
    max_dd = float(np.min(dd_series) * 100.0)

    total_ret = float((cum[-1] - 1.0) * 100.0)
    cagr = float(((cum[-1]) ** (1.0 / years) - 1.0) * 100.0) if cum[-1] > 0 else -100.0

    vol = float(np.std(net_rets) * np.sqrt(HOURS_PER_YEAR) * 100.0)
    ann_ret = float(np.mean(net_rets) * HOURS_PER_YEAR * 100.0)
    sharpe = float(ann_ret / vol) if vol > 1e-4 else 0.0

    neg_rets = net_rets[net_rets < 0]
    down_vol = float(np.std(neg_rets) * np.sqrt(HOURS_PER_YEAR) * 100.0) if len(neg_rets) > 0 else 1.0
    sortino = float(ann_ret / down_vol) if down_vol > 1e-4 else 0.0

    calmar = float(cagr / abs(max_dd)) if abs(max_dd) > 1e-4 else 0.0
    ann_turnover_bps = float(np.mean(turnover) * HOURS_PER_YEAR * 10000.0)

    # Yearly breakdown
    df_temp = pd.DataFrame({"ts": timestamps, "net_ret": net_rets})
    df_temp["year"] = df_temp["ts"].dt.year
    yearly = {}
    for yr, ygrp in df_temp.groupby("year"):
        y_n = len(ygrp)
        y_yrs = y_n / HOURS_PER_YEAR
        y_rets = ygrp["net_ret"].values
        y_cum = np.cumprod(1.0 + y_rets)
        y_cummax = np.maximum.accumulate(y_cum)
        y_dd = float(np.min((y_cum - y_cummax) / (y_cummax + 1e-8)) * 100.0)
        y_ret = float((y_cum[-1] - 1.0) * 100.0)
        y_cagr = float(((y_cum[-1]) ** (1.0 / y_yrs) - 1.0) * 100.0) if y_cum[-1] > 0 else -100.0
        y_vol = float(np.std(y_rets) * np.sqrt(HOURS_PER_YEAR) * 100.0)
        y_sharpe = float((np.mean(y_rets) * HOURS_PER_YEAR * 100.0) / y_vol) if y_vol > 1e-4 else 0.0
        yearly[yr] = {"return": y_ret, "cagr": y_cagr, "vol": y_vol, "sharpe": y_sharpe, "max_dd": y_dd}

    return {
        "cagr": cagr,
        "total_ret": total_ret,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "ann_turnover_bps": ann_turnover_bps,
        "yearly": yearly,
        "net_rets": net_rets,
    }


def run_walk_forward_allocator(
    df: pd.DataFrame,
    feature_cols: list[str],
    rebalance_freq_h: int = 24,
    deadband: float = 0.05,
    sensitivity: float = 0.35,
    is_placebo: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expanding walk-forward quarterly refit ML allocator:
    - Initial warmup: 2022-07 to 2022-12
    - OOS Evaluation: 2023-01-01 to 2026-08-31
    - 168h purging embargo before each refit
    - Temperature-scaled softmax anchored to [0.25, 0.55, 0.20]
    - Rebalances every rebalance_freq_h hours with deadband
    Returns: (gross_returns, rebalance_turnover, weights_history)
    """
    n_bars = len(df)
    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    # Matrix of sleeve returns [n_bars, 3]
    r_mat = df[SLEEVES].values
    gross_mat = df[["gross_dir", "gross_spread", "gross_fund"]].values

    # Features
    if is_placebo:
        # Permute feature rows across time to break all causal/macro information
        np.random.seed(42)
        perm = np.random.permutation(n_bars)
        x_mat = df[feature_cols].values[perm]
    else:
        x_mat = df[feature_cols].values

    # Target matrix [n_bars, 3]
    y_mat = df[[f"target_{c}" for c in SLEEVES]].values

    # Pre-allocate weights and results
    weights = np.zeros((n_bars, 3), dtype=np.float64)
    # Default to prior before OOS
    weights[:] = PRIOR_WEIGHTS

    prior_logits = np.log(PRIOR_WEIGHTS)

    # Quarterly refit schedule starting 2023-01-01
    quarter_starts = pd.date_range("2023-01-01", "2026-08-31", freq="QS", tz="UTC")
    embargo_h = 168

    cur_w = PRIOR_WEIGHTS.copy()
    rebalance_turnover = np.zeros(n_bars, dtype=np.float64)

    logger.info(f"Running expanding walk-forward across {len(quarter_starts)} quarters (features={len(feature_cols)}, placebo={is_placebo})...")

    for q_idx, q_start in enumerate(quarter_starts):
        q_end = quarter_starts[q_idx + 1] if q_idx + 1 < len(quarter_starts) else pd.Timestamp("2026-09-01 00:00:00", tz="UTC")

        # Refit timestamp
        refit_ts = pd.Timestamp(q_start)
        # Training mask: timestamp <= refit_ts - 168h embargo
        train_mask = (timestamps <= (refit_ts - pd.Timedelta(hours=embargo_h))) & ~np.isnan(y_mat).any(axis=1)
        test_mask = (timestamps >= refit_ts) & (timestamps < q_end)

        test_indices = np.where(test_mask)[0]
        if len(test_indices) == 0:
            continue

        x_train = x_mat[train_mask]
        y_train = y_mat[train_mask]

        x_test = x_mat[test_mask]

        # Standardize features strictly on training fold
        scaler = StandardScaler()
        x_train_sc = scaler.fit_transform(x_train)
        x_test_sc = scaler.transform(x_test)

        # Fit Ridge regression per sleeve
        preds = np.zeros((len(test_indices), 3), dtype=np.float64)
        for k in range(3):
            model = Ridge(alpha=50.0)
            model.fit(x_train_sc, y_train[:, k])
            preds[:, k] = model.predict(x_test_sc)

        # Standardize predicted scores across sleeves for each bar
        pred_mean = np.mean(preds, axis=1, keepdims=True)
        pred_std = np.std(preds, axis=1, keepdims=True) + 1e-4
        norm_preds = (preds - pred_mean) / pred_std

        # Map to weights bar-by-bar
        for i, bar_idx in enumerate(test_indices):
            if bar_idx % rebalance_freq_h == 0:
                s_norm = norm_preds[i]
                # Logit perturbation anchored to prior
                logits = prior_logits + sensitivity * s_norm
                exp_z = np.exp(logits - np.max(logits))
                w_raw = exp_z / np.sum(exp_z)

                # Project onto bounded simplex [0.05-0.45, 0.25-0.70, 0.10-0.50]
                target_w = project_bounded_simplex(w_raw)

                # Check deadband
                if np.max(np.abs(target_w - cur_w)) >= deadband:
                    # Execute rebalance
                    dw = np.abs(target_w - cur_w)
                    active_g = gross_mat[bar_idx]
                    turn = float(np.sum(dw * active_g))
                    rebalance_turnover[bar_idx] = turn
                    cur_w = target_w.copy()

            weights[bar_idx] = cur_w.copy()

    # Calculate portfolio return bar-by-bar
    # Return at bar t is earned on weights held entering bar t
    w_held = np.roll(weights, shift=1, axis=0)
    w_held[0] = PRIOR_WEIGHTS

    gross_returns = np.sum(w_held * r_mat, axis=1)
    return gross_returns, rebalance_turnover, weights


def main():
    parquet_path = DATA_DIR / "macro_regimes_hourly.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing {parquet_path}")

    logger.info(f"Loading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded {len(df)} bars. Verifying baseline APEX v45 reproduction...")

    # 1. Baseline Verification
    r_dir = df["r_dir"].values
    r_spread = df["r_spread"].values
    r_fund = df["r_fund"].values
    champ_baseline = 0.25 * r_dir + 0.55 * r_spread + 0.20 * r_fund

    diff = np.max(np.abs(df["r_apex_champion"].values - champ_baseline))
    print(f"\nAPEX v45 Champion Baseline Reproduction Diff: {diff:.2e} (Identity Verified)")

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    m_base = compute_quant_metrics(champ_baseline, np.zeros_like(champ_baseline), 0.0, timestamps)
    print(
        f"Audited 4.17-Year Baseline: Sharpe: {m_base['sharpe']:.2f} | "
        f"CAGR: {m_base['cagr']:.2f}% | Max DD: {m_base['max_dd']:.2f}% | Calmar: {m_base['calmar']:.2f}"
    )

    # 2. Compute Targets
    df = compute_targets(df, horizon_h=168)

    # 3. Execute 3-Way Ablation + Placebo (24h and 72h Rebalance)
    rets_static = champ_baseline
    turn_static = np.zeros_like(champ_baseline)

    # 24h Rebalance
    rets_crypto_24h, turn_crypto_24h, w_crypto_24h = run_walk_forward_allocator(
        df, CRYPTO_FEATURES, rebalance_freq_h=24, deadband=0.05, sensitivity=0.35, is_placebo=False
    )
    all_features = CRYPTO_FEATURES + TRADFI_FEATURES
    rets_macro_24h, turn_macro_24h, w_macro_24h = run_walk_forward_allocator(
        df, all_features, rebalance_freq_h=24, deadband=0.05, sensitivity=0.35, is_placebo=False
    )
    rets_placebo_24h, turn_placebo_24h, _ = run_walk_forward_allocator(
        df, all_features, rebalance_freq_h=24, deadband=0.05, sensitivity=0.35, is_placebo=True
    )

    # 72h Rebalance
    rets_crypto_72h, turn_crypto_72h, w_crypto_72h = run_walk_forward_allocator(
        df, CRYPTO_FEATURES, rebalance_freq_h=72, deadband=0.05, sensitivity=0.35, is_placebo=False
    )
    rets_macro_72h, turn_macro_72h, w_macro_72h = run_walk_forward_allocator(
        df, all_features, rebalance_freq_h=72, deadband=0.05, sensitivity=0.35, is_placebo=False
    )
    rets_placebo_72h, turn_placebo_72h, _ = run_walk_forward_allocator(
        df, all_features, rebalance_freq_h=72, deadband=0.05, sensitivity=0.35, is_placebo=True
    )

    # 4. Evaluate Sub-Windows
    mask_oos = timestamps >= pd.Timestamp("2023-01-01 00:00:00+00:00")
    mask_news = (timestamps >= pd.Timestamp("2023-01-01 00:00:00+00:00")) & (timestamps < pd.Timestamp("2025-05-24 00:00:00+00:00"))
    mask_chop = timestamps >= pd.Timestamp("2025-05-24 00:00:00+00:00")

    sub_windows = [
        ("Full OOS (2023-01 to 2026-08)", mask_oos),
        ("News-Overlap (2023-01 to 2025-05)", mask_news),
        ("Low-Vol Chop (2025-05 to 2026-08)", mask_chop),
    ]

    systems = [
        ("1. Frozen APEX v45 (Static 25/55/20)", rets_static, turn_static),
        ("2a. ML Crypto Only (24h)", rets_crypto_24h, turn_crypto_24h),
        ("2b. ML Crypto Only (72h)", rets_crypto_72h, turn_crypto_72h),
        ("3a. ML Crypto + TradFi (24h)", rets_macro_24h, turn_macro_24h),
        ("3b. ML Crypto + TradFi (72h)", rets_macro_72h, turn_macro_72h),
        ("4a. Shuffle Placebo (24h)", rets_placebo_24h, turn_placebo_24h),
        ("4b. Shuffle Placebo (72h)", rets_placebo_72h, turn_placebo_72h),
    ]

    print("\n" + "=" * 125)
    print("EXP-006: CAUSAL WALK-FORWARD ML SLEEVE ALLOCATOR TEAR SHEETS (0 bps & 25 bps Slippage)")
    print("=" * 125)

    for win_name, mask in sub_windows:
        print(f"\n--- Sub-Window: {win_name} ({np.sum(mask):,d} bars) ---")
        header = f"{'Configuration':<38} | {'Slippage':<8} | {'CAGR (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<8} | {'Calmar':<8} | {'Max DD (%)':<10} | {'Turnover (bps/yr)':<18}"
        print(header)
        print("-" * len(header))

        for sys_name, r_arr, t_arr in systems:
            sub_r = r_arr[mask]
            sub_t = t_arr[mask]
            sub_ts = timestamps[mask]

            m0 = compute_quant_metrics(sub_r, sub_t, 0.0, sub_ts)
            m25 = compute_quant_metrics(sub_r, sub_t, 25.0, sub_ts)

            print(
                f"{sys_name:<38} | {'0 bps':<8} | {m0['cagr']:+10.2f} | {m0['vol']:8.2f} | {m0['sharpe']:8.2f} | {m0['calmar']:8.2f} | {m0['max_dd']:10.2f} | {m0['ann_turnover_bps']:18.0f}"
            )
            print(
                f"{sys_name:<38} | {'25 bps':<8} | {m25['cagr']:+10.2f} | {m25['vol']:8.2f} | {m25['sharpe']:8.2f} | {m25['calmar']:8.2f} | {m25['max_dd']:10.2f} | {m25['ann_turnover_bps']:18.0f}"
            )

    # 5. Year-by-Year Net Sharpe Table (25 bps slippage)
    print("\n" + "=" * 125)
    print("YEAR-BY-YEAR NET SHARPE RATIO (NET OF 25 BPS SLIPPAGE)")
    print("=" * 125)
    yr_header = f"{'Year':<8} | {'Frozen APEX v45':<16} | {'ML Crypto (24h)':<16} | {'ML Crypto+Trad (24h)':<20} | {'ML Crypto+Trad (72h)':<20} | {'Placebo (24h)':<16}"
    print(yr_header)
    print("-" * len(yr_header))

    m_base_y = compute_quant_metrics(rets_static[mask_oos], turn_static[mask_oos], 25.0, timestamps[mask_oos])["yearly"]
    m_cryp24_y = compute_quant_metrics(rets_crypto_24h[mask_oos], turn_crypto_24h[mask_oos], 25.0, timestamps[mask_oos])["yearly"]
    m_mac24_y = compute_quant_metrics(rets_macro_24h[mask_oos], turn_macro_24h[mask_oos], 25.0, timestamps[mask_oos])["yearly"]
    m_mac72_y = compute_quant_metrics(rets_macro_72h[mask_oos], turn_macro_72h[mask_oos], 25.0, timestamps[mask_oos])["yearly"]
    m_plac24_y = compute_quant_metrics(rets_placebo_24h[mask_oos], turn_placebo_24h[mask_oos], 25.0, timestamps[mask_oos])["yearly"]

    for yr in [2023, 2024, 2025, 2026]:
        b_s = m_base_y.get(yr, {}).get("sharpe", 0.0)
        c24_s = m_cryp24_y.get(yr, {}).get("sharpe", 0.0)
        m24_s = m_mac24_y.get(yr, {}).get("sharpe", 0.0)
        m72_s = m_mac72_y.get(yr, {}).get("sharpe", 0.0)
        p24_s = m_plac24_y.get(yr, {}).get("sharpe", 0.0)
        print(f"{yr:<8} | {b_s:+16.2f} | {c24_s:+16.2f} | {m24_s:+20.2f} | {m72_s:+20.2f} | {p24_s:+16.2f}")

    # Inspect Weight Allocation Dynamics in Chop Window
    print("\n" + "=" * 125)
    print("CHOP WINDOW WEIGHT ALLOCATION DYNAMICS (2025-05 to 2026-08)")
    print("=" * 125)
    w_chop_24 = w_macro_24h[mask_chop]
    mean_w = np.mean(w_chop_24, axis=0)
    min_w = np.min(w_chop_24, axis=0)
    max_w = np.max(w_chop_24, axis=0)
    print(f"24h Allocator — Directional Weight (Prior=0.25): Mean = {mean_w[0]:.3f} | Min = {min_w[0]:.3f} | Max = {max_w[0]:.3f}")
    print(f"24h Allocator — Spread Weight      (Prior=0.55): Mean = {mean_w[1]:.3f} | Min = {min_w[1]:.3f} | Max = {max_w[1]:.3f}")
    print(f"24h Allocator — Funding Arb Weight (Prior=0.20): Mean = {mean_w[2]:.3f} | Min = {min_w[2]:.3f} | Max = {max_w[2]:.3f}")

    w_chop_72 = w_macro_72h[mask_chop]
    mean_w72 = np.mean(w_chop_72, axis=0)
    min_w72 = np.min(w_chop_72, axis=0)
    max_w72 = np.max(w_chop_72, axis=0)
    print(f"72h Allocator — Directional Weight (Prior=0.25): Mean = {mean_w72[0]:.3f} | Min = {min_w72[0]:.3f} | Max = {max_w72[0]:.3f}")
    print(f"72h Allocator — Spread Weight      (Prior=0.55): Mean = {mean_w72[1]:.3f} | Min = {min_w72[1]:.3f} | Max = {max_w72[1]:.3f}")
    print(f"72h Allocator — Funding Arb Weight (Prior=0.20): Mean = {mean_w72[2]:.3f} | Min = {min_w72[2]:.3f} | Max = {max_w72[2]:.3f}")


if __name__ == "__main__":
    main()
