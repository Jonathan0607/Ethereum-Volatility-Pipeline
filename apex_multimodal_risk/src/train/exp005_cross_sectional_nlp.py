"""
EXP-005 Cross-Sectional NLP Alpha Evaluation.

1. Cross-Sectional Spearman Rank IC, ICIR, and Newey-West t-stats across forward 24h, 48h, 72h
   excess returns (full panel and yearly: 2022, 2023, 2024, 2025).
2. News-Timestamp Shuffle Placebo test (verifying IC collapses to ~0).
3. Sleeve 1: 15-Coin Dollar-Neutral Long/Short Post-News Drift (Top-3 Long, Bottom-3 Short,
   rebalance 24h & 12h, vol-normalized weights, net of 0 bps and 25 bps slippage + funding fees).
4. Sleeve 2: Exposure-Neutral Conviction Tilt on APEX v45 Directional engine ([0.65x, 1.35x]).
5. Portfolio Blend: 90/10 and 80/20 APEX v45 + Sleeve 1 under 0 bps and 25 bps slippage.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
DATA_DIR = PROJECT_ROOT / "apex_multimodal_risk" / "data" / "processed"
HOURS_PER_YEAR = 8760.0

ASSETS_15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT",
    "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT",
    "UNIUSDT", "ATOMUSDT", "NEARUSDT", "DOTUSDT", "ETCUSDT"
]


def compute_newey_west_t(series: np.ndarray, max_lags: int) -> tuple[float, float, float]:
    """Compute mean, Newey-West standard error, and t-statistic for a 1D time series."""
    clean = series[~np.isnan(series)]
    n = len(clean)
    if n <= max_lags + 1:
        return 0.0, 0.0, 0.0

    mean_val = float(np.mean(clean))
    demeaned = clean - mean_val

    # Gamma 0
    gamma0 = float(np.mean(demeaned**2))

    # Gammas with Bartlett kernel weights
    gamma_sum = 0.0
    for lag in range(1, max_lags + 1):
        weight = 1.0 - (lag / (max_lags + 1.0))
        gamma_l = float(np.mean(demeaned[lag:] * demeaned[:-lag]))
        gamma_sum += 2.0 * weight * gamma_l

    var_nw = (gamma0 + gamma_sum) / float(n)
    if var_nw <= 0.0:
        se_nw = float(np.std(clean) / np.sqrt(n))
    else:
        se_nw = float(np.sqrt(var_nw))

    t_stat = mean_val / se_nw if se_nw > 1e-12 else 0.0
    return mean_val, se_nw, t_stat


def compute_vectorized_rank_ic(sig_mat: np.ndarray, tgt_mat: np.ndarray) -> np.ndarray:
    """Computes cross-sectional Spearman Rank IC across columns for each row."""
    # Rank across assets (axis=1)
    ra = rankdata(sig_mat, axis=1)
    rb = rankdata(tgt_mat, axis=1)

    ra_dm = ra - ra.mean(axis=1, keepdims=True)
    rb_dm = rb - rb.mean(axis=1, keepdims=True)

    cov = (ra_dm * rb_dm).sum(axis=1)
    var_a = (ra_dm**2).sum(axis=1)
    var_b = (rb_dm**2).sum(axis=1)

    denom = np.sqrt(var_a * var_b)
    valid = denom > 1e-12
    ic = np.zeros(len(sig_mat), dtype=np.float64)
    ic[valid] = cov[valid] / denom[valid]
    return ic


def evaluate_ic_gate(
    df: pd.DataFrame, signals: list[str], horizons: list[int] | None = None
) -> dict[str, Any]:
    """
    Computes cross-sectional Spearman Rank IC across 15 assets for each horizon.
    Returns full panel and yearly breakdown (2022, 2023, 2024, 2025).
    """
    if horizons is None:
        horizons = [24, 48, 72]
    logger.info("Evaluating Cross-Sectional Information Coefficient (IC) Gate (Vectorized)...")
    results = {}

    timestamps = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()
    years = timestamps.year
    year_labels = ["Full", 2022, 2023, 2024, 2025]

    for sig in signals:
        results[sig] = {}
        # Pivot signal [T, 15]
        sig_pivot = df.pivot(index="timestamp", columns="asset", values=sig).reindex(index=timestamps, columns=ASSETS_15).fillna(0.0).values

        for h in horizons:
            tgt_col = f"xs_ret_{h}h"
            tgt_pivot = df.pivot(index="timestamp", columns="asset", values=tgt_col).reindex(index=timestamps, columns=ASSETS_15).values

            # Exclude last h bars where forward return is NaN
            sig_slice = sig_pivot[:-h]
            tgt_slice = tgt_pivot[:-h]
            ts_years = years[:-h]

            ic_series = compute_vectorized_rank_ic(sig_slice, tgt_slice)

            results[sig][f"{h}h"] = {}
            for yr in year_labels:
                if yr == "Full":
                    sub_ic = ic_series
                else:
                    sub_ic = ic_series[ts_years == yr]

                if len(sub_ic) > 100:
                    mean_ic, se_nw, t_stat = compute_newey_west_t(sub_ic, max_lags=h)
                    std_ic = float(np.std(sub_ic))
                    ann_factor = np.sqrt(HOURS_PER_YEAR / float(h))
                    icir = (mean_ic / std_ic * ann_factor) if std_ic > 1e-6 else 0.0
                    results[sig][f"{h}h"][yr] = {
                        "mean_ic": mean_ic * 100.0,
                        "std_ic": std_ic * 100.0,
                        "icir": icir,
                        "t_stat": t_stat,
                        "n_bars": len(sub_ic),
                    }
                else:
                    results[sig][f"{h}h"][yr] = {
                        "mean_ic": 0.0,
                        "std_ic": 0.0,
                        "icir": 0.0,
                        "t_stat": 0.0,
                        "n_bars": len(sub_ic),
                    }

    return results


def run_shuffle_placebo(df: pd.DataFrame, signal: str = "rel_sent", h: int = 24) -> dict[str, Any]:
    """
    News-Timestamp Shuffle Placebo:
    1. Cross-Sectional Asset Shuffle Placebo (destroys contemporaneous signal-asset mapping).
    2. Time-Row Permutation Placebo (destroys causal forward timing).
    """
    logger.info(f"Running Shuffle Placebos on {signal} (h={h}h)...")
    timestamps = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()
    sig_pivot = df.pivot(index="timestamp", columns="asset", values=signal).reindex(index=timestamps, columns=ASSETS_15).fillna(0.0).values
    tgt_pivot = df.pivot(index="timestamp", columns="asset", values=f"xs_ret_{h}h").reindex(index=timestamps, columns=ASSETS_15).values

    sig_slice = sig_pivot[:-h]
    tgt_slice = tgt_pivot[:-h]

    np.random.seed(42)

    # Placebo 1: Cross-Sectional Asset Shuffle
    shuffled_xs = np.array([np.random.permutation(row) for row in sig_slice])
    ic_xs = compute_vectorized_rank_ic(shuffled_xs, tgt_slice)
    mean_xs, se_xs, t_xs = compute_newey_west_t(ic_xs, max_lags=h)
    std_xs = float(np.std(ic_xs))
    icir_xs = (mean_xs / std_xs * np.sqrt(HOURS_PER_YEAR / float(h))) if std_xs > 1e-6 else 0.0

    # Placebo 2: Time-Row Permutation
    perm = np.random.permutation(len(sig_slice))
    shuffled_time = sig_slice[perm]
    ic_time = compute_vectorized_rank_ic(shuffled_time, tgt_slice)
    mean_time, se_time, t_time = compute_newey_west_t(ic_time, max_lags=h)
    std_time = float(np.std(ic_time))
    icir_time = (mean_time / std_time * np.sqrt(HOURS_PER_YEAR / float(h))) if std_time > 1e-6 else 0.0

    return {
        "xs_shuffle": {
            "mean_ic": mean_xs * 100.0,
            "std_ic": std_xs * 100.0,
            "icir": icir_xs,
            "t_stat": t_xs,
            "n_bars": len(ic_xs),
        },
        "time_shuffle": {
            "mean_ic": mean_time * 100.0,
            "std_ic": std_time * 100.0,
            "icir": icir_time,
            "t_stat": t_time,
            "n_bars": len(ic_time),
        },
    }


def compute_quant_tearsheet(
    returns: np.ndarray,
    turnover: np.ndarray,
    slippage_bps: float,
    timestamps: pd.DatetimeIndex,
    funding_costs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Computes full institutional quant tear sheet."""
    n = len(returns)
    years = n / HOURS_PER_YEAR
    fee_drag = turnover * (slippage_bps / 10000.0)
    net_rets = returns - fee_drag
    if funding_costs is not None:
        net_rets = net_rets - funding_costs

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

    df_temp = pd.DataFrame({"ts": timestamps, "net_ret": net_rets})
    df_temp["year"] = df_temp["ts"].dt.year
    yearly = {}
    for yr, ygrp in df_temp.groupby("year"):
        y_cum = np.cumprod(1.0 + ygrp["net_ret"].values)
        y_cummax = np.maximum.accumulate(y_cum)
        y_dd = float(np.min((y_cum - y_cummax) / (y_cummax + 1e-8)) * 100.0)
        y_ret = float((y_cum[-1] - 1.0) * 100.0)
        y_vol = float(np.std(ygrp["net_ret"].values) * np.sqrt(HOURS_PER_YEAR) * 100.0)
        y_sharpe = float(y_ret / y_vol) if y_vol > 1e-4 else 0.0
        yearly[yr] = {"return": y_ret, "vol": y_vol, "sharpe": y_sharpe, "max_dd": y_dd}

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
        "cum_equity": cum,
    }


def backtest_sleeve_1(
    df: pd.DataFrame,
    rebalance_freq_h: int = 24,
    signal_col: str = "rel_sent",
    top_k: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    15-Coin Dollar-Neutral Long/Short Post-News Drift:
    - Long top-k, Short bottom-k
    - Vol-normalized weights: w_i = sign / vol_24h, normalized so sum(long) = +0.5, sum(short) = -0.5
    - Rebalances every rebalance_freq_h hours
    """
    logger.info(f"Backtesting Sleeve 1 (freq={rebalance_freq_h}h, signal={signal_col}, top_k={top_k})...")
    df = df.sort_values(["timestamp", "asset"]).reset_index(drop=True)

    df["ret_1h"] = df.groupby("asset")["close"].transform(lambda s: s.pct_change().fillna(0.0))
    df["vol_24h"] = (
        df.groupby("asset")["ret_1h"]
        .transform(lambda s: s.shift(1).rolling(24, min_periods=12).std().fillna(0.02))
    )
    df["vol_24h"] = np.maximum(df["vol_24h"].values, 0.005)

    timestamps = pd.DatetimeIndex(df["timestamp"].unique()).sort_values()
    n_bars = len(timestamps)

    p_signal = df.pivot(index="timestamp", columns="asset", values=signal_col).reindex(index=timestamps, columns=ASSETS_15).fillna(0.0).values
    p_ret1h = df.pivot(index="timestamp", columns="asset", values="ret_1h").reindex(index=timestamps, columns=ASSETS_15).fillna(0.0).values
    p_fund = df.pivot(index="timestamp", columns="asset", values="funding_rate").reindex(index=timestamps, columns=ASSETS_15).fillna(0.0).values
    p_vol = df.pivot(index="timestamp", columns="asset", values="vol_24h").reindex(index=timestamps, columns=ASSETS_15).fillna(0.02).values

    gross_returns = np.zeros(n_bars, dtype=np.float64)
    turnover = np.zeros(n_bars, dtype=np.float64)
    funding_costs = np.zeros(n_bars, dtype=np.float64)

    cur_weights = np.zeros(len(ASSETS_15), dtype=np.float64)
    funding_hours = timestamps.hour.isin([0, 8, 16])

    for t in range(n_bars):
        if t % rebalance_freq_h == 0:
            sig_t = p_signal[t]
            vol_t = p_vol[t]

            sorted_idx = np.argsort(sig_t)
            short_idx = sorted_idx[:top_k]
            long_idx = sorted_idx[-top_k:]

            target_w = np.zeros(len(ASSETS_15), dtype=np.float64)

            w_long_raw = 1.0 / vol_t[long_idx]
            target_w[long_idx] = 0.5 * (w_long_raw / np.sum(w_long_raw))

            w_short_raw = 1.0 / vol_t[short_idx]
            target_w[short_idx] = -0.5 * (w_short_raw / np.sum(w_short_raw))

            turn = np.sum(np.abs(target_w - cur_weights))
            turnover[t] = turn
            cur_weights = target_w.copy()
        else:
            turnover[t] = 0.0

        gross_returns[t] = np.sum(cur_weights * p_ret1h[t])
        if funding_hours[t]:
            funding_costs[t] = np.sum(cur_weights * p_fund[t])

    return gross_returns, turnover, funding_costs, timestamps


def backtest_sleeve_2_tilt(
    df: pd.DataFrame,
    common_idx: pd.DatetimeIndex,
    pos_dir_base_c: dict[str, np.ndarray],
    hourly_ret_c: dict[str, np.ndarray],
    funding_rate_c: dict[str, np.ndarray],
    macro_regime_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sleeve 2: Exposure-Neutral Conviction Tilt on APEX Directional:
    Multiplies base directional position by c_t in [0.65x, 1.35x] based on relative sentiment.
    """
    logger.info("Backtesting Sleeve 2 (APEX Directional Conviction Tilt [0.65x, 1.35x])...")
    from research.apex_v45_champion import (
        DIR_TAKER_FEE,
        UNIVERSE,
    )

    n = len(common_idx)
    funding_hours = common_idx.hour.isin([0, 8, 16])

    p_sig = df.pivot(index="timestamp", columns="asset", values="rel_sent").reindex(index=common_idx, columns=ASSETS_15).fillna(0.0)

    port_ret = np.zeros(n, dtype=np.float64)
    total_turn = np.zeros(n, dtype=np.float64)
    cur_pos = {s: 0.0 for s in UNIVERSE}

    for t in range(n):
        for s in UNIVERSE:
            base_pos = pos_dir_base_c[s][t]
            sig_val = p_sig.loc[common_idx[t], s] if s in p_sig.columns else 0.0
            tilt_mult = np.clip(1.0 + 0.35 * sig_val, 0.65, 1.35)
            tilted_pos = base_pos * tilt_mult * macro_regime_c[t]

            turn = abs(tilted_pos - cur_pos[s])
            fee = turn * DIR_TAKER_FEE if turn > 0.001 else 0.0
            cur_pos[s] = tilted_pos
            total_turn[t] += turn

            ret = hourly_ret_c[s][t]
            p_ret = ret * cur_pos[s]
            p_fund = -cur_pos[s] * funding_rate_c[s][t] if funding_hours[t] else 0.0
            port_ret[t] += p_ret - fee + p_fund

    return port_ret, total_turn


def run_apex_v45_blend(
    df: pd.DataFrame,
    sleeve1_rets: np.ndarray,
    sleeve1_turn: np.ndarray,
    timestamps: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Loads APEX v45 Champion baseline and evaluates 90/10 and 80/20 blends + Sleeve 2 tilt."""
    logger.info("Loading APEX v45 Champion baseline for portfolio blend...")
    from apex_multimodal_risk.src.train.walkforward_equity_curve import (
        run_custom_directional_engine,
        run_custom_spread_engine,
    )
    from research.apex_v45_champion import (
        UNIVERSE,
        W_ARB,
        W_DIR,
        W_SPREAD,
        WARMUP_BARS,
        load_data_and_features,
        run_funding_arb_engine,
    )

    dataset, processed, garch_wts, eval_index, n, _split_idx, macro_regime, mtf_skew_mult, arb_mag_scale = (
        load_data_and_features()
    )

    dir_ret, _, pos_dir_base = run_custom_directional_engine(
        processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult, gate_mask=None
    )
    sp_ret, _, _ = run_custom_spread_engine(
        processed, dataset, UNIVERSE, eval_index, n, gate_mask=None
    )
    arb_ret, _ = run_funding_arb_engine(dataset, UNIVERSE, eval_index, n, arb_mag_scale)

    dir_filtered = dir_ret * macro_regime
    sp_filtered = sp_ret * macro_regime
    apex_ret_unhedged = W_DIR * dir_filtered + W_SPREAD * sp_filtered + W_ARB * arb_ret

    common_idx = eval_index.intersection(timestamps)
    # Strictly OOS 2022-11-01 to 2025-05-23
    oos_mask = common_idx >= pd.Timestamp("2022-11-01 00:00:00+00:00")
    common_idx_oos = common_idx[oos_mask]
    logger.info(f"OOS evaluation bars: {len(common_idx_oos)} ({common_idx_oos[0]} to {common_idx_oos[-1]})")

    loc_eval = eval_index.get_indexer(common_idx_oos)
    s_apex = apex_ret_unhedged[loc_eval]
    sp_filtered_c = sp_filtered[loc_eval]
    arb_ret_c = arb_ret[loc_eval]
    macro_regime_c = macro_regime[loc_eval]
    pos_dir_base_c = {s: pos_dir_base[s][loc_eval] for s in UNIVERSE}
    hourly_ret_c = {s: processed[s].iloc[WARMUP_BARS:]["hourly_return"].values[loc_eval] for s in UNIVERSE}
    funding_rate_c = {s: dataset[s].iloc[WARMUP_BARS:]["funding_rate"].values[loc_eval] for s in UNIVERSE}

    s_sleeve_ret = pd.Series(sleeve1_rets, index=timestamps).loc[common_idx_oos].values
    s_sleeve_turn = pd.Series(sleeve1_turn, index=timestamps).loc[common_idx_oos].values

    corr = float(np.corrcoef(s_apex, s_sleeve_ret)[0, 1])
    logger.info(f"Pairwise hourly return correlation (APEX v45 vs Sleeve 1): {corr:.4f}")

    apex_0bps = compute_quant_tearsheet(s_apex, np.zeros_like(s_apex), 0.0, common_idx_oos)
    apex_25bps = compute_quant_tearsheet(s_apex, np.zeros_like(s_apex), 25.0, common_idx_oos)

    slv_0bps = compute_quant_tearsheet(s_sleeve_ret, s_sleeve_turn, 0.0, common_idx_oos)
    slv_25bps = compute_quant_tearsheet(s_sleeve_ret, s_sleeve_turn, 25.0, common_idx_oos)

    # Sleeve 2: Conviction tilt
    dir_tilt_ret, dir_tilt_turn = backtest_sleeve_2_tilt(
        df, common_idx_oos, pos_dir_base_c, hourly_ret_c, funding_rate_c, macro_regime_c
    )
    apex_tilt_ret = W_DIR * dir_tilt_ret + W_SPREAD * sp_filtered_c + W_ARB * arb_ret_c
    slv2_0bps = compute_quant_tearsheet(apex_tilt_ret, dir_tilt_turn * W_DIR, 0.0, common_idx_oos)
    slv2_25bps = compute_quant_tearsheet(apex_tilt_ret, dir_tilt_turn * W_DIR, 25.0, common_idx_oos)

    # 90/10 Blend
    ret_90_10 = 0.90 * s_apex + 0.10 * s_sleeve_ret
    turn_90_10 = 0.10 * s_sleeve_turn
    b90_0bps = compute_quant_tearsheet(ret_90_10, turn_90_10, 0.0, common_idx_oos)
    b90_25bps = compute_quant_tearsheet(ret_90_10, turn_90_10, 25.0, common_idx_oos)

    # 80/20 Blend
    ret_80_20 = 0.80 * s_apex + 0.20 * s_sleeve_ret
    turn_80_20 = 0.20 * s_sleeve_turn
    b80_0bps = compute_quant_tearsheet(ret_80_20, turn_80_20, 0.0, common_idx_oos)
    b80_25bps = compute_quant_tearsheet(ret_80_20, turn_80_20, 25.0, common_idx_oos)

    return {
        "correlation": corr,
        "apex_0bps": apex_0bps,
        "apex_25bps": apex_25bps,
        "slv_0bps": slv_0bps,
        "slv_25bps": slv_25bps,
        "slv2_0bps": slv2_0bps,
        "slv2_25bps": slv2_25bps,
        "b90_0bps": b90_0bps,
        "b90_25bps": b90_25bps,
        "b80_0bps": b80_0bps,
        "b80_25bps": b80_25bps,
    }


def main():
    feat_path = DATA_DIR / "universe_15_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}")

    logger.info(f"Loading {feat_path}...")
    df = pd.read_parquet(feat_path)
    logger.info(f"Loaded {len(df)} rows across {df['asset'].nunique()} assets.")

    # 1. Signal IC Gate
    signals = ["rel_sent", "rel_surprise", "sent_24h"]
    ic_results = evaluate_ic_gate(df, signals=signals, horizons=[24, 48, 72])

    print("\n" + "=" * 80)
    print("MANDATORY CROSS-SECTIONAL SIGNAL IC GATE (15-COIN PANEL)")
    print("=" * 80)
    for sig in signals:
        print(f"\n--- Signal: {sig} ---")
        for h in [24, 48, 72]:
            print(f"Horizon {h}h:")
            for yr in ["Full", 2022, 2023, 2024, 2025]:
                res = ic_results[sig][f"{h}h"][yr]
                print(
                    f"  {str(yr):<6} | Mean IC: {res['mean_ic']:+6.2f}% | "
                    f"Std IC: {res['std_ic']:5.2f}% | ICIR: {res['icir']:+5.2f} | "
                    f"NW t-stat: {res['t_stat']:+5.2f} | Bars: {res['n_bars']}"
                )

    # 2. Shuffle Placebo
    print("\n" + "=" * 80)
    print("NEWS-TIMESTAMP SHUFFLE PLACEBO TEST")
    print("=" * 80)
    placebo_res = run_shuffle_placebo(df, signal="rel_sent", h=24)
    p_xs = placebo_res["xs_shuffle"]
    p_t = placebo_res["time_shuffle"]
    print(
        f"Cross-Sectional Shuffle Placebo (rel_sent, 24h): Mean IC: {p_xs['mean_ic']:+6.2f}% | "
        f"Std IC: {p_xs['std_ic']:5.2f}% | ICIR: {p_xs['icir']:+5.2f} | "
        f"NW t-stat: {p_xs['t_stat']:+5.2f} | Bars: {p_xs['n_bars']}"
    )
    print(
        f"Time-Permutation Placebo (rel_sent, 24h):         Mean IC: {p_t['mean_ic']:+6.2f}% | "
        f"Std IC: {p_t['std_ic']:5.2f}% | ICIR: {p_t['icir']:+5.2f} | "
        f"NW t-stat: {p_t['t_stat']:+5.2f} | Bars: {p_t['n_bars']}"
    )

    # 3. Sleeve 1 Backtest
    print("\n" + "=" * 80)
    print("SLEEVE 1: 15-COIN DOLLAR-NEUTRAL LONG/SHORT POST-NEWS DRIFT (24h Rebalance)")
    print("=" * 80)
    g_ret, turn, fund, ts = backtest_sleeve_1(df, rebalance_freq_h=24, signal_col="rel_sent", top_k=3)

    ts_oos = ts >= pd.Timestamp("2022-11-01 00:00:00+00:00")
    g_ret_oos = g_ret[ts_oos]
    turn_oos = turn[ts_oos]
    fund_oos = fund[ts_oos]
    ts_oos_dt = ts[ts_oos]

    slv1_0 = compute_quant_tearsheet(g_ret_oos, turn_oos, 0.0, ts_oos_dt, funding_costs=fund_oos)
    slv1_25 = compute_quant_tearsheet(g_ret_oos, turn_oos, 25.0, ts_oos_dt, funding_costs=fund_oos)

    print(f"Sleeve 1 (0 bps):  CAGR: {slv1_0['cagr']:+6.2f}% | Vol: {slv1_0['vol']:5.2f}% | Sharpe: {slv1_0['sharpe']:5.2f} | Max DD: {slv1_0['max_dd']:6.2f}% | Turn: {slv1_0['ann_turnover_bps']:.0f} bps/yr")
    print(f"Sleeve 1 (25 bps): CAGR: {slv1_25['cagr']:+6.2f}% | Vol: {slv1_25['vol']:5.2f}% | Sharpe: {slv1_25['sharpe']:5.2f} | Max DD: {slv1_25['max_dd']:6.2f}% | Turn: {slv1_25['ann_turnover_bps']:.0f} bps/yr")

    # 4. APEX v45 Blend
    print("\n" + "=" * 80)
    print("APEX v45 CHAMPION + SLEEVE 1 & 2 PORTFOLIO BLEND")
    print("=" * 80)
    blend_res = run_apex_v45_blend(df, g_ret, turn, ts)

    print(f"Correlation (APEX v45 vs Sleeve 1): {blend_res['correlation']:+.4f}")
    print("\n--- Summary Performance Table (OOS 2022-11-01 to 2025-05-23) ---")
    rows = [
        ("APEX v45 Champion (0 bps)", blend_res["apex_0bps"]),
        ("APEX v45 Champion (25 bps)", blend_res["apex_25bps"]),
        ("Sleeve 1 Standalone (0 bps)", blend_res["slv_0bps"]),
        ("Sleeve 1 Standalone (25 bps)", blend_res["slv_25bps"]),
        ("Sleeve 2 Directional Tilt (0 bps)", blend_res["slv2_0bps"]),
        ("Sleeve 2 Directional Tilt (25 bps)", blend_res["slv2_25bps"]),
        ("90% APEX / 10% Sleeve 1 (0 bps)", blend_res["b90_0bps"]),
        ("90% APEX / 10% Sleeve 1 (25 bps)", blend_res["b90_25bps"]),
        ("80% APEX / 20% Sleeve 1 (0 bps)", blend_res["b80_0bps"]),
        ("80% APEX / 20% Sleeve 1 (25 bps)", blend_res["b80_25bps"]),
    ]

    header = f"{'Configuration':<35} | {'CAGR (%)':<10} | {'Vol (%)':<8} | {'Sharpe':<8} | {'Calmar':<8} | {'Max DD (%)':<10} | {'Turnover (bps/yr)':<18}"
    print(header)
    print("-" * len(header))
    for name, m in rows:
        print(
            f"{name:<35} | {m['cagr']:+10.2f} | {m['vol']:8.2f} | {m['sharpe']:8.2f} | {m['calmar']:8.2f} | {m['max_dd']:10.2f} | {m['ann_turnover_bps']:18.0f}"
        )


if __name__ == "__main__":
    main()
