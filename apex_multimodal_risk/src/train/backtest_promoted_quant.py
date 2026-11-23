"""
Cost-Aware Quant Backtest & Institutional Tear Sheet (EXP-003b).
Evaluates 5 strategies on the 2.90-year news overlap window (2022-07-01 to 2025-05-23)
under 0 bps and realistic 25 bps one-way slippage:
1. Unhedged APEX v45 Baseline
2. Naive Active Liquidation Overlay (p_crash_cal > 0.15 -> scale = 0.0, else 1.0)
3. NLP + Tail-Risk Entry-Gating Overlay (alert_flag == 1 or p_crash_cal > 0.07 -> suppress new risk-increasing entries; hold existing positions)
4. Delayed-Feedback ACI VaR-Budget Sizing (scale = clip(var_budget / |aci_var_5pct|, 0.25, 1.0) with 0.15 deadband)
5. Combined Quant Risk System (Entry-Gating + ACI VaR Deadband Sizing)
"""

import pickle
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from apex_multimodal_risk.src.features.shared_features import MARKET_NUMERICAL_COLS
from apex_multimodal_risk.src.models.conformal import MultiAssetDelayedFeedbackACI
from apex_multimodal_risk.src.models.itransformer import iTransformerOnlyModel
from apex_multimodal_risk.src.train.exp002e_nlp_angles import (
    engineer_nlp_features_and_targets,
)
from apex_multimodal_risk.src.train.train_ablation import (
    build_lookback_tensor,
    eval_itransformer_batched,
)
from apex_multimodal_risk.src.utils import PROJECT_ROOT, get_device, setup_logger

logger = setup_logger("backtest_promoted_quant")

# Ensure research module importable without modifying frozen files
sys.path.insert(0, str(PROJECT_ROOT / "research"))
sys.path.insert(0, str(PROJECT_ROOT))
from research.apex_v45_champion import (
    DIR_HYST,
    DIR_TAKER_FEE,
    DIR_WC,
    DIR_WO,
    DIR_WT,
    HOURS_PER_YEAR,
    LEVERAGE,
    UNIVERSE,
    W_ARB,
    W_DIR,
    W_SPREAD,
    WARMUP_BARS,
    load_data_and_features,
    run_funding_arb_engine,
    run_spread_engine,
)


def run_custom_directional_engine(
    processed: dict[str, pd.DataFrame],
    dataset: dict[str, pd.DataFrame],
    univ: list[str],
    eval_index: pd.DatetimeIndex,
    n: int,
    wts: dict[str, np.ndarray],
    skew_mult_dict: dict[str, np.ndarray],
    entry_gate_mask: np.ndarray | None = None,
    scale_overlay: np.ndarray | None = None,
    slippage_bps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parametric Directional Engine supporting:
    - entry_gate_mask: Suppresses new risk-increasing positions when True (holds existing open trades).
    - scale_overlay: Continuous exposure multiplier.
    - slippage_bps: Execution slippage drag applied to turnover (e.g. 25.0 bps).
    Returns (port_ret, turnover, positions).
    """
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = DIR_WT * df["tsmom"].values + DIR_WC * df["carry"].values + DIR_WO * df["oi_regime"].values

    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps[s][t - 1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5
        xs[sc[1][0]][t] = 1.0
        xs[sc[2][0]][t] = 0.5

    port_ret = np.zeros(n)
    port_turn = np.zeros(n)
    all_shifted_pos = {s: np.zeros(n) for s in univ}

    total_fee_rate = DIR_TAKER_FEE + (slippage_bps / 10000.0)

    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df["hourly_return"].values
        fr = dataset[s].iloc[WARMUP_BARS:]["funding_rate"].values
        w = wts[s]
        comp = comps[s]
        x = xs[s]
        sm = skew_mult_dict[s]

        cur = 0.0
        tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * sm[t] * LEVERAGE

            # 1. Entry Gating Logic: suppress new risk-increasing entries
            if entry_gate_mask is not None and entry_gate_mask[t]:
                if cur == 0.0:
                    ideal = 0.0
                elif np.sign(ideal) != np.sign(cur):
                    ideal = 0.0  # Allow closing current position, but do not flip into opposite new exposure
                elif abs(ideal) > abs(cur):
                    ideal = cur  # Cannot scale up existing exposure

            # 2. Scale Overlay Logic (e.g. Sizing or Liquidation)
            if scale_overlay is not None:
                ideal = ideal * scale_overlay[t]

            # 3. Directional Hysteresis
            if abs(ideal - cur) >= DIR_HYST:
                cur = ideal
            tgt[t] = cur

        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        all_shifted_pos[s] = shifted
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))

        p_ret = ret * shifted
        p_fee = turn * total_fee_rate
        p_fund = np.where(funding_hours, -shifted * fr, 0.0)

        step_pnl = p_ret - p_fee + p_fund
        port_ret += step_pnl
        port_turn += turn

    return port_ret, port_turn, all_shifted_pos


def compute_quant_metrics(
    returns: np.ndarray,
    turnover: np.ndarray,
    slippage_bps: float,
    hedged_mask: np.ndarray,
) -> dict[str, float]:
    """Computes Institutional Quant Tear Sheet Metrics."""
    n = len(returns)
    if n == 0:
        return {}

    years = n / float(HOURS_PER_YEAR)
    cum = np.cumprod(1.0 + returns)
    cummax = np.maximum.accumulate(cum)
    dd_series = (cum - cummax) / (cummax + 1e-8)
    max_dd = float(np.min(dd_series) * 100.0)
    total_ret = float((cum[-1] - 1.0) * 100.0)

    # True CAGR
    if cum[-1] > 0:
        cagr = float(((cum[-1]) ** (1.0 / years) - 1.0) * 100.0)
    else:
        cagr = -100.0

    # Codebase Calmar & True CAGR Calmar
    codebase_calmar = total_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    cagr_calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # Sharpe & Sortino
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns)) + 1e-8
    sharpe = float((mean_ret / std_ret) * np.sqrt(HOURS_PER_YEAR))

    downside = returns[returns < 0]
    downside_std = float(np.std(downside)) + 1e-8 if len(downside) > 0 else 1e-8
    sortino = float((mean_ret * HOURS_PER_YEAR) / (downside_std * np.sqrt(HOURS_PER_YEAR)))

    # CVaR (Expected Shortfall)
    p5 = np.percentile(returns, 5.0)
    p1 = np.percentile(returns, 1.0)
    cvar_95 = float(np.mean(returns[returns <= p5]) * 100.0)
    cvar_99 = float(np.mean(returns[returns <= p1]) * 100.0)

    # Annual Turnover Drag (bps/year under slippage)
    ann_turnover = float(np.mean(turnover) * HOURS_PER_YEAR)
    turnover_drag_bps = float(ann_turnover * slippage_bps)

    # % Hours Hedged / Deleveraged
    pct_hedged = float(np.mean(hedged_mask) * 100.0)

    return {
        "cagr": cagr,
        "codebase_calmar": codebase_calmar,
        "cagr_calmar": cagr_calmar,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "cvar_95": cvar_95,
        "cvar_99": cvar_99,
        "turnover_drag_bps": turnover_drag_bps,
        "pct_hedged": pct_hedged,
        "cum_series": cum,
        "dd_series": dd_series,
    }


def compute_crisis_attribution(
    eval_index: pd.DatetimeIndex,
    returns: np.ndarray,
    episodes: dict[str, tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Calculates cumulative net return and max drawdown across specific crisis windows."""
    ret_series = pd.Series(returns, index=eval_index)
    results = {}
    for name, (start_str, end_str) in episodes.items():
        sub = ret_series.loc[start_str:end_str]
        if len(sub) == 0:
            results[name] = {"net_return": 0.0, "max_dd": 0.0}
            continue
        cum = np.cumprod(1.0 + sub.to_numpy())
        cummax = np.maximum.accumulate(cum)
        dd = float(np.min((cum - cummax) / (cummax + 1e-8)) * 100.0)
        net_ret = float((cum[-1] - 1.0) * 100.0)
        results[name] = {"net_return": net_ret, "max_dd": dd}
    return results


def main():
    logger.info("=" * 80)
    logger.info("STARTING COST-AWARE QUANT BACKTEST & CRISIS ATTRIBUTION (EXP-003b)")
    logger.info("=" * 80)
    t0 = time.time()

    # 1. Load APEX v45 Champion Data
    logger.info("Loading APEX v45 champion datasets and features...")
    dataset, processed, garch_wts, eval_index, n, _split_idx, macro_regime, mtf_skew_mult, arb_mag_scale = (
        load_data_and_features()
    )

    # Component engines for Spread and Funding Arb (unchanged baseline)
    spread_stream, _sp_decomp = run_spread_engine(processed, dataset, UNIVERSE, eval_index, n)
    arb_stream, _arb_decomp = run_funding_arb_engine(dataset, UNIVERSE, eval_index, n, arb_mag_scale)
    spread_filtered = spread_stream * macro_regime

    # 2. News Overlap Mask (2.90 Years: 2022-07-01 to 2025-05-23)
    overlap_mask = (eval_index >= "2022-07-01") & (eval_index <= "2025-05-23 23:59:59")
    eval_overlap = eval_index[overlap_mask]
    n_overlap = len(eval_overlap)
    logger.info(f"News Overlap Window: {eval_overlap[0]} to {eval_overlap[-1]} ({n_overlap} bars, {n_overlap/8760:.2f} years)")

    # 3. Load Calibrated Multimodal Risk Models
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts"
    with open(artifacts_dir / "best_lightgbm_risk.pkl", "rb") as f:
        lgb_bundle = pickle.load(f)
    lgb_model = lgb_bundle["model"]
    platt_scaler = lgb_bundle["platt_scaler"]
    tau_nov = lgb_bundle["gate_thresholds"]["tau_nov"]
    tau_proto = lgb_bundle["gate_thresholds"]["tau_proto"]

    # NLP & Feature Panel
    eval_panel, _meta = engineer_nlp_features_and_targets()

    X_mkt = eval_panel[MARKET_NUMERICAL_COLS].to_numpy(dtype=np.float32)
    nov = eval_panel["embedding_novelty_score"].to_numpy(dtype=np.float32)
    p_max = eval_panel["max_proto_sim_24h"].to_numpy(dtype=np.float32)
    burst = eval_panel["news_burst_ratio"].to_numpy(dtype=np.float32).reshape(-1, 1)

    g_nov = np.maximum(0.0, nov - tau_nov).reshape(-1, 1)
    g_proto = np.maximum(0.0, p_max - tau_proto).reshape(-1, 1)
    alert = ((nov > tau_nov) | (p_max > tau_proto)).astype(np.float32).reshape(-1, 1)

    X_20 = np.concatenate([X_mkt, g_nov, g_proto, alert, burst], axis=1)
    raw_logits = lgb_model.predict(X_20, raw_score=True)
    p_cal = platt_scaler.transform(raw_logits)
    eval_panel["p_crash_cal"] = p_cal
    eval_panel["alert_flag"] = alert.flatten()

    # 4. iTransformer & Delayed-Feedback ACI 5% VaR
    logger.info("Computing Delayed-Feedback ACI 5% VaR across panel...")
    lookback_tensor = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)
    ckpt = torch.load(artifacts_dir / "best_itransformer_var.pt", map_location="cpu", weights_only=False)
    itr_model = iTransformerOnlyModel(lookback_len=48, num_variates=16, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    itr_model.load_state_dict(ckpt["model_state_dict"])
    itr_model.eval()

    means = np.array(ckpt["scaler_means"], dtype=np.float32).reshape(1, 1, 16)
    stds = np.array(ckpt["scaler_stds"], dtype=np.float32).reshape(1, 1, 16)
    norm_lb = (lookback_tensor - means) / (stds + 1e-8)

    device = get_device()
    itr_model.to(device)
    _, _, raw_vars = eval_itransformer_batched(itr_model, torch.tensor(norm_lb, dtype=torch.float32), device, batch_size=256)
    eval_panel["raw_var"] = raw_vars
    eval_panel = eval_panel.sort_values("timestamp").reset_index(drop=True)

    fwd_ret_map = dict(zip(zip(eval_panel["asset"], eval_panel["timestamp"]), eval_panel["fwd_min_ret_6h"]))
    aci = MultiAssetDelayedFeedbackACI(nominal_alpha=0.05, gamma=0.01, window_size=500, delay_hours=6)

    n_panel = len(eval_panel)
    assets_arr = eval_panel["asset"].to_numpy()
    ts_arr = eval_panel["timestamp"].to_numpy()
    rvar_arr = eval_panel["raw_var"].to_numpy()
    aci_vars = np.zeros(n_panel, dtype=np.float32)

    for i in range(n_panel):
        sym = assets_arr[i]
        t_ts = pd.Timestamp(ts_arr[i])
        r_var = float(rvar_arr[i])
        t_del = t_ts - timedelta(hours=6)
        prev_val = fwd_ret_map.get((sym, t_del))
        if prev_val is not None:
            aci.step_feedback(sym, t_del, t_ts, float(prev_val))
        v_adj, _ = aci.predict(sym, r_var, t_ts)
        aci_vars[i] = v_adj

    eval_panel["aci_var_5pct"] = aci_vars

    # Align risk signals hourly
    p_hourly = eval_panel.groupby("timestamp")["p_crash_cal"].max()
    alert_hourly = eval_panel.groupby("timestamp")["alert_flag"].max()
    var_hourly = eval_panel.groupby("timestamp")["aci_var_5pct"].min()  # Conservative worst-asset VaR

    p_aligned = p_hourly.reindex(eval_index).fillna(0.0412).to_numpy()
    alert_aligned = alert_hourly.reindex(eval_index).fillna(0.0).to_numpy()
    var_aligned = var_hourly.reindex(eval_index).fillna(-0.040).to_numpy()

    # 5. Define Strategies & Signals
    # S1: Unhedged Baseline
    # S2: Naive Active Liquidation (p_crash_cal > 0.15 -> scale = 0.0)
    scale_s2 = np.where(p_aligned > 0.15, 0.0, 1.0)
    hedged_s2 = scale_s2 < 1.0

    # S3: NLP + Tail-Risk Entry Gating (alert == 1 or p_crash_cal > 0.07)
    gate_mask_s3 = (alert_aligned == 1.0) | (p_aligned > 0.07)

    # S4: Delayed-Feedback ACI VaR-Budget Sizing (0.15 deadband)
    var_budget = 0.040  # 4.0% nominal 5% VaR budget
    raw_scale_s4 = np.clip(var_budget / np.abs(var_aligned), 0.25, 1.0)
    scale_s4 = np.ones(n, dtype=np.float64)
    s_curr = 1.0
    for t in range(n):
        s_tgt = raw_scale_s4[t]
        if abs(s_tgt - s_curr) >= 0.15:
            s_curr = s_tgt
        scale_s4[t] = s_curr
    hedged_s4 = scale_s4 < 1.0

    # S5: Combined Quant Risk System (Entry Gating + ACI VaR Deadband Sizing)
    # Uses gate_mask_s3 + scale_s4

    strategies = [
        ("1. Unhedged APEX v45 Baseline", None, None, np.zeros(n, dtype=bool)),
        ("2. Naive Active Liquidation Overlay", None, scale_s2, hedged_s2),
        ("3. NLP + Tail-Risk Entry-Gating Overlay", gate_mask_s3, None, gate_mask_s3),
        ("4. Delayed ACI VaR-Budget Sizing", None, scale_s4, hedged_s4),
        ("5. Combined Quant Risk System", gate_mask_s3, scale_s4, gate_mask_s3 | hedged_s4),
    ]

    # Crisis episodes
    crisis_episodes = {
        "FTX Collapse (Nov 2022)": ("2022-11-06", "2022-11-14"),
        "SVB / USDC Depeg (Mar 2023)": ("2023-03-09", "2023-03-14"),
        "SEC Lawsuit (Jun 2023)": ("2023-06-05", "2023-06-12"),
        "Aug 2024 Global Unwind": ("2024-08-02", "2024-08-07"),
    }

    # Evaluate across 0 bps and 25 bps slippage
    results_0bps = {}
    results_25bps = {}
    crisis_res_25bps = {}

    for name, gate, scale, hedge_mask in strategies:
        # 0 bps slippage
        dir_ret_0, dir_turn_0, _ = run_custom_directional_engine(
            processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult,
            entry_gate_mask=gate, scale_overlay=scale, slippage_bps=0.0
        )
        dir_filt_0 = dir_ret_0 * macro_regime
        champ_0 = W_DIR * dir_filt_0 + W_SPREAD * spread_filtered + W_ARB * arb_stream
        m0 = compute_quant_metrics(
            champ_0[overlap_mask],
            W_DIR * dir_turn_0[overlap_mask],
            slippage_bps=0.0,
            hedged_mask=hedge_mask[overlap_mask],
        )
        results_0bps[name] = m0

        # 25 bps slippage
        dir_ret_25, dir_turn_25, _ = run_custom_directional_engine(
            processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult,
            entry_gate_mask=gate, scale_overlay=scale, slippage_bps=25.0
        )
        dir_filt_25 = dir_ret_25 * macro_regime
        champ_25 = W_DIR * dir_filt_25 + W_SPREAD * spread_filtered + W_ARB * arb_stream
        m25 = compute_quant_metrics(
            champ_25[overlap_mask],
            W_DIR * dir_turn_25[overlap_mask],
            slippage_bps=25.0,
            hedged_mask=hedge_mask[overlap_mask],
        )
        results_25bps[name] = m25

        # Crisis attribution under 25 bps
        cr = compute_crisis_attribution(eval_index, champ_25, crisis_episodes)
        crisis_res_25bps[name] = cr

    elapsed = time.time() - t0
    logger.info(f"Quant backtest completed in {elapsed:.1f}s")

    # 6. Print Institutional Quant Tear Sheet Table
    print("\n" + "=" * 140)
    print("EXP-003b: INSTITUTIONAL QUANT TEAR SHEET TABLE (2.90-Year News Overlap: 2022-07-01 to 2025-05-23)")
    print("=" * 140)
    print(f"{'Strategy Sleeve':40s} | {'Slippage':8s} | {'CAGR %':8s} | {'Code Cal':8s} | {'True Cal':8s} | {'Sharpe':8s} | {'Sortino':8s} | {'MaxDD %':8s} | {'95% CVaR':8s} | {'99% CVaR':8s} | {'Turn Drag':9s} | {'% Hedged':8s}")
    print("-" * 140)

    for name, _, _, _ in strategies:
        m0 = results_0bps[name]
        m25 = results_25bps[name]
        # 0 bps row
        print(
            f"{name:40s} | {'0 bps':8s} | "
            f"{m0['cagr']:<8.2f} | {m0['codebase_calmar']:<8.2f} | {m0['cagr_calmar']:<8.2f} | "
            f"{m0['sharpe']:<8.2f} | {m0['sortino']:<8.2f} | {m0['max_dd']:<8.1f} | "
            f"{m0['cvar_95']:<8.2f} | {m0['cvar_99']:<8.2f} | {m0['turnover_drag_bps']:<9.1f} | {m0['pct_hedged']:<8.1f}%"
        )
        # 25 bps row
        print(
            f"{'  + 25 bps Slippage Stress':40s} | {'25 bps':8s} | "
            f"{m25['cagr']:<8.2f} | {m25['codebase_calmar']:<8.2f} | {m25['cagr_calmar']:<8.2f} | "
            f"{m25['sharpe']:<8.2f} | {m25['sortino']:<8.2f} | {m25['max_dd']:<8.1f} | "
            f"{m25['cvar_95']:<8.2f} | {m25['cvar_99']:<8.2f} | {m25['turnover_drag_bps']:<9.1f} | {m25['pct_hedged']:<8.1f}%"
        )
        print("-" * 140)
    print("=" * 140)

    # 7. Print 4-Shock Crisis Attribution Table
    print("\n" + "=" * 135)
    print("EXP-003b: 4-SHOCK CRISIS ATTRIBUTION TABLE (Net Return % / Max Drawdown % Under 25 bps Slippage)")
    print("=" * 135)
    print(f"{'Strategy Sleeve':38s} | {'FTX Collapse (Nov 22)':22s} | {'SVB / USDC (Mar 23)':22s} | {'SEC Lawsuit (Jun 23)':22s} | {'Aug 2024 Unwind':22s}")
    print("-" * 135)

    for name, _, _, _ in strategies:
        cr = crisis_res_25bps[name]
        col_ftx = f"{cr['FTX Collapse (Nov 2022)']['net_return']:+.2f}% / {cr['FTX Collapse (Nov 2022)']['max_dd']:.1f}%"
        col_svb = f"{cr['SVB / USDC Depeg (Mar 2023)']['net_return']:+.2f}% / {cr['SVB / USDC Depeg (Mar 2023)']['max_dd']:.1f}%"
        col_sec = f"{cr['SEC Lawsuit (Jun 2023)']['net_return']:+.2f}% / {cr['SEC Lawsuit (Jun 2023)']['max_dd']:.1f}%"
        col_aug = f"{cr['Aug 2024 Global Unwind']['net_return']:+.2f}% / {cr['Aug 2024 Global Unwind']['max_dd']:.1f}%"
        print(f"{name:38s} | {col_ftx:22s} | {col_svb:22s} | {col_sec:22s} | {col_aug:22s}")
    print("=" * 135)

    return {
        "results_0bps": results_0bps,
        "results_25bps": results_25bps,
        "crisis_res_25bps": crisis_res_25bps,
    }


if __name__ == "__main__":
    main()
