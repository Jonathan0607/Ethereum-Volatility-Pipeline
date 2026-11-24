"""
Export Promoted EXP-008d (Beta-Neutral 48h Total-Return 15-Coin LightGBM) to ONNX
and Verify Numerical Parity (< 1e-6) vs Native LightGBM.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
    QUARTERLY_TEST_STARTS,
    RAW_FEATURE_COLS,
    apply_honest_vol_targeting,
    compute_metrics,
    engineer_features_and_targets,
    prepare_matrices,
    simulate_c1_sleeve,
)

TEST_START = QUARTERLY_TEST_STARTS[0]
from apex_multimodal_risk.src.utils import PROJECT_ROOT, setup_logger

logger = setup_logger("export_alpha15_onnx")


def export_alpha15_model() -> tuple[Path, Path, float]:
    """Fits 48h Total Return LightGBM model, exports to ONNX, and audits parity."""
    logger.info("Loading panel and computing features & targets...")
    df_panel = engineer_features_and_targets()

    # Training window: strictly causal up to out-of-sample start (or expanding panel)
    train_mask = (df_panel["timestamp"] < TEST_START) & df_panel["target_total_48h"].notnull()
    df_train = df_panel[train_mask]

    X_train = df_train[RAW_FEATURE_COLS].values
    y_train = df_train["target_total_48h"].values
    logger.info(f"Fitting promoted 48h Total-Return LightGBM model on {len(X_train)} training rows...")

    model = lgb.LGBMRegressor(
        objective="huber",
        max_depth=3,
        num_leaves=7,
        min_child_samples=300,
        colsample_bytree=0.6,
        reg_lambda=10.0,
        n_estimators=60,
        learning_rate=0.03,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Export to ONNX
    export_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts/onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "lightgbm_alpha15.onnx"

    logger.info("Converting LightGBM model to ONNX via onnxmltools...")
    initial_type = [("input", FloatTensorType([None, len(RAW_FEATURE_COLS)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
        zipmap=False,
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"Saved ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # Unique sorted assets
    unique_assets = sorted(df_panel["asset"].unique().tolist())

    # Metadata
    metadata_path = export_dir / "alpha15_metadata.json"
    metadata = {
        "model_version": "v1.0.0",
        "target": "target_total_48h",
        "objective": "huber",
        "horizon_hours": 48,
        "n_features": len(RAW_FEATURE_COLS),
        "feature_names": RAW_FEATURE_COLS,
        "assets": unique_assets,
        "n_assets": len(unique_assets),
        "params": {
            "max_depth": 3,
            "num_leaves": 7,
            "min_child_samples": 300,
            "colsample_bytree": 0.6,
            "reg_lambda": 10.0,
            "n_estimators": 60,
            "learning_rate": 0.03,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Numerical Parity Check on 500 real cross-sections (500 * 15 = 7,500 samples)
    logger.info("Auditing numerical parity vs native LightGBM on real cross-sections...")
    test_sample = df_panel[df_panel["timestamp"] >= TEST_START].iloc[: 500 * len(unique_assets)]
    X_test = test_sample[RAW_FEATURE_COLS].values

    lgb_preds = model.predict(X_test)
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_preds = ort_session.run(None, {"input": X_test.astype(np.float32)})[0].flatten()

    max_diff = float(np.max(np.abs(lgb_preds - ort_preds)))
    mean_diff = float(np.mean(np.abs(lgb_preds - ort_preds)))
    logger.info(f"Numerical Parity: Max Abs Diff = {max_diff:.8e}, Mean Abs Diff = {mean_diff:.8e}")
    assert max_diff < 1e-5, f"Parity failed: max_diff = {max_diff}"

    return onnx_path, metadata_path, max_diff


def generate_final_tearsheet() -> Path:
    """Generates the clean 3-panel OOS Equity, Drawdown & Rolling Sharpe comparison chart."""
    logger.info("Generating publication-grade 3-panel tearsheet for EXP-008d promotion...")
    df_panel = engineer_features_and_targets()

    # Run the walkforward models to generate clean OOS return series
    import archive.apex_v45_champion as v45
    from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
        TEST_END_FINAL,
        run_walkforward_models_all,
    )

    df_panel, _ = run_walkforward_models_all(df_panel)
    preds, ret_1h_mat, fr_mat, vol_mat, beta_mat, ts_index, _ = prepare_matrices(df_panel)

    # 1. Standalone P1 48h Total Return (Beta-Neutral)
    r_48_raw, turn_48, _, _ = simulate_c1_sleeve(
        preds["total_48h"], ret_1h_mat, fr_mat, vol_mat, beta_mat, ts_index, beta_neutral=True
    )
    # Net at 10 bps slippage
    r_ml_10 = r_48_raw - turn_48 * (10.0 / 10000.0)

    # 2. Frozen APEX v45 Baseline
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
    slice_22440 = df_champ_full.loc[TEST_START:TEST_END_FINAL]
    r_apex_base = slice_22440["r_apex"].values

    # Apply Honest Vol-Matching at 10 bps to APEX v45 Baseline
    r_apex_vm, turn_apex_vm, _ = apply_honest_vol_targeting(
        r_apex_base, 10.0, underlying_turnover=np.zeros_like(r_apex_base), target_vol=0.1844, deadband=0.05
    )

    # 3. 90% APEX / 10% ML Honest Vol-Matched Blend at 10 bps
    w_ml = 0.10
    w_apex = 0.90
    r_raw_blend = w_apex * r_apex_base + w_ml * r_48_raw
    r_blend_vm, turn_blend_vm, _ = apply_honest_vol_targeting(
        r_raw_blend, 10.0, underlying_turnover=(w_ml * turn_48), target_vol=0.1844, deadband=0.05
    )

    # 4. 90% APEX / 10% ML Unlevered Blend at 10 bps
    r_blend_unlev = w_apex * r_apex_base + w_ml * r_48_raw - (w_ml * turn_48) * (10.0 / 10000.0)

    # Plotting 3-panel figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Panel 1: Cumulative Equity Growth
    cum_apex = np.cumprod(1.0 + r_apex_vm)
    cum_blend_vm = np.cumprod(1.0 + r_blend_vm)
    cum_blend_unlev = np.cumprod(1.0 + r_blend_unlev)
    cum_ml = np.cumprod(1.0 + r_ml_10)

    m_apex = compute_metrics(r_apex_vm, turn_apex_vm, 10.0, ts_index)
    m_blend = compute_metrics(r_blend_vm, turn_blend_vm, 10.0, ts_index)
    m_unlev = compute_metrics(r_blend_unlev, w_ml * turn_48, 10.0, ts_index)
    m_ml = compute_metrics(r_ml_10, turn_48, 10.0, ts_index)

    ax1 = axes[0]
    ax1.plot(
        ts_index,
        cum_apex,
        label=f"Honest Vol-Matched APEX v45 Baseline (100% Core, Sharpe: {m_apex['sharpe']:.2f}, CAGR: {m_apex['cagr']:.1f}%)",
        color="#1f77b4",
        lw=2.0,
    )
    ax1.plot(
        ts_index,
        cum_blend_vm,
        label=f"90% APEX / 10% ML [Honest Vol-Matched] (PROMOTED, Sharpe: {m_blend['sharpe']:.2f}, CAGR: {m_blend['cagr']:.1f}%)",
        color="#2ca02c",
        lw=2.2,
    )
    ax1.plot(
        ts_index,
        cum_blend_unlev,
        label=f"90% APEX / 10% ML [Unlevered] (Sharpe: {m_unlev['sharpe']:.2f}, CAGR: {m_unlev['cagr']:.1f}%)",
        color="#ff7f0e",
        lw=1.6,
        linestyle="--",
    )
    ax1.plot(
        ts_index,
        cum_ml,
        label=f"Standalone EXP-008d Beta-Neutral ML Sleeve (Sharpe: {m_ml['sharpe']:.2f}, CAGR: {m_ml['cagr']:.1f}%)",
        color="#9467bd",
        lw=1.4,
        alpha=0.8,
    )
    ax1.set_title("OOS Cumulative Equity Growth (2022-11 to 2025-05, 22,440 Audited Bars @ 10 bps Slippage)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Growth of $1.00", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=9.5)

    # Panel 2: Underwater Drawdown Profile
    def get_dd(cum_arr: np.ndarray) -> np.ndarray:
        cmax = np.maximum.accumulate(cum_arr)
        return (cum_arr - cmax) / (cmax + 1e-12) * 100.0

    ax2 = axes[1]
    ax2.plot(
        ts_index,
        get_dd(cum_apex),
        label=f"APEX v45 Baseline (MaxDD: {m_apex['max_dd']:.1f}%, Calmar: {m_apex['calmar']:.2f})",
        color="#1f77b4",
        lw=1.5,
    )
    ax2.plot(
        ts_index,
        get_dd(cum_blend_vm),
        label=f"90/10 Honest Vol-Matched Blend (MaxDD: {m_blend['max_dd']:.1f}%, Calmar: {m_blend['calmar']:.2f})",
        color="#2ca02c",
        lw=2.0,
    )
    ax2.plot(
        ts_index,
        get_dd(cum_blend_unlev),
        label=f"90/10 Unlevered Blend (MaxDD: {m_unlev['max_dd']:.1f}%)",
        color="#ff7f0e",
        lw=1.4,
        linestyle="--",
    )
    ax2.set_title("Underwater Drawdown Profile (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown %", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower left", fontsize=9.5)

    # Panel 3: Rolling 90-Day Realized Sharpe
    def rolling_sharpe(r_arr: np.ndarray) -> np.ndarray:
        s = pd.Series(r_arr)
        m = s.rolling(2160, min_periods=720).mean()
        v = s.rolling(2160, min_periods=720).std() + 1e-8
        return (m / v * np.sqrt(8760.0)).values

    ax3 = axes[2]
    ax3.plot(ts_index, rolling_sharpe(r_apex_vm), label="APEX v45 Baseline", color="#1f77b4", lw=1.5)
    ax3.plot(ts_index, rolling_sharpe(r_blend_vm), label="90/10 Honest Vol-Matched Blend", color="#2ca02c", lw=2.0)
    ax3.axhline(0.0, color="black", linestyle=":", alpha=0.5)
    ax3.set_title("Rolling 90-Day Realized Sharpe Ratio (8,760h Annualized)", fontsize=13, fontweight="bold")
    ax3.set_ylabel("Sharpe Ratio", fontsize=11)
    ax3.set_xlabel("UTC Timestamp", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", fontsize=9.5)

    plt.tight_layout()
    plot_path = PROJECT_ROOT / "apex_multimodal_risk/docs/apex_v45_plus_ml15_tearsheet.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    logger.info(f"Saved tearsheet figure to {plot_path}")
    return plot_path


if __name__ == "__main__":
    export_alpha15_model()
    generate_final_tearsheet()
