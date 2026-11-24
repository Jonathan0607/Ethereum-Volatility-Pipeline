"""
Export Promoted EXP-012 (R2: LightGBM Dynamic Beta Forecaster) to ONNX Runtime
and Verify Numerical Parity (< 1e-6) vs Native LightGBM on 500 Real Cross-Sections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Bootstrap project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import onnxmltools
import onnxruntime as ort
from onnxmltools.convert.common.data_types import FloatTensorType

from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
    QUARTERLY_TEST_STARTS,
)
from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
    engineer_features_and_targets as load_base_panel,
)
from apex_multimodal_risk.src.train.exp012_dynamic_covariance import (
    RISK_FEATURE_COLS,
    build_risk_features_and_targets,
)
from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("export_beta15_onnx")
TEST_START = QUARTERLY_TEST_STARTS[0]


def export_beta15_model() -> tuple[Path, Path, float]:
    """Fits 48h Dynamic Beta LightGBM model, exports to ONNX, and audits parity."""
    logger.info("Loading panel and computing risk features & targets...")
    df_panel = load_base_panel()
    df_panel, assets, ts_index, ret_mat, fwd_beta_48, fwd_beta_168, fwd_cov_168 = build_risk_features_and_targets(df_panel)

    # Training window: strictly causal prior to OOS evaluation window
    train_mask = (df_panel["timestamp"] < TEST_START) & df_panel["target_beta_48h"].notnull()
    df_train = df_panel[train_mask]

    X_train = df_train[RISK_FEATURE_COLS].values
    y_train = df_train["target_beta_48h"].values
    logger.info(f"Fitting promoted R2 Dynamic Beta LightGBM model on {len(X_train)} training rows...")

    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=75,
        learning_rate=0.04,
        num_leaves=12,
        max_depth=4,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=5.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    # Export to ONNX
    export_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts/onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / "lightgbm_beta15.onnx"

    logger.info("Converting LightGBM Beta model to ONNX via onnxmltools...")
    initial_type = [("input", FloatTensorType([None, len(RISK_FEATURE_COLS)]))]
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=15,
        zipmap=False,
    )

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"Saved ONNX model to {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    unique_assets = sorted(df_panel["asset"].unique().tolist())
    btc_asset = "BTCUSDT"

    # Save metadata
    metadata_path = export_dir / "beta15_metadata.json"
    metadata = {
        "model_version": "v1.0.0",
        "model_name": "lightgbm_beta15",
        "target": "target_beta_48h",
        "objective": "huber",
        "horizon_hours": 48,
        "n_features": len(RISK_FEATURE_COLS),
        "feature_names": RISK_FEATURE_COLS,
        "assets": unique_assets,
        "n_assets": len(unique_assets),
        "btc_asset": btc_asset,
        "clip_bounds": [0.3, 2.5],
        "params": {
            "objective": "huber",
            "n_estimators": 75,
            "learning_rate": 0.04,
            "num_leaves": 12,
            "max_depth": 4,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 5.0,
            "random_state": 42,
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Numerical Parity Check on 500 real cross-sections (500 * 15 = 7,500 samples)
    logger.info("Auditing numerical parity vs native LightGBM on 500 real cross-sections...")
    test_sample = df_panel[df_panel["timestamp"] >= TEST_START].iloc[: 500 * len(unique_assets)]
    X_test = test_sample[RISK_FEATURE_COLS].values
    asset_test = test_sample["asset"].values

    lgb_preds = model.predict(X_test)
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_preds = ort_session.run(None, {"input": X_test.astype(np.float32)})[0].flatten()

    # Enforce clamping on both
    lgb_preds = np.clip(lgb_preds, 0.3, 2.5)
    ort_preds = np.clip(ort_preds, 0.3, 2.5)

    # Enforce BTCUSDT == 1.0 on both
    btc_mask = asset_test == btc_asset
    lgb_preds[btc_mask] = 1.0
    ort_preds[btc_mask] = 1.0

    max_diff = float(np.max(np.abs(lgb_preds - ort_preds)))
    mean_diff = float(np.mean(np.abs(lgb_preds - ort_preds)))
    logger.info(f"Numerical Parity: Max Abs Diff = {max_diff:.8e}, Mean Abs Diff = {mean_diff:.8e}")
    assert max_diff < 1e-6, f"Parity audit failed: max_diff = {max_diff} exceeds 1e-6!"
    logger.info(f"PARITY AUDIT PASSED: max_diff = {max_diff:.4e} < 1e-6 on {len(X_test)} predictions.")

    return onnx_path, metadata_path, max_diff


if __name__ == "__main__":
    export_beta15_model()
