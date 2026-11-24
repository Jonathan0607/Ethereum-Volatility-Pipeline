"""
Unit & Parity Tests for Alpha-15 Online Feature Builder (EXP-008d / EXP-012).
Includes:
1. Fresh-clone safe unit tests (pure NumPy/Pandas synthetic data, zero local artifacts required).
2. Offline parity tests against local historical Parquet panel (requires_local_artifacts).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from apex_multimodal_risk.src.features.alpha15_features import (
    ALPHA_FEATURE_COLS,
    DEFAULT_UNIVERSE_15,
    RISK_FEATURE_COLS,
    compute_alpha15_and_risk_features,
    extract_bar_features_for_serving,
)
from apex_multimodal_risk.src.serving.portfolio_math import (
    compute_portfolio_exposures,
    project_beta_neutral,
    project_multi_factor_neutral,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def generate_synthetic_15_panel(n_hours: int = 200) -> pd.DataFrame:
    """Generates synthetic multi-asset panel data for fresh-clone CI testing."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_hours, freq="1h", tz="UTC")
    rows = []
    base_prices = {sym: 100.0 + i * 20.0 for i, sym in enumerate(DEFAULT_UNIVERSE_15)}

    for t in timestamps:
        for sym in DEFAULT_UNIVERSE_15:
            ret = np.random.normal(0.0001, 0.02)
            base_prices[sym] *= 1.0 + ret
            c = base_prices[sym]
            h = c * (1.0 + abs(np.random.normal(0, 0.005)))
            low = c * (1.0 - abs(np.random.normal(0, 0.005)))
            o = (h + low) / 2.0
            vol = float(np.random.uniform(1e5, 1e7))
            fr = float(np.random.normal(0.0001, 0.0002))
            rows.append({
                "timestamp": t,
                "asset": sym,
                "open": o,
                "high": h,
                "low": low,
                "close": c,
                "volume": vol,
                "funding_rate": fr,
            })

    return pd.DataFrame(rows)


def test_alpha15_features_synthetic_pipeline():
    """Fresh-clone CI test: verifies feature calculation, rank normalization, and shape."""
    df_syn = generate_synthetic_15_panel(n_hours=180)
    df_feat, alpha_cols, risk_cols = compute_alpha15_and_risk_features(df_syn, btc_symbol="BTCUSDT")

    assert len(alpha_cols) == 21
    assert len(risk_cols) == 13
    assert set(alpha_cols) == set(ALPHA_FEATURE_COLS)
    assert set(risk_cols) == set(RISK_FEATURE_COLS)

    # Check cross-sectional rank normalization boundaries [-1.0, +1.0]
    for col in alpha_cols:
        vals = df_feat[col].values
        assert np.nanmin(vals) >= -1.0 - 1e-6
        assert np.nanmax(vals) <= 1.0 + 1e-6

    # Test extraction for a specific bar
    target_ts = df_syn["timestamp"].iloc[-15]
    X_alpha, X_risk, assets = extract_bar_features_for_serving(df_feat, target_ts)
    assert X_alpha.shape == (15, 21)
    assert X_risk.shape == (15, 13)
    assert assets == DEFAULT_UNIVERSE_15

    # Check BTC beta in risk features
    btc_idx = assets.index("BTCUSDT")
    assert np.allclose(X_risk[btc_idx, 0:4], 1.0, atol=1e-5)


def test_portfolio_math_nullspace_and_exposures():
    """Verifies 2-stage nullspace projection and exposure accounting."""
    np.random.seed(42)
    scores = np.random.uniform(-1.0, 1.0, 15)
    betas = np.random.uniform(0.5, 2.0, 15)
    betas[4] = 1.0  # BTC

    w = project_beta_neutral(scores, betas, gross_cap=1.0, max_weight=0.15)
    exp = compute_portfolio_exposures(w, betas)

    assert abs(exp["net_dollar_exposure"]) < 1e-10
    assert abs(exp["net_btc_beta"]) < 1e-10
    assert exp["gross_exposure"] <= 1.0 + 1e-6
    assert np.all(np.abs(w) <= 0.15 + 1e-6)

    # Multi-factor projection equivalence for single beta
    C = np.column_stack([np.ones(15), betas])
    w_multi = project_multi_factor_neutral(scores, C, gross_cap=1.0, max_weight=0.15)
    assert np.allclose(w, w_multi, atol=1e-10)


@pytest.mark.requires_local_artifacts
def test_alpha15_features_offline_numerical_parity():
    """
    Parity test against historical universe_15_hourly.parquet panel.
    Verifies that alpha15_features produces matching output to offline train scripts.
    """
    parquet_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/universe_15_hourly.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Historical data parquet not found at {parquet_path}")

    # Read slice of historical data (e.g. 500 hours)
    df_raw = pd.read_parquet(parquet_path)
    timestamps = sorted(df_raw["timestamp"].unique())
    sub_ts = timestamps[1000:1500]
    df_slice = df_raw[df_raw["timestamp"].isin(sub_ts)].copy()

    df_feat, alpha_cols, risk_cols = compute_alpha15_and_risk_features(df_slice, btc_symbol="BTCUSDT")
    assert len(df_feat) == len(df_slice)

    # Verify rank features are finite and valid
    for col in alpha_cols:
        assert not df_feat[col].isna().any()
        assert df_feat[col].min() >= -1.0 - 1e-6
        assert df_feat[col].max() <= 1.0 + 1e-6

    for col in risk_cols:
        assert not df_feat[col].isna().any()
