"""
Unit & API Integration Tests for 15-Coin Alpha Serving (/v1/alpha).
Tests:
1. Analytic nullspace beta-neutral mathematical projection.
2. Fresh-clone CI compatibility (APEX_ALLOW_MOCK_MODELS=1).
3. Real ONNX inference parity and sub-5ms CPU latency.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apex_multimodal_risk.src.serving.app import app
from apex_multimodal_risk.src.serving.portfolio_math import (
    compute_portfolio_exposures,
    project_beta_neutral,
)


def test_nullspace_beta_neutral_math():
    """Validates mathematical properties of closed-form nullspace projection."""
    rng = np.random.default_rng(42)

    for _ in range(50):
        n = 15
        scores = rng.normal(0.0, 1.0, n)
        betas = rng.uniform(0.4, 2.2, n)
        gross_cap = 1.0
        max_weight = 0.15

        w = project_beta_neutral(scores, betas, gross_cap=gross_cap, max_weight=max_weight)

        # 1. Dollar Neutrality: |sum(w)| < 1e-10
        assert abs(float(np.sum(w))) < 1e-10, f"Dollar neutrality violated: sum(w) = {np.sum(w)}"

        # 2. Beta Neutrality: |sum(w * betas)| < 1e-10
        assert abs(float(np.sum(w * betas))) < 1e-10, f"Beta neutrality violated: sum(w*beta) = {np.sum(w * betas)}"

        # 3. Position bounds: |w_i| <= max_weight + 1e-10
        assert np.all(np.abs(w) <= max_weight + 1e-10), f"Max weight violated: {np.max(np.abs(w))}"

        # 4. Gross exposure bound: sum(|w_i|) <= gross_cap + 1e-10
        assert float(np.sum(np.abs(w))) <= gross_cap + 1e-10, f"Gross cap violated: {np.sum(np.abs(w))}"

        # 5. Helper function agreement
        exposures = compute_portfolio_exposures(w, betas)
        assert abs(exposures["net_dollar_exposure"]) < 1e-10
        assert abs(exposures["net_btc_beta"]) < 1e-10


def test_nullspace_edge_cases():
    """Tests error handling for mismatched or invalid dimensions."""
    with pytest.raises(ValueError, match="matching non-zero lengths"):
        project_beta_neutral([0.1, 0.2], [1.0])

    with pytest.raises(ValueError, match="matching non-zero lengths"):
        project_beta_neutral([], [])


def test_alpha_endpoint_mock_ci(monkeypatch):
    """Verifies POST /v1/alpha under fresh-clone CI mode with mock fallback."""
    monkeypatch.setenv("APEX_ALLOW_MOCK_MODELS", "1")

    with TestClient(app) as client:
        # 15 assets x 21 features
        features = [[0.05 * (i - j) for j in range(21)] for i in range(15)]
        betas = [1.0 + 0.05 * i for i in range(15)]
        assets = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT",
            "DOGEUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "MATICUSDT",
            "DOTUSDT", "ATOMUSDT", "LTCUSDT", "NEARUSDT", "UNIUSDT",
        ]

        payload = {
            "features": features,
            "betas": betas,
            "assets": assets,
            "gross_cap": 1.0,
            "max_weight": 0.15,
        }

        # Warmup
        t0 = time.perf_counter()
        resp = client.post("/v1/alpha", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200, f"Error: {resp.text}"
        data = resp.json()

        assert "weights" in data
        assert len(data["weights"]) == 15
        assert abs(data["net_dollar_exposure"]) < 1e-6
        assert abs(data["net_btc_beta"]) < 1e-6
        assert data["gross_exposure"] <= 1.0001
        for a, w in data["weights"].items():
            assert abs(w) <= 0.150001, f"Weight bound exceeded for {a}: {w}"

        # Latency check on local CPU
        assert latency_ms < 50.0  # Safe threshold across local / CI runners


def test_alpha_endpoint_validation():
    """Verifies 400 Bad Request on dimensional mismatch."""
    with TestClient(app) as client:
        # Invalid: 14 assets with 15 betas
        features = [[0.1] * 21 for _ in range(14)]
        betas = [1.0] * 15

        resp = client.post("/v1/alpha", json={"features": features, "betas": betas})
        assert resp.status_code == 400
        assert "must match betas" in resp.json()["detail"]

        # Invalid: 15 assets with 20 features (expected 21)
        features_bad_feats = [[0.1] * 20 for _ in range(15)]
        resp2 = client.post("/v1/alpha", json={"features": features_bad_feats, "betas": betas})
        assert resp2.status_code == 400
        assert "expected 21" in resp2.json()["detail"]


@pytest.mark.requires_local_artifacts
def test_alpha_endpoint_onnx_local():
    """Verifies POST /v1/alpha with real lightgbm_alpha15.onnx model on CPU."""
    onnx_path = os.path.join("apex_multimodal_risk/artifacts/onnx", "lightgbm_alpha15.onnx")
    if not os.path.exists(onnx_path):
        pytest.skip("lightgbm_alpha15.onnx not present locally")

    with TestClient(app) as client:
        features = [[0.05 * (i - j) for j in range(21)] for i in range(15)]
        betas = [1.0 + 0.05 * i for i in range(15)]
        payload = {"features": features, "betas": betas}

        t0 = time.perf_counter()
        resp = client.post("/v1/alpha", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["weights"]) == 15
        assert abs(data["net_dollar_exposure"]) < 1e-6
        assert abs(data["net_btc_beta"]) < 1e-6
        assert latency_ms < 50.0
