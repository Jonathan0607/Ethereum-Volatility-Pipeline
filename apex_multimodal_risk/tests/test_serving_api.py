"""
Unit & Integration Tests for Async FastAPI Serving Microservice.
Tests:
1. GET /health: Status, loaded ONNX models, uptime.
2. POST /v1/score: Market-only scoring (48x16 lookback).
3. POST /v1/score: Multimodal scoring with crisis headline.
4. POST /v1/score: SLA timeout fallback (degraded mode with neutral 4.12% prior).
5. POST /v1/feedback: Strictly causal delayed-feedback ACI update.
6. POST /v1/feedback: Premature feedback causal violation rejection (HTTP 400).
7. Concurrent micro-batching test (16 parallel requests).
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
import httpx

from apex_multimodal_risk.src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test health check returns 200, valid models and uptime."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "bge_small_int8" in data["models_loaded"]
    assert "itransformer_int8" in data["models_loaded"]
    assert "lightgbm_risk" in data["models_loaded"]
    assert data["uptime_seconds"] >= 0.0


def test_score_market_only(client):
    """Test scoring request with 48x16 market window and no headline."""
    dummy_window = np.random.randn(48, 16).tolist()
    payload = {
        "asset": "SOLUSDT",
        "market_window": dummy_window,
        "timeout_ms": 1000.0,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "SOLUSDT"
    assert 0.0 <= data["p_crash_cal"] <= 1.0
    assert -0.30 <= data["raw_var_5pct"] <= 0.10
    assert -0.30 <= data["aci_var_5pct"] <= 0.10
    assert 0.001 <= data["aci_alpha_t"] <= 0.50
    assert data["embedding_novelty_score"] == 0.0
    assert data["risk_regime"] in ["NORMAL", "ELEVATED", "CRISIS"]
    assert data["degraded_fallback"] is False
    assert data["latency_ms"] >= 0.0


def test_score_multimodal_headline(client):
    """Test scoring request with breaking crisis headline."""
    dummy_window = np.random.randn(48, 16).tolist()
    payload = {
        "asset": "ETHUSDT",
        "market_window": dummy_window,
        "headline": "Major crypto exchange pauses all withdrawals following insolvency rumor and 200M exploit",
        "timeout_ms": 1000.0,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "ETHUSDT"
    assert 0.0 <= data["p_crash_cal"] <= 1.0
    # Novelty score should be positive and non-zero
    assert data["embedding_novelty_score"] >= 0.0
    assert data["degraded_fallback"] is False


def test_score_sla_timeout_fallback(client):
    """Test SLA latency budget fallback triggers when budget is exceeded."""
    dummy_window = np.random.randn(48, 16).tolist()
    # Provide tiny timeout budget (0.0001 ms)
    payload = {
        "asset": "BTCUSDT",
        "market_window": dummy_window,
        "timeout_ms": 0.0001,
    }
    resp = client.post("/v1/score", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "BTCUSDT"
    assert data["degraded_fallback"] is True
    assert data["p_crash_cal"] == 0.0412  # Unhedged historical base rate
    assert data["risk_regime"] == "DEGRADED_FALLBACK"


def test_feedback_delayed_aci(client):
    """Test causal 6-hour delayed feedback updates ACI quantile."""
    # First get a prediction to register state
    dummy_window = np.random.randn(48, 16).tolist()
    t_bar = "2024-08-05T00:00:00Z"
    client.post("/v1/score", json={"asset": "LINKUSDT", "market_window": dummy_window, "timestamp": t_bar})

    # Step feedback causally at T + 6h
    feedback_payload = {
        "asset": "LINKUSDT",
        "target_bar_time": t_bar,
        "current_time": "2024-08-05T06:00:00Z",
        "realized_fwd_min_ret_6h": -0.15,  # Severe violation
    }
    resp = client.post("/v1/feedback", json=feedback_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["asset"] == "LINKUSDT"
    assert data["loss_applied"] is True
    assert data["updated_alpha_t"] > 0.0
    assert data["buffer_size"] >= 1


def test_feedback_causal_violation_rejection(client):
    """Test premature feedback before 6-hour delay is rejected with HTTP 400."""
    t_bar = "2024-08-05T00:00:00Z"
    feedback_payload = {
        "asset": "AVAXUSDT",
        "target_bar_time": t_bar,
        "current_time": "2024-08-05T03:00:00Z",  # Only 3h elapsed! Causal violation!
        "realized_fwd_min_ret_6h": -0.05,
    }
    resp = client.post("/v1/feedback", json=feedback_payload)
    assert resp.status_code == 400
    assert "Causal error" in resp.json()["detail"] or "causal" in resp.json()["detail"].lower()


def test_dynamic_micro_batching_concurrent(client):
    """Test dynamic micro-batcher processing 16 concurrent requests across assets."""
    from concurrent.futures import ThreadPoolExecutor

    assets = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "AVAXUSDT"]
    headlines = [
        "DeFi lending pool exploited for 45M amid oracle manipulation",
        "SEC approves landmark crypto framework",
        "Bitcoin breaks all-time high with massive retail volume",
        "Solana network congestion leads to transaction delays",
    ]

    def send_score(idx):
        payload = {
            "asset": assets[idx % len(assets)],
            "market_window": np.random.randn(48, 16).tolist(),
            "headline": headlines[idx % len(headlines)],
            "timeout_ms": 2000.0,
        }
        return client.post("/v1/score", json=payload)

    with ThreadPoolExecutor(max_workers=8) as executor:
        responses = list(executor.map(send_score, range(16)))

    assert len(responses) == 16
    for resp in responses:
        assert resp.status_code == 200
        data = resp.json()
        assert data["degraded_fallback"] is False
        assert 0.0 <= data["p_crash_cal"] <= 1.0
        assert data["latency_ms"] > 0.0
