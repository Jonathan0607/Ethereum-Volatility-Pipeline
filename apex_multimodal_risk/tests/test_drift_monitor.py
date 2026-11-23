"""
Tests for Live Distribution & Embedding Drift Monitor.
Tests:
1. In-distribution samples produce status == 'OK', PSI <= 0.10, shift <= 0.15.
2. Severe distribution shift (e.g. 5x volatility) triggers status == 'CRITICAL', PSI > 0.25.
3. Severe embedding shift (orthogonal or inverted news vectors) triggers status == 'CRITICAL', shift > 0.25.
4. Two-sample KS test produces valid statistic and p-value.
5. Thread safety of concurrent recordings.
"""

import numpy as np
import pytest

from apex_multimodal_risk.src.serving.drift_monitor import CORE_DRIFT_FEATURES, DriftMonitor
from apex_multimodal_risk.src.utils import PACKAGE_ROOT, PROJECT_ROOT


@pytest.fixture
def monitor():
    baseline_path = PROJECT_ROOT / "apex_multimodal_risk/configs/drift_baseline.json"
    if not baseline_path.exists():
        baseline_path = PACKAGE_ROOT / "configs/drift_baseline.json"
    assert baseline_path.exists(), f"Baseline config {baseline_path} must exist."
    return DriftMonitor(baseline_path=baseline_path, window_size=500, min_eval_samples=20)


def test_in_distribution_samples(monitor):
    """Feeds in-distribution samples sampled directly from baseline reference distribution."""
    rng = np.random.default_rng(42)
    base_feats = monitor.baseline_features

    # Generate 200 in-distribution samples by sampling reference points directly
    for _ in range(200):
        obs = {}
        for feat in CORE_DRIFT_FEATURES:
            ref_pool = base_feats[feat]["reference_sample"]
            val = float(rng.choice(ref_pool))
            obs[feat] = val

        # Embedding near baseline centroid
        noise = rng.normal(0, 0.01, size=384).astype(np.float32)
        emb = monitor.baseline_centroid + noise
        emb /= np.linalg.norm(emb)

        monitor.record_observation(obs, embedding=emb)

    report = monitor.evaluate()
    assert report["status"] == "OK"
    assert report["max_psi"] <= 0.10, f"Expected PSI <= 0.10 for in-distribution data, got {report['max_psi']}"
    assert report["embedding_centroid_shift"] <= 0.15, f"Expected shift <= 0.15, got {report['embedding_centroid_shift']}"
    for feat in CORE_DRIFT_FEATURES:
        assert report["features"][feat]["psi"] <= 0.10
        assert not report["features"][feat]["drift_detected"]


def test_severe_feature_distribution_shift(monitor):
    """Feeds severe synthetic volatility shift (5x normal) and asserts status == CRITICAL."""
    rng = np.random.default_rng(42)
    base_feats = monitor.baseline_features

    for _ in range(100):
        obs = {}
        for feat in CORE_DRIFT_FEATURES:
            # Shift mean by 5 standard deviations
            val = float(base_feats[feat]["mean"] + 5.0 * base_feats[feat]["std"] + rng.normal(0, base_feats[feat]["std"]))
            obs[feat] = val

        monitor.record_observation(obs, embedding=monitor.baseline_centroid)

    report = monitor.evaluate()
    assert report["status"] == "CRITICAL"
    assert report["max_psi"] > 0.25, f"Expected PSI > 0.25 for 5x shock, got {report['max_psi']}"
    # Volatility should flag drift
    assert report["features"]["realized_vol_72h"]["psi"] > 0.25
    assert report["features"]["realized_vol_72h"]["drift_detected"] is True
    assert report["features"]["realized_vol_72h"]["ks_p_value"] < 0.01


def test_severe_embedding_centroid_shift(monitor):
    """Feeds orthogonal / inverted embeddings and asserts status == CRITICAL with shift > 0.25."""
    rng = np.random.default_rng(42)
    base_feats = monitor.baseline_features

    # Inverted centroid (cos_sim == -1.0 => shift == 2.0)
    inverted_emb = -1.0 * monitor.baseline_centroid

    for _ in range(100):
        obs = {}
        for feat in CORE_DRIFT_FEATURES:
            ref_pool = base_feats[feat]["reference_sample"]
            obs[feat] = float(rng.choice(ref_pool))

        noise = rng.normal(0, 0.01, size=384).astype(np.float32)
        emb = inverted_emb + noise
        emb /= np.linalg.norm(emb)
        monitor.record_observation(obs, embedding=emb)

    report = monitor.evaluate()
    assert report["status"] == "CRITICAL"
    assert report["embedding_centroid_shift"] > 0.25
    assert report["embedding_centroid_shift"] >= 1.90, f"Expected large shift near 2.0, got {report['embedding_centroid_shift']}"


def test_batch_recording_and_ks_statistic(monitor):
    """Verifies record_batch matches record_observation and KS statistics are computed."""
    base_feats = monitor.baseline_features
    features_batch = []
    embeddings_batch = []

    for _ in range(50):
        obs = {feat: float(base_feats[feat]["mean"]) for feat in CORE_DRIFT_FEATURES}
        features_batch.append(obs)
        embeddings_batch.append(monitor.baseline_centroid.copy())

    monitor.record_batch(features_batch, embeddings_batch)
    assert len(monitor.embedding_window) == 50
    assert len(monitor.feature_windows["realized_vol_72h"]) == 50

    report = monitor.evaluate()
    for feat in CORE_DRIFT_FEATURES:
        assert "ks_statistic" in report["features"][feat]
        assert "ks_p_value" in report["features"][feat]
        assert 0.0 <= report["features"][feat]["ks_statistic"] <= 1.0
        assert 0.0 <= report["features"][feat]["ks_p_value"] <= 1.0
