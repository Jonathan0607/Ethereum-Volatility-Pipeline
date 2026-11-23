"""
Live Distribution & Embedding Drift Monitoring Module.
Computes:
1. Rolling Population Stability Index (PSI) against reference baseline quantiles.
2. Two-sample Kolmogorov-Smirnov (KS) test statistic and p-value via scipy.stats.ks_2samp.
3. Rolling embedding centroid cosine shift against training baseline centroid.
4. Status classification: OK (PSI <= 0.10, shift <= 0.15), WARN (>0.10 / >0.15), CRITICAL (>0.25 / >0.25).
5. Exposes hooks for Prometheus gauges and GET /v1/drift.
"""

import json
import threading
from collections import deque
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from apex_multimodal_risk.src.utils import PROJECT_ROOT, setup_logger

logger = setup_logger("drift_monitor")

CORE_DRIFT_FEATURES = [
    "realized_vol_72h",
    "skew_72h",
    "funding_zscore",
    "oi_delta_48h",
    "embedding_novelty_score",
]


def calculate_psi(
    observed_values: np.ndarray,
    bin_edges: np.ndarray,
    expected_proportions: np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes Population Stability Index (PSI) between observed samples and expected proportions.
    PSI = sum((P_b - E_b) * ln((P_b + eps) / (E_b + eps)))
    """
    if len(observed_values) == 0:
        return 0.0

    counts, _ = np.histogram(observed_values, bins=bin_edges)
    observed_proportions = counts / len(observed_values)

    exp_p = np.array(expected_proportions, dtype=np.float64)
    obs_p = np.array(observed_proportions, dtype=np.float64)

    psi_terms = (obs_p - exp_p) * np.log((obs_p + epsilon) / (exp_p + epsilon))
    return float(max(0.0, np.sum(psi_terms)))


class DriftMonitor:
    """
    Thread-safe live distribution and embedding drift monitor for production serving.
    Maintains rolling windows of served feature values and headline embeddings.
    """

    def __init__(
        self,
        baseline_path: Path | str | None = None,
        window_size: int = 500,
        min_eval_samples: int = 20,
    ):
        self.window_size = window_size
        self.min_eval_samples = min_eval_samples
        self.lock = threading.Lock()

        if baseline_path is None:
            baseline_path = PROJECT_ROOT / "apex_multimodal_risk/configs/drift_baseline.json"
        self.baseline_path = Path(baseline_path)

        self.baseline_features: dict[str, dict[str, Any]] = {}
        self.baseline_centroid: np.ndarray | None = None

        self._load_baseline()

        # Rolling windows for core features
        self.feature_windows: dict[str, deque[float]] = {
            f: deque(maxlen=self.window_size) for f in CORE_DRIFT_FEATURES
        }

        # Rolling window for 384-dim headline embeddings
        self.embedding_window: deque[np.ndarray] = deque(maxlen=self.window_size)

    def _load_baseline(self):
        if not self.baseline_path.exists():
            logger.warning(f"Drift baseline file {self.baseline_path} not found. Running with empty baseline.")
            return

        with open(self.baseline_path) as f:
            data = json.load(f)

        self.baseline_features = data.get("features", {})
        if "embedding_centroid" in data:
            self.baseline_centroid = np.array(data["embedding_centroid"], dtype=np.float32)
            c_norm = np.linalg.norm(self.baseline_centroid)
            if c_norm > 1e-6:
                self.baseline_centroid /= c_norm
        logger.info(f"Loaded drift baseline from {self.baseline_path} ({len(self.baseline_features)} features)")

    def record_observation(
        self,
        feature_dict: dict[str, float],
        embedding: np.ndarray | None = None,
    ):
        """Records a single scoring request's features and optional headline embedding."""
        with self.lock:
            for feat in CORE_DRIFT_FEATURES:
                if feat in feature_dict:
                    val = float(feature_dict[feat])
                    if not np.isnan(val):
                        self.feature_windows[feat].append(val)

            if embedding is not None and isinstance(embedding, np.ndarray) and embedding.size == 384:
                self.embedding_window.append(np.asarray(embedding, dtype=np.float32))

    def record_batch(
        self,
        features_list: list[dict[str, float]],
        embeddings_list: list[np.ndarray | None] | None = None,
    ):
        """Records a micro-batch of requests atomically."""
        with self.lock:
            for i, feat_dict in enumerate(features_list):
                for feat in CORE_DRIFT_FEATURES:
                    if feat in feat_dict:
                        val = float(feat_dict[feat])
                        if not np.isnan(val):
                            self.feature_windows[feat].append(val)

                if embeddings_list and i < len(embeddings_list):
                    emb = embeddings_list[i]
                    if emb is not None and isinstance(emb, np.ndarray) and emb.size == 384:
                        self.embedding_window.append(np.asarray(emb, dtype=np.float32))

    def compute_embedding_shift(self) -> float:
        """
        Computes cosine distance shift between rolling live embedding centroid and baseline centroid:
        shift = 1.0 - (live_centroid . baseline_centroid)
        """
        if self.baseline_centroid is None or len(self.embedding_window) == 0:
            return 0.0

        embs = np.array(list(self.embedding_window), dtype=np.float32)
        live_centroid = np.mean(embs, axis=0)
        norm = np.linalg.norm(live_centroid)
        if norm > 1e-6:
            live_centroid /= norm

        cos_sim = float(np.dot(live_centroid, self.baseline_centroid))
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))

    def evaluate(self) -> dict[str, Any]:
        """
        Evaluates PSI, KS-test, and embedding centroid shift across rolling windows.
        Returns detailed report and status classification.
        """
        with self.lock:
            window_len = max(len(w) for w in self.feature_windows.values()) if self.feature_windows else 0
            emb_count = len(self.embedding_window)

            results: dict[str, dict[str, Any]] = {}
            max_psi = 0.0

            for feat in CORE_DRIFT_FEATURES:
                feat_obs = np.array(list(self.feature_windows[feat]), dtype=np.float64)
                base_cfg = self.baseline_features.get(feat, {})

                if len(feat_obs) < self.min_eval_samples or not base_cfg:
                    results[feat] = {
                        "psi": 0.0,
                        "ks_statistic": 0.0,
                        "ks_p_value": 1.0,
                        "drift_detected": False,
                        "sample_count": len(feat_obs),
                    }
                    continue

                bin_edges = np.array(base_cfg["bin_edges"], dtype=np.float64)
                exp_props = np.array(base_cfg["expected_proportions"], dtype=np.float64)
                ref_sample = np.array(base_cfg.get("reference_sample", []), dtype=np.float64)

                # 1. Compute PSI
                psi_val = calculate_psi(feat_obs, bin_edges, exp_props)
                max_psi = max(max_psi, psi_val)

                # 2. Compute 2-sample KS test
                if len(ref_sample) > 0:
                    ks_res = ks_2samp(feat_obs, ref_sample)
                    ks_stat = float(ks_res.statistic)
                    ks_pval = float(ks_res.pvalue)
                else:
                    ks_stat, ks_pval = 0.0, 1.0

                drift_detected = bool(psi_val > 0.10 or ks_pval < 0.01)

                results[feat] = {
                    "psi": round(psi_val, 4),
                    "ks_statistic": round(ks_stat, 4),
                    "ks_p_value": round(ks_pval, 4),
                    "drift_detected": drift_detected,
                    "sample_count": len(feat_obs),
                }

            # 3. Compute Embedding Centroid Shift
            emb_shift = self.compute_embedding_shift()

            # 4. Status Classification:
            # OK: PSI <= 0.10, shift <= 0.15
            # WARN: PSI > 0.10 or shift > 0.15
            # CRITICAL: PSI > 0.25 or shift > 0.25
            if max_psi > 0.25 or emb_shift > 0.25:
                status = "CRITICAL"
            elif max_psi > 0.10 or emb_shift > 0.15:
                status = "WARN"
            else:
                status = "OK"

            return {
                "status": status,
                "window_size": window_len,
                "embedding_count": emb_count,
                "embedding_centroid_shift": round(emb_shift, 4),
                "max_psi": round(max_psi, 4),
                "features": results,
                "timestamp": pd.Timestamp.now(tz=UTC).isoformat(),
            }
