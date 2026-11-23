"""
Generate reference drift baseline statistics for live feature and embedding monitoring.
Computes:
1. 10-quantile bin edges, expected proportions, and reference samples for:
   - realized_vol_72h
   - skew_72h
   - funding_zscore
   - oi_delta_48h
   - embedding_novelty_score
2. 384-dimensional normalized news embedding centroid.
Saves output to configs/drift_baseline.json.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Ensure apex_multimodal_risk is importable when executed directly
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from apex_multimodal_risk.src.features.shared_features import load_and_merge_features_and_targets
from apex_multimodal_risk.src.utils import PROJECT_ROOT, read_parquet_utc, setup_logger

logger = setup_logger("generate_drift_baseline")

CORE_DRIFT_FEATURES = [
    "realized_vol_72h",
    "skew_72h",
    "funding_zscore",
    "oi_delta_48h",
    "embedding_novelty_score",
]


def generate_drift_baseline(output_path: Path | str | None = None) -> dict[str, Any]:
    if output_path is None:
        output_path = PROJECT_ROOT / "apex_multimodal_risk/configs/drift_baseline.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading merged panel for drift baseline calculation...")
    panel_df = load_and_merge_features_and_targets()

    # 1. Feature Baselines
    feature_baselines = {}
    rng = np.random.default_rng(42)

    for feat in CORE_DRIFT_FEATURES:
        if feat not in panel_df.columns:
            raise ValueError(f"Feature {feat} not found in merged panel.")
        vals = panel_df[feat].dropna().to_numpy(dtype=np.float64)
        if len(vals) == 0:
            raise ValueError(f"Feature {feat} has no non-NaN values.")

        # Compute 10 quantile bin edges (11 points)
        edges = np.percentile(vals, np.linspace(0, 100, 11))
        # Ensure strictly monotonic increasing bin edges
        for idx in range(1, len(edges)):
            if edges[idx] <= edges[idx - 1]:
                edges[idx] = edges[idx - 1] + 1e-6

        # Extend extreme outer boundaries to handle future test extremes
        edges[0] = -1e9
        edges[-1] = 1e9

        counts, _ = np.histogram(vals, bins=edges)
        proportions = (counts / len(vals)).tolist()

        # Sample 500 reference points for two-sample KS testing
        ref_sample = rng.choice(vals, size=min(500, len(vals)), replace=False).tolist()

        feature_baselines[feat] = {
            "bin_edges": edges.tolist(),
            "expected_proportions": proportions,
            "reference_sample": ref_sample,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        logger.info(f"Generated baseline for {feat}: {len(edges)-1} bins, {len(ref_sample)} ref samples.")

    # 2. Embedding Centroid Baseline
    news_parquet = PROJECT_ROOT / "apex_multimodal_risk/data/processed/news_dedup.parquet"
    logger.info(f"Loading deduplicated news from {news_parquet}...")
    news_df = read_parquet_utc(str(news_parquet))
    embs = np.array(news_df["embedding"].tolist(), dtype=np.float32)
    centroid = np.mean(embs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 1e-6:
        centroid /= norm
    logger.info(f"Computed reference embedding centroid across {len(embs)} news items (dim {len(centroid)}).")

    baseline_data = {
        "features": feature_baselines,
        "embedding_centroid": centroid.tolist(),
        "n_samples": len(panel_df),
        "n_news_items": len(news_df),
        "created_at": "2026-11-23 UTC",
    }

    with open(output_path, "w") as f:
        json.dump(baseline_data, f, indent=2)

    logger.info(f"Saved drift baseline to {output_path}")
    return baseline_data


if __name__ == "__main__":
    generate_drift_baseline()
