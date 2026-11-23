"""
Probability Calibration and Metric Evaluation Module.
Implements:
- Temperature Scaling for binary classification calibrated strictly on inner train validation slice.
- Expected Calibration Error (ECE, 10 bins) computation.
- PR-AUC, ROC-AUC, Brier Score.
- 5% Pinball Loss and empirical VaR violation rate.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    brier_score_loss,
)

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("calibration")


class TemperatureScaler:
    """
    Leak-free Temperature Scaling for binary tail risk classification.
    Calibrated strictly on inner validation fold:
      p_calibrated = sigmoid(logit / T)
    Optimizes scalar T > 0 to minimize Negative Log Likelihood (NLL).
    """

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)

        def nll_loss(T: float) -> float:
            if T <= 0.01:
                return 1e9
            scaled = logits / T
            # Numerically stable BCE loss
            log_p = -np.log1p(np.exp(-scaled))
            log_1_p = -np.log1p(np.exp(scaled))
            bce = -(y * log_p + (1.0 - y) * log_1_p)
            return float(np.mean(bce))

        res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))


def compute_ece(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Computes Expected Calibration Error (ECE) across n_bins equal intervals.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.asarray(probs, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= low) & (probs <= high if i == n_bins - 1 else probs < high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probs[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_all_metrics(
    y_clf_true: np.ndarray,
    clf_logits: np.ndarray,
    clf_probs_raw: np.ndarray,
    clf_probs_cal: np.ndarray,
    y_var_true: Optional[np.ndarray] = None,
    var_preds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Computes complete audit metric suite:
    - PR-AUC (primary metric)
    - ROC-AUC
    - Brier score (raw vs calibrated)
    - ECE (raw vs calibrated)
    - 5% Pinball Loss and VaR violation rate
    """
    metrics: Dict[str, float] = {}

    # PR-AUC
    prec, rec, _ = precision_recall_curve(y_clf_true, clf_probs_cal)
    metrics["pr_auc"] = float(auc(rec, prec))

    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_clf_true, clf_probs_cal))
    except Exception:
        metrics["roc_auc"] = 0.5

    # Brier Score
    metrics["brier_raw"] = float(brier_score_loss(y_clf_true, clf_probs_raw))
    metrics["brier_calibrated"] = float(brier_score_loss(y_clf_true, clf_probs_cal))

    # ECE (10 bins)
    metrics["ece_raw"] = compute_ece(y_clf_true, clf_probs_raw, n_bins=10)
    metrics["ece_calibrated"] = compute_ece(y_clf_true, clf_probs_cal, n_bins=10)

    # VaR metrics
    if y_var_true is not None and var_preds is not None:
        diff = y_var_true - var_preds
        pinball = np.mean(np.maximum(0.05 * diff, -0.95 * diff))
        violation_rate = np.mean((y_var_true < var_preds).astype(float))
        metrics["pinball_loss_5pct"] = float(pinball)
        metrics["var_violation_rate"] = float(violation_rate)
    else:
        metrics["pinball_loss_5pct"] = float("nan")
        metrics["var_violation_rate"] = float("nan")

    return metrics
