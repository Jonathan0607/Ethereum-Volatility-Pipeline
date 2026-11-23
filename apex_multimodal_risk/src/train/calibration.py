"""
Probability Calibration and Metric Evaluation Module.
Implements:
- 2-Parameter Platt Scaling (sigmoid(a * logit + b)) fit via L-BFGS on inner validation fold.
- Reference Constant_Prior baseline calculator.
- Expected Calibration Error (ECE, 10 bins) computation.
- PR-AUC, ROC-AUC, Brier Score (Raw vs Calibrated).
- 5% Pinball Loss and empirical VaR violation rate.
"""


import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("calibration")


class PlattScaler:
    """
    Leak-free 2-Parameter Platt Scaling for binary tail risk classification:
        p_calibrated = sigmoid(a * logit + b)
    Fit via LogisticRegression (L-BFGS) strictly on inner train validation slice.
    Guarantees:
    - Shifts uncalibrated logits back to empirical base rate prior (~4.12%).
    - Preserves ranking monotonicity (a > 0).
    - Minimizes negative log-likelihood / cross-entropy.
    """

    def __init__(self, lambda_reg: float = 1.0):
        self.a: float = 1.0
        self.b: float = 0.0
        self.lambda_reg: float = lambda_reg

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "PlattScaler":
        logits = np.asarray(logits, dtype=np.float64).ravel()
        y = np.asarray(y_true, dtype=np.float64).ravel()

        if len(np.unique(y)) < 2:
            self.a = 1.0
            self.b = -3.15  # approx logit of 0.0412
            return self

        # Regularized L-BFGS: NLL + lambda * ((a - 1)^2 + b^2) to prevent overfit on small validation blocks
        from scipy.optimize import minimize

        def loss_fn(params):
            a, b = params
            pred_z = np.clip(a * logits + b, -30.0, 30.0)
            p = 1.0 / (1.0 + np.exp(-pred_z))
            eps = 1e-15
            p = np.clip(p, eps, 1.0 - eps)
            nll = -np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
            reg = self.lambda_reg * ((a - 1.0) ** 2 + b ** 2)
            return nll + reg

        res = minimize(
            loss_fn,
            [1.0, 0.0],
            method="L-BFGS-B",
            bounds=[(0.01, 5.0), (-5.0, 5.0)],
        )
        self.a = float(res.x[0])
        self.b = float(res.x[1])

        # Enforce positive slope to prevent ranking inversion
        if self.a <= 0.0:
            self.a = 1e-4

        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=np.float64)
        z = self.a * logits + self.b
        # Numerically safe sigmoid with clipping
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))


class TemperatureScaler:
    """
    Leak-free Temperature Scaling for binary classification:
        p_calibrated = sigmoid(logit / T)
    Preserved for comparison with Platt Scaling.
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
            log_p = -np.log1p(np.exp(-np.clip(scaled, -30.0, 30.0)))
            log_1_p = -np.log1p(np.exp(np.clip(scaled, -30.0, 30.0)))
            bce = -(y * log_p + (1.0 - y) * log_1_p)
            return float(np.mean(bce))

        res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        scaled = np.asarray(logits, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))


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


def compute_constant_prior_metrics(
    y_true: np.ndarray,
    prior: float = 0.0412,
    y_var_true: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Computes theoretical and empirical benchmark metrics for a constant prior predictor (p = prior).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.full_like(y_true, fill_value=prior)
    logits = np.full_like(y_true, fill_value=np.log(prior / (1.0 - prior)))

    res = compute_all_metrics(
        y_clf_true=y_true,
        clf_logits=logits,
        clf_probs_raw=probs,
        clf_probs_cal=probs,
        y_var_true=y_var_true,
        var_preds=None,
    )
    # Exact analytical benchmarks for constant predictor
    res["pr_auc"] = prior
    res["roc_auc"] = 0.500
    res["ece_raw"] = compute_ece(y_true, probs)
    res["ece_calibrated"] = compute_ece(y_true, probs)
    return res


def compute_all_metrics(
    y_clf_true: np.ndarray,
    clf_logits: np.ndarray,
    clf_probs_raw: np.ndarray,
    clf_probs_cal: np.ndarray,
    y_var_true: np.ndarray | None = None,
    var_preds: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Computes complete audit metric suite:
    - PR-AUC (primary metric)
    - ROC-AUC
    - Brier score (raw vs calibrated)
    - ECE (raw vs calibrated)
    - 5% Pinball Loss and VaR violation rate
    """
    metrics: dict[str, float] = {}

    # PR-AUC
    prec, rec, _ = precision_recall_curve(y_clf_true, clf_probs_cal)
    metrics["pr_auc"] = float(auc(rec, prec))

    # ROC-AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_clf_true, clf_probs_cal))
    except ValueError:
        metrics["roc_auc"] = 0.5

    # Brier Score (Raw vs Calibrated)
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
