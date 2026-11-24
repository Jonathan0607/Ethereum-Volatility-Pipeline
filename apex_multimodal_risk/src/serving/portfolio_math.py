"""
Pure NumPy Portfolio Mathematics & Analytic Nullspace Beta-Neutral Projection.
STRICTLY zero PyTorch and zero LightGBM imports for ultralight CPU serving.
"""

from __future__ import annotations

import numpy as np


def project_beta_neutral(
    scores: np.ndarray,
    betas: np.ndarray,
    gross_cap: float = 1.0,
    max_weight: float = 0.15,
) -> np.ndarray:
    """
    Analytically projects score/weight vector onto the null space of [1, betas],
    strictly enforcing:
      1. Dollar neutrality: |sum(w)| < 1e-10
      2. BTC beta neutrality: |sum(w * betas)| < 1e-10
      3. Bounded individual positions: |w_i| <= max_weight
      4. Bounded gross exposure: sum(|w_i|) <= gross_cap

    Mathematical formulation:
      C = [1, betas] (n x 2 constraint matrix)
      P = I - C (C^T C)^(-1) C^T (nullspace projection operator)
      w = P @ scores
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    betas_arr = np.asarray(betas, dtype=np.float64)
    n = len(scores_arr)

    if n == 0 or len(betas_arr) != n:
        raise ValueError(f"scores and betas must have matching non-zero lengths (got {len(scores_arr)} and {len(betas_arr)})")

    C = np.column_stack([np.ones(n, dtype=np.float64), betas_arr])
    # Compute pseudo-inverse for numerical stability
    inv_CC = np.linalg.pinv(C.T @ C)
    P = np.eye(n, dtype=np.float64) - C @ inv_CC @ C.T
    w = P @ scores_arr

    gross = float(np.sum(np.abs(w)))
    if gross > 1e-12:
        w = w * (gross_cap / gross)

    # Clip positions and re-project to preserve nullspace orthogonality
    w_clipped = np.clip(w, -max_weight, max_weight)
    w = P @ w_clipped

    gross2 = float(np.sum(np.abs(w)))
    if gross2 > 1e-12:
        w = w * (gross_cap / gross2)

    # Exact bound guarantee via scalar scaling (scalar multiplication strictly preserves nullspace)
    max_val = float(np.max(np.abs(w)))
    if max_val > max_weight:
        w = w * (max_weight / max_val)

    return w


def compute_portfolio_exposures(
    weights: np.ndarray,
    betas: np.ndarray,
) -> dict[str, float]:
    """Computes net dollar exposure, net BTC beta, and gross exposure."""
    w = np.asarray(weights, dtype=np.float64)
    b = np.asarray(betas, dtype=np.float64)
    return {
        "net_dollar_exposure": float(np.sum(w)),
        "net_btc_beta": float(np.sum(w * b)),
        "gross_exposure": float(np.sum(np.abs(w))),
    }
