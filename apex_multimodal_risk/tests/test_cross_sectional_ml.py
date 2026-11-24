"""Unit tests for EXP-008 15-Coin Cross-Sectional ML Alpha Factor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr


def test_cross_sectional_rank_normalization():
    """Verifies that cross-sectional ranking produces [-1, +1] range and ~0 mean."""
    n_assets = 15
    np.random.seed(42)
    raw_vals = np.random.randn(n_assets)
    s = pd.Series(raw_vals)
    ranks = s.rank(method="average")
    norm_vals = (2.0 * (ranks - 1.0) / (n_assets - 1)) - 1.0

    assert norm_vals.min() == -1.0
    assert norm_vals.max() == 1.0
    assert pytest.approx(norm_vals.mean(), abs=1e-6) == 0.0


def test_beta_neutral_reversal_demeaning():
    """Verifies that subtracting cross-sectional mean ensures zero net market beta."""
    np.random.seed(42)
    raw_returns = np.random.randn(15) * 0.02 + 0.01  # Market drifted +1%
    demeaned = raw_returns - np.mean(raw_returns)

    assert pytest.approx(np.sum(demeaned), abs=1e-6) == 0.0


def test_dollar_neutral_portfolio_weights():
    """Verifies dollar-neutrality (+0.50 long, -0.50 short, net 0.00, gross 1.00)."""
    n_assets = 15
    preds = np.linspace(-1.0, 1.0, n_assets)
    vols = np.full(n_assets, 0.05)
    inv_vols = 1.0 / vols

    ranked = np.argsort(preds)
    shorts = ranked[:4]
    longs = ranked[-4:]

    w = np.zeros(n_assets)
    long_inv = [inv_vols[i] for i in longs]
    short_inv = [inv_vols[i] for i in shorts]

    for idx in longs:
        w[idx] = +0.50 * (inv_vols[idx] / sum(long_inv))
    for idx in shorts:
        w[idx] = -0.50 * (inv_vols[idx] / sum(short_inv))

    assert pytest.approx(np.sum(w[longs]), 1e-6) == +0.50
    assert pytest.approx(np.sum(w[shorts]), 1e-6) == -0.50
    assert pytest.approx(np.sum(w), 1e-6) == 0.00  # Dollar-neutral
    assert pytest.approx(np.sum(np.abs(w)), 1e-6) == 1.00  # 1.0x Gross exposure


def test_spearman_ic_computation():
    """Verifies Spearman rank IC calculation."""
    preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    sp = spearmanr(preds, targets).correlation
    assert pytest.approx(sp, 1e-6) == 1.0

    targets_rev = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
    sp_rev = spearmanr(preds, targets_rev).correlation
    assert pytest.approx(sp_rev, 1e-6) == -1.0


def test_ema_smoothed_scores_half_life():
    """Verifies EMA smoothing formula with 72h half-life."""
    half_life = 72.0
    theta = float(np.exp(-np.log(2.0) / half_life))
    assert 0.990 < theta < 0.991

    # Constant signal should remain constant
    scores = np.ones((100, 15))
    cur = scores[0].copy()
    for t in range(100):
        cur = theta * cur + (1.0 - theta) * scores[t]
    assert np.allclose(cur, 1.0)


def test_convex_l1_optimizer_dollar_neutral():
    """Verifies that SLSQP QP produces dollar-neutral, bounded weights with L1 penalty."""
    from scipy.optimize import LinearConstraint, minimize

    n = 15
    gamma = 2.0
    lambda_tc = 0.0020
    sigma_horizon = np.diag(np.full(n, 0.0168))
    alpha_horizon = np.linspace(-0.005, 0.005, n)
    w_curr = np.zeros(n)

    A_eq = np.zeros((1, 3 * n))
    A_eq[0, :n] = 1.0
    bounds = [(-0.15, 0.15)] * n + [(0.0, None)] * n + [(0.0, 0.15)] * n

    A_ineq = np.zeros((1 + 4 * n, 3 * n))
    A_ineq[0, 2 * n :] = 1.0
    for i in range(n):
        A_ineq[1 + i, i] = -1.0
        A_ineq[1 + i, n + i] = 1.0
        A_ineq[1 + n + i, i] = 1.0
        A_ineq[1 + n + i, n + i] = 1.0
        A_ineq[1 + 2 * n + i, i] = -1.0
        A_ineq[1 + 2 * n + i, 2 * n + i] = 1.0
        A_ineq[1 + 3 * n + i, i] = 1.0
        A_ineq[1 + 3 * n + i, 2 * n + i] = 1.0

    lc_eq = LinearConstraint(A_eq, [0.0], [0.0])

    def obj_fn(x: np.ndarray) -> float:
        w = x[:n]
        u = x[n : 2 * n]
        return float(-np.dot(w, alpha_horizon) + 0.5 * gamma * np.dot(w, sigma_horizon @ w) + lambda_tc * np.sum(u))

    def grad_fn(x: np.ndarray) -> np.ndarray:
        w = x[:n]
        gw = -alpha_horizon + gamma * (sigma_horizon @ w)
        gu = np.full(n, lambda_tc)
        gv = np.zeros(n)
        return np.concatenate([gw, gu, gv])

    lb_ineq = np.zeros(1 + 4 * n)
    ub_ineq = np.full(1 + 4 * n, np.inf)
    lb_ineq[0] = 0.0
    ub_ineq[0] = 1.0
    for i in range(n):
        lb_ineq[1 + i] = -w_curr[i]
        lb_ineq[1 + n + i] = w_curr[i]

    lc_ineq = LinearConstraint(A_ineq, lb_ineq, ub_ineq)
    x0 = np.concatenate([w_curr, np.zeros(n), np.abs(w_curr)])
    res = minimize(obj_fn, x0, jac=grad_fn, method="SLSQP", bounds=bounds, constraints=[lc_eq, lc_ineq], options={"maxiter": 80})

    assert res.success
    w_opt = res.x[:n]
    assert pytest.approx(np.sum(w_opt), abs=1e-5) == 0.0  # Dollar neutral
    assert np.sum(np.abs(w_opt)) <= 1.0001  # Max gross <= 1.0
    assert np.all(w_opt >= -0.1501) and np.all(w_opt <= 0.1501)  # Max position bound
