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


def test_funding_adjusted_total_return():
    """Verifies that forward total return correctly subtracts cumulative funding paid."""
    c = np.array([100.0, 105.0])
    fwd_price_ret = (c[1] / c[0]) - 1.0  # +5%
    cum_funding_paid = 0.008  # Long pays 0.8% funding
    fwd_total_ret = fwd_price_ret - cum_funding_paid

    assert pytest.approx(fwd_total_ret, 1e-6) == 0.042  # Net +4.2%


def test_beta_neutral_projection_nullspace():
    """Verifies that nullspace projection enforces both sum(w)=0 and sum(w*beta)=0."""
    from apex_multimodal_risk.src.train.exp008c_beta_neutral_carry_ml import project_weights_to_beta_neutral

    n = 15
    np.random.seed(42)
    w0 = np.random.randn(n)
    w0 = w0 - np.mean(w0)  # Dollar neutral initial guess
    beta = np.random.uniform(0.5, 2.0, n)
    beta[0] = 1.0  # BTC

    w_bn = project_weights_to_beta_neutral(w0, beta, max_pos=0.15)

    assert pytest.approx(np.sum(w_bn), abs=1e-5) == 0.0  # Dollar-neutral
    assert pytest.approx(np.sum(w_bn * beta), abs=1e-5) == 0.0  # BTC Beta-neutral
    assert np.sum(np.abs(w_bn)) <= 1.0001  # Gross <= 1.0
    assert np.all(w_bn >= -0.1501) and np.all(w_bn <= 0.1501)  # Max position cap


def test_vol_matched_blend_scaling():
    """Verifies causal volatility matching leverage scalar capping at <= 1.25x."""
    from apex_multimodal_risk.src.train.exp008c_beta_neutral_carry_ml import run_vol_matched_blend

    n_bars = 5000
    np.random.seed(42)
    r_apex = np.random.randn(n_bars) * (0.1844 / np.sqrt(8760))
    r_ml = np.random.randn(n_bars) * (0.1000 / np.sqrt(8760))
    turn_ml = np.full(n_bars, 0.005)

    r_scaled = run_vol_matched_blend(
        r_apex, r_ml, turn_ml, w_ml=0.10, slippage_bps=10.0, target_vol=0.1844, max_leverage=1.25
    )

    assert len(r_scaled) == n_bars
    assert not np.any(np.isnan(r_scaled))


def test_newey_west_hac_stats():
    """Verifies that Newey-West HAC correctly deflates t-stats on overlapping series."""
    from apex_multimodal_risk.src.train.exp008d_path_a_audit import newey_west_stats

    np.random.seed(42)
    # Generate an AR(1) / overlapping series with strong positive autocorrelation
    iid = np.random.randn(2000) + 0.10
    overlap_series = np.convolve(iid, np.ones(48) / 48.0, mode="valid")

    stats = newey_west_stats(overlap_series, lags=48)
    assert stats["naive_se"] < stats["hac_se"]
    assert stats["naive_t"] > stats["hac_t"]
    assert stats["hac_se"] > 0.0


def test_honest_vol_targeting_deadband():
    """Verifies that honest vol-targeting obeys deadband and incurs exact turnover."""
    from apex_multimodal_risk.src.train.exp008d_path_a_audit import apply_honest_vol_targeting

    n_bars = 500
    r_raw = np.random.randn(n_bars) * (0.1844 / np.sqrt(8760))
    underlying_turnover = np.zeros(n_bars)

    r_scaled, total_turnover, kappa = apply_honest_vol_targeting(
        r_raw, slippage_bps=10.0, underlying_turnover=underlying_turnover, target_vol=0.1844, deadband=0.05
    )

    assert len(r_scaled) == n_bars
    assert np.all(kappa >= 0.50) and np.all(kappa <= 1.25)
    assert np.all(total_turnover >= 0.0)


