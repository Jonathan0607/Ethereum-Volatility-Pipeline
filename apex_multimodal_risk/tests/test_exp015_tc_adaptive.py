#!/usr/bin/env python3
"""
test_exp015_tc_adaptive.py
==========================
Fresh-clone-safe unit tests for EXP-015 pure mathematical primitives:
  1. Exponential time-decay sample weight monotonicity and normalization.
  2. Continuous rank-Gaussian EMA nullspace weights (< 1e-10 dollar & beta neutrality, bounds).
  3. Causal dispersion-gated gross exposure bounds [0.25, 1.0] and strict causality (zero lookahead).
"""

import numpy as np
import pandas as pd

from apex_multimodal_risk.src.serving.tc_adaptive_math import (
    compute_causal_dispersion_scale,
    compute_exponential_time_decay_weights,
    compute_rank_gaussian_scores,
    simulate_continuous_ema_sleeve,
)


def test_exponential_time_decay_monotonicity_and_bounds():
    """Verifies exponential sample weights decay strictly monotonically with age and handle uniform prior."""
    ages = np.array([0.0, 90.0, 180.0, 365.0, 730.0])

    # Uniform prior
    w_unif = compute_exponential_time_decay_weights(ages, halflife_days=None)
    assert np.allclose(w_unif, 1.0), "Uniform prior must return all ones"

    # 180-day and 365-day exponential decay
    for hl in [180.0, 365.0]:
        w = compute_exponential_time_decay_weights(ages, halflife_days=hl)
        assert np.all(w > 0.0), "All weights must be strictly positive"
        assert np.isclose(np.mean(w), 1.0, atol=1e-12), "Weights must normalize to mean 1.0"
        # Strict monotonic decay with increasing age
        assert np.all(np.diff(w) < 0.0), f"Weights failed monotonic decay for half-life {hl}"

    # Verify exact half-life ratio w(hl) / w(0) == 0.5
    w_180 = compute_exponential_time_decay_weights(np.array([0.0, 180.0]), halflife_days=180.0)
    assert np.isclose(w_180[1] / w_180[0], 0.5, atol=1e-12)


def test_continuous_rank_gaussian_ema_neutrality_and_deadband():
    """Verifies K1/K2 continuous rank-Gaussian EMA weights maintain < 1e-10 neutrality and bounded leverage."""
    rng = np.random.RandomState(42)
    n_ts, n_assets = 72, 15
    ts_index = pd.date_range("2024-01-01", periods=n_ts, freq="1h", tz="UTC")

    pred_mat = rng.randn(n_ts, n_assets)
    ret_1h_mat = rng.randn(n_ts, n_assets) * 0.005
    fr_mat = rng.uniform(-0.0002, 0.0003, size=(n_ts, n_assets))
    beta_mat = rng.uniform(0.5, 2.0, size=(n_ts, n_assets))
    beta_mat[:, 4] = 1.0  # BTCUSDT

    # Rank-Gaussian symmetry & zero mean
    z_mat = compute_rank_gaussian_scores(pred_mat)
    assert np.allclose(np.mean(z_mat, axis=1), 0.0, atol=1e-12)

    # K1 (unscaled gross cap 1.0)
    gross_k1, turn_k1, pos_k1, w_k1 = simulate_continuous_ema_sleeve(
        pred_mat, ret_1h_mat, fr_mat, beta_mat, ts_index, rebal_hours=12, ema_halflife=48.0, deadband=0.015
    )
    # Check active bars (every rebal_hours=12) strictly satisfy < 1e-10 neutrality against beta_mat[t]
    for t in range(0, n_ts, 12):
        dollar_net = float(np.sum(w_k1[t]))
        beta_net = float(np.sum(w_k1[t] * beta_mat[t]))
        gross_lev = float(np.sum(np.abs(w_k1[t])))
        max_w = float(np.max(np.abs(w_k1[t])))
        assert abs(dollar_net) < 1e-10, f"Dollar neutrality violated at t={t}: {dollar_net}"
        assert abs(beta_net) < 1e-10, f"Beta neutrality violated at t={t}: {beta_net}"
        assert gross_lev <= 1.0 + 1e-9, f"Gross leverage exceeded at t={t}: {gross_lev}"
        assert max_w <= 0.15 + 1e-9, f"Position limit exceeded at t={t}: {max_w}"

    # Verify strict 1-bar execution lag
    assert np.allclose(pos_k1[0], 0.0), "Bar 0 position must be zero (1-bar execution lag)"
    assert np.allclose(pos_k1[1:], w_k1[:-1]), "Shifted positions must lag weights by exactly 1 hour"
    assert np.all(np.isfinite(gross_k1)) and np.all(turn_k1 >= 0.0)


def test_causal_dispersion_scale_bounds_and_zero_lookahead():
    """Verifies K2 dispersion scaling stays in [0.25, 1.0] and has strictly zero future lookahead."""
    rng = np.random.RandomState(42)
    n_ts, n_assets = 800, 15
    ret_mat = rng.randn(n_ts, n_assets) * 0.01
    # Inject low dispersion chop region in bars 500..700
    ret_mat[500:700] *= 0.10

    scale_base, trail_disp, exp_med = compute_causal_dispersion_scale(ret_mat)

    assert len(scale_base) == n_ts
    assert np.all(scale_base >= 0.25 - 1e-12), "Dispersion scale violated 0.25 floor"
    assert np.all(scale_base <= 1.00 + 1e-12), "Dispersion scale violated 1.00 ceiling"
    assert np.all(np.isfinite(trail_disp)) and np.all(np.isfinite(exp_med))
    # Verify scale throttles down during chop regime
    assert float(np.min(scale_base[550:700])) < 0.75, "Scale should throttle downward in low-dispersion chop"

    # Zero-lookahead causality check: wildly perturb returns after t_cutoff=450
    t_cutoff = 450
    ret_perturbed = ret_mat.copy()
    ret_perturbed[t_cutoff + 1 :] *= 50.0

    scale_pert, _, _ = compute_causal_dispersion_scale(ret_perturbed)
    max_causal_diff = float(np.max(np.abs(scale_base[: t_cutoff + 1] - scale_pert[: t_cutoff + 1])))
    assert max_causal_diff < 1e-14, f"Future lookahead detected in dispersion scale! diff={max_causal_diff}"
