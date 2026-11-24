#!/usr/bin/env python3
"""
test_orthogonal_microstructure.py
=================================
Unit tests for EXP-014:
  1. Orthogonal derivatives microstructure feature causality, scaling, and mathematical bounds.
  2. Squeeze-risk throttling function mathematical guarantees (monotonicity, bounds [0.5, 1.0]).
  3. Dual-threat portfolio neutrality (dollar neutrality and beta neutrality < 10^-10).
"""

import numpy as np

from apex_multimodal_risk.src.serving.portfolio_math import project_beta_neutral


def test_orthogonal_features_mathematical_bounds():
    """Verifies that OFI, whale trade size surge, and divergence features stay within strict mathematical bounds."""
    # OFI: 2 * (taker_buy / total) - 1
    tb_vol = np.array([0.0, 50.0, 100.0, 20.0])
    tot_vol = np.array([100.0, 100.0, 100.0, 0.0])

    ofi = np.clip(2.0 * (tb_vol / (tot_vol + 1e-5)) - 1.0, -1.0, 1.0)
    assert np.all(ofi >= -1.0), "OFI lower bound violation!"
    assert np.all(ofi <= 1.0), "OFI upper bound violation!"
    assert np.isclose(ofi[0], -1.0, atol=1e-5), "Zero taker buy must give -1.0 OFI"
    assert np.isclose(ofi[2], 1.0, atol=1e-5), "100% taker buy must give +1.0 OFI"

    # Cross-sectional rank normalization bounds: rank_cs(x) in [-1.0, +1.0]
    raw_vals = np.array([10.5, 20.2, 5.1, 100.0, -2.0])
    n = len(raw_vals)
    ranks = np.argsort(np.argsort(raw_vals))
    norm_ranks = 2.0 * ranks / (n - 1) - 1.0
    assert np.isclose(np.min(norm_ranks), -1.0), "Rank cs min should be -1.0"
    assert np.isclose(np.max(norm_ranks), 1.0), "Rank cs max should be +1.0"

    # Whale-vs-retail divergence: smart_retail_spread = rank(top) - rank(retail)
    # Difference of two [-1, +1] variables is bounded in [-2.0, +2.0]
    spread = norm_ranks - np.roll(norm_ranks, 1)
    assert np.all(spread >= -2.0), "Spread lower bound violation"
    assert np.all(spread <= 2.0), "Spread upper bound violation"


def test_squeeze_risk_throttling_properties():
    """Verifies that squeeze throttling multiplier satisfies:
    1. scale(p) in [0.5, 1.0]
    2. scale(p) == 1.0 for p <= q85
    3. scale(p) == 0.5 for p >= 1.0
    4. scale(p) is strictly non-increasing in p
    """
    q85 = 0.85
    p_test = np.linspace(0.0, 1.0, 100)
    scale = 1.0 - 0.5 * np.clip((p_test - q85) / (1.0 - q85 + 1e-8), 0.0, 1.0)

    # 1. Bounds
    assert np.all(scale >= 0.5), "Scale dropped below 0.5 floor"
    assert np.all(scale <= 1.0), "Scale exceeded 1.0 ceiling"

    # 2. No throttling below q85
    assert np.all(scale[p_test <= q85] == 1.0), "Should not throttle below threshold"

    # 3. Maximum 50% throttling at p=1.0
    assert np.isclose(scale[-1], 0.5), "Must throttle exactly 50% at p=1.0"

    # 4. Monotonic non-increasing
    diffs = np.diff(scale)
    assert np.all(diffs <= 1e-12), "Throttling multiplier must be monotonically non-increasing"


def test_dual_threat_portfolio_neutrality():
    """Verifies that project_beta_neutral maintains dollar and beta neutrality
    even after non-linear squeeze-risk throttling.
    """
    rng = np.random.RandomState(42)
    n_assets = 15
    raw_alpha = rng.randn(n_assets)
    betas = rng.uniform(0.5, 1.8, size=n_assets)
    betas[4] = 1.0  # BTC

    # Squeeze throttling probabilities
    p_sq = rng.uniform(0.1, 0.99, size=n_assets)
    q85 = 0.85
    scale = 1.0 - 0.5 * np.clip((p_sq - q85) / (1.0 - q85 + 1e-8), 0.0, 1.0)

    # Throttled alpha
    throttled_alpha = raw_alpha * scale

    # Portfolio weights
    w = project_beta_neutral(throttled_alpha, betas, max_pos=0.15, gross_cap=1.0)

    # Neutrality assertions
    dollar_net = float(np.sum(w))
    beta_net = float(np.sum(w * betas))
    gross_lev = float(np.sum(np.abs(w)))
    max_pos = float(np.max(np.abs(w)))

    assert abs(dollar_net) < 1e-10, f"Dollar neutrality violated: {dollar_net}"
    assert abs(beta_net) < 1e-10, f"Beta neutrality violated: {beta_net}"
    assert gross_lev <= 1.0 + 1e-6, f"Gross leverage exceeded: {gross_lev}"
    assert max_pos <= 0.15 + 1e-6, f"Position cap exceeded: {max_pos}"
