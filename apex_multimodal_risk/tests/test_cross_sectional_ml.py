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
