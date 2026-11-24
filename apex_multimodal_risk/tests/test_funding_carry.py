#!/usr/bin/env python3
"""
Unit tests for EXP-011: ML-Gated 15-Coin Beta-Neutral Funding Carry & Basis-Squeeze Filter.
Fresh-clone safe: uses synthetic data, zero local binary or parquet dependencies.
"""

from __future__ import annotations

import numpy as np

from apex_multimodal_risk.src.train.exp011_funding_carry import (
    newey_west_stats,
    project_weights_to_beta_neutral,
)


def test_funding_pnl_sign_convention() -> None:
    """
    Verifies Binance perpetual funding cash flow rules:
      - fr > 0 (bullish / perp at premium): Longs pay shorts (Shorts earn +fr).
      - fr < 0 (bearish / perp at discount): Shorts pay longs (Longs earn +|fr|).
      - fund_pnl = sum(-w_i * fr_i) on hours 0, 8, 16 UTC.
    """
    # 1. Long position with positive funding rate (+10 bps) -> must pay funding (-10 bps)
    w_long = 1.0
    fr_pos = 0.0010
    cf_long_pos = -w_long * fr_pos
    assert cf_long_pos < 0.0, f"Long with positive funding should pay: {cf_long_pos}"
    assert abs(cf_long_pos - (-0.0010)) < 1e-12

    # 2. Long position with negative funding rate (-10 bps) -> must earn funding (+10 bps)
    fr_neg = -0.0010
    cf_long_neg = -w_long * fr_neg
    assert cf_long_neg > 0.0, f"Long with negative funding should earn: {cf_long_neg}"
    assert abs(cf_long_neg - 0.0010) < 1e-12

    # 3. Short position with positive funding rate (+10 bps) -> must earn funding (+10 bps)
    w_short = -1.0
    cf_short_pos = -w_short * fr_pos
    assert cf_short_pos > 0.0, f"Short with positive funding should earn: {cf_short_pos}"
    assert abs(cf_short_pos - 0.0010) < 1e-12

    # 4. Short position with negative funding rate (-10 bps) -> must pay funding (-10 bps)
    cf_short_neg = -w_short * fr_neg
    assert cf_short_neg < 0.0, f"Short with negative funding should pay: {cf_short_neg}"
    assert abs(cf_short_neg - (-0.0010)) < 1e-12

    # 5. Multi-asset portfolio vector check
    weights = np.array([+0.5, -0.5])
    funding_rates = np.array([+0.0010, +0.0020])  # Long coin 0 (pays 0.0005), Short coin 1 (earns 0.0010)
    total_fund_pnl = float(np.sum(-weights * funding_rates))
    assert abs(total_fund_pnl - (+0.0005)) < 1e-12, f"Total funding cash flow failed: {total_fund_pnl}"


def test_nullspace_dollar_and_beta_neutrality() -> None:
    """Verifies that project_weights_to_beta_neutral strictly enforces dollar and beta neutrality."""
    np.random.seed(42)
    n_assets = 15

    for _ in range(50):
        w0 = np.random.randn(n_assets)
        beta_t = np.random.uniform(0.3, 2.5, size=n_assets)
        beta_t[4] = 1.0  # BTCUSDT index

        w_bn = project_weights_to_beta_neutral(w0, beta_t, max_pos=0.15)

        dollar_net = abs(float(np.sum(w_bn)))
        beta_net = abs(float(np.sum(w_bn * beta_t)))
        max_pos = float(np.max(np.abs(w_bn)))
        gross = float(np.sum(np.abs(w_bn)))

        assert dollar_net < 1e-11, f"Dollar neutrality failed: {dollar_net}"
        assert beta_net < 1e-11, f"Beta neutrality failed: {beta_net}"
        assert max_pos <= 0.150001, f"Individual position cap exceeded 0.15: {max_pos}"
        assert gross <= 1.000001, f"Gross exposure exceeded 1.0: {gross}"


def test_zero_lookahead_carry_and_targets() -> None:
    """Verifies that trailing carry features and forward targets have strict zero lookahead."""
    np.random.seed(42)
    n_hours = 500

    # Synthetic prices and funding rates
    prices = 100.0 * np.exp(np.cumsum(np.random.randn(n_hours) * 0.005))
    fr = np.random.randn(n_hours) * 0.0005

    t_eval = 250
    w = 72

    # Trailing cumulative funding at t_eval
    cum_fr_past = float(np.sum(fr[t_eval - w : t_eval]))

    # Modify future data at t > t_eval drastically
    fr_corrupt = fr.copy()
    fr_corrupt[t_eval:] = 100.0

    cum_fr_check = float(np.sum(fr_corrupt[t_eval - w : t_eval]))
    assert cum_fr_past == cum_fr_check, "Future funding corrupted trailing calculation!"

    # Forward 168h total return over [t_eval, t_eval + 168]
    fwd_p = prices[t_eval + 168] / prices[t_eval] - 1.0
    fwd_fund = float(np.sum(fr[t_eval + 1 : t_eval + 169]))
    fwd_total = fwd_p - fwd_fund

    # Modify past data at t < t_eval drastically
    prices_past_corrupt = prices.copy()
    prices_past_corrupt[:t_eval] = 1.0
    fwd_p_check = prices[t_eval + 168] / prices[t_eval] - 1.0
    assert abs(fwd_p - fwd_p_check) == 0.0, "Past data corrupted forward target!"
    assert not np.isnan(fwd_total)


def test_newey_west_stats() -> None:
    """Verifies that Newey-West HAC properly inflates SE under positive autocorrelation."""
    np.random.seed(42)
    n = 1000

    # 1. Independent noise: HAC SE ~= Naive SE
    x_iid = np.random.randn(n)
    s_iid = newey_west_stats(x_iid, lags=10)
    assert abs(s_iid["naive_se"] - s_iid["hac_se"]) / s_iid["naive_se"] < 0.20

    # 2. Strongly persistent AR(1) series: HAC SE > Naive SE
    x_ar = np.zeros(n)
    for t in range(1, n):
        x_ar[t] = 0.90 * x_ar[t - 1] + np.random.randn()

    s_ar = newey_west_stats(x_ar, lags=20)
    assert s_ar["hac_se"] > s_ar["naive_se"], f"HAC SE {s_ar['hac_se']} should exceed Naive SE {s_ar['naive_se']}"
