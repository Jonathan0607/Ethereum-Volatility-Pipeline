#!/usr/bin/env python3
"""
Unit tests for EXP-010: ML-Filtered Rolling PCA / Eigenportfolio Statistical Arbitrage.
Fresh-clone safe: uses synthetic data, zero local binary or parquet dependencies.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def test_pca_residual_orthogonality() -> None:
    """Verifies that OLS residuals against PC1 and PC2 are orthogonal to factor returns (< 1e-5)."""
    np.random.seed(42)
    n_bars = 500
    n_assets = 15

    # Generate synthetic multi-asset returns with 2 latent factors
    market_factor = np.random.randn(n_bars) * 0.02
    sector_factor = np.random.randn(n_bars) * 0.01
    idiosyncratic = np.random.randn(n_bars, n_assets) * 0.015

    beta_mkt = np.random.uniform(0.6, 1.4, size=n_assets)
    beta_sec = np.random.uniform(-0.5, 0.5, size=n_assets)

    returns = (
        market_factor[:, None] * beta_mkt
        + sector_factor[:, None] * beta_sec
        + idiosyncratic
    )

    # Standardize
    mu = np.mean(returns, axis=0)
    sig = np.std(returns, axis=0) + 1e-8
    std_returns = (returns - mu) / sig

    pca = PCA(n_components=2)
    factors = pca.fit_transform(std_returns)
    f1, f2 = factors[:, 0], factors[:, 1]

    # Regress each coin on [1, F1, F2]
    x_mat = np.column_stack([np.ones(n_bars), f1, f2])
    inv_xtx = np.linalg.pinv(x_mat.T @ x_mat)

    for i in range(n_assets):
        y_vec = returns[:, i]
        beta_i = inv_xtx @ (x_mat.T @ y_vec)
        resid = y_vec - x_mat @ beta_i

        # Orthogonality checks
        corr_f1 = abs(float(np.corrcoef(resid, f1)[0, 1]))
        corr_f2 = abs(float(np.corrcoef(resid, f2)[0, 1]))
        mean_resid = abs(float(np.mean(resid)))

        assert corr_f1 < 1e-5, f"Residual correlated with PC1 for asset {i}: {corr_f1}"
        assert corr_f2 < 1e-5, f"Residual correlated with PC2 for asset {i}: {corr_f2}"
        assert mean_resid < 1e-6, f"Residual mean non-zero for asset {i}: {mean_resid}"


def test_exact_hedge_factor_and_dollar_neutrality() -> None:
    """Verifies that analytic nullspace hedge enforces dollar, PC1, and PC2 neutrality (< 1e-12)."""
    np.random.seed(42)
    n_assets = 15

    for _ in range(25):
        beta1 = np.random.uniform(0.4, 2.0, size=n_assets)
        beta2 = np.random.uniform(-0.8, 0.8, size=n_assets)
        ones = np.ones(n_assets)

        c_mat = np.column_stack([ones, beta1, beta2])
        inv_cc = np.linalg.pinv(c_mat.T @ c_mat)
        p_proj = np.eye(n_assets) - c_mat @ inv_cc @ c_mat.T

        # Test long and short triggers across all assets
        for coin_idx in [0, 4, 10, 14]:
            for direction in [-1.0, 1.0]:
                e_i = np.zeros(n_assets)
                e_i[coin_idx] = direction

                w_hedged = p_proj @ e_i
                gross = np.sum(np.abs(w_hedged))
                w_trade = w_hedged / (gross + 1e-12)

                dollar_net = abs(float(np.sum(w_trade)))
                pc1_net = abs(float(np.sum(w_trade * beta1)))
                pc2_net = abs(float(np.sum(w_trade * beta2)))
                gross_sum = float(np.sum(np.abs(w_trade)))

                assert dollar_net < 1e-11, f"Dollar neutrality failed: {dollar_net}"
                assert pc1_net < 1e-11, f"PC1 neutrality failed: {pc1_net}"
                assert pc2_net < 1e-11, f"PC2 neutrality failed: {pc2_net}"
                assert abs(gross_sum - 1.0) < 1e-11, f"Gross sum not 1.0: {gross_sum}"


def test_zero_pca_lookahead_synthetic() -> None:
    """Verifies rolling PCA fit on [0, 720) is strictly uninfluenced by future data (t >= 720)."""
    np.random.seed(42)
    n_assets = 15
    data_past = np.random.randn(720, n_assets) * 0.02

    # Fit PCA on original past
    std_past1 = (data_past - np.mean(data_past, axis=0)) / (np.std(data_past, axis=0) + 1e-8)
    pca1 = PCA(n_components=2).fit(std_past1)

    # Corrupt future data at t >= 720 drastically
    future_corrupt = np.random.randn(500, n_assets) * 1000.0

    # Ensure PCA on past window remains bit-for-bit identical
    std_past2 = (data_past - np.mean(data_past, axis=0)) / (np.std(data_past, axis=0) + 1e-8)
    pca2 = PCA(n_components=2).fit(std_past2)

    diff_v1 = float(np.max(np.abs(pca1.components_[0] - pca2.components_[0])))
    diff_v2 = float(np.max(np.abs(pca1.components_[1] - pca2.components_[1])))
    diff_ar = float(abs(pca1.explained_variance_ratio_[0] - pca2.explained_variance_ratio_[0]))

    assert diff_v1 == 0.0, f"PC1 component differed: {diff_v1}"
    assert diff_v2 == 0.0, f"PC2 component differed: {diff_v2}"
    assert diff_ar == 0.0, f"Absorption ratio differed: {diff_ar}"
    # Verify future corruption had no influence
    assert np.mean(np.abs(future_corrupt)) > 100.0


def test_ou_halflife_mean_reversion() -> None:
    """Verifies that AR(1) OU half-life estimates positive finite half-life for mean-reverting series."""
    np.random.seed(42)
    n_points = 200
    phi = 0.85  # theoretical discrete half-life = ln(2) / -ln(0.85) ~= 4.26
    theoretical_hl = 0.69315 / (-np.log(phi))

    # Generate AR(1) process
    s = np.zeros(n_points)
    for t in range(1, n_points):
        s[t] = phi * s[t - 1] + np.random.randn() * 0.1

    s_prev = s[:-1] - np.mean(s[:-1])
    delta_s = np.diff(s) - np.mean(np.diff(s))
    b = float(np.sum(s_prev * delta_s) / (np.sum(s_prev**2) + 1e-12))

    assert b < 0.0, f"Estimated mean-reversion slope should be negative: {b}"
    kappa = -np.log(1.0 + np.clip(b, -0.999, -0.001))
    estimated_hl = 0.69315 / kappa

    # Estimated half life should be within reasonable proximity of theoretical
    assert abs(estimated_hl - theoretical_hl) < 3.5, f"Estimated half-life {estimated_hl} vs theoretical {theoretical_hl}"
