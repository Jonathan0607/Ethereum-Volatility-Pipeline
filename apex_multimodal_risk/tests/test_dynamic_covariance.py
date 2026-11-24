"""Unit tests for EXP-012: Dynamic Covariance & Latent-Factor Beta Forecasting."""

from __future__ import annotations

import numpy as np
import pytest


def project_multi_factor_neutral_numpy(
    w0: np.ndarray,
    C: np.ndarray,
    gross_cap: float = 1.0,
    max_weight: float = 0.15,
) -> np.ndarray:
    """Pure NumPy implementation of multi-factor nullspace projection."""
    n = len(w0)
    K = C.shape[1]
    CtC = C.T @ C
    CtC_inv = np.linalg.inv(CtC + np.eye(K) * 1e-12)
    P = np.eye(n) - C @ CtC_inv @ C.T
    w_proj = P @ w0

    sum_abs = np.sum(np.abs(w_proj))
    if sum_abs < 1e-12:
        return np.zeros(n)
    w_scaled = w_proj * (gross_cap / sum_abs)

    max_w = np.max(np.abs(w_scaled))
    if max_w > max_weight:
        w_scaled = w_scaled * (max_weight / max_w)

    return w_scaled


def test_multi_factor_nullspace_projection_neutrality():
    """Verifies that nullspace projection achieves exact dollar and multi-factor neutrality."""
    np.random.seed(42)
    n_assets = 15

    # Raw signal weights
    w0 = np.random.randn(n_assets)
    # Constraints: Column 0 = Dollar neutrality (ones), Column 1 = Beta, Column 2 = Factor 2
    ones = np.ones((n_assets, 1))
    beta = np.random.uniform(0.5, 2.0, (n_assets, 1))
    factor2 = np.random.randn(n_assets, 1)

    C = np.hstack([ones, beta, factor2])  # Shape (15, 3)

    w_opt = project_multi_factor_neutral_numpy(w0, C, gross_cap=1.0, max_weight=0.15)

    # 1. Neutrality check: C.T @ w_opt == 0
    residuals = C.T @ w_opt
    assert np.all(np.abs(residuals) < 1e-11), f"Residuals exceed machine precision: {residuals}"

    # 2. Dollar neutrality
    assert pytest.approx(np.sum(w_opt), abs=1e-11) == 0.0

    # 3. Factor neutrality
    assert pytest.approx(float(w_opt @ beta.squeeze()), abs=1e-11) == 0.0
    assert pytest.approx(float(w_opt @ factor2.squeeze()), abs=1e-11) == 0.0

    # 4. Gross exposure & position boundary
    assert np.sum(np.abs(w_opt)) <= 1.0 + 1e-11
    assert np.all(np.abs(w_opt) <= 0.15 + 1e-11)


def test_woodbury_matrix_inversion_and_determinant_parity():
    """Verifies Woodbury 2-factor matrix inversion lemma and determinant lemma parity."""
    np.random.seed(42)
    n_assets = 15
    k_factors = 2

    # Factor loadings B and idiosyncratic variance d^2
    B = np.random.randn(n_assets, k_factors) * 0.1
    d = np.random.uniform(0.05, 0.20, n_assets)
    d_sq = d**2
    D = np.diag(d_sq)
    D_inv = np.diag(1.0 / d_sq)

    # Full covariance Sigma = D + B @ B.T
    Sigma = D + B @ B.T

    # 1. Direct inversion
    inv_direct = np.linalg.inv(Sigma)

    # 2. Woodbury inversion: D^-1 - D^-1 @ B @ (I_k + B.T @ D^-1 @ B)^-1 @ B.T @ D^-1
    D_inv_B = D_inv @ B
    M = np.eye(k_factors) + B.T @ D_inv_B
    M_inv = np.linalg.inv(M)
    inv_woodbury = D_inv - D_inv_B @ M_inv @ D_inv_B.T

    max_inv_err = np.max(np.abs(inv_direct - inv_woodbury))
    assert max_inv_err < 1e-10, f"Woodbury inversion error {max_inv_err} exceeds tolerance"

    # 3. Matrix determinant lemma: det(D + B @ B.T) = det(D) * det(I_k + B.T @ D^-1 @ B)
    logdet_direct = float(np.linalg.slogdet(Sigma)[1])
    logdet_woodbury = float(np.sum(np.log(d_sq)) + np.linalg.slogdet(M)[1])

    assert pytest.approx(logdet_direct, rel=1e-6) == logdet_woodbury


def test_qlike_loss_minimum_at_identity():
    """Verifies that Gaussian NLL / QLIKE loss achieves exact 0 minimum when Sigma_hat == Sigma_true."""
    np.random.seed(42)
    n_assets = 15
    A = np.random.randn(n_assets, n_assets) * 0.1
    Sigma_true = A @ A.T + np.eye(n_assets) * 0.05

    def qlike(pred: np.ndarray, true: np.ndarray) -> float:
        inv_pred = np.linalg.inv(pred)
        prod = inv_pred @ true
        tr = np.trace(prod)
        _, logdet = np.linalg.slogdet(prod)
        return float((tr - logdet - n_assets) / n_assets)

    # Exact match -> loss must be exactly 0
    loss_exact = qlike(Sigma_true, Sigma_true)
    assert pytest.approx(loss_exact, abs=1e-10) == 0.0

    # Perturbed prediction -> loss must be strictly positive
    Sigma_perturbed = Sigma_true * 1.25 + np.eye(n_assets) * 0.02
    loss_perturbed = qlike(Sigma_perturbed, Sigma_true)
    assert loss_perturbed > 0.0, f"QLIKE loss should be positive for imperfect forecast: {loss_perturbed}"


def test_vasicek_shrinkage_bounds():
    """Verifies that Vasicek Bayesian shrinkage shrinks extreme beta estimates toward prior mean."""
    np.random.seed(42)
    raw_betas = np.array([0.1, 0.4, 0.8, 1.0, 1.2, 1.5, 2.8, 3.5] + [1.0] * 7)

    cs_var = float(np.var(raw_betas))
    est_var = 0.05
    w_vasicek = cs_var / (cs_var + est_var)
    prior_mean = 1.0

    shrunk_betas = np.clip(w_vasicek * raw_betas + (1.0 - w_vasicek) * prior_mean, 0.2, 3.0)

    # Shrunk betas must be strictly between raw and prior
    assert shrunk_betas[0] > raw_betas[0], "Low beta should be pulled up toward 1.0"
    assert shrunk_betas[7] < raw_betas[7], "Extreme high beta should be pulled down toward 1.0"
    assert np.all(shrunk_betas >= 0.2)
    assert np.all(shrunk_betas <= 3.0)


@pytest.mark.requires_local_artifacts
def test_graph_beta_cov_net_forward_and_spd():
    """Verifies GraphBetaCovNet architecture, parameter count, forward pass, and guaranteed SPD covariance."""
    import torch

    from apex_multimodal_risk.src.train.exp012_dynamic_covariance import (
        GraphBetaCovNet,
        compute_woodbury_cov_and_loss,
    )

    torch.manual_seed(42)
    n_features = 13
    d_model = 16
    n_factors = 2
    n_assets = 15
    btc_idx = 4

    model = GraphBetaCovNet(in_features=n_features, d_model=d_model, n_factors=n_factors)

    # Verify model is ultra-compact (< 10,000 parameters)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params < 10000, f"Model has {total_params} parameters, exceeding 10,000 budget!"

    batch_size = 4
    x = torch.randn(batch_size, n_assets, n_features)
    # Adjacency: row-normalized
    adj_raw = torch.rand(batch_size, n_assets, n_assets)
    adj = adj_raw / (adj_raw.sum(dim=-1, keepdim=True) + 1e-6)
    base_beta = torch.ones(batch_size, n_assets)

    beta_hat, B, d = model(x, adj, base_beta, btc_idx=btc_idx)

    # 1. Output shapes
    assert beta_hat.shape == (batch_size, n_assets)
    assert B.shape == (batch_size, n_assets, n_factors)
    assert d.shape == (batch_size, n_assets)

    # 2. BTC Beta constraint
    assert torch.all(beta_hat[:, btc_idx] == 1.0)

    # 3. Covariance reconstruction & SPD guarantee
    Sigma_true = torch.eye(n_assets).unsqueeze(0).expand(batch_size, -1, -1)
    Sigma_hat, loss = compute_woodbury_cov_and_loss(B, d, Sigma_true)

    assert Sigma_hat.shape == (batch_size, n_assets, n_assets)
    assert not torch.isnan(loss)

    # Check minimum eigenvalue across all batch elements > 0 (strictly SPD)
    for b in range(batch_size):
        sigma_np = Sigma_hat[b].detach().numpy()
        eigvals = np.linalg.eigvalsh(sigma_np)
        assert np.all(eigvals > 0.0), f"Covariance matrix is not SPD, min eig: {np.min(eigvals)}"

    # 4. Backward pass (joint beta + cov loss)
    loss_total = beta_hat.sum() + loss
    loss_total.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradient for {name} is None"
        assert not torch.isnan(p.grad).any(), f"Gradient for {name} contains NaN"
