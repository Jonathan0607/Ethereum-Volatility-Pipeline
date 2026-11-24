#!/usr/bin/env python3
"""
Unit tests for EXP-009: Decision-Focused Learning Policy & Differentiable Projection
Fresh-clone safe: uses synthetic tensors, zero local data artifacts required.
"""

from __future__ import annotations

import torch
from torch.autograd import grad

from apex_multimodal_risk.src.train.exp009_decision_focused import (
    DecisionFocusedPolicy,
    compute_nullspace_projection_matrix,
    project_weights_diff,
)


def test_differentiable_nullspace_dollar_and_beta_neutrality() -> None:
    """Verifies that project_weights_diff strictly enforces dollar and beta neutrality (< 1e-5)."""
    torch.manual_seed(42)
    n_assets = 15

    for _ in range(50):
        s = torch.randn(n_assets, requires_grad=True)
        # Random betas between 0.3 and 2.5 with BTC = 1.0
        beta = torch.rand(n_assets) * 2.2 + 0.3
        beta[4] = 1.0  # BTC index

        p_t = compute_nullspace_projection_matrix(beta)
        w = project_weights_diff(s, p_t, max_pos=0.15)

        dollar_net = float(torch.abs(torch.sum(w)).item())
        beta_net = float(torch.abs(torch.sum(w * beta)).item())

        assert dollar_net < 1e-5, f"Dollar neutrality violated: {dollar_net}"
        assert beta_net < 1e-5, f"Beta neutrality violated: {beta_net}"


def test_gross_cap_and_position_bounds() -> None:
    """Verifies gross cap sum(|w|) <= 1.0 and individual asset cap |w_i| <= 0.15."""
    torch.manual_seed(42)
    n_assets = 15
    beta = torch.ones(n_assets)
    beta[0] = 1.2
    beta[1] = 0.8
    p_t = compute_nullspace_projection_matrix(beta)

    # Test extreme positive, negative, mixed, and small scores
    test_scores = [
        torch.ones(n_assets) * 100.0,
        torch.ones(n_assets) * -50.0,
        torch.tensor([10.0 if i % 2 == 0 else -10.0 for i in range(n_assets)]),
        torch.randn(n_assets) * 5.0,
        torch.randn(n_assets) * 0.001,
    ]

    for s in test_scores:
        w = project_weights_diff(s, p_t, max_pos=0.15)
        gross = float(torch.sum(torch.abs(w)).item())
        max_pos = float(torch.max(torch.abs(w)).item())

        assert gross <= 1.000001, f"Gross exposure exceeded 1.0: {gross}"
        assert max_pos <= 0.150001, f"Individual position exceeded 0.15: {max_pos}"


def test_gradient_flow_through_l1_turnover() -> None:
    """Verifies strict gradient flow through the L1 turnover penalty term with zero NaNs."""
    torch.manual_seed(42)
    n_assets = 15
    n_features = 21

    policy = DecisionFocusedPolicy(in_features=22, hidden=16)
    x_t = torch.randn(n_assets, n_features)
    w_prev = torch.randn(n_assets) * 0.05
    beta = torch.rand(n_assets) * 1.5 + 0.5
    beta[0] = 1.0
    p_t = compute_nullspace_projection_matrix(beta)

    # Forward step
    w_t = policy(x_t, w_prev, p_t, max_pos=0.15)
    turnover = torch.sum(torch.abs(w_t - w_prev))

    # Test gradient of turnover w.r.t policy parameters
    grads = grad(turnover, policy.parameters(), create_graph=False)

    for i, g in enumerate(grads):
        assert not torch.isnan(g).any(), f"NaN detected in parameter gradient {i}"
        assert not torch.isinf(g).any(), f"Inf detected in parameter gradient {i}"
        grad_norm = float(torch.norm(g).item())
        assert grad_norm > 0.0, f"Gradient vanished (norm=0) for parameter {i}"


def test_nullspace_projection_matrix_properties() -> None:
    """Verifies that P_t is symmetric, idempotent (P^2 = P), and satisfies P C = 0."""
    torch.manual_seed(42)
    n_assets = 15
    beta = torch.rand(n_assets) * 1.8 + 0.4
    beta[0] = 1.0

    p_t = compute_nullspace_projection_matrix(beta)

    # Symmetry: P == P^T
    sym_err = float(torch.max(torch.abs(p_t - p_t.T)).item())
    assert sym_err < 1e-5, f"Projection matrix not symmetric: {sym_err}"

    # Idempotence: P @ P == P
    idem_err = float(torch.max(torch.abs(p_t @ p_t - p_t)).item())
    assert idem_err < 1e-5, f"Projection matrix not idempotent: {idem_err}"

    # Nullspace condition: P @ C == 0 (< 1e-5 as required by prompt)
    ones = torch.ones_like(beta)
    c_mat = torch.stack([ones, beta], dim=1)
    null_err = float(torch.max(torch.abs(p_t @ c_mat)).item())
    assert null_err < 1e-5, f"Projection matrix does not annihilate constraints: {null_err}"
