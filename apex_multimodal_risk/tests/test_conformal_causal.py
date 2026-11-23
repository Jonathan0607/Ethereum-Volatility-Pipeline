"""
Unit Tests for Delayed-Feedback Adaptive Conformal Inference (ACI).
Verifies:
1. Strict 6-hour delay queue causality (raising CausalViolationError on early feedback).
2. Adaptive alpha_t update dynamics (tightening on failure, creeping on success).
3. Rolling buffer window capacity enforcement (W = 500).
4. Multi-asset state isolation.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from apex_multimodal_risk.src.models.conformal import (
    DelayedFeedbackACI,
    MultiAssetDelayedFeedbackACI,
    CausalViolationError,
    kupiec_pof_test,
)


def test_causal_delay_enforcement():
    """Verify that delivering feedback before 6 hours strictly raises CausalViolationError."""
    aci = DelayedFeedbackACI(delay_hours=6)

    t0 = pd.Timestamp("2024-01-01 12:00:00+00:00")
    aci.predict(raw_var=-0.05, bar_time=t0)

    # Attempt feedback at +1h, +5h, and +5h59m -> ALL MUST FAIL
    for early_minutes in [60, 300, 359]:
        t_early = t0 + timedelta(minutes=early_minutes)
        with pytest.raises(CausalViolationError) as exc_info:
            aci.step_feedback(target_bar_time=t0, current_time=t_early, realized_y=-0.04)
        assert "Causal violation" in str(exc_info.value)

    # Feedback at exactly +6h must SUCCEED
    t_exact_6h = t0 + timedelta(hours=6)
    res = aci.step_feedback(target_bar_time=t0, current_time=t_exact_6h, realized_y=-0.04)
    assert res is not None
    assert "alpha_t" in res
    assert res["buffer_len"] == 1


def test_adaptive_alpha_dynamics():
    """Verify alpha_t decreases on crash violations and increases on calm non-violations."""
    aci = DelayedFeedbackACI(nominal_alpha=0.05, gamma=0.01, delay_hours=6)

    t_start = pd.Timestamp("2024-01-01 00:00:00+00:00")

    # Step 1: Normal non-violation feedback
    # realized_y (-0.02) > aci_var (-0.05) -> err = 0
    # alpha_t should increase: alpha + gamma * (0.05 - 0) = 0.05 + 0.01 * 0.05 = 0.0505
    aci.predict(raw_var=-0.05, bar_time=t_start)
    res1 = aci.step_feedback(
        target_bar_time=t_start,
        current_time=t_start + timedelta(hours=6),
        realized_y=-0.02,
    )
    assert res1["err"] == 0.0
    assert res1["alpha_t"] > 0.05

    # Step 2: Severe tail crash violation
    # realized_y (-0.12) < aci_var (-0.05) -> err = 1
    # alpha_t should sharply decrease (become more conservative / protective)
    t_crash = t_start + timedelta(hours=1)
    aci.predict(raw_var=-0.05, bar_time=t_crash)
    res2 = aci.step_feedback(
        target_bar_time=t_crash,
        current_time=t_crash + timedelta(hours=6),
        realized_y=-0.12,
    )
    assert res2["err"] == 1.0
    assert res2["alpha_t"] < res1["alpha_t"]


def test_alpha_clipping_bounds():
    """Verify alpha_t stays strictly within [0.005, 0.25]."""
    aci = DelayedFeedbackACI(nominal_alpha=0.05, gamma=0.05, alpha_min=0.005, alpha_max=0.25, delay_hours=6)

    t0 = pd.Timestamp("2024-01-01 00:00:00+00:00")

    # Repeat 20 violations -> alpha_t should hit alpha_min (0.005)
    for i in range(20):
        t_i = t0 + timedelta(hours=i)
        aci.predict(raw_var=-0.05, bar_time=t_i)
        aci.step_feedback(target_bar_time=t_i, current_time=t_i + timedelta(hours=6), realized_y=-0.20)

    assert aci.alpha_t >= 0.005
    assert pytest.approx(aci.alpha_t, abs=1e-5) == 0.005

    # Repeat 100 calm steps -> alpha_t should hit alpha_max (0.25)
    for i in range(20, 120):
        t_i = t0 + timedelta(hours=i)
        aci.predict(raw_var=-0.05, bar_time=t_i)
        aci.step_feedback(target_bar_time=t_i, current_time=t_i + timedelta(hours=6), realized_y=0.0)

    assert aci.alpha_t <= 0.25
    assert pytest.approx(aci.alpha_t, abs=1e-5) == 0.25


def test_buffer_capacity_limit():
    """Verify rolling residual buffer stays bounded by window_size."""
    window_size = 50
    aci = DelayedFeedbackACI(window_size=window_size, delay_hours=6)

    t0 = pd.Timestamp("2024-01-01 00:00:00+00:00")
    for i in range(120):
        t_i = t0 + timedelta(hours=i)
        aci.predict(raw_var=-0.05, bar_time=t_i)
        aci.step_feedback(target_bar_time=t_i, current_time=t_i + timedelta(hours=6), realized_y=-0.03)

    assert len(aci.residual_buffer) == window_size


def test_multi_asset_isolation():
    """Verify that feedback on one asset does not leak into another asset's buffer or alpha."""
    manager = MultiAssetDelayedFeedbackACI(nominal_alpha=0.05, gamma=0.01, delay_hours=6)

    t0 = pd.Timestamp("2024-01-01 00:00:00+00:00")
    manager.predict("SOLUSDT", raw_var=-0.05, bar_time=t0)
    manager.predict("AVAXUSDT", raw_var=-0.05, bar_time=t0)

    # Deliver crash on SOLUSDT only
    manager.step_feedback("SOLUSDT", t0, t0 + timedelta(hours=6), realized_y=-0.15)

    sol_engine = manager.get_engine("SOLUSDT")
    avax_engine = manager.get_engine("AVAXUSDT")

    assert sol_engine.alpha_t < 0.05  # Tightened
    assert avax_engine.alpha_t == 0.05  # Untouched
    assert len(sol_engine.residual_buffer) == 1
    assert len(avax_engine.residual_buffer) == 0


def test_kupiec_pof_test():
    """Verify Kupiec likelihood ratio test on controlled distributions."""
    # Scenario 1: Exactly 5% violations out of 1000 bars
    y = np.zeros(1000)
    v = np.ones(1000)
    # Put 50 violations (y < v)
    y[:50] = -1.0
    v[:50] = 0.0
    y[50:] = 1.0
    v[50:] = 0.0

    p_val, p_hat, x = kupiec_pof_test(y, v, alpha=0.05)
    assert p_hat == 0.05
    assert x == 50
    assert p_val > 0.95  # Perfect match, high p-value

    # Scenario 2: Severe violation rate (15% violations)
    y[:150] = -1.0
    v[:150] = 0.0
    p_val_bad, p_hat_bad, x_bad = kupiec_pof_test(y, v, alpha=0.05)
    assert p_hat_bad == 0.15
    assert p_val_bad < 0.001  # Strongly rejected
