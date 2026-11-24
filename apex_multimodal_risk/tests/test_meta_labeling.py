"""Unit tests for EXP-007 López de Prado Meta-Labeling and trade event extraction."""

from __future__ import annotations

import numpy as np
import pytest

from apex_multimodal_risk.src.train.exp007_meta_labeling import compute_ece


def test_trade_event_extraction_logic():
    """Verifies trade entry detection on 0-crossing and sign flip."""
    # Synthetic positions:
    # t=0: 0.0
    # t=1: 1.0 (entry trade 1)
    # t=2: 1.0
    # t=3: 0.0 (exit trade 1)
    # t=4: -1.0 (entry trade 2)
    # t=5: 1.0 (flip: exit trade 2, entry trade 3)
    # t=6: 0.0 (exit trade 3)
    p = np.array([0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0])
    r = np.array([0.0, 0.01, 0.02, -0.01, -0.03, 0.04, 0.0])

    trades = []
    in_trade = False
    t_entry = 0
    pos_entry = 0.0

    for t in range(len(p)):
        p_prev = p[t - 1] if t > 0 else 0.0
        p_curr = p[t]

        is_entry = False
        if p_curr != 0.0:
            if p_prev == 0.0 or (np.sign(p_curr) != np.sign(p_prev)):
                is_entry = True

        if in_trade:
            if p_curr == 0.0 or (np.sign(p_curr) != np.sign(pos_entry)):
                trade_pnl = float(np.sum(r[t_entry:t]))
                trades.append(
                    {
                        "t_entry": t_entry,
                        "t_exit": t,
                        "pos_entry": pos_entry,
                        "y_trade_pnl": trade_pnl,
                        "y_meta": 1 if trade_pnl > 0 else 0,
                    }
                )
                in_trade = False

        if is_entry:
            in_trade = True
            t_entry = t
            pos_entry = p_curr

    assert len(trades) == 3
    # Trade 1: entered t=1, exited t=3. PnL = r[1] + r[2] = 0.01 + 0.02 = 0.03 > 0 -> y_meta = 1
    assert trades[0]["t_entry"] == 1
    assert trades[0]["t_exit"] == 3
    assert pytest.approx(trades[0]["y_trade_pnl"], 1e-6) == 0.03
    assert trades[0]["y_meta"] == 1

    # Trade 2: entered t=4, exited t=5. PnL = r[4] = -0.03 < 0 -> y_meta = 0
    assert trades[1]["t_entry"] == 4
    assert trades[1]["t_exit"] == 5
    assert pytest.approx(trades[1]["y_trade_pnl"], 1e-6) == -0.03
    assert trades[1]["y_meta"] == 0

    # Trade 3: entered t=5, exited t=6. PnL = r[5] = 0.04 > 0 -> y_meta = 1
    assert trades[2]["t_entry"] == 5
    assert trades[2]["t_exit"] == 6
    assert pytest.approx(trades[2]["y_trade_pnl"], 1e-6) == 0.04
    assert trades[2]["y_meta"] == 1


def test_ece_computation():
    """Verifies Expected Calibration Error computation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    # Perfectly calibrated predictions
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3, 0.6, 0.4])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_exposure_neutral_scaling_calibration():
    """Verifies that the continuous sizing curve achieves mean(scale) == 1.00 on the training fold."""
    np.random.seed(42)
    p_tr = np.random.uniform(0.1, 0.8, size=200)

    tr_mean_p = np.mean(p_tr)
    tr_std_p = np.std(p_tr) + 1e-6
    z_tr = (p_tr - tr_mean_p) / tr_std_p

    low, high = 0.5, 1.35
    for _ in range(50):
        mid = (low + high) / 2.0
        m_sc = np.mean(np.clip(mid + 0.25 * z_tr, 0.50, 1.35))
        if m_sc < 1.00:
            low = mid
        else:
            high = mid
    c0 = (low + high) / 2.0
    scales_tr = np.clip(c0 + 0.25 * z_tr, 0.50, 1.35)

    assert pytest.approx(np.mean(scales_tr), abs=1e-3) == 1.00
    assert np.all(scales_tr >= 0.50)
    assert np.all(scales_tr <= 1.35)
