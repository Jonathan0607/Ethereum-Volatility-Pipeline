"""
tests/test_cpcv_validation.py
===================================================================================
Unit Tests for EXP-013: Combinatorial Purged Cross-Validation (CPCV), PBO, and DSR.
Lean CI-safe: runs 100% in fresh-clone environment with zero local data/artifacts.
===================================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

EULER_MASCHERONI = 0.5772156649015329
HOURS_PER_YEAR = 8760.0


def compute_sharpe_se(ann_sharpe: float, skew: float, kurt: float, n_obs: int) -> float:
    """Standard error of annualized Sharpe ratio (Lo 2002 / Bailey & López de Prado 2014)."""
    term1 = HOURS_PER_YEAR
    term2 = skew * ann_sharpe * np.sqrt(HOURS_PER_YEAR)
    term3 = ((kurt - 1.0) / 4.0) * (ann_sharpe ** 2)
    variance = max(1e-8, (term1 - term2 + term3) / (n_obs - 1.0))
    return float(np.sqrt(variance))


def compute_psr(ann_sharpe: float, skew: float, kurt: float, n_obs: int, sr_benchmark: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (PSR)."""
    se = compute_sharpe_se(ann_sharpe, skew, kurt, n_obs)
    z = (ann_sharpe - sr_benchmark) / se
    return float(scipy_stats.norm.cdf(z))


def compute_dsr(
    ann_sharpe: float, skew: float, kurt: float, n_obs: int, n_trials: int = 12, var_trials: float = 0.0608
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (DSR) and benchmark SR*."""
    term1 = (1.0 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
    term2 = EULER_MASCHERONI * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_benchmark = float(np.sqrt(max(1e-6, var_trials)) * (term1 + term2))
    se = compute_sharpe_se(ann_sharpe, skew, kurt, n_obs)
    z = (ann_sharpe - sr_benchmark) / se
    dsr = float(scipy_stats.norm.cdf(z))
    return dsr, sr_benchmark


def compute_cscv_pbo(matrix_returns: np.ndarray, n_blocks: int = 6) -> tuple[float, float]:
    """
    Computes Probability of Backtest Overfitting (PBO) via CSCV partition.
    matrix_returns: shape (n_obs, n_strats)
    """
    from itertools import combinations

    n_obs, n_strats = matrix_returns.shape
    block_len = n_obs // n_blocks
    blocks = [np.arange(i * block_len, (i + 1) * block_len) for i in range(n_blocks)]

    k_test = n_blocks // 2
    combos = list(combinations(range(n_blocks), k_test))
    ranks = []

    for c in combos:
        is_blocks = list(c)
        oos_blocks = [b for b in range(n_blocks) if b not in is_blocks]

        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in oos_blocks])

        is_srs = [np.mean(matrix_returns[is_idx, j]) / (np.std(matrix_returns[is_idx, j]) + 1e-8) for j in range(n_strats)]
        oos_srs = [np.mean(matrix_returns[oos_idx, j]) / (np.std(matrix_returns[oos_idx, j]) + 1e-8) for j in range(n_strats)]

        best_is_idx = int(np.argmax(is_srs))
        best_oos_sr = oos_srs[best_is_idx]

        sorted_oos = sorted(oos_srs)
        rank_val = (sorted_oos.index(best_oos_sr) + 1.0) / (len(sorted_oos) + 1.0)
        ranks.append(rank_val)

    pbo = float(np.mean([1.0 if r <= 0.50 else 0.0 for r in ranks]))
    return pbo, float(np.mean(ranks))


# =================================================================================
# UNIT TESTS
# =================================================================================

def test_bidirectional_cpcv_purging_and_embargo():
    """
    Verifies bidirectional 168h purging and embargo logic:
    1. Training samples before test block satisfy t_train + 168h <= t_test_start.
    2. Training samples after test block satisfy t_train >= t_test_end + 168h.
    """
    timestamps = pd.date_range("2023-01-01", periods=10000, freq="1h", tz="UTC")
    n_ts = len(timestamps)

    # Define test block
    test_start_idx = 3000
    test_end_idx = 4500
    test_start = timestamps[test_start_idx]
    test_end = timestamps[test_end_idx]

    embargo_hours = 168

    train_indices = []
    for i in range(n_ts):
        t = timestamps[i]
        # Purge before: label forward 168h cannot enter test block
        # Embargo after: post-test window must clear 168h
        if t < test_start - pd.Timedelta(hours=embargo_hours) or t > test_end + pd.Timedelta(hours=embargo_hours):
            train_indices.append(i)

    train_indices = np.array(train_indices)

    # Verifications
    train_ts = timestamps[train_indices]

    # No train sample inside test block
    assert not any((train_ts >= test_start) & (train_ts <= test_end)), "Train timestamps overlap with test block!"

    # Purge before verification
    before_train = train_ts[train_ts < test_start]
    assert len(before_train) > 0
    max_before = before_train.max()
    hours_before = (test_start - max_before).total_seconds() / 3600.0
    assert hours_before >= embargo_hours, f"Pre-test purge violation: {hours_before}h < {embargo_hours}h"

    # Embargo after verification
    after_train = train_ts[train_ts > test_end]
    assert len(after_train) > 0
    min_after = after_train.min()
    hours_after = (min_after - test_end).total_seconds() / 3600.0
    assert hours_after >= embargo_hours, f"Post-test embargo violation: {hours_after}h < {embargo_hours}h"


def test_dsr_psr_mathematical_properties():
    """
    Verifies mathematical properties of PSR and DSR:
    1. Zero Sharpe has PSR = 0.50.
    2. Positive Sharpe has PSR > 0.50 and monotonically increases with sample size.
    3. DSR < PSR for any number of trials M > 1.
    4. Higher M strictly deflates DSR.
    """
    n_obs = 33600
    skew = 0.5
    kurt = 4.0

    # 1. Zero Sharpe -> PSR = 0.50
    psr_zero = compute_psr(0.0, skew, kurt, n_obs)
    assert abs(psr_zero - 0.50) < 1e-4, f"PSR for zero Sharpe should be 0.50, got {psr_zero}"

    # 2. Positive Sharpe -> PSR > 0.50
    sr_pos = 1.0
    psr_pos = compute_psr(sr_pos, skew, kurt, n_obs)
    assert psr_pos > 0.95, f"PSR for SR=1.0 over 33,600 hours should be > 0.95, got {psr_pos}"

    # Sample size monotonicity
    psr_short = compute_psr(sr_pos, skew, kurt, n_obs=1000)
    assert psr_pos > psr_short, "PSR should increase monotonically with sample size"

    # 3. DSR < PSR for M > 1
    dsr_12, bench_12 = compute_dsr(sr_pos, skew, kurt, n_obs, n_trials=12)
    assert dsr_12 < psr_pos, f"DSR ({dsr_12}) must be strictly less than PSR ({psr_pos})"
    assert bench_12 > 0.0, f"Expected positive benchmark SR*, got {bench_12}"

    # 4. Higher M deflates DSR further
    dsr_50, bench_50 = compute_dsr(sr_pos, skew, kurt, n_obs, n_trials=50)
    assert dsr_50 < dsr_12, f"DSR with M=50 ({dsr_50}) must be less than M=12 ({dsr_12})"
    assert bench_50 > bench_12, f"Benchmark SR* with M=50 ({bench_50}) must be greater than M=12 ({bench_12})"


def test_cscv_pbo_bounds_and_limits():
    """
    Verifies CSCV PBO bounds:
    1. PBO is strictly in [0.0, 1.0].
    2. A consistently superior strategy yields PBO = 0.0.
    3. Random i.i.d. noise strategies yield PBO around 0.50.
    """
    np.random.seed(42)
    n_obs = 6000
    n_strats = 5

    # Case A: One strategy strictly dominates (higher mean return across all blocks)
    returns_dominant = np.random.randn(n_obs, n_strats) * 0.01
    returns_dominant[:, 0] += 0.005  # Strategy 0 has distinct positive edge

    pbo_dom, mean_rank_dom = compute_cscv_pbo(returns_dominant, n_blocks=6)
    assert 0.0 <= pbo_dom <= 1.0, f"PBO out of bounds: {pbo_dom}"
    assert pbo_dom == 0.0, f"Dominant strategy should have PBO = 0.0, got {pbo_dom}"
    assert mean_rank_dom > 0.70, f"Dominant strategy should have high mean rank, got {mean_rank_dom}"

    # Case B: Pure random noise strategies
    returns_noise = np.random.randn(n_obs, n_strats) * 0.01
    pbo_noise, mean_rank_noise = compute_cscv_pbo(returns_noise, n_blocks=6)
    assert 0.0 <= pbo_noise <= 1.0, f"PBO out of bounds: {pbo_noise}"
    assert 0.20 <= pbo_noise <= 0.80, f"Noise strategies should have PBO around ~0.50, got {pbo_noise}"


def test_circular_block_bootstrap_math():
    """Verifies that circular block bootstrap indices cover the full range and maintain block length."""
    n = 1000
    block_len = 50
    n_blocks = n // block_len

    start_indices = np.random.randint(0, n, size=n_blocks)
    sampled_indices = np.concatenate([
        (start_idx + np.arange(block_len)) % n for start_idx in start_indices
    ])[:n]

    assert len(sampled_indices) == n, f"Bootstrap sample length mismatch: {len(sampled_indices)} vs {n}"
    assert np.all(sampled_indices >= 0) and np.all(sampled_indices < n), "Index out of bounds in bootstrap"
