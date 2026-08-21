"""
Discrete Execution & Volatility-Scaled Sizing Module
Supports:
1. Discrete Microstructure Funding Squeeze Allocations (-1.0, 0.0, +1.0)
2. Cointegrated Statistical Arbitrage Dynamic Beta-Neutral Allocations
3. Volatility-Scaled Continuous Position Sizing
"""

import numpy as np
import pandas as pd


def compute_volatility_multiplier(sigma_spread, target_vol=0.20, min_mult=0.2, max_mult=1.0):
    """
    Calculates the volatility scaling multiplier:
        gross_mult = clip(target_vol / (sigma_spread * sqrt(24 * 365)), min_mult, max_mult)
    """
    ann_factor = np.sqrt(24 * 365)
    if isinstance(sigma_spread, (pd.Series, np.ndarray)):
        ann_vol = np.asarray(sigma_spread, dtype=np.float64) * ann_factor
        mult = target_vol / (ann_vol + 1e-8)
        return np.clip(mult, min_mult, max_mult)
    else:
        ann_vol = float(sigma_spread) * ann_factor
        mult = target_vol / (ann_vol + 1e-8)
        return float(np.clip(mult, min_mult, max_mult))


def execute_discrete_allocation(current_pos, target_pos, rebalance_deadband=0.15):
    """
    Executes discrete allocation with deadband filter:
    Only trades if |target_pos - current_pos| > rebalance_deadband.
    """
    if np.isnan(target_pos):
        return current_pos, False
    if abs(target_pos - current_pos) > rebalance_deadband:
        return target_pos, True
    return current_pos, False


def execute_discrete_state_machine(
    z_scores, 
    mu_residuals, 
    betas, 
    sigmas_spread, 
    entry_threshold=1.75, 
    stop_loss_threshold=3.5, 
    target_vol=0.20,
    initial_direction=0.0,
    initial_beta=1.0
):
    """
    Executes discrete Cointegrated StatArb state machine.
    """
    n = len(z_scores)
    pos_eth = np.zeros(n, dtype=np.float64)
    pos_btc = np.zeros(n, dtype=np.float64)
    directions = np.zeros(n, dtype=np.float64)
    gross_mults = np.zeros(n, dtype=np.float64)

    curr_dir = float(initial_direction)
    locked_beta = float(initial_beta)

    for t in range(n):
        z_t = float(z_scores[t]) if not np.isnan(z_scores[t]) else 0.0
        mu_t = float(mu_residuals[t]) if not np.isnan(mu_residuals[t]) else 0.0
        beta_t = float(betas[t]) if not np.isnan(betas[t]) else locked_beta
        sigma_t = float(sigmas_spread[t]) if not np.isnan(sigmas_spread[t]) else 0.01

        if curr_dir == 0.0:
            if z_t < -entry_threshold and mu_t > 0:
                curr_dir = 1.0
                locked_beta = beta_t
            elif z_t > entry_threshold and mu_t < 0:
                curr_dir = -1.0
                locked_beta = beta_t
        elif curr_dir == 1.0:
            if z_t >= 0.0 or abs(z_t) > stop_loss_threshold:
                curr_dir = 0.0
        elif curr_dir == -1.0:
            if z_t <= 0.0 or abs(z_t) > stop_loss_threshold:
                curr_dir = 0.0

        g_mult = compute_volatility_multiplier(sigma_t, target_vol=target_vol)

        if curr_dir != 0.0:
            p_eth = g_mult * curr_dir
            p_btc = -g_mult * curr_dir * locked_beta
        else:
            p_eth = 0.0
            p_btc = 0.0

        pos_eth[t] = p_eth
        pos_btc[t] = p_btc
        directions[t] = curr_dir
        gross_mults[t] = g_mult

    return pos_eth, pos_btc, directions, gross_mults


def calculate_target_position(
    mu, 
    sigma=None, 
    target_leverage=0.25, 
    max_leverage=1.0, 
    min_volatility=1e-8, 
    noise_threshold=1e-4,
    kelly_fraction=None,
    sigma_sq=None
):
    if kelly_fraction is not None:
        target_leverage = kelly_fraction
    if sigma is None and sigma_sq is not None:
        sigma = np.sqrt(np.maximum(sigma_sq, 1e-12))
    elif sigma is None:
        sigma = 1e-4

    if isinstance(mu, (pd.Series, np.ndarray)) or isinstance(sigma, (pd.Series, np.ndarray)):
        mu_arr = np.asarray(mu, dtype=np.float64)
        vol_arr = np.asarray(sigma, dtype=np.float64)
        vol_arr = np.maximum(vol_arr, min_volatility) + 1e-8
        raw = target_leverage * (mu_arr / vol_arr)
        if noise_threshold > 0:
            raw = np.where(np.abs(mu_arr) < noise_threshold, 0.0, raw)
        return np.clip(raw, -max_leverage, max_leverage)
    else:
        s_vol = max(float(sigma), min_volatility) + 1e-8
        mu_val = float(mu)
        if abs(mu_val) < noise_threshold:
            return 0.0
        return float(np.clip(target_leverage * (mu_val / s_vol), -max_leverage, max_leverage))


def execute_kelly_rebalance(current_position, target_position, rebalance_deadband=0.15, rebalance_threshold=None):
    deadband = rebalance_deadband if rebalance_threshold is None else rebalance_threshold
    if np.isnan(target_position):
        return current_position, False
    if abs(target_position - current_position) > deadband:
        return target_position, True
    return current_position, False


def calculate_delta_neutral_positions(f_star, beta=1.0):
    f_val = float(f_star)
    return {
        'eth_target': f_val,
        'btc_target': -f_val * float(beta),
        'net_exposure': 0.0,
        'gross_leverage': abs(f_val) + abs(f_val * float(beta))
    }
