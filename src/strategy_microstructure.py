"""
Structural Market-Microstructure Strategy Module
Implements Perpetual Funding Rate Mean-Reversion Squeeze Strategy
gated by a Discrete 2-State Gaussian HMM / GARCH Volatility Regime Filter:
- Regime 0: Low/Normal Volatility (Active Trading)
- Regime 1: High Volatility / Liquidation Cascade (Risk-Off / Flat)

Signals:
- LONG ETH:  Regime == 0 AND Z_funding < -entry_threshold (Shorts trapped, squeeze imminent)
- SHORT ETH: Regime == 0 AND Z_funding > +entry_threshold (Longs over-leveraged, liquidation dump imminent)
- FLAT:      Regime == 1 OR |Z_funding| <= exit_threshold OR bars_held >= holding_period_max
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.stats import norm, skew, kurtosis
import warnings

warnings.filterwarnings("ignore")

FEE_PCT = 0.0020  # 20 bps transaction fee per leg


class MicrostructureRegimeFilter:
    """
    2-State Gaussian Hidden Markov Model (HMM) on hourly log returns.
    Enforces state ordering:
    - State 0 = Low/Normal Volatility (Active Trading)
    - State 1 = High Volatility / Liquidation Cascade (Risk-Off)
    """
    def __init__(self, n_iter=100, random_state=42):
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.state_map = {0: 0, 1: 1}  # Maps raw HMM state to ordered state

    def fit(self, returns):
        arr = np.asarray(returns, dtype=np.float64).flatten()
        valid = arr[~np.isnan(arr)]
        if len(valid) < 50:
            raise ValueError(f"Insufficient returns data to fit HMM ({len(valid)} rows)")

        X = valid.reshape(-1, 1)
        self.model = GaussianHMM(
            n_components=2, 
            covariance_type='full', 
            n_iter=self.n_iter, 
            random_state=self.random_state
        )
        self.model.fit(X)

        # Enforce State 0 is Low Vol, State 1 is High Vol
        covars = self.model.covars_.flatten()
        vols = np.sqrt(np.maximum(covars, 1e-12))
        
        if vols[0] > vols[1]:
            # Raw state 0 is higher vol -> map raw 0 -> 1, raw 1 -> 0
            self.state_map = {0: 1, 1: 0}
        else:
            self.state_map = {0: 0, 1: 1}

        return self

    def predict_states(self, returns):
        if self.model is None:
            raise RuntimeError("HMM must be fitted before predict_states")
        arr = np.asarray(returns, dtype=np.float64).flatten()
        valid_mask = ~np.isnan(arr)
        
        raw_states = np.zeros(len(arr), dtype=int)
        if np.sum(valid_mask) > 0:
            pred_raw = self.model.predict(arr[valid_mask].reshape(-1, 1))
            ordered_pred = np.array([self.state_map[s] for s in pred_raw], dtype=int)
            raw_states[valid_mask] = ordered_pred

        return raw_states

    def predict_state_probs(self, returns):
        if self.model is None:
            raise RuntimeError("HMM must be fitted before predict_state_probs")
        arr = np.asarray(returns, dtype=np.float64).flatten()
        valid_mask = ~np.isnan(arr)
        
        probs = np.zeros((len(arr), 2), dtype=np.float64)
        if np.sum(valid_mask) > 0:
            raw_post = self.model.predict_proba(arr[valid_mask].reshape(-1, 1))
            if self.state_map[0] == 1:
                # Swapped
                probs[valid_mask, 0] = raw_post[:, 1]
                probs[valid_mask, 1] = raw_post[:, 0]
            else:
                probs[valid_mask] = raw_post
        return probs


def execute_microstructure_state_machine(
    z_funding, 
    regimes, 
    entry_threshold=1.8, 
    exit_threshold=0.2, 
    holding_period_max=12,
    initial_pos=0.0,
    initial_bars_held=0
):
    """
    Executes the discrete funding squeeze state machine.
    
    Signals:
    - LONG ETH (+1.0):  Regime == 0 AND Z_funding < -entry_threshold
    - SHORT ETH (-1.0): Regime == 0 AND Z_funding > +entry_threshold
    - FLAT (0.0):       Regime == 1 OR |Z_funding| <= exit_threshold OR bars_held >= holding_period_max
    
    Returns:
        positions (np.ndarray): Target position for ETH in {-1.0, 0.0, +1.0}
        bars_held_series (np.ndarray): Duration in active trade
    """
    n = len(z_funding)
    positions = np.zeros(n, dtype=np.float64)
    bars_held_arr = np.zeros(n, dtype=int)

    curr_pos = float(initial_pos)
    bars_held = int(initial_bars_held)

    for t in range(n):
        z_t = float(z_funding[t]) if not np.isnan(z_funding[t]) else 0.0
        reg_t = int(regimes[t]) if not np.isnan(regimes[t]) else 1

        if curr_pos == 0.0:
            if reg_t == 0:
                if z_t < -entry_threshold:
                    curr_pos = 1.0
                    bars_held = 0
                elif z_t > entry_threshold:
                    curr_pos = -1.0
                    bars_held = 0
        else:
            bars_held += 1
            # Exit conditions
            if reg_t == 1 or abs(z_t) <= exit_threshold or bars_held >= holding_period_max:
                curr_pos = 0.0
                bars_held = 0

        positions[t] = curr_pos
        bars_held_arr[t] = bars_held

    return positions, bars_held_arr


def simulate_microstructure_in_sample(
    z_funding, 
    regimes, 
    r_eth_1h, 
    entry_threshold=1.8, 
    exit_threshold=0.2, 
    holding_period_max=12,
    fee_pct=FEE_PCT
):
    """
    Fast in-sample simulation of the Microstructure Funding Rate Mean-Reversion strategy.
    """
    n = len(r_eth_1h)
    if n < 50:
        return None

    positions, _ = execute_microstructure_state_machine(
        z_funding=z_funding,
        regimes=regimes,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        holding_period_max=holding_period_max
    )

    # Next-bar execution
    pos_exec = np.roll(positions, 1)
    pos_exec[0] = 0.0

    gross_ret = pos_exec * r_eth_1h
    turnover = np.abs(np.diff(pos_exec, prepend=0.0))
    trans_costs = turnover * fee_pct
    net_ret = gross_ret - trans_costs

    return net_ret, pos_exec


def evaluate_microstructure_metrics(net_returns, positions, step_num=0, n_trials=300):
    """
    Computes annualized Sharpe, Sortino, Max Drawdown, Total Return, and Deflated Sharpe Ratio (DSR).
    """
    if net_returns is None or len(net_returns) < 50 or np.std(net_returns) < 1e-10:
        return {'sharpe': -999.0, 'sortino': -999.0, 'max_drawdown': -999.0, 'strategy_return': -999.0, 'dsr': 0.0}

    total_trades = np.sum(np.abs(np.diff(positions)) > 1e-4)
    if total_trades == 0:
        return {'sharpe': -999.0, 'sortino': -999.0, 'max_drawdown': -999.0, 'strategy_return': -999.0, 'dsr': 0.0}

    mean_ret = np.mean(net_returns) * 24 * 365
    std_ret = np.std(net_returns) * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)

    negative_returns = np.minimum(net_returns, 0.0)
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ret / (downside_dev + 1e-9)

    cum_ret = np.cumprod(1.0 + net_returns)
    strat_return = cum_ret[-1] - 1.0

    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = np.min(drawdown)

    sr_observed = np.mean(net_returns) / (np.std(net_returns) + 1e-9)
    n = len(net_returns)
    skewness = skew(net_returns)
    excess_kurt = kurtosis(net_returns, fisher=True)

    std_sr_unann = np.sqrt(max(1e-9, (1.0 + 0.5 * sr_observed**2 - skewness * sr_observed + (excess_kurt / 4.0) * sr_observed**2) / (n - 1.0)))
    std_sr_ann = std_sr_unann * np.sqrt(24 * 365)

    n_trials_total = max(1, step_num) * n_trials
    var_sharpes = 0.5
    gamma_const = 0.57721566490153286
    expected_max_ann = 0.0 + np.sqrt(var_sharpes) * (
        (1.0 - gamma_const) * norm.ppf(1.0 - 1.0 / n_trials_total) +
        gamma_const * norm.ppf(1.0 - 1.0 / (n_trials_total * np.e))
    )
    dsr = norm.cdf((sharpe - expected_max_ann) / (std_sr_ann + 1e-9))

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'strategy_return': float(strat_return),
        'dsr': float(dsr)
    }
