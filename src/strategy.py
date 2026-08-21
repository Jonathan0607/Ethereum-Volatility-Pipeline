"""
Structural Market-Microstructure Strategy Module
Combines:
- Perpetual Funding Rate 72-Hour Rolling Z-Score (Z_funding)
- Discrete 2-State Gaussian HMM Volatility Regime Filter
- Discrete Squeeze / Liquidation Dump Execution State Machine
"""

import os
import json
import ast
import pandas as pd
import numpy as np

from strategy_microstructure import (
    MicrostructureRegimeFilter,
    execute_microstructure_state_machine,
    FEE_PCT
)
from data import compute_rolling_funding_zscore


def load_best_params():
    """
    Loads optimized hyperparameters from best_params.txt with robust fallbacks.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    defaults = {
        'entry_threshold': 1.80,
        'exit_threshold': 0.20,
        'holding_period_max': 12
    }
    if not os.path.exists(params_path):
        return defaults

    with open(params_path, 'r') as f:
        content = f.read().strip()
        try:
            params = json.loads(content)
        except Exception:
            try:
                params = ast.literal_eval(content)
            except Exception:
                params = defaults

    if isinstance(params, dict):
        defaults.update(params)
    return defaults


def compute_microstructure_signal(
    df_window, 
    hmm_model, 
    entry_threshold=1.80,
    exit_threshold=0.20,
    holding_period_max=12,
    current_pos=0.0,
    bars_held=0
):
    """
    Calculates the live/recent microstructure funding squeeze signal.
    """
    c_eth = df_window['eth_close'].values if 'eth_close' in df_window.columns else df_window['close'].values
    log_ret = np.diff(np.log(c_eth), prepend=0.0)

    # 1. HMM Volatility Regime
    regime = hmm_model.predict_states(log_ret)[-1]

    # 2. Funding Rate & 72h Z-score
    if 'z_funding' in df_window.columns:
        z_t = float(df_window['z_funding'].iloc[-1])
        fr_t = float(df_window['funding_rate'].iloc[-1])
    elif 'funding_rate' in df_window.columns:
        fr = df_window['funding_rate'].values
        z_fr = compute_rolling_funding_zscore(fr, window=72)
        z_t = float(z_fr[-1])
        fr_t = float(fr[-1])
    else:
        z_t = 0.0
        fr_t = 0.0

    # 3. Discrete State Machine
    new_pos = current_pos
    new_bars_held = bars_held

    if current_pos == 0.0:
        if regime == 0:
            if z_t < -entry_threshold:
                new_pos = 1.0
                new_bars_held = 0
            elif z_t > entry_threshold:
                new_pos = -1.0
                new_bars_held = 0
    else:
        new_bars_held += 1
        if regime == 1 or abs(z_t) <= exit_threshold or new_bars_held >= holding_period_max:
            new_pos = 0.0
            new_bars_held = 0

    return {
        'funding_rate': float(fr_t),
        'z_funding': float(z_t),
        'regime': int(regime),
        'active_regime': 'High-Vol' if regime == 1 else 'Low-Vol',
        'target_position': float(new_pos),
        'eth_position': float(new_pos),
        'btc_position': 0.0,
        'bars_held': int(new_bars_held)
    }


def compute_unified_signal(df_window, predictor, msgarch_engine, **kwargs):
    """
    Backward-compatibility alias.
    """
    hmm = MicrostructureRegimeFilter()
    c_eth = df_window['eth_close'].values if 'eth_close' in df_window.columns else df_window['close'].values
    hmm.fit(np.diff(np.log(c_eth), prepend=0.0))
    return compute_microstructure_signal(df_window, hmm, **kwargs)
