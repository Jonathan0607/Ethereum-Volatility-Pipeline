#!/usr/bin/env python3
"""
==============================================================================
WALK-FORWARD MODEL TOURNAMENT
Evaluates 3 Directional Alpha Models against the 2-State MS-GARCH Engine:
1. Model 1 (Microstructure Funding Squeeze): Direction from z_funding
2. Model 2 (Cointegrated StatArb): Beta-neutral spread arbitrage from spread_zscore
3. Model 3 (Ensemble): Equal-weighted blend of Model 1 and Model 2
==============================================================================
- Rolling 180d In-Sample / 7d Out-of-Sample Architecture across 26 weekly folds
- MS-GARCH-X Volatility Scaling with Risk-Off Override (prob_high_vol >= 0.5 -> 0.0)
- 15% Deadband Filter on active rebalancing (bypassed on liquidations to 0.0)
- 20 bps (0.0020) Transaction Fee applied on executed turnover
- Benchmark: 100% Buy & Hold ETH
==============================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data import load_pairs_data
from ms_garch_engine import MSGARCHX
from cointegration import compute_rolling_cointegration
from kelly_execution import execute_kelly_rebalance, compute_volatility_multiplier

FEE_PCT = 0.0020          # 20 bps transaction fee
TARGET_VOLATILITY = 0.20  # Target volatility for Kelly multiplier
DEADBAND = 0.15           # 15% rebalance deadband


def apply_deadband_filter(curr_pos, target_pos, deadband=0.15):
    """
    Applies the deadband filter with Risk-Off 0.0 override:
    If target is exactly 0.0, bypass deadband to liquidate immediately.
    Otherwise, only trade if |target - current| > deadband.
    """
    if target_pos == 0.0:
        return 0.0
    if abs(target_pos - curr_pos) > deadband:
        return target_pos
    return curr_pos


def simulate_tournament_step(slice_df, test_start, trained_msgarch, initial_states):
    """
    Simulates Model 1 (Funding), Model 2 (StatArb), and Model 3 (Ensemble)
    over a 7-day Out-of-Sample window with exact live execution mechanics.
    """
    # 1. Filter MS-GARCH-X across slice
    filtered_garch = trained_msgarch.filter_full_series(slice_df)
    slice_df = slice_df.join(filtered_garch[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
    slice_df['prob_high_vol'] = slice_df['prob_high_vol'].ffill().fillna(1.0)
    slice_df['ms_garch_volatility'] = slice_df['ms_garch_volatility'].ffill().fillna(0.01)

    test_df = slice_df[slice_df.index >= test_start].copy()
    if test_df.empty:
        return test_df, initial_states

    n = len(test_df)

    # State tracking for the 3 models
    m1_curr_eth = float(initial_states.get('m1_eth', 0.0))
    m2_curr_eth = float(initial_states.get('m2_eth', 0.0))
    m2_curr_btc = float(initial_states.get('m2_btc', 0.0))
    m3_curr_eth = float(initial_states.get('m3_eth', 0.0))
    m3_curr_btc = float(initial_states.get('m3_btc', 0.0))

    m1_eth_targets = np.zeros(n, dtype=np.float64)
    m2_eth_targets = np.zeros(n, dtype=np.float64)
    m2_btc_targets = np.zeros(n, dtype=np.float64)
    m3_eth_targets = np.zeros(n, dtype=np.float64)
    m3_btc_targets = np.zeros(n, dtype=np.float64)

    for i in range(n):
        p_high = float(test_df['prob_high_vol'].iloc[i])
        g_vol = float(test_df['ms_garch_volatility'].iloc[i])
        z_fund = float(test_df['z_funding'].iloc[i]) if 'z_funding' in test_df.columns else 0.0
        z_spread = float(test_df['spread_zscore'].iloc[i])
        beta = float(test_df['ols_beta'].iloc[i])

        # Target size scaler: compute_volatility_multiplier * (1.0 - prob_high_vol)
        if p_high >= 0.50:
            vol_scaler = 0.0
        else:
            base_mult = compute_volatility_multiplier(g_vol, target_vol=TARGET_VOLATILITY)
            vol_scaler = float(base_mult * (1.0 - p_high))

        # --- MODEL 1: Microstructure Funding Squeeze ---
        if z_fund < -1.8:
            m1_dir = 1.0
        elif z_fund > 1.8:
            m1_dir = -1.0
        else:
            m1_dir = 0.0

        m1_raw_eth = m1_dir * vol_scaler
        m1_curr_eth = apply_deadband_filter(m1_curr_eth, m1_raw_eth, DEADBAND)
        m1_eth_targets[i] = m1_curr_eth

        # --- MODEL 2: Cointegrated StatArb ---
        if z_spread < -2.0:
            m2_dir = 1.0
        elif z_spread > 2.0:
            m2_dir = -1.0
        else:
            m2_dir = 0.0

        m2_raw_eth = m2_dir * vol_scaler
        m2_raw_btc = -m2_dir * vol_scaler * beta

        m2_curr_eth = apply_deadband_filter(m2_curr_eth, m2_raw_eth, DEADBAND)
        m2_curr_btc = apply_deadband_filter(m2_curr_btc, m2_raw_btc, DEADBAND)
        m2_eth_targets[i] = m2_curr_eth
        m2_btc_targets[i] = m2_curr_btc

        # --- MODEL 3: Ensemble (Equal Blend) ---
        m3_raw_eth = (m1_raw_eth + m2_raw_eth) / 2.0
        m3_raw_btc = (0.0 + m2_raw_btc) / 2.0

        m3_curr_eth = apply_deadband_filter(m3_curr_eth, m3_raw_eth, DEADBAND)
        m3_curr_btc = apply_deadband_filter(m3_curr_btc, m3_raw_btc, DEADBAND)
        m3_eth_targets[i] = m3_curr_eth
        m3_btc_targets[i] = m3_curr_btc

    # Record Target Positions
    test_df['m1_target_eth'] = m1_eth_targets
    test_df['m1_target_btc'] = 0.0

    test_df['m2_target_eth'] = m2_eth_targets
    test_df['m2_target_btc'] = m2_btc_targets

    test_df['m3_target_eth'] = m3_eth_targets
    test_df['m3_target_btc'] = m3_btc_targets

    # Execution positions are shifted by 1 bar
    test_df['m1_pos_eth'] = test_df['m1_target_eth'].shift(1).fillna(initial_states.get('m1_eth', 0.0))
    test_df['m1_pos_btc'] = 0.0

    test_df['m2_pos_eth'] = test_df['m2_target_eth'].shift(1).fillna(initial_states.get('m2_eth', 0.0))
    test_df['m2_pos_btc'] = test_df['m2_target_btc'].shift(1).fillna(initial_states.get('m2_btc', 0.0))

    test_df['m3_pos_eth'] = test_df['m3_target_eth'].shift(1).fillna(initial_states.get('m3_eth', 0.0))
    test_df['m3_pos_btc'] = test_df['m3_target_btc'].shift(1).fillna(initial_states.get('m3_btc', 0.0))

    # Asset Returns
    eth_c = test_df['eth_close'] if 'eth_close' in test_df.columns else test_df['close']
    btc_c = test_df['btc_close']
    test_df['r_eth'] = eth_c.pct_change().fillna(0.0)
    test_df['r_btc'] = btc_c.pct_change().fillna(0.0)

    # Benchmark: Buy & Hold ETH
    test_df['benchmark_returns'] = test_df['r_eth']

    # Friction & Net Returns: Model 1
    m1_turnover = test_df['m1_pos_eth'].diff().abs().fillna(0.0)
    test_df['m1_fees'] = m1_turnover * FEE_PCT
    test_df['m1_net_returns'] = (test_df['m1_pos_eth'] * test_df['r_eth']) - test_df['m1_fees']

    # Friction & Net Returns: Model 2
    m2_turnover = test_df['m2_pos_eth'].diff().abs().fillna(0.0) + test_df['m2_pos_btc'].diff().abs().fillna(0.0)
    test_df['m2_fees'] = m2_turnover * FEE_PCT
    test_df['m2_net_returns'] = (test_df['m2_pos_eth'] * test_df['r_eth']) + (test_df['m2_pos_btc'] * test_df['r_btc']) - test_df['m2_fees']

    # Friction & Net Returns: Model 3
    m3_turnover = test_df['m3_pos_eth'].diff().abs().fillna(0.0) + test_df['m3_pos_btc'].diff().abs().fillna(0.0)
    test_df['m3_fees'] = m3_turnover * FEE_PCT
    test_df['m3_net_returns'] = (test_df['m3_pos_eth'] * test_df['r_eth']) + (test_df['m3_pos_btc'] * test_df['r_btc']) - test_df['m3_fees']

    next_states = {
        'm1_eth': m1_eth_targets[-1] if n > 0 else 0.0,
        'm2_eth': m2_eth_targets[-1] if n > 0 else 0.0,
        'm2_btc': m2_btc_targets[-1] if n > 0 else 0.0,
        'm3_eth': m3_eth_targets[-1] if n > 0 else 0.0,
        'm3_btc': m3_btc_targets[-1] if n > 0 else 0.0,
    }
    return test_df, next_states


def compute_performance_metrics(returns_series):
    """
    Computes Annualized Sharpe, Sortino, Max Drawdown, and Cumulative Return.
    """
    r = returns_series.dropna()
    if len(r) == 0:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0, 'total_return': 0.0}

    cum = (1.0 + r).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0) * 100

    mean_ann = r.mean() * 24 * 365
    std_ann = r.std() * np.sqrt(24 * 365)
    sharpe = mean_ann / (std_ann + 1e-9)

    neg_r = np.minimum(r.values, 0.0)
    downside_dev = np.sqrt(np.mean(neg_r ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ann / (downside_dev + 1e-9)

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min()) * 100

    return {
        'total_return': round(total_ret, 2),
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'max_drawdown': round(max_dd, 2)
    }


def main():
    print("=" * 85)
    print("  WALK-FORWARD MODEL TOURNAMENT: 3 DIRECTIONAL ALPHA ENGINES")
    print("  Engine 1: Funding Squeeze | Engine 2: StatArb | Engine 3: Ensemble")
    print("  Volatility Regimes: 2-State MS-GARCH(1,1)-X with 15% Deadband")
    print("=" * 85)

    # 1. Load Synchronous Dataset
    df = load_pairs_data()
    print(f"Loaded {len(df)} synchronous hourly bars from {df.index.min()} to {df.index.max()}.")

    # 2. Compute Rolling Cointegration features (OLS Beta & Spread Z-Score)
    print("Computing 72h Rolling Cointegration features...")
    coint_df = compute_rolling_cointegration(df, window=72)
    
    # Map column names for tournament
    df['spread_zscore'] = coint_df['coint_zscore'] if 'coint_zscore' in coint_df.columns else coint_df['spread_zscore']
    df['ols_beta'] = coint_df['coint_beta'] if 'coint_beta' in coint_df.columns else coint_df['ols_beta']

    # 3. Setup Walk-Forward Rolling Windows (180d IS / 7d OOS, 26 weeks)
    end_test_date = df.index.max()
    total_oos_weeks = 26
    start_test_date = end_test_date - pd.Timedelta(weeks=total_oos_weeks)

    step_days = 7
    train_days = 180

    current_test_start = start_test_date
    oos_results = []
    
    states = {
        'm1_eth': 0.0,
        'm2_eth': 0.0,
        'm2_btc': 0.0,
        'm3_eth': 0.0,
        'm3_btc': 0.0
    }
    step_num = 0

    while current_test_start + pd.Timedelta(days=step_days) <= end_test_date:
        current_test_end = current_test_start + pd.Timedelta(days=step_days)
        current_train_start = current_test_start - pd.Timedelta(days=train_days)
        current_train_end = current_test_start

        print(f"--- TOURNAMENT STEP {step_num:02d} --- "
              f"Train: {current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')} | "
              f"Test: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")

        train_df = df[(df.index >= current_train_start) & (df.index < current_train_end)].copy()

        # In-Sample: Fit MS-GARCH-X
        msgarch = MSGARCHX()
        msgarch.fit(train_df)

        # Out-of-Sample: 7-Day Execution (with 100h lookback for filter continuity)
        lookback_delta = pd.Timedelta(hours=100)
        slice_df = df[(df.index >= current_test_start - lookback_delta) & (df.index < current_test_end)].copy()

        test_step, states = simulate_tournament_step(
            slice_df=slice_df,
            test_start=current_test_start,
            trained_msgarch=msgarch,
            initial_states=states
        )

        test_step['step'] = step_num
        oos_results.append(test_step)

        current_test_start = current_test_end
        step_num += 1

    # 4. Stitch Out-of-Sample Slices
    final_oos = pd.concat(oos_results)
    final_oos.sort_index(inplace=True)
    final_oos = final_oos[~final_oos.index.duplicated(keep='first')]

    # Cumulative Return Series
    final_oos['cum_benchmark'] = (1.0 + final_oos['benchmark_returns']).cumprod()
    final_oos['cum_m1'] = (1.0 + final_oos['m1_net_returns']).cumprod()
    final_oos['cum_m2'] = (1.0 + final_oos['m2_net_returns']).cumprod()
    final_oos['cum_m3'] = (1.0 + final_oos['m3_net_returns']).cumprod()

    # 5. Calculate Metrics
    bench_m = compute_performance_metrics(final_oos['benchmark_returns'])
    m1_m = compute_performance_metrics(final_oos['m1_net_returns'])
    m2_m = compute_performance_metrics(final_oos['m2_net_returns'])
    m3_m = compute_performance_metrics(final_oos['m3_net_returns'])

    print("\n" + "=" * 85)
    print("                    WALK-FORWARD MODEL TOURNAMENT RESULTS")
    print("=" * 85)
    print(f"{'Strategy / Model':<35} | {'Cum Return':<12} | {'Sharpe':<8} | {'Sortino':<8} | {'Max Drawdown':<12}")
    print("-" * 85)
    print(f"{'Benchmark (Buy & Hold ETH)':<35} | {bench_m['total_return']:>10.2f}% | {bench_m['sharpe']:>8.2f} | {bench_m['sortino']:>8.2f} | {bench_m['max_drawdown']:>10.2f}%")
    print(f"{'Model 1 (Funding Squeeze)':<35} | {m1_m['total_return']:>10.2f}% | {m1_m['sharpe']:>8.2f} | {m1_m['sortino']:>8.2f} | {m1_m['max_drawdown']:>10.2f}%")
    print(f"{'Model 2 (Cointegrated StatArb)':<35} | {m2_m['total_return']:>10.2f}% | {m2_m['sharpe']:>8.2f} | {m2_m['sortino']:>8.2f} | {m2_m['max_drawdown']:>10.2f}%")
    print(f"{'Model 3 (Ensemble Blend)':<35} | {m3_m['total_return']:>10.2f}% | {m3_m['sharpe']:>8.2f} | {m3_m['sortino']:>8.2f} | {m3_m['max_drawdown']:>10.2f}%")
    print("=" * 85)

    # 6. Save JSON Metrics
    summary = {
        'benchmark': bench_m,
        'model_1_funding': m1_m,
        'model_2_statarb': m2_m,
        'model_3_ensemble': m3_m
    }
    json_path = os.path.join(PROJECT_ROOT, 'research', 'tournament_results.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=4)

    # 7. Generate Tournament Performance Plot
    plt.figure(figsize=(14, 8))
    plt.plot(final_oos.index, final_oos['cum_benchmark'], label=f"Benchmark: Buy & Hold ETH ({bench_m['total_return']:+.1f}%, SR={bench_m['sharpe']})", color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.7)
    plt.plot(final_oos.index, final_oos['cum_m1'], label=f"Model 1: Funding Squeeze ({m1_m['total_return']:+.1f}%, SR={m1_m['sharpe']})", color='#2ca02c', linewidth=2.0)
    plt.plot(final_oos.index, final_oos['cum_m2'], label=f"Model 2: StatArb Beta-Neutral ({m2_m['total_return']:+.1f}%, SR={m2_m['sharpe']})", color='#1f77b4', linewidth=2.0)
    plt.plot(final_oos.index, final_oos['cum_m3'], label=f"Model 3: Ensemble Blend ({m3_m['total_return']:+.1f}%, SR={m3_m['sharpe']})", color='#d62728', linewidth=2.4)

    plt.title('Walk-Forward Model Tournament: Directional Alpha Comparison with MS-GARCH Engine', fontsize=13, fontweight='bold')
    plt.ylabel('Cumulative Equity Multiplier')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', framealpha=0.9, fontsize=10)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    png_path = os.path.join(PROJECT_ROOT, 'research', 'tournament_results.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved tournament performance chart to: {png_path}")
    print(f"Saved tournament metrics payload to:    {json_path}")


if __name__ == "__main__":
    main()
