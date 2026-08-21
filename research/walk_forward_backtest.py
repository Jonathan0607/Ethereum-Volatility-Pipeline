#!/usr/bin/env python3
"""
==============================================================================
FAST MOMENTUM (7D / 7D) LONG/CASH TREND FOLLOWING ENGINE (FULL EXPOSURE)
==============================================================================
1. Fast Momentum Donchian Architecture:
   - Entry: 7-Day (168-hour) Donchian Breakout (Upper 168) -> Long (+1.0)
   - Exit: 7-Day (168-hour) Donchian Trailing Exit (Lower 168) -> Cash (0.0)
   - Long / Cash Only (No shorting)
2. Full Exposure Sizing (Zero Volatility De-leveraging):
   - Target ETH = Direction (100% exposure in Long, 0% in Cash)
   - No GARCH de-leveraging during secular bull trends
3. Friction & Execution:
   - 15% Deadband Filter
   - 20 bps (0.0020) Transaction Fee applied on executed turnover
4. 180-Day In-Sample Window | 7-Day Out-of-Sample Window (26 Weekly Folds)
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

from data import load_microstructure_data
from kelly_execution import execute_kelly_rebalance

CHANNEL_WINDOW = 168  # 7 Days = 168 Hours
DEADBAND = 0.15       # 15% Rebalance Deadband
FEE_PCT = 0.0020      # 20 bps Transaction Fee


def run_oos_step(slice_df, test_start, initial_direction=0.0, initial_position=0.0):
    """
    Executes the 7-day Out-of-Sample window:
    - 7-Day (168h) Donchian Upper Entry / Lower Exit
    - Full 100% Exposure when Long, 0% when Cash (No GARCH scaling)
    - 15% Deadband Filter
    - 20 bps Transaction Fee on executed turnover
    """
    eth_c = slice_df['eth_close'] if 'eth_close' in slice_df.columns else slice_df['close']

    # 1. Compute 7-Day (168h) Donchian Channel Bounds (strictly prior to bar t)
    slice_df['upper_168'] = eth_c.shift(1).rolling(CHANNEL_WINDOW, min_periods=CHANNEL_WINDOW).max()
    slice_df['lower_168'] = eth_c.shift(1).rolling(CHANNEL_WINDOW, min_periods=CHANNEL_WINDOW).min()

    # Filter to strictly the Out-of-Sample window
    test_df = slice_df[slice_df.index >= test_start].copy()
    if test_df.empty:
        return test_df, initial_direction, initial_position

    n = len(test_df)
    target_pos_arr = np.zeros(n, dtype=np.float64)
    direction_arr = np.zeros(n, dtype=np.float64)

    curr_dir = float(initial_direction)
    curr_pos = float(initial_position)

    for i in range(n):
        c_t = float(test_df['eth_close'].iloc[i]) if 'eth_close' in test_df.columns else float(test_df['close'].iloc[i])
        u_168 = float(test_df['upper_168'].iloc[i])
        l_168 = float(test_df['lower_168'].iloc[i])

        # Fast 7D / 7D Long-Cash State Machine
        if not np.isnan(u_168) and not np.isnan(l_168):
            if curr_dir == 0.0:  # Currently FLAT / CASH
                if c_t > u_168:
                    curr_dir = 1.0   # 7-Day Long Breakout Entry
            elif curr_dir == 1.0: # Currently LONG
                if c_t < l_168:
                    curr_dir = 0.0   # 7-Day Trailing Exit to Cash

        # Full Exposure Sizing (100% in Long, 0% in Cash)
        raw_target = curr_dir

        # 15% Deadband Filter
        new_pos, _ = execute_kelly_rebalance(curr_pos, raw_target, rebalance_deadband=DEADBAND)
        curr_pos = new_pos

        target_pos_arr[i] = curr_pos
        direction_arr[i] = curr_dir

    test_df['direction'] = direction_arr
    test_df['target_position'] = target_pos_arr

    # Shift execution position by 1 bar
    test_df['eth_position'] = test_df['target_position'].shift(1).fillna(initial_position)

    # Returns & Transaction Costs (20 bps)
    test_df['eth_returns'] = eth_c.loc[test_df.index].pct_change().fillna(0.0)
    test_df['benchmark_returns'] = test_df['eth_returns']
    test_df['gross_returns'] = test_df['eth_position'] * test_df['eth_returns']

    # Turnover occurs only when executed position changes
    test_df['turnover'] = test_df['eth_position'].diff().abs().fillna(0.0)
    test_df['transaction_costs'] = test_df['turnover'] * FEE_PCT
    test_df['strategy_returns'] = test_df['gross_returns'] - test_df['transaction_costs']

    return test_df, curr_dir, curr_pos


def compute_metrics(returns_series):
    r = returns_series.dropna()
    if len(r) == 0:
        return {'total_return': 0.0, 'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0}

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
    print("  FAST MOMENTUM (7D/7D) LONG/CASH TREND FOLLOWING ENGINE (FULL EXPOSURE)")
    print("  Direction: 7D Entry (Upper 168) / 7D Exit (Lower 168) | Long/Cash Only")
    print("  Sizer: Full 100% Exposure (No Vol De-leveraging) | Friction: 15% Deadband, 20 bps Fee")
    print("=" * 85)

    df = load_microstructure_data()
    print(f"Loaded {len(df)} synchronous hourly bars from {df.index.min()} to {df.index.max()}.\n")

    end_test_date = df.index.max()
    total_oos_weeks = 26
    start_test_date = end_test_date - pd.Timedelta(weeks=total_oos_weeks)

    step_days = 7
    train_days = 180

    current_test_start = start_test_date
    oos_results = []
    step_num = 0
    state_direction = 0.0
    state_position = 0.0

    while current_test_start + pd.Timedelta(days=step_days) <= end_test_date:
        current_test_end = current_test_start + pd.Timedelta(days=step_days)
        current_train_start = current_test_start - pd.Timedelta(days=train_days)
        current_train_end = current_test_start

        # Out-of-Sample Step: Slice with 250h lookback for 168h channel
        lookback_delta = pd.Timedelta(hours=250)
        slice_df = df[(df.index >= current_test_start - lookback_delta) & (df.index < current_test_end)].copy()

        test_step, state_direction, state_position = run_oos_step(
            slice_df=slice_df,
            test_start=current_test_start,
            initial_direction=state_direction,
            initial_position=state_position
        )

        test_step['step'] = step_num
        oos_results.append(test_step)

        dir_label = "LONG (+1.0)" if state_direction == 1.0 else "CASH (0.0)"
        print(f"--- WFA STEP {step_num:02d} --- "
              f"Test: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')} | "
              f"Direction: {dir_label} | Position: {state_position:+.2f}")

        current_test_start = current_test_end
        step_num += 1

    # Stitch Out-of-Sample Slices
    final_oos = pd.concat(oos_results)
    final_oos.sort_index(inplace=True)
    final_oos = final_oos[~final_oos.index.duplicated(keep='first')]

    final_oos['cum_benchmark'] = (1.0 + final_oos['benchmark_returns']).cumprod()
    final_oos['cum_strategy'] = (1.0 + final_oos['strategy_returns']).cumprod()
    final_oos['cum_gross'] = (1.0 + final_oos['gross_returns']).cumprod()

    # Performance Metrics
    bench_m = compute_metrics(final_oos['benchmark_returns'])
    strat_m = compute_metrics(final_oos['strategy_returns'])
    gross_m = compute_metrics(final_oos['gross_returns'])

    total_fees_paid = float(final_oos['transaction_costs'].sum()) * 100
    total_trades = int((final_oos['turnover'] > 0).sum())

    print("\n" + "=" * 85)
    print("      FAST MOMENTUM (7D/7D) FULL EXPOSURE: NET OUT-OF-SAMPLE RESULTS (26 WEEKS)")
    print("=" * 85)
    print(f"Benchmark Return (Buy & Hold ETH): {bench_m['total_return']:>10.2f}% | Sharpe: {bench_m['sharpe']:>6.2f} | Max DD: {bench_m['max_drawdown']:>8.2f}%")
    print(f"Gross Strategy Return (0 Fees):    {gross_m['total_return']:>10.2f}% | Sharpe: {gross_m['sharpe']:>6.2f} | Max DD: {gross_m['max_drawdown']:>8.2f}%")
    print(f"Net Strategy Return (20 bps Fees): {strat_m['total_return']:>10.2f}% | Sharpe: {strat_m['sharpe']:>6.2f} | Max DD: {strat_m['max_drawdown']:>8.2f}%")
    print(f"Strategy Sortino Ratio:            {strat_m['sortino']:>10.2f}")
    print(f"Total Cumulative Fees Paid:        {total_fees_paid:>10.2f}% ({total_trades} round-trip adjustments)")
    print("=" * 85)

    # Save JSON Metrics
    results_payload = {
        'benchmark': bench_m,
        'strategy_net': strat_m,
        'strategy_gross': gross_m,
        'model': 'Fast Momentum Long/Cash Trend Following (7D Entry / 7D Exit) with Full 100% Exposure',
        'channel_window_hours': CHANNEL_WINDOW,
        'deadband': DEADBAND,
        'fee_pct': FEE_PCT,
        'total_fees_pct': round(total_fees_paid, 2),
        'total_rebalance_trades': total_trades
    }
    json_path = os.path.join(PROJECT_ROOT, 'research', 'walk_forward_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_payload, f, indent=4)

    # Generate Performance Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1.2]})

    ax1.plot(final_oos.index, final_oos['cum_benchmark'], label=f"Benchmark: Buy & Hold ETH ({bench_m['total_return']:+.1f}%, SR={bench_m['sharpe']})", color='#7f7f7f', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.plot(final_oos.index, final_oos['cum_gross'], label=f"Gross Fast 7D/7D (0 Fees: {gross_m['total_return']:+.1f}%, SR={gross_m['sharpe']})", color='#17becf', linestyle=':', linewidth=1.8)
    ax1.plot(final_oos.index, final_oos['cum_strategy'], label=f"Net Fast 7D/7D (20 bps Fees: {strat_m['total_return']:+.1f}%, SR={strat_m['sharpe']})", color='#2ca02c', linewidth=2.4)

    ax1.set_title('Walk-Forward Analysis: Fast Momentum (7D Entry / 7D Exit) Long/Cash Engine', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Return Multiplier')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10)

    # Subplot 2: Full Exposure Position
    ax2.plot(final_oos.index, final_oos['eth_position'], color='#ff7f0e', linewidth=1.5, label='Executed Long Position (0.0 or 1.0)')
    ax2.axhline(0.0, color='black', linestyle=':', alpha=0.5)
    ax2.set_ylabel('Position Exposure')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_title('Full 100% Long / Cash Position Exposure', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    png_path = os.path.join(PROJECT_ROOT, 'research', 'walk_forward_results.png')
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(PROJECT_ROOT, 'walk_forward_results.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved performance chart to: {png_path}")
    print(f"Saved metrics payload to:    {json_path}")


if __name__ == "__main__":
    main()
