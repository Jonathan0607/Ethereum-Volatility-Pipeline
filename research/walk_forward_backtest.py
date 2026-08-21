#!/usr/bin/env python3
"""
==============================================================================
PURE 2-STATE MS-GARCH-X WALK-FORWARD ANALYSIS (WFA) BACKTESTER
Volatility-Scaled Kelly Execution Engine with 15% Deadband & Risk-Off Override
==============================================================================
- 180-Day In-Sample Window | 7-Day Out-of-Sample Window
- In-Sample: Fits 2-State MS-GARCH(1,1)-X on historical OHLCV
- Out-of-Sample: Evaluates pure structural volatility edge without hyperparameter tuning
- Live API Replication: Target sizing scaled by (1 - prob_high_vol) * (TARGET_VOL / ann_vol)
- Risk-Off Override: Prob_high_vol >= 0.50 immediately liquidates to 0.0 (CASH)
- Friction: 20 bps transaction fee charged only on executed size deltas
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
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data import load_microstructure_data
from ms_garch_engine import MSGARCHX

FEE_PCT = 0.0020          # 20 bps transaction cost
TARGET_VOLATILITY = 0.06  # Target volatility for Kelly scaling
DEADBAND = 0.15           # 15% rebalance deadband filter


def run_oos_step(slice_df, test_start, trained_msgarch, initial_pos=0.0):
    """
    Executes the 7-day Out-of-Sample window with exact live API state machine replication.
    """
    # 1. Filter MS-GARCH-X across the slice
    filtered_garch = trained_msgarch.filter_full_series(slice_df)
    slice_df = slice_df.join(filtered_garch[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
    slice_df['prob_high_vol'] = slice_df['prob_high_vol'].ffill().fillna(1.0)
    slice_df['ms_garch_volatility'] = slice_df['ms_garch_volatility'].ffill().fillna(0.01)

    test_df = slice_df[slice_df.index >= test_start].copy()
    if test_df.empty:
        return test_df, initial_pos

    ann_factor = np.sqrt(24 * 365)
    target_pos_list = []
    actions_list = []
    curr_pos = float(initial_pos)

    for idx in range(len(test_df)):
        p_high = float(test_df['prob_high_vol'].iloc[idx])
        g_vol = float(test_df['ms_garch_volatility'].iloc[idx])

        # Calculate volatility multiplier
        ann_vol = g_vol * ann_factor
        vol_scaler = TARGET_VOLATILITY / (ann_vol + 1e-8)

        # Risk-Off Deadband Override
        if p_high >= 0.50:
            # Bypass deadband and immediately liquidate
            ideal_target = 0.0
            if curr_pos != 0.0:
                action = "CASH"
            else:
                action = "FLAT"
            curr_pos = 0.0
        else:
            # Low-vol regime: scale target by (1.0 - prob_high_vol) * vol_scaler
            raw_target = (1.0 - p_high) * vol_scaler
            ideal_target = float(np.clip(raw_target, 0.0, 1.0))

            # Active position rebalancing with 15% deadband
            if abs(ideal_target - curr_pos) > DEADBAND:
                curr_pos = ideal_target
                action = "BUY"
            else:
                action = "HOLDING"

        target_pos_list.append(curr_pos)
        actions_list.append(action)

    test_df['target_position'] = target_pos_list
    test_df['action'] = actions_list

    # Execution position is shifted by 1 bar (trade executed on close t, earns return t+1)
    test_df['eth_position'] = test_df['target_position'].shift(1).fillna(initial_pos)

    eth_c = test_df['eth_close'] if 'eth_close' in test_df.columns else test_df['close']
    test_df['eth_returns'] = eth_c.pct_change().fillna(0.0)

    # Benchmark: 100% Buy & Hold ETH
    test_df['market_returns'] = test_df['eth_returns']
    test_df['gross_strategy_returns'] = test_df['eth_position'] * test_df['eth_returns']
    test_df['turnover'] = test_df['eth_position'].diff().abs().fillna(0.0)
    test_df['transaction_costs'] = test_df['turnover'] * FEE_PCT
    test_df['strategy_returns'] = test_df['gross_strategy_returns'] - test_df['transaction_costs']
    test_df['active_regime'] = np.where(test_df['prob_high_vol'] >= 0.50, 'High-Vol', 'Low-Vol')

    next_pos = target_pos_list[-1] if len(target_pos_list) > 0 else 0.0
    return test_df, next_pos


def calculate_metrics(returns_series):
    """
    Computes annualized Sharpe, Sortino, Max Drawdown, and Deflated Sharpe Ratio.
    """
    returns = returns_series.dropna()
    if len(returns) == 0:
        return {'sharpe': 0.0, 'sortino': 0.0, 'max_drawdown': 0.0, 'deflated_sharpe': 0.0}

    mean_ret = returns.mean() * 24 * 365
    std_ret = returns.std() * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)

    negative_returns = np.minimum(returns, 0.0)
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ret / (downside_dev + 1e-9)

    from scipy.stats import norm, skew, kurtosis
    dsr = 0.0
    if returns.std() > 1e-9 and len(returns) > 5:
        sr_observed = returns.mean() / returns.std()
        n = len(returns)
        skewness = skew(returns)
        excess_kurt = kurtosis(returns, fisher=True)
        std_sr_unann = np.sqrt(max(1e-9, (1.0 + 0.5 * sr_observed**2 - skewness * sr_observed + (excess_kurt / 4.0) * sr_observed**2) / (n - 1.0)))
        std_sr_ann = std_sr_unann * np.sqrt(24 * 365)
        dsr = float(norm.cdf(sharpe / (std_sr_ann + 1e-9)))

    return {
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'deflated_sharpe': round(float(dsr), 4)
    }


def main():
    print("=" * 85)
    print("  PURE 2-STATE MS-GARCH-X WALK-FORWARD BACKTESTER")
    print("  Kelly Volatility-Scaled Execution with 15% Deadband & Risk-Off Override")
    print("  In-Sample: 180 Days | Out-of-Sample Step: 7 Days | Total Test: 6 Months")
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
    wfa_log = []
    
    current_pos = 0.0
    step_num = 0

    while current_test_start + pd.Timedelta(days=step_days) <= end_test_date:
        current_test_end = current_test_start + pd.Timedelta(days=step_days)
        current_train_start = current_test_start - pd.Timedelta(days=train_days)
        current_train_end = current_test_start

        print(f"--- WFA STEP {step_num} ---")
        print(f"  Train: {current_train_start.strftime('%Y-%m-%d')} to {current_train_end.strftime('%Y-%m-%d')} | Test: {current_test_start.strftime('%Y-%m-%d')} to {current_test_end.strftime('%Y-%m-%d')}")

        train_df = df[(df.index >= current_train_start) & (df.index < current_train_end)].copy()

        # 1. Fit MS-GARCH-X model on In-Sample window
        msgarch = MSGARCHX()
        msgarch.fit(train_df)

        # 2. Out-of-Sample 7-Day Execution (with 100h lookback for filter continuity)
        lookback_delta = pd.Timedelta(hours=100)
        slice_df = df[(df.index >= current_test_start - lookback_delta) & (df.index < current_test_end)].copy()

        test_df_step, current_pos = run_oos_step(
            slice_df=slice_df,
            test_start=current_test_start,
            trained_msgarch=msgarch,
            initial_pos=current_pos
        )

        test_df_step['step'] = step_num
        oos_results.append(test_df_step)

        wfa_log.append({
            'step': step_num,
            'test_start': current_test_start.strftime('%Y-%m-%d'),
            'test_end': current_test_end.strftime('%Y-%m-%d')
        })

        current_test_start = current_test_end
        step_num += 1

    # 3. Stitch OOS results together
    final_oos_df = pd.concat(oos_results)
    final_oos_df.sort_index(inplace=True)
    final_oos_df = final_oos_df[~final_oos_df.index.duplicated(keep='first')]

    final_oos_df['cumulative_market'] = (1.0 + final_oos_df['market_returns']).cumprod()
    final_oos_df['cumulative_strategy'] = (1.0 + final_oos_df['strategy_returns']).cumprod()

    market_ret = (final_oos_df['cumulative_market'].iloc[-1] - 1.0) * 100
    strat_ret = (final_oos_df['cumulative_strategy'].iloc[-1] - 1.0) * 100

    metrics = calculate_metrics(final_oos_df['strategy_returns'])
    
    peak = final_oos_df['cumulative_strategy'].cummax()
    drawdown = (final_oos_df['cumulative_strategy'] - peak) / peak
    max_dd = drawdown.min() * 100

    print("\n" + "=" * 85)
    print("  PURE MS-GARCH-X VOLATILITY STRATEGY: OUT-OF-SAMPLE RESULTS")
    print("=" * 85)
    print(f"Benchmark Return (Buy & Hold ETH): {market_ret:.2f}%")
    print(f"Strategy Return:                   {strat_ret:.2f}%")
    print(f"Annualized Sharpe:                 {metrics['sharpe']:.2f}")
    print(f"Sortino Ratio:                     {metrics['sortino']:.2f}")
    print(f"Deflated Sharpe (DSR):             {metrics['deflated_sharpe']:.4f}")
    print(f"Max Drawdown:                      {max_dd:.2f}%")
    print("=" * 85)

    # 4. Save JSON results
    results_payload = {
        'metrics': {
            'market_return': round(float(market_ret), 2),
            'strategy_return': round(float(strat_ret), 2),
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'deflated_sharpe': metrics['deflated_sharpe'],
            'max_drawdown': round(float(max_dd), 2)
        },
        'wfa_log': wfa_log
    }

    out_json = os.path.join(PROJECT_ROOT, 'walk_forward_results.json')
    with open(out_json, 'w') as f:
        json.dump(results_payload, f, indent=4)

    alt_json = os.path.join(PROJECT_ROOT, 'walk_forward_microstructure_results.json')
    with open(alt_json, 'w') as f:
        json.dump(results_payload, f, indent=4)

    # 5. Plot Performance Chart
    plt.figure(figsize=(14, 8))
    plt.plot(final_oos_df.index, final_oos_df['cumulative_market'], label='Benchmark (Buy & Hold ETH)', color='gray', alpha=0.6, linewidth=1.4)
    plt.plot(final_oos_df.index, final_oos_df['cumulative_strategy'], label='2-State MS-GARCH Strategy (OOS)', color='#1f77b4', linewidth=2.0)

    # Highlight High-Vol Regimes
    ax = plt.gca()
    changes = final_oos_df['active_regime'] != final_oos_df['active_regime'].shift(1)
    change_indices = final_oos_df.index[changes].tolist()
    time_diff = final_oos_df.index[1] - final_oos_df.index[0] if len(final_oos_df) > 1 else pd.Timedelta(hours=1)
    change_indices.append(final_oos_df.index[-1] + time_diff)

    for k in range(len(change_indices) - 1):
        start_time = change_indices[k]
        end_time = change_indices[k+1]
        regime = final_oos_df.loc[start_time, 'active_regime']
        if isinstance(regime, pd.Series):
            regime = regime.iloc[0]
        if regime == 'High-Vol':
            ax.axvspan(start_time, end_time, color='crimson', alpha=0.08)

    high_vol_patch = mpatches.Patch(color='crimson', alpha=0.15, label='MS-GARCH High-Vol Risk-Off Regime')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(high_vol_patch)
    ax.legend(handles=handles, loc='upper left', framealpha=0.9)

    plt.title('Out-of-Sample Walk-Forward Performance: MS-GARCH(1,1)-X Volatility Strategy', fontsize=13, fontweight='bold')
    plt.ylabel('Cumulative Return Multiplier')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    out_png = os.path.join(PROJECT_ROOT, 'walk_forward_results.png')
    plt.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(PROJECT_ROOT, 'walk_forward_microstructure_results.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"\nSaved performance chart to {out_png}")
    print(f"Saved metrics payload to {out_json}")


if __name__ == "__main__":
    main()
