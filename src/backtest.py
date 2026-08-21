import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data import split_data, load_microstructure_data
from ms_garch_engine import MSGARCHX

FEE_PCT = 0.0020          # 20 bps transaction friction
TARGET_VOLATILITY = 0.06  # Target volatility for Kelly scaling
DEADBAND = 0.15           # 15% deadband filter


def load_data():
    return load_microstructure_data()


def run_backtest(df):
    """
    Executes Out-of-Sample backtest using pure 2-State MS-GARCH-X
    with Volatility-Scaled Kelly Sizing and 15% Deadband.
    """
    train_df, test_df = split_data(df, verbose=False)
    if test_df.empty:
        raise ValueError("Test set is empty!")

    # 1. Fit 2-State MS-GARCH-X on In-Sample (75% history)
    msgarch = MSGARCHX()
    msgarch.fit(train_df)

    # 2. Filter MS-GARCH-X over the full dataset
    filtered = msgarch.filter_full_series(df)
    df = df.join(filtered[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
    df['prob_high_vol'] = df['prob_high_vol'].ffill().fillna(1.0)
    df['ms_garch_volatility'] = df['ms_garch_volatility'].ffill().fillna(0.01)

    test_indices = np.where(df.index >= test_df.index[0])[0]
    out_test_df = df.iloc[test_indices].copy()

    ann_factor = np.sqrt(24 * 365)
    target_pos_list = []
    actions_list = []
    curr_pos = 0.0

    for idx in range(len(out_test_df)):
        p_high = float(out_test_df['prob_high_vol'].iloc[idx])
        g_vol = float(out_test_df['ms_garch_volatility'].iloc[idx])

        ann_vol = g_vol * ann_factor
        vol_scaler = TARGET_VOLATILITY / (ann_vol + 1e-8)

        # Risk-Off Deadband Override
        if p_high >= 0.50:
            ideal_target = 0.0
            if curr_pos != 0.0:
                action = "CASH"
            else:
                action = "FLAT"
            curr_pos = 0.0
        else:
            # Low-vol regime: scale by (1 - prob_high_vol) * vol_scaler
            raw_target = (1.0 - p_high) * vol_scaler
            ideal_target = float(np.clip(raw_target, 0.0, 1.0))

            # 15% Deadband Filter
            if abs(ideal_target - curr_pos) > DEADBAND:
                curr_pos = ideal_target
                action = "BUY"
            else:
                action = "HOLDING"

        target_pos_list.append(curr_pos)
        actions_list.append(action)

    out_test_df['target_position'] = target_pos_list
    out_test_df['action'] = actions_list
    out_test_df['eth_position'] = out_test_df['target_position'].shift(1).fillna(0.0)

    eth_c = out_test_df['eth_close'] if 'eth_close' in out_test_df.columns else out_test_df['close']
    out_test_df['eth_returns'] = eth_c.pct_change().fillna(0.0)

    # Benchmark: Buy & Hold ETH
    out_test_df['market_returns'] = out_test_df['eth_returns']
    out_test_df['gross_strategy_returns'] = out_test_df['eth_position'] * out_test_df['eth_returns']
    out_test_df['turnover'] = out_test_df['eth_position'].diff().abs().fillna(0.0)
    out_test_df['transaction_costs'] = out_test_df['turnover'] * FEE_PCT
    out_test_df['strategy_returns'] = out_test_df['gross_strategy_returns'] - out_test_df['transaction_costs']

    out_test_df['cumulative_market'] = (1.0 + out_test_df['market_returns']).cumprod()
    out_test_df['cumulative_strategy'] = (1.0 + out_test_df['strategy_returns']).cumprod()
    out_test_df['close'] = eth_c
    out_test_df['active_regime'] = np.where(out_test_df['prob_high_vol'] >= 0.50, 'High-Vol', 'Low-Vol')

    return out_test_df


def calculate_metrics(df, verbose=True):
    returns = df['strategy_returns'].dropna()
    mean_ret = returns.mean() * 24 * 365
    std_ret = returns.std() * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)

    negative_returns = np.minimum(returns, 0.0)
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ret / (downside_dev + 1e-9)

    cumulative = df['cumulative_strategy']
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    market_ret = (df['cumulative_market'].iloc[-1] - 1.0) * 100
    strat_ret = (df['cumulative_strategy'].iloc[-1] - 1.0) * 100

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

    if verbose:
        longs = (df['eth_position'] > 0).sum()
        flat = (df['eth_position'] == 0).sum()
        print(f"\n=== PURE MS-GARCH-X VOLATILITY BACKTEST (OUT OF SAMPLE) ===")
        print(f"Bars: LONG={longs} | FLAT={flat}")
        print(f"Benchmark Return (Buy & Hold): {market_ret:.2f}%")
        print(f"Strategy Return:               {strat_ret:.2f}%")
        print(f"Annualized Sharpe:             {sharpe:.2f}")
        print(f"Sortino Ratio:                 {sortino:.2f}")
        print(f"Deflated Sharpe:               {dsr:.4f}")
        print(f"Max Drawdown:                  {max_drawdown * 100:.2f}%")

    return {
        'market_return': round(float(market_ret), 2),
        'strategy_return': round(float(strat_ret), 2),
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'deflated_sharpe': round(float(dsr), 4),
        'max_drawdown': round(float(max_drawdown * 100), 2),
    }


def export_json(df, metrics):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    step = max(1, len(df) // 400)
    sampled = df.iloc[::step].copy()
    sampled = sampled.dropna(subset=['cumulative_market', 'cumulative_strategy'])

    records = []
    for ts, row in sampled.iterrows():
        records.append({
            'timestamp': ts.isoformat(),
            'close': round(float(row['close']), 2),
            'cumulative_market': round(float(row['cumulative_market']), 4),
            'cumulative_strategy': round(float(row['cumulative_strategy']), 4),
            'eth_position': round(float(row.get('eth_position', 0.0)), 4),
            'prob_high_vol': round(float(row.get('prob_high_vol', 0.0)), 4),
            'ms_garch_volatility': round(float(row.get('ms_garch_volatility', 0.0)), 6),
            'active_regime': str(row.get('active_regime', 'Low-Vol'))
        })

    payload = {'metrics': metrics, 'series': records}
    out_path = os.path.join(current_dir, '..', 'backtest_results.json')
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=4)


def plot_results(df):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['cumulative_market'], label='Benchmark (Buy & Hold ETH)', color='gray', alpha=0.6, linewidth=1.2)
    plt.plot(df.index, df['cumulative_strategy'], label='2-State MS-GARCH Strategy (OOS)', color='#1f77b4', linewidth=1.8)
    
    ax = plt.gca()
    changes = df['active_regime'] != df['active_regime'].shift(1)
    change_indices = df.index[changes].tolist()
    time_diff = df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta(hours=1)
    change_indices.append(df.index[-1] + time_diff)

    for k in range(len(change_indices) - 1):
        start_time = change_indices[k]
        end_time = change_indices[k+1]
        regime = df.loc[start_time, 'active_regime']
        if isinstance(regime, pd.Series):
            regime = regime.iloc[0]
        if regime == 'High-Vol':
            ax.axvspan(start_time, end_time, color='crimson', alpha=0.08)

    high_vol_patch = mpatches.Patch(color='crimson', alpha=0.15, label='MS-GARCH High-Vol Risk-Off Regime')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(high_vol_patch)
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

    plt.title('MS-GARCH(1,1)-X Volatility-Scaled Sizing (Out of Sample)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Return Multiplier')
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(current_dir, '..', 'backtest_results.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_dashboard(df):
    subset = df.tail(2000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(subset.index, subset['close'], color='gray', alpha=0.5, label='ETH Price')
    long_pos = subset[subset['eth_position'] > 0]
    flat_pos = subset[(subset['eth_position'] == 0) & (subset['eth_position'].shift(1) > 0)]

    ax1.scatter(long_pos.index, long_pos['close'], color='green', s=8, alpha=0.6, label='Active Long Position')
    ax1.scatter(flat_pos.index, flat_pos['close'], color='red', s=16, alpha=0.9, label='Risk-Off Liquidation (CASH)')

    ax1.set_title('MS-GARCH Volatility Regime & Execution Allocations', fontsize=14, fontweight='bold')
    ax1.set_ylabel('ETH Price (USD)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.15)

    if 'prob_high_vol' in subset.columns:
        ax2.plot(subset.index, subset['prob_high_vol'], color='#d62728', alpha=0.85, label='P(High-Vol Regime)')
        ax2.axhline(0.50, color='black', linestyle='--', alpha=0.6, label='Risk-Off Threshold (0.50)')
        ax2.fill_between(subset.index, 0.50, subset['prob_high_vol'], where=subset['prob_high_vol'] >= 0.50, color='crimson', alpha=0.2)
        ax2.set_ylabel('Regime Probability')
        ax2.set_ylim(-0.05, 1.05)

    ax2.set_title('2-State MS-GARCH-X Posterior Probability', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.15)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model_dashboard.png')
    plt.savefig(output_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    try:
        data = load_data()
        results = run_backtest(data)
        metrics = calculate_metrics(results)
        export_json(results, metrics)
        plot_results(results)
        plot_dashboard(results)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()