#!/usr/bin/env python3
"""
===================================================================================
APEX v29: THE TRUE HISTORY HEAD-TO-HEAD
===================================================================================
Loads the True Historical Dataset downloaded directly from Binance Vision CSVs.
No synthetic open interest. No synthetic funding rates.
Evaluates the Static Baseline Triple Ensemble to find the absolute truth.
===================================================================================
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

from multi_asset_ablation import HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
TSMOM_WINDOWS = [24, 72, 168, 336, 720]

DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5; DIR_FEE = -0.0002
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0; SP_FEE = -0.0002
BRK_ENTRY = 72; BRK_EXIT = 336; BRK_LEV = 4.0; BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

BEST_DH = 0.60
BEST_SH = 0.40
BEST_BOI = 0.15
BEST_D, BEST_S, BEST_B = 0.20, 0.55, 0.25

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR
    av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    return am / (av + 1e-8)

def fast_metrics(h_rets):
    if len(h_rets) == 0: return 0.0, 0.0, 0.0
    sh = fast_sharpe(h_rets)
    cum = np.cumprod(1.0 + h_rets)
    cummax = np.maximum.accumulate(cum)
    dd = np.min((cum - cummax) / (cummax + 1e-8)) * 100.0
    ret = (cum[-1] - 1.0) * 100.0
    return sh, ret, dd

def compute_indicators(dataset, univ):
    processed = {}
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']
        fr = df['funding_rate'].fillna(0.0)
        oi = df['open_interest'].fillna(method='ffill')
        cs = c.shift(1)
        df[f'upper_{BRK_ENTRY}'] = h.rolling(BRK_ENTRY).max().shift(1)
        df[f'lower_{BRK_EXIT}'] = l.rolling(BRK_EXIT).min().shift(1)
        df[f'lower_72'] = l.rolling(72).min().shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / (cs + 1e-8)
        df['inv_vol'] = 1.0 / df['atr_pct'].replace(0, np.nan).fillna(0.0)
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w)
            vw = c.pct_change().rolling(w).std() * np.sqrt(w)
            parts.append(np.tanh(rw / (vw + 1e-8)))
        df['tsmom'] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
        mu = fr.rolling(720, min_periods=72).mean()
        sig = fr.rolling(720, min_periods=72).std()
        df['carry'] = (-np.tanh((fr - mu) / (sig + 1e-8))).fillna(0.0)
        doi = oi.pct_change(48).fillna(0.0)
        df['doi_48'] = doi.shift(1).fillna(0.0)
        df['oi_regime'] = np.tanh(df['doi_48'] * 5.0).fillna(0.0)
        df['hourly_return'] = c.pct_change().fillna(0.0)
        processed[s] = df
    return processed

def cross_weights(processed, univ, eval_index):
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in univ}, index=eval_index)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {s: w[s].values for s in univ}

def run_directional(processed, wts, univ, eval_index, n, hyst):
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = DIR_WT*df['tsmom'].values + DIR_WC*df['carry'].values + DIR_WO*df['oi_regime'].values
    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values; w = wts[s]; comp = comps[s]; x = xs[s]
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * DIR_LEV
            if abs(ideal - cur) >= hyst: cur = ideal
            tgt[t] = cur
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * DIR_FEE)
    return port

def run_spread(processed, univ, eval_index, n, hyst):
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = SP_WT*df['tsmom'].values + SP_WC*df['carry'].values + SP_WO*df['oi_regime'].values
    cur_pos = {s: 0.0 for s in univ}
    port = np.zeros(n)
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        top_s = sc[0][0]; bot_s = sc[-1][0]
        conv = abs(sc[0][1] - sc[-1][1])
        ideal = {s: 0.0 for s in univ}
        ideal[top_s] = conv * SP_LEV; ideal[bot_s] = -conv * SP_LEV
        for s in univ:
            old = cur_pos[s]
            if abs(ideal[s] - cur_pos[s]) >= hyst: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port[t] += turn * SP_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port[t] += ret * cur_pos[s]
    return port

def run_breakout(processed, wts, univ, n, oi_gate):
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        cv = df['close'].values; ret = df['hourly_return'].values
        upper = df[f'upper_{BRK_ENTRY}'].values; lower_base = df[f'lower_{BRK_EXIT}'].values
        lower_fast = df['lower_72'].values
        doi = df['doi_48'].values; w = wts[s]
        pos = 0.0; tgt = np.zeros(n)
        entry_price = 0.0; profit_lock_active = False
        for t in range(n):
            if np.isnan(cv[t]) or np.isnan(upper[t]): continue
            if pos > 0:
                open_profit = (cv[t] - entry_price) / entry_price
                if open_profit > PROFIT_LOCK_THRESHOLD: profit_lock_active = True
                exit_price = lower_fast[t] if profit_lock_active else lower_base[t]
                if cv[t] < exit_price: pos = 0.0; profit_lock_active = False
            elif cv[t] > upper[t] and doi[t] >= oi_gate:
                pos = 1.0; entry_price = cv[t]; profit_lock_active = False
            tgt[t] = pos * (w[t] if not np.isnan(w[t]) else 0.0) * BRK_LEV
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * BRK_FEE)
    return port

def main():
    print("=" * 80)
    print("APEX v29: THE TRUE HISTORY EVALUATION")
    print("=" * 80)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    print(f"\n[LOADING] True dataset from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    wts = cross_weights(processed, UNIVERSE, eval_index)

    print(f"  Dataset Range: {eval_index[0]} to {eval_index[-1]} ({n} bars)")
    print(f"  Configuration: DH={BEST_DH} SH={BEST_SH} BOI={BEST_BOI}")
    print(f"  Weights: {int(BEST_D*100)}D / {int(BEST_S*100)}S / {int(BEST_B*100)}B")

    print("\n[STEP 1] Running Triple Ensemble on 100% Real Historical Data...")
    
    d_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, BEST_DH)
    s_rets = run_spread(processed, UNIVERSE, eval_index, n, BEST_SH)
    b_rets = run_breakout(processed, wts, UNIVERSE, n, BEST_BOI)
    
    ens_rets = (BEST_D * d_rets) + (BEST_S * s_rets) + (BEST_B * b_rets)

    sh_full, net_full, dd_full = fast_metrics(ens_rets)

    print("\n" + "=" * 80)
    print("THE TRUTH REVEALED")
    print("=" * 80)
    print(f"True Sharpe Ratio: {sh_full:.2f}")
    print(f"True Net Return:   {net_full:+.2f}%")
    print(f"True Max Drawdown: {dd_full:.2f}%")
    print("=" * 80)

    # Visualization
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))

    ens_cum = np.cumprod(1 + ens_rets)
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1 + bench_h.values)

    ax.plot(eval_index, ens_cum, color='#00E5FF', linewidth=2.5,
             label=f'Triple Ensemble (Sharpe: {sh_full:.2f}, Net: {net_full:+.1f}%)')
    ax.plot(eval_index, bench_cum, color='#78909C', linewidth=1.5, linestyle=':', alpha=0.5,
             label='Buy & Hold Benchmark')
             
    ax.set_title('Apex v29: True Historical Data Performance (100% Real Binance Vision Data)', 
                 color='white', fontsize=14, fontweight='bold')
    ax.grid(color='gray', linestyle=':', alpha=0.2)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1E222D')
    
    # Add watermark
    ax.text(0.5, 0.5, 'ZERO SYNTHETIC DATA', transform=ax.transAxes,
            fontsize=40, color='white', alpha=0.05,
            ha='center', va='center', rotation=15)

    out_path = os.path.join(RESULTS_DIR, 'apex_v29_true_history.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] Equity curve to {out_path}")

if __name__ == '__main__':
    main()
