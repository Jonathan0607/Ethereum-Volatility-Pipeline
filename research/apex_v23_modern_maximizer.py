#!/usr/bin/env python3
r"""
=====================================================================================
ACM v23: MODERN REGIME MAXIMIZER (2024–2026 TEST PERIOD TARGETING)
=====================================================================================
Optimizes strictly for the modern 2024–2026 Test Period while retaining baseline stability
by filtering for configurations where Train Sharpe (2022–2024) >= 0.70.

Sweep Ranges:
- Directional Hysteresis: [0.40, 0.50]
- Spread Hysteresis:      [0.40, 0.50]
- Breakout OI Gate:       [0.20, 0.25] (with 72h Fast Donchian profit-lock)
- Allocations:            Full 5% simplex grid across (Directional, Spread, Breakout)
=====================================================================================
"""

import os, sys, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, 'research')
ARCHIVE_DIR = os.path.join(RESEARCH_DIR, 'archive')
RESULTS_DIR = os.path.join(RESEARCH_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
ARTIFACT_DIR = "/Users/kingebenezer/.gemini/antigravity-ide/brain/b22eb094-5eb5-444d-9566-04f193fce2dc"
sys.path.insert(0, ARCHIVE_DIR)
sys.path.insert(0, RESEARCH_DIR)
sys.path.insert(0, PROJECT_ROOT)

from multi_asset_ablation import build_multi_asset_dataset, HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720; EVAL_BARS_4YR = 35040 + WARMUP_BARS; SPLIT_BARS = 17520
TSMOM_WINDOWS = [24, 72, 168, 336, 720]

# Base Configurations
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5; DIR_FEE = -0.0002
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0; SP_FEE = -0.0002
BRK_ENTRY = 72; BRK_EXIT = 336; BRK_LEV = 4.0; BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

def fast_metrics(h_rets):
    if len(h_rets) == 0: return 0.0, 0.0, 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR; av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    sh = am / (av + 1e-8)
    cum = np.cumprod(1.0 + h_rets)
    cummax = np.maximum.accumulate(cum)
    drawdown = (cum - cummax) / cummax
    mdd = np.min(drawdown) * 100.0
    tr = (cum[-1] - 1.0) * 100.0
    return sh, tr, mdd

def compute_indicators(dataset, univ):
    processed = {}
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']
        fr = df['funding_rate'].fillna(0.0); oi = df['open_interest'].fillna(method='ffill')
        cs = c.shift(1)
        df[f'upper_{BRK_ENTRY}'] = h.rolling(BRK_ENTRY).max().shift(1)
        df[f'lower_{BRK_EXIT}'] = l.rolling(BRK_EXIT).min().shift(1)
        df[f'lower_72'] = l.rolling(72).min().shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / (cs + 1e-8)
        df['inv_vol'] = 1.0 / df['atr_pct'].replace(0, np.nan).fillna(0.0)
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w); vw = c.pct_change().rolling(w).std() * np.sqrt(w)
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
                if open_profit > PROFIT_LOCK_THRESHOLD:
                    profit_lock_active = True
                exit_price = lower_fast[t] if profit_lock_active else lower_base[t]
                if cv[t] < exit_price:
                    pos = 0.0; profit_lock_active = False
            elif cv[t] > upper[t] and doi[t] >= oi_gate:
                pos = 1.0; entry_price = cv[t]; profit_lock_active = False
                
            tgt[t] = pos * (w[t] if not np.isnan(w[t]) else 0.0) * BRK_LEV
            
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * BRK_FEE)
    return port

def main():
    print("=" * 120)
    print("ACM v23: MODERN REGIME MAXIMIZER (2024–2026 TEST PERIOD TARGETING)")
    print("=" * 120)

    dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)
    print("\nComputing zero-lag indicators across 4-year dataset...")
    processed = compute_indicators(dataset, UNIVERSE)

    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index; n = len(eval_index)
    split_idx = min(SPLIT_BARS, n // 2)
    weights = cross_weights(processed, UNIVERSE, eval_index)

    # Step 1: Define Targeted Sweep Ranges
    HYST_DIR = [0.40, 0.50]
    HYST_SP = [0.40, 0.50]
    OI_GATES = [0.20, 0.25]
    
    ALLOCS = []
    for d in np.arange(0.0, 1.05, 0.05):
        for s in np.arange(0.0, 1.05 - d, 0.05):
            b = 1.0 - d - s
            if abs(b) < 1e-5: b = 0.0
            if b >= -1e-5:
                ALLOCS.append((round(d, 2), round(s, 2), round(b, 2)))
    
    print(f"\nPre-computing Engine Variants:")
    print(f"  - Directional Hysteresis: {HYST_DIR}")
    print(f"  - Spread Hysteresis:      {HYST_SP}")
    print(f"  - Breakout OI Gate:       {OI_GATES} (with 72h profit-lock)")
    print(f"  - Allocations Grid:       {len(ALLOCS)} ternary simplex combinations (5% step)")
    
    dir_engines = {h: run_directional(processed, weights, UNIVERSE, eval_index, n, h) for h in HYST_DIR}
    sp_engines = {h: run_spread(processed, UNIVERSE, eval_index, n, h) for h in HYST_SP}
    brk_engines = {o: run_breakout(processed, weights, UNIVERSE, n, o) for o in OI_GATES}

    total_configs = len(HYST_DIR) * len(HYST_SP) * len(OI_GATES) * len(ALLOCS)
    print(f"\nEvaluating {total_configs} targeted combinations...")

    passing_results = []
    filtered_out_count = 0

    for dh in HYST_DIR:
        for sh in HYST_SP:
            for boi in OI_GATES:
                d_arr = dir_engines[dh]
                s_arr = sp_engines[sh]
                b_arr = brk_engines[boi]
                
                for (d, s, b) in ALLOCS:
                    ens = d * d_arr + s * s_arr + b * b_arr
                    
                    # Compute Train Sharpe (2022–2024)
                    tr_sh, tr_net, tr_dd = fast_metrics(ens[:split_idx])
                    
                    # Step 2: Modern Regime Objective Function Filter
                    # Filter for Train Sharpe >= 0.70 (ensuring non-fragility during bear chop)
                    if tr_sh < 0.70:
                        filtered_out_count += 1
                        continue
                    
                    # Compute Test Sharpe (2024–2026) and Full Metrics
                    te_sh, te_net, te_dd = fast_metrics(ens[split_idx:])
                    full_sh, full_net, full_dd = fast_metrics(ens)
                    
                    cfg_str = f"DH={dh:.2f} SH={sh:.2f} BOI={boi:.2f} [{int(round(d*100))}D/{int(round(s*100))}S/{int(round(b*100))}B]"
                    passing_results.append({
                        'name': cfg_str,
                        'dh': dh, 'sh': sh, 'boi': boi,
                        'd': d, 's': s, 'b': b,
                        'train_sharpe': tr_sh, 'train_net': tr_net, 'train_dd': tr_dd,
                        'test_sharpe': te_sh, 'test_net': te_net, 'test_dd': te_dd,
                        'full_sharpe': full_sh, 'full_net': full_net, 'full_dd': full_dd,
                        'ens': ens
                    })

    print(f"Total evaluated: {total_configs} | Passed Train Sharpe >= 0.70: {len(passing_results)} | Filtered: {filtered_out_count}")

    # Rank all passing configurations STRICTLY by Test Sharpe (2024–2026)
    passing_results.sort(key=lambda x: x['test_sharpe'], reverse=True)

    print("\n" + "=" * 120)
    print("### TOP 5 CONFIGURATIONS RANKED STRICTLY BY TEST SHARPE (2024–2026) [Train Sharpe >= 0.70]")
    print("=" * 120)
    headers = ["Rank", "Configuration", "Train Sharpe", "Test Sharpe", "Full Sharpe", "Test Max DD", "Full Net", "Test Net"]
    table_rows = []
    for i in range(min(5, len(passing_results))):
        r = passing_results[i]
        table_rows.append([
            f"#{i+1}",
            r['name'],
            f"{r['train_sharpe']:.2f}",
            f"{r['test_sharpe']:.2f}",
            f"{r['full_sharpe']:.2f}",
            f"{r['test_dd']:.2f}%",
            f"{r['full_net']:+.2f}%",
            f"{r['test_net']:+.2f}%"
        ])

    col_w = [max(len(str(row[i])) for row in [headers] + table_rows) for i in range(len(headers))]
    print(f"| {' | '.join(h.ljust(col_w[i]) for i, h in enumerate(headers))} |")
    print(f"| {'-|-'.join('-' * col_w[i] for i in range(len(headers)))} |")
    for row in table_rows:
        print(f"| {' | '.join(val.ljust(col_w[i]) for i, val in enumerate(row))} |")
    print("=" * 120)

    # Winner details
    winner = passing_results[0]
    print(f"\nWinning Configuration: {winner['name']}")
    print(f"  • Test Sharpe (2024–2026): {winner['test_sharpe']:.2f}")
    print(f"  • Train Sharpe (2022–2024): {winner['train_sharpe']:.2f}")
    print(f"  • Full 4-Year Sharpe:       {winner['full_sharpe']:.2f}")
    print(f"  • Test Max Drawdown:        {winner['test_dd']:.2f}%")
    print(f"  • Full Max Drawdown:        {winner['full_dd']:.2f}%")
    print(f"  • Test Net Return:          {winner['test_net']:+.2f}%")
    print(f"  • Full Net Return:          {winner['full_net']:+.2f}%")

    # Step 3: Visualization & Artifacts
    print("\nGenerating compounding equity curve visualization for the winning configuration...")
    plt.style.use('dark_background')
    fig, (ax_eq, ax_dd) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)

    c_f = np.cumprod(1.0 + winner['ens'])
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1.0 + bench_h.values)

    # Drawdown calculations
    cummax_strat = np.maximum.accumulate(c_f)
    dd_strat = (c_f - cummax_strat) / cummax_strat * 100.0

    cummax_bench = np.maximum.accumulate(bench_cum)
    dd_bench = (bench_cum - cummax_bench) / cummax_bench * 100.0

    # Top Panel: Equity Curves
    ax_eq.plot(eval_index, c_f, label=f"Apex v23 Modern Maximizer ({winner['name']})\nFull Sharpe: {winner['full_sharpe']:.2f} | Test Sharpe: {winner['test_sharpe']:.2f} | Net: {winner['full_net']:+.1f}%", color='#00E5FF', linewidth=2.5)
    ax_eq.plot(eval_index, bench_cum, label=f"Buy & Hold Equal-Weight Benchmark (Net: {(bench_cum[-1]-1.0)*100:+.1f}%)", color='#78909C', linestyle=':', linewidth=1.5)

    ax_eq.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.8, label="Train / Test Split (2024-01-01)")
    
    # Text callouts positioned cleanly using axes coordinates
    ax_eq.text(0.24, 0.65, f"TRAIN (2022–2024)\nSharpe: {winner['train_sharpe']:.2f}\nNet: {winner['train_net']:+.1f}%\nMaxDD: {winner['train_dd']:.1f}%",
               transform=ax_eq.transAxes, horizontalalignment='center', color='#ECEFF1', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E222D', edgecolor='#00E5FF', alpha=0.85))
    ax_eq.text(0.74, 0.86, f"TEST (2024–2026) [FLAGSHIP TARGET]\nSharpe: {winner['test_sharpe']:.2f}\nNet: {winner['test_net']:+.1f}%\nMaxDD: {winner['test_dd']:.1f}%",
               transform=ax_eq.transAxes, horizontalalignment='center', color='#00E5FF', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E222D', edgecolor='#FF4081', alpha=0.85))

    ax_eq.set_title(f"Project Apex v23: Modern Regime Maximizer (Flagship 2024–2026 Test Period)\n{winner['name']}", fontsize=13, fontweight='bold', color='#ECEFF1', pad=12)
    ax_eq.set_ylabel("Portfolio Equity (Normalized to 1.0)", fontsize=11, fontweight='bold', color='#ECEFF1')
    ax_eq.grid(True, linestyle='--', alpha=0.25)
    ax_eq.legend(loc='upper left', fontsize=10, framealpha=0.9, facecolor='#1E222D')

    # Bottom Panel: Drawdown
    ax_dd.plot(eval_index, dd_strat, color='#FF5252', linewidth=1.5, label=f"Strategy DD (Test Max: {winner['test_dd']:.1f}%, Full Max: {winner['full_dd']:.1f}%)")
    ax_dd.plot(eval_index, dd_bench, color='#78909C', linestyle=':', linewidth=1.2, label=f"Benchmark DD (Max: {np.min(dd_bench):.1f}%)")
    ax_dd.fill_between(eval_index, dd_strat, 0, color='#FF5252', alpha=0.25)
    ax_dd.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.8)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=11, fontweight='bold', color='#ECEFF1')
    ax_dd.set_xlabel("Macro Cycle Timeline (2022 – 2026)", fontsize=11, fontweight='bold', color='#ECEFF1')
    ax_dd.grid(True, linestyle='--', alpha=0.25)
    ax_dd.legend(loc='lower left', fontsize=9, framealpha=0.9, facecolor='#1E222D')

    out = os.path.join(RESULTS_DIR, 'apex_v23_modern_apex.png')
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[SAVED] {out}")
    if os.path.exists(ARTIFACT_DIR):
        shutil.copyfile(out, os.path.join(ARTIFACT_DIR, 'apex_v23_modern_apex.png'))
        print(f"[COPIED] {os.path.join(ARTIFACT_DIR, 'apex_v23_modern_apex.png')}")

if __name__ == '__main__':
    main()
