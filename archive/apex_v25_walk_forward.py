#!/usr/bin/env python3
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

from multi_asset_ablation import build_multi_asset_dataset, HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
EVAL_BARS_4YR = 35040 + WARMUP_BARS

TSMOM_WINDOWS = [24, 72, 168, 336, 720]

DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5
DIR_FEE = -0.0002

SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0
SP_FEE = -0.0002

BRK_ENTRY = 72; BRK_EXIT = 336; BRK_LEV = 4.0; BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

# Static v22 Hyperparameters
DH = 0.75
SH = 0.40
BOI = 0.25

# WFA Parameters
HOURS_PER_MONTH = 730
LOOKBACK_MONTHS = 6
LOOKBACK_BARS = HOURS_PER_MONTH * LOOKBACK_MONTHS
STEP_BARS = HOURS_PER_MONTH

def fast_metrics(h_rets):
    if len(h_rets) == 0: return 0.0, 0.0, 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR
    av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    sh = am / (av + 1e-8)
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
    print("=" * 60)
    print("APEX v25: WALK-FORWARD DYNAMIC ALLOCATION ANALYSIS")
    print("=" * 60)

    dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)
    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in UNIVERSE}, index=eval_index)
    weights_df = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    wts = {s: weights_df[s].values for s in UNIVERSE}
    
    print("\n[1] Pre-computing Base Engines (Static Hyperparameters)...")
    dir_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, DH)
    sp_rets = run_spread(processed, UNIVERSE, eval_index, n, SH)
    brk_rets = run_breakout(processed, wts, UNIVERSE, n, BOI)
    
    # Static Baseline v22 (15D, 60S, 25B)
    static_rets = (0.15 * dir_rets) + (0.60 * sp_rets) + (0.25 * brk_rets)

    ALLOCS = []
    for d in np.arange(0.0, 1.05, 0.05):
        for s in np.arange(0.0, 1.05 - d, 0.05):
            b = 1.0 - d - s
            if abs(b) < 1e-5: b = 0.0
            if b >= -1e-5:
                ALLOCS.append((round(d,2), round(s,2), round(b,2)))
    
    print(f"\n[2] Executing Walk-Forward Optimization...")
    print(f"    Lookback: {LOOKBACK_MONTHS} Months | Step: 1 Month")
    
    wfa_rets = np.zeros(n)
    
    # We start OOS testing after the initial lookback period
    start_idx = LOOKBACK_BARS
    
    # Copy the first LOOKBACK_BARS to be flat or match baseline (we won't score them in OOS anyway)
    # We will score performance ONLY from start_idx onwards to be fair.
    
    num_steps = (n - start_idx) // STEP_BARS
    
    weight_history = []
    
    for step in range(num_steps + 1):
        # In-sample window
        is_end = start_idx + step * STEP_BARS
        is_start = is_end - LOOKBACK_BARS
        
        # OOS window
        oos_start = is_end
        oos_end = min(oos_start + STEP_BARS, n)
        
        if oos_start >= n: break
        
        # Optimize over IS window
        best_sh = -999; best_alloc = (0.15, 0.60, 0.25)
        for (d, s, b) in ALLOCS:
            is_ens = (d * dir_rets[is_start:is_end]) + (s * sp_rets[is_start:is_end]) + (b * brk_rets[is_start:is_end])
            sh, _, _ = fast_metrics(is_ens)
            if sh > best_sh:
                best_sh = sh
                best_alloc = (d, s, b)
                
        # Apply to OOS window
        d, s, b = best_alloc
        wfa_rets[oos_start:oos_end] = (d * dir_rets[oos_start:oos_end]) + (s * sp_rets[oos_start:oos_end]) + (b * brk_rets[oos_start:oos_end])
        weight_history.append({'date': eval_index[oos_start], 'd': d, 's': s, 'b': b})
        
        if step % 6 == 0:
            print(f"    Step {step}/{num_steps} | Optimizing {eval_index[oos_start].date()} -> Best IS Sharpe: {best_sh:.2f} | Alloc: {d:.2f}D, {s:.2f}S, {b:.2f}B")

    print("\n[3] Calculating OOS Performance Metrics...")
    
    # Evaluate ONLY the out-of-sample period (from start_idx to end)
    eval_static = static_rets[start_idx:]
    eval_wfa = wfa_rets[start_idx:]
    
    static_sh, static_net, static_dd = fast_metrics(eval_static)
    wfa_sh, wfa_net, wfa_dd = fast_metrics(eval_wfa)
    
    print("\n==================================================")
    print("RESULTS (Out-Of-Sample Period Only: Month 7 to 48)")
    print("==================================================")
    print(f"STATIC BASELINE (v22: 15D/60S/25B):")
    print(f"  - Sharpe:     {static_sh:.2f}")
    print(f"  - Net Return: {static_net:+.2f}%")
    print(f"  - Max DD:     {static_dd:.2f}%")
    print(f"\nWALK-FORWARD DYNAMIC (6m Lookback / 1m Step):")
    print(f"  - Sharpe:     {wfa_sh:.2f}")
    print(f"  - Net Return: {wfa_net:+.2f}%")
    print(f"  - Max DD:     {wfa_dd:.2f}%")
    print("==================================================")
    
    # Generate visualization
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    oos_index = eval_index[start_idx:]
    
    static_cum = np.cumprod(1 + eval_static)
    wfa_cum = np.cumprod(1 + eval_wfa)
    
    ax1.plot(oos_index, wfa_cum, color='#FF5252', linewidth=2.0, label=f'Walk-Forward Dynamic (Sharpe: {wfa_sh:.2f})')
    ax1.plot(oos_index, static_cum, color='#00E5FF', linewidth=1.5, label=f'Static Baseline v22 (Sharpe: {static_sh:.2f})', alpha=0.8)
    
    ax1.set_title('Out-of-Sample Performance: Walk-Forward vs Static Baseline', color='white', fontsize=14)
    ax1.grid(color='gray', linestyle=':', alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot weights
    wh_df = pd.DataFrame(weight_history).set_index('date')
    wh_df = wh_df.reindex(oos_index).fillna(method='ffill')
    
    ax2.stackplot(wh_df.index, wh_df['d'], wh_df['s'], wh_df['b'], labels=['Directional', 'Spread', 'Breakout'], colors=['#2962FF', '#00C853', '#FFAB00'], alpha=0.8)
    ax2.set_title('Dynamic Portfolio Allocation (100%)', color='white', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.margins(x=0)
    ax2.legend(loc='lower left')
    
    os.makedirs('archive/results', exist_ok=True)
    out_path = 'archive/results/apex_v25_walk_forward.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    print(f"\n[SAVED] {out_path}")

if __name__ == '__main__':
    main()
