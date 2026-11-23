#!/usr/bin/env python3
"""
===================================================================================
APEX v40: ALPHA LAB - THE MICRO-REGIME MAXIMIZER
===================================================================================
Uses pure synthetic alpha proxies for feature generation.
Employs a 3-Month Lookback, 1-Month Step Walk-Forward Optimization to rapidly
adapt to crypto micro-regimes.
Uses fixed global ensemble weights (20D / 55S / 25B).
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
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
BRK_ENTRY = 72; BRK_EXIT = 336
PROFIT_LOCK_THRESHOLD = 0.20

TAKER_FEE = -0.0010
MAKER_FEE = -0.0002

HYST_BANDS = [0.25, 0.40, 0.50, 0.60]
OI_GATES = [0.10, 0.15, 0.20, 0.25]

HOURS_PER_MONTH = 730
LOOKBACK_MONTHS = 3
LOOKBACK_BARS = HOURS_PER_MONTH * LOOKBACK_MONTHS
STEP_BARS = HOURS_PER_MONTH * 1

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
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        
        vol_z = (v - v.rolling(720, min_periods=24).mean()) / (v.rolling(720, min_periods=24).std() + 1e-8)
        ret_cum = c.pct_change(48).fillna(0.0)
        synthetic_oi = 5000000.0 * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
        
        ret_24 = c.pct_change(24).fillna(0.0)
        synthetic_fr = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)

        cs = c.shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / (cs + 1e-8)
        df['inv_vol'] = 1.0 / df['atr_pct'].replace(0, np.nan).fillna(0.0)
        
        df[f'upper_{BRK_ENTRY}'] = h.rolling(BRK_ENTRY).max().shift(1)
        df[f'lower_{BRK_EXIT}'] = l.rolling(BRK_EXIT).min().shift(1)
        
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w)
            vw = c.pct_change().rolling(w).std() * np.sqrt(w)
            parts.append(np.tanh(rw / (vw + 1e-8)))
        df['tsmom'] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
        
        mu = synthetic_fr.rolling(720, min_periods=72).mean()
        sig = synthetic_fr.rolling(720, min_periods=72).std()
        df['carry'] = (-np.tanh((synthetic_fr - mu) / (sig + 1e-8))).fillna(0.0)
        
        doi = synthetic_oi.pct_change(48).fillna(0.0)
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
            ideal = cv * wv * x[t] * 2.5
            if abs(ideal - cur) >= hyst: cur = ideal
            tgt[t] = cur
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * MAKER_FEE)
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
        ideal[top_s] = conv * 1.0; ideal[bot_s] = -conv * 1.0
        for s in univ:
            old = cur_pos[s]
            if abs(ideal[s] - cur_pos[s]) >= hyst: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port[t] += turn * MAKER_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port[t] += ret * cur_pos[s]
    return port

def run_breakout(processed, wts, univ, n, oi_gate):
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        cv = df['close'].values; ret = df['hourly_return'].values
        upper = df[f'upper_{BRK_ENTRY}'].values; lower_base = df[f'lower_{BRK_EXIT}'].values
        doi = df['doi_48'].values; w = wts[s]
        
        pos = 0.0; tgt = np.zeros(n)
        for t in range(n):
            if np.isnan(cv[t]) or np.isnan(upper[t]): continue
            if pos > 0:
                if cv[t] < lower_base[t]: pos = 0.0
            elif cv[t] > upper[t] and doi[t] >= oi_gate:
                pos = 1.0
            
            tgt[t] = pos * (w[t] if not np.isnan(w[t]) else 0.0) * 4.0
            
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * TAKER_FEE)
    return port

def main():
    print("=" * 80)
    print("APEX v40: MICRO-REGIME WFA (3M Lookback / 1M Step)")
    print("=" * 80)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found.")
        return
        
    print(f"\n[LOADING] Dataset from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    wts = cross_weights(processed, UNIVERSE, eval_index)
    split_idx = min(17520, n // 2)

    print(f"  Dataset: {eval_index[0]} to {eval_index[-1]} ({n} bars)")

    # =========================================================================
    # Pre-compute Engine Variants
    # =========================================================================
    print("\n[STEP 1] Pre-computing WFA Engine Variants...")
    
    dir_engines = {}
    sp_engines = {}
    brk_engines = {}
    
    for dh in HYST_BANDS:
        dir_engines[dh] = run_directional(processed, wts, UNIVERSE, eval_index, n, dh)
    for sh in HYST_BANDS:
        sp_engines[sh] = run_spread(processed, UNIVERSE, eval_index, n, sh)
    for boi in OI_GATES:
        brk_engines[boi] = run_breakout(processed, wts, UNIVERSE, n, boi)

    # =========================================================================
    # Walk-Forward Optimization (Static Weights)
    # =========================================================================
    print(f"\n[STEP 2] Walk-Forward Optimization ({LOOKBACK_MONTHS}-Month Lookback, {STEP_BARS//730}-Month Step)...")
    
    wfa_rets = np.zeros(n)
    start_idx = LOOKBACK_BARS
    num_steps = (n - start_idx) // STEP_BARS

    def_dh = 0.50; def_sh = 0.40; def_boi = 0.20
    w_d, w_s, w_b = 0.20, 0.55, 0.25 # Static global optimum
    
    warmup_ens = (w_d * dir_engines[def_dh][:start_idx]) + \
                 (w_s * sp_engines[def_sh][:start_idx]) + \
                 (w_b * brk_engines[def_boi][:start_idx])
    wfa_rets[:start_idx] = warmup_ens

    param_history = []
    
    for step in range(num_steps + 1):
        is_end = start_idx + step * STEP_BARS
        is_start = is_end - LOOKBACK_BARS
        oos_start = is_end
        oos_end = min(oos_start + STEP_BARS, n)
        if oos_start >= n: break

        best_is_sh = -999; best_params = None
        
        for dh in HYST_BANDS:
            d_arr = dir_engines[dh][is_start:is_end]
            for sh in HYST_BANDS:
                s_arr = sp_engines[sh][is_start:is_end]
                for boi in OI_GATES:
                    b_arr = brk_engines[boi][is_start:is_end]
                    
                    is_ens = (w_d * d_arr) + (w_s * s_arr) + (w_b * b_arr)
                    sh_val = fast_sharpe(is_ens)
                    
                    if sh_val > best_is_sh:
                        best_is_sh = sh_val
                        best_params = (dh, sh, boi)

        opt_dh, opt_sh, opt_boi = best_params
        
        wfa_rets[oos_start:oos_end] = (w_d * dir_engines[opt_dh][oos_start:oos_end]) + \
                                      (w_s * sp_engines[opt_sh][oos_start:oos_end]) + \
                                      (w_b * brk_engines[opt_boi][oos_start:oos_end])
        
        param_history.append({'date': eval_index[oos_start], 'sh': best_is_sh, 
                              'dh': opt_dh, 'sh_val': opt_sh, 'boi': opt_boi})
                              
        if step % 5 == 0 or step == num_steps:
            print(f"    Step {step}/{num_steps} | {eval_index[oos_start].date()} | IS Sharpe: {best_is_sh:.2f} | "
                  f"DH={opt_dh} SH={opt_sh} BOI={opt_boi}")

    # =========================================================================
    # Evaluation
    # =========================================================================
    wfa_sh_full, wfa_net, wfa_dd = fast_metrics(wfa_rets)
    wfa_sh_train = fast_sharpe(wfa_rets[:split_idx])
    wfa_sh_test = fast_sharpe(wfa_rets[split_idx:])

    print("\n" + "=" * 80)
    print("V40 ALPHA LAB: WALK-FORWARD RESULTS")
    print("=" * 80)
    print(f"Full 4-Year Sharpe:       {wfa_sh_full:.2f}")
    print(f"Train Sharpe (2022-2024): {wfa_sh_train:.2f}")
    print(f"Test Sharpe (2024-2026):  {wfa_sh_test:.2f}")
    print(f"Net Return:               {wfa_net:+.2f}%")
    print(f"Maximum Drawdown:         {wfa_dd:.2f}%")
    print("=" * 80)

    # Output Plot
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    wfa_cum = np.cumprod(1 + wfa_rets)
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1 + bench_h.values)

    ax1.plot(eval_index, wfa_cum, color='#00E5FF', linewidth=2.5,
             label=f'Apex v40 Micro-Regime (Sharpe: {wfa_sh_full:.2f}, Net: {wfa_net:+.1f}%)')
    ax1.plot(eval_index, bench_cum, color='#78909C', linewidth=1.5, linestyle=':', alpha=0.5,
             label='Buy & Hold Benchmark')
             
    ax1.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_title(f'Apex v40: Micro-Regime WFA (3M Lookback)', color='white', fontsize=14, fontweight='bold')
    ax1.grid(color='gray', linestyle=':', alpha=0.2)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1E222D')
    
    ph_df = pd.DataFrame(param_history).set_index('date')
    ph_df = ph_df.reindex(eval_index).fillna(method='ffill')
    plot_idx = eval_index[start_idx:]
    ph_plot = ph_df.loc[plot_idx]
    
    ax2.plot(plot_idx, ph_plot['dh'], label='DH Parameter', color='#2962FF', linewidth=2)
    ax2.plot(plot_idx, ph_plot['sh_val'], label='SH Parameter', color='#00C853', linewidth=2)
    ax2.plot(plot_idx, ph_plot['boi'], label='BOI Parameter', color='#FFAB00', linewidth=2)
    ax2.set_title('WFA Parameter Evolution (Micro-Regime Shifts)', color='white', fontsize=11)
    ax2.set_ylim(0, 0.8)
    ax2.grid(color='gray', linestyle=':', alpha=0.2)
    ax2.legend(loc='upper left', fontsize=9, ncol=3)

    out_path = os.path.join(RESULTS_DIR, 'apex_v40_alpha_lab.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] Equity curve to {out_path}")

if __name__ == '__main__':
    main()
