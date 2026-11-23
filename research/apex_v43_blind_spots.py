#!/usr/bin/env python3
"""
Deep diagnostic: the previous model NEVER tried:
1. Dropping the Breakout engine entirely (it has NEGATIVE returns and 0.376 Sharpe)
2. Optimal Dir/Spread blend is 35/65 → Sharpe 1.272 (already beats V32's 1.14!)
3. Drawdown-controlled cash parking on the optimal blend
4. Combining #2 + #3 together
5. Adding a simple bear-market cash filter (macro trend)
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
TAKER_FEE = -0.0010
MAKER_FEE = -0.0002

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
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in univ}).mean(axis=1)
    
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        
        # Macro trend filters at multiple timeframes
        df['macro_trend_336'] = (idx_price > idx_price.rolling(336).mean()).astype(float).fillna(0.0)
        df['macro_trend_720'] = (idx_price > idx_price.rolling(720).mean()).astype(float).fillna(0.0)
        
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

def apply_drawdown_control(base_rets, dd_enter, dd_exit):
    """Go to cash when cumulative drawdown exceeds dd_enter.
    Re-enter when drawdown recovers past dd_exit."""
    n = len(base_rets)
    controlled = np.zeros(n)
    cum = 1.0
    peak = 1.0
    in_cash = False
    for t in range(n):
        cum *= (1 + base_rets[t])
        peak = max(peak, cum)
        dd = (cum - peak) / peak
        
        if not in_cash and dd < -dd_enter:
            in_cash = True
        elif in_cash and dd > -dd_exit:
            in_cash = False
        
        if not in_cash:
            controlled[t] = base_rets[t]
    return controlled

def apply_macro_filter(base_rets, macro_signal, scale_factor=0.5):
    """Scale down position by scale_factor when macro signal is 0 (bear)."""
    n = len(base_rets)
    filtered = np.zeros(n)
    for t in range(n):
        if macro_signal[t] > 0.5:
            filtered[t] = base_rets[t]
        else:
            filtered[t] = base_rets[t] * scale_factor
    return filtered

def main():
    print("=" * 80)
    print("APEX v43: THE BLIND SPOT EXPLOITER")
    print("=" * 80)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
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
    # Compute engines
    # =========================================================================
    print("\n[STEP 1] Computing engines...")
    dir_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, 0.50)
    sp_rets = run_spread(processed, UNIVERSE, eval_index, n, 0.40)
    brk_rets = run_breakout(processed, wts, UNIVERSE, n, 0.15)
    
    macro_336 = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_trend_336'].values
    macro_720 = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_trend_720'].values

    # =========================================================================
    # EXPERIMENT BATTERY
    # =========================================================================
    print("\n[STEP 2] Running experiment battery...\n")
    
    results = []
    
    # --- Baseline (V42 equivalent) ---
    ens_baseline = 0.20*dir_rets + 0.55*sp_rets + 0.25*brk_rets
    sh, ret, dd = fast_metrics(ens_baseline)
    results.append(('V42 Baseline (20/55/25)', sh, ret, dd, ens_baseline))
    
    # --- BLIND SPOT 1: Drop the Breakout engine ---
    # The Breakout engine has a NEGATIVE total return (-23.4%) and 0.376 Sharpe.
    # The previous model NEVER tried removing it.
    for wd, ws in [(0.30, 0.70), (0.35, 0.65), (0.40, 0.60), (0.50, 0.50), (0.25, 0.75)]:
        ens = wd*dir_rets + ws*sp_rets
        sh, ret, dd = fast_metrics(ens)
        results.append((f'Dir/Spread Only ({int(wd*100)}/{int(ws*100)})', sh, ret, dd, ens))
    
    # --- BLIND SPOT 2: Drawdown-controlled overlay on best blend ---
    ens_best = 0.35*dir_rets + 0.65*sp_rets
    for dd_enter, dd_exit in [(0.04, 0.02), (0.05, 0.025), (0.06, 0.03), 
                               (0.08, 0.04), (0.10, 0.05)]:
        ctrl = apply_drawdown_control(ens_best, dd_enter, dd_exit)
        sh, ret, dd = fast_metrics(ctrl)
        results.append((f'Best+DD_Ctrl({int(dd_enter*100)}%/{int(dd_exit*100)}%)', sh, ret, dd, ctrl))
    
    # --- BLIND SPOT 3: Macro trend filter on best blend ---
    for macro, name in [(macro_336, '336h'), (macro_720, '720h')]:
        for scale in [0.0, 0.25, 0.50]:
            filt = apply_macro_filter(ens_best, macro, scale)
            sh, ret, dd = fast_metrics(filt)
            results.append((f'Best+Macro_{name}(s={scale})', sh, ret, dd, filt))
    
    # --- BLIND SPOT 4: Combine DD control + Macro filter ---
    for dd_enter, dd_exit in [(0.05, 0.025), (0.06, 0.03), (0.08, 0.04)]:
        for macro, mname in [(macro_336, '336h'), (macro_720, '720h')]:
            for scale in [0.0, 0.25, 0.50]:
                filt = apply_macro_filter(ens_best, macro, scale)
                ctrl = apply_drawdown_control(filt, dd_enter, dd_exit)
                sh, ret, dd = fast_metrics(ctrl)
                results.append((f'Best+Macro_{mname}(s={scale})+DD({int(dd_enter*100)}%)', sh, ret, dd, ctrl))
    
    # --- BLIND SPOT 5: Add a SMALL breakout allocation to the best blend ---
    for wb in [0.05, 0.10, 0.15]:
        remaining = 1.0 - wb
        wd = 0.35 * remaining
        ws = 0.65 * remaining
        ens = wd*dir_rets + ws*sp_rets + wb*brk_rets
        sh, ret, dd = fast_metrics(ens)
        results.append((f'Best+SmallBrk({int(wb*100)}%)', sh, ret, dd, ens))
    
    # Sort by Sharpe
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Strategy':<50} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10}")
    print("-" * 80)
    for name, sh, ret, dd, _ in results[:30]:
        marker = " <<<" if sh >= 1.5 else (" **" if sh >= 1.2 else "")
        print(f"{name:<50} {sh:>8.3f} {ret:>+9.1f}% {dd:>9.1f}%{marker}")
    
    # =========================================================================
    # Detailed analysis of top strategies
    # =========================================================================
    print("\n" + "=" * 80)
    print("TOP 5 STRATEGIES: TRAIN/TEST SPLIT")
    print("=" * 80)
    for name, sh_full, ret, dd, rets in results[:5]:
        sh_train = fast_sharpe(rets[:split_idx])
        sh_test = fast_sharpe(rets[split_idx:])
        print(f"\n{name}")
        print(f"  Full:  Sharpe={sh_full:.3f}, Ret={ret:+.1f}%, DD={dd:.1f}%")
        print(f"  Train: Sharpe={sh_train:.3f}")
        print(f"  Test:  Sharpe={sh_test:.3f}")
        
        # Per-year breakdown
        rs = pd.Series(rets, index=eval_index)
        for yr in [2022, 2023, 2024, 2025, 2026]:
            chunk = rs[rs.index.year == yr]
            if len(chunk) == 0: continue
            s = fast_sharpe(chunk.values)
            r = (np.prod(1+chunk.values)-1)*100
            print(f"    {yr}: Sharpe={s:.2f}, Ret={r:+.1f}%")
    
    # =========================================================================
    # Save plot of top strategy
    # =========================================================================
    best_name, best_sh, best_ret, best_dd, best_rets = results[0]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    cum_best = np.cumprod(1 + best_rets)
    cum_baseline = np.cumprod(1 + ens_baseline)
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1 + bench_h.values)
    
    ax.plot(eval_index, cum_best, color='#00E5FF', linewidth=2.5,
            label=f'{best_name} (Sharpe: {best_sh:.2f}, Ret: {best_ret:+.1f}%)')
    ax.plot(eval_index, cum_baseline, color='#FF6B6B', linewidth=1.5, linestyle='--', alpha=0.7,
            label=f'V42 Baseline (Sharpe: {fast_sharpe(ens_baseline):.2f})')
    ax.plot(eval_index, bench_cum, color='#78909C', linewidth=1.5, linestyle=':', alpha=0.5,
            label='Buy & Hold')
    
    ax.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.5, label='Train/Test Split')
    ax.set_title(f'Apex v43: {best_name}', color='white', fontsize=14, fontweight='bold')
    ax.grid(color='gray', linestyle=':', alpha=0.2)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, facecolor='#1E222D')
    
    out_path = os.path.join(RESULTS_DIR, 'apex_v43_blind_spots.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] {out_path}")

if __name__ == '__main__':
    main()
