#!/usr/bin/env python3
"""
APEX v43 DEEP AUDIT: Validate the top strategies for robustness.
Focus on strategies with the best Train/Test stability (least overfitting).
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

def main():
    print("=" * 80)
    print("APEX v43 DEEP AUDIT: ROBUSTNESS VALIDATION")
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

    dir_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, 0.50)
    sp_rets = run_spread(processed, UNIVERSE, eval_index, n, 0.40)
    
    macro_336 = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_trend_336'].values
    
    ens_best = 0.35*dir_rets + 0.65*sp_rets

    # =========================================================================
    # CHAMPION: Macro Filter (336h, scale=0) + DD Control (5%/2.5%)
    # Train Sharpe: 2.335, Test Sharpe: 1.321
    # =========================================================================
    # But let's audit whether DD thresholds are overfit.
    # Run a parameter stability test: sweep DD thresholds and see if the 
    # result is robust across a RANGE of thresholds, not just one.
    
    print("\n" + "=" * 80)
    print("AUDIT 1: DD Threshold Stability (with Macro 336h, scale=0)")
    print("=" * 80)
    
    # Apply macro filter first
    macro_filtered = np.zeros(n)
    for t in range(n):
        if macro_336[t] > 0.5:
            macro_filtered[t] = ens_best[t]
        else:
            macro_filtered[t] = 0.0  # fully cash in bear
    
    for dd_enter in np.arange(0.03, 0.16, 0.01):
        dd_exit = dd_enter * 0.5
        cum = 1.0; peak = 1.0; in_cash = False
        ctrl = np.zeros(n)
        for t in range(n):
            cum *= (1 + macro_filtered[t])
            peak = max(peak, cum)
            dd = (cum - peak) / peak
            if not in_cash and dd < -dd_enter:
                in_cash = True
            elif in_cash and dd > -dd_exit:
                in_cash = False
            if not in_cash:
                ctrl[t] = macro_filtered[t]
        
        sh_full = fast_sharpe(ctrl)
        sh_train = fast_sharpe(ctrl[:split_idx])
        sh_test = fast_sharpe(ctrl[split_idx:])
        ret_full = (np.prod(1+ctrl)-1)*100
        dd_full = np.min((np.cumprod(1+ctrl) - np.maximum.accumulate(np.cumprod(1+ctrl))) / 
                         (np.maximum.accumulate(np.cumprod(1+ctrl))+1e-8)) * 100
        marker = " <<<" if sh_full >= 1.5 else ""
        print(f"  DD={dd_enter*100:5.1f}%: Full={sh_full:.3f}, Train={sh_train:.3f}, Test={sh_test:.3f}, "
              f"Ret={ret_full:+.1f}%, MaxDD={dd_full:.1f}%{marker}")

    # =========================================================================
    # AUDIT 2: Macro SMA period sensitivity 
    # =========================================================================
    print("\n" + "=" * 80)
    print("AUDIT 2: Macro SMA Period Stability (no DD control, scale=0)")
    print("=" * 80)
    
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in UNIVERSE}).mean(axis=1)
    
    for sma_period in [168, 240, 336, 480, 600, 720, 960, 1440]:
        macro_sig = (idx_price > idx_price.rolling(sma_period).mean()).astype(float).fillna(0.0)
        macro_vals = macro_sig.iloc[WARMUP_BARS + (len(dataset[UNIVERSE[0]]) - len(idx_price)):].values
        # Align to eval_index length
        macro_vals = macro_vals[:n]
        
        filt = np.where(macro_vals > 0.5, ens_best, 0.0)
        sh_full = fast_sharpe(filt)
        sh_train = fast_sharpe(filt[:split_idx])
        sh_test = fast_sharpe(filt[split_idx:])
        ret_full = (np.prod(1+filt)-1)*100
        marker = " <<<" if sh_full >= 1.5 else ""
        print(f"  SMA={sma_period:5d}h: Full={sh_full:.3f}, Train={sh_train:.3f}, Test={sh_test:.3f}, "
              f"Ret={ret_full:+.1f}%{marker}")

    # =========================================================================
    # AUDIT 3: Weight sensitivity for Dir/Spread (no Breakout)
    # =========================================================================
    print("\n" + "=" * 80)
    print("AUDIT 3: Dir/Spread Weight Stability (no macro, no DD)")
    print("=" * 80)
    
    for wd in np.arange(0.20, 0.55, 0.05):
        ws = 1.0 - wd
        ens = wd*dir_rets + ws*sp_rets
        sh_full = fast_sharpe(ens)
        sh_train = fast_sharpe(ens[:split_idx])
        sh_test = fast_sharpe(ens[split_idx:])
        ret_full = (np.prod(1+ens)-1)*100
        marker = " <<<" if sh_full >= 1.5 else ""
        print(f"  Dir={wd:.2f}/Spread={ws:.2f}: Full={sh_full:.3f}, Train={sh_train:.3f}, Test={sh_test:.3f}, "
              f"Ret={ret_full:+.1f}%{marker}")

    # =========================================================================
    # AUDIT 4: Rolling 12-month Sharpe
    # =========================================================================
    print("\n" + "=" * 80)
    print("AUDIT 4: Rolling 12-Month Sharpe (Champion Strategy)")
    print("=" * 80)
    
    # Champion: Macro 336h scale=0, DD 5%/2.5%, weights 35/65
    macro_filtered2 = np.where(macro_336 > 0.5, ens_best, 0.0)
    cum = 1.0; peak = 1.0; in_cash = False
    champion = np.zeros(n)
    for t in range(n):
        cum *= (1 + macro_filtered2[t])
        peak = max(peak, cum)
        dd = (cum - peak) / peak
        if not in_cash and dd < -0.05:
            in_cash = True
        elif in_cash and dd > -0.025:
            in_cash = False
        if not in_cash:
            champion[t] = macro_filtered2[t]
    
    rs = pd.Series(champion, index=eval_index)
    roll_sh = rs.rolling(8760).apply(lambda x: fast_sharpe(x.values), raw=False)
    
    for date in roll_sh.dropna().index[::2190]:  # every ~3 months
        print(f"  {date.date()}: Rolling 12M Sharpe = {roll_sh.loc[date]:.2f}")
    
    # Count months with negative returns
    monthly = rs.resample('M').apply(lambda x: (np.prod(1+x)-1)*100)
    neg_months = (monthly < 0).sum()
    pos_months = (monthly >= 0).sum()
    print(f"\n  Monthly Win Rate: {pos_months}/{pos_months+neg_months} = {pos_months/(pos_months+neg_months)*100:.1f}%")
    print(f"  Best Month:  {monthly.max():+.1f}%")
    print(f"  Worst Month: {monthly.min():+.1f}%")
    
    # =========================================================================
    # FINAL: The most robust compromise
    # Best Test Sharpe stability: DD_Ctrl(4%/2%) has Test=1.419
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON: TOP CANDIDATES")
    print("=" * 80)
    
    candidates = {
        'A: Macro336+DD5%': champion,
        'B: DD_Ctrl(4%/2%) only': None,
        'C: Macro336 only (scale=0)': macro_filtered2,
        'D: Pure Dir35/Sp65': ens_best,
    }
    
    # Build candidate B
    cum = 1.0; peak = 1.0; in_cash = False
    ctrl_b = np.zeros(n)
    for t in range(n):
        cum *= (1 + ens_best[t])
        peak = max(peak, cum)
        dd = (cum - peak) / peak
        if not in_cash and dd < -0.04:
            in_cash = True
        elif in_cash and dd > -0.02:
            in_cash = False
        if not in_cash:
            ctrl_b[t] = ens_best[t]
    candidates['B: DD_Ctrl(4%/2%) only'] = ctrl_b
    
    print(f"{'Candidate':<30} {'Full':>7} {'Train':>7} {'Test':>7} {'Ret':>9} {'MaxDD':>8} {'Test/Train':>10}")
    print("-" * 85)
    for name, rets in candidates.items():
        sh_full, ret_full, dd_full = fast_metrics(rets)
        sh_train = fast_sharpe(rets[:split_idx])
        sh_test = fast_sharpe(rets[split_idx:])
        ratio = sh_test / sh_train if sh_train > 0 else 0
        print(f"{name:<30} {sh_full:>7.3f} {sh_train:>7.3f} {sh_test:>7.3f} {ret_full:>+8.1f}% {dd_full:>7.1f}% {ratio:>9.2f}")
    
    # =========================================================================
    # Plot all candidates
    # =========================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    colors = ['#00E5FF', '#FF6B6B', '#00C853', '#FFAB00']
    for (name, rets), color in zip(candidates.items(), colors):
        cum = np.cumprod(1 + rets)
        sh = fast_sharpe(rets)
        ret = (cum[-1] - 1)*100
        ax.plot(eval_index, cum, color=color, linewidth=2,
                label=f'{name} (Sharpe: {sh:.2f}, Ret: {ret:+.0f}%)')
    
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1 + bench_h.values)
    ax.plot(eval_index, bench_cum, color='#78909C', linewidth=1, linestyle=':', alpha=0.4, label='Buy & Hold')
    
    ax.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1, alpha=0.5, label='Train/Test Split')
    ax.set_title('Apex v43: Blind Spot Exploiter — Candidate Comparison', color='white', fontsize=14, fontweight='bold')
    ax.grid(color='gray', linestyle=':', alpha=0.2)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, facecolor='#1E222D')
    
    out_path = os.path.join(RESULTS_DIR, 'apex_v43_audit.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] {out_path}")

if __name__ == '__main__':
    main()
