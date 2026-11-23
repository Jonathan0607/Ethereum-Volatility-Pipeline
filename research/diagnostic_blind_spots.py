#!/usr/bin/env python3
"""
Diagnostic script: analyze individual engine performance and identify blind spots.
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd

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

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR
    av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    return am / (av + 1e-8)

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

d = pickle.load(open(os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl'), 'rb'))
processed = compute_indicators(d, UNIVERSE)
first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
eval_index = first_df.index
n = len(eval_index)
wts = cross_weights(processed, UNIVERSE, eval_index)

dir_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, 0.50)
sp_rets = run_spread(processed, UNIVERSE, eval_index, n, 0.40)
brk_rets = run_breakout(processed, wts, UNIVERSE, n, 0.15)

print("=" * 70)
print("INDIVIDUAL ENGINE ANALYSIS")
print("=" * 70)
for name, rets in [('Directional', dir_rets), ('Spread', sp_rets), ('Breakout', brk_rets)]:
    rs = pd.Series(rets, index=eval_index)
    sh = fast_sharpe(rets)
    ret = (np.prod(1+rets)-1)*100
    print(f'\n{name}: Sharpe={sh:.3f}, TotalRet={ret:+.1f}%')
    for yr in [2022,2023,2024,2025,2026]:
        chunk = rs[rs.index.year == yr]
        if len(chunk) == 0: continue
        s = fast_sharpe(chunk.values)
        r = (np.prod(1+chunk.values)-1)*100
        print(f'  {yr}: Sharpe={s:.2f}, Ret={r:+.1f}%')

print("\n" + "=" * 70)
print("ENGINE CORRELATION MATRIX")
print("=" * 70)
ec = pd.DataFrame({'Dir':dir_rets, 'Spread':sp_rets, 'Brk':brk_rets})
print(ec.corr().round(3))

# CRITICAL: What fraction of time is each engine actually exposed to the market?
print("\n" + "=" * 70)
print("TURNOVER & FEE ANALYSIS")
print("=" * 70)

# Calculate turnover for directional engine
for name, rets in [('Directional', dir_rets), ('Spread', sp_rets), ('Breakout', brk_rets)]:
    # Count non-zero return hours (proxy for time in market)
    in_market = np.sum(np.abs(rets) > 1e-10) / len(rets) * 100
    total_abs_ret = np.sum(np.abs(rets))
    mean_abs_ret = np.mean(np.abs(rets))
    print(f'{name}: In-Market={in_market:.1f}%, TotalAbsRet={total_abs_ret:.4f}, MeanAbsRet={mean_abs_ret*10000:.2f}bps')

# NEW: Test what happens with different ensemble weights
print("\n" + "=" * 70)
print("WEIGHT SWEEP (static params DH=0.5, SH=0.4, BOI=0.15)")
print("=" * 70)
best_sh = -999
best_wts_combo = None
for wd in np.arange(0.0, 1.05, 0.05):
    for ws in np.arange(0.0, 1.05-wd, 0.05):
        wb = 1.0 - wd - ws
        if wb < -0.01: continue
        ens = wd*dir_rets + ws*sp_rets + wb*brk_rets
        sh = fast_sharpe(ens)
        if sh > best_sh:
            best_sh = sh
            best_wts_combo = (wd, ws, wb)

print(f"Best Static Weights: Dir={best_wts_combo[0]:.2f}, Spread={best_wts_combo[1]:.2f}, Brk={best_wts_combo[2]:.2f}")
print(f"Best Sharpe: {best_sh:.3f}")

# Also test: what if we just run the best single engine?
print("\n" + "=" * 70)
print("BEST SINGLE ENGINE vs ENSEMBLE")
print("=" * 70)
ens_20_55_25 = 0.20*dir_rets + 0.55*sp_rets + 0.25*brk_rets
print(f"Ensemble 20/55/25: Sharpe={fast_sharpe(ens_20_55_25):.3f}")
ens_best = best_wts_combo[0]*dir_rets + best_wts_combo[1]*sp_rets + best_wts_combo[2]*brk_rets
print(f"Ensemble Best:     Sharpe={fast_sharpe(ens_best):.3f}")

# CRITICAL NEW IDEA: What about a drawdown-controlled overlay?
# When the equity curve drops X% from peak, go to cash
print("\n" + "=" * 70)
print("DRAWDOWN-CONTROLLED OVERLAY (Cash Parking when DD > threshold)")
print("=" * 70)
for dd_thresh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    base = ens_20_55_25.copy()
    cum = np.ones(len(base))
    peak = 1.0
    controlled = np.zeros(len(base))
    in_cash = False
    recovery_bars = 0
    for t in range(len(base)):
        if t > 0:
            cum[t] = cum[t-1] * (1 + base[t])
        else:
            cum[t] = 1 + base[t]
        peak = max(peak, cum[t])
        dd = (cum[t] - peak) / peak
        
        if dd < -dd_thresh:
            in_cash = True
            recovery_bars = 0
        elif in_cash and dd > -dd_thresh * 0.5:
            # Re-enter when DD recovers to half the threshold
            in_cash = False
            
        if not in_cash:
            controlled[t] = base[t]
    
    sh_ctrl = fast_sharpe(controlled)
    ret_ctrl = (np.prod(1+controlled)-1)*100
    dd_ctrl = np.min((np.cumprod(1+controlled) - np.maximum.accumulate(np.cumprod(1+controlled))) / (np.maximum.accumulate(np.cumprod(1+controlled))+1e-8)) * 100
    print(f"  DD Threshold={dd_thresh*100:.0f}%: Sharpe={sh_ctrl:.3f}, Ret={ret_ctrl:+.1f}%, MaxDD={dd_ctrl:.1f}%")
