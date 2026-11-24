#!/usr/bin/env python3
"""
Quick sweep of nearby SMA periods to find the robust plateau.
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, PROJECT_ROOT)
from multi_asset_ablation import HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
TSMOM_WINDOWS = [24, 72, 168, 336, 720]
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
MAKER_FEE = -0.0002

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    return (np.mean(h_rets)*HOURS_PER_YEAR) / (np.std(h_rets)*np.sqrt(HOURS_PER_YEAR)+1e-8)

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
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w); vw = c.pct_change().rolling(w).std() * np.sqrt(w)
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

d = pickle.load(open(os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl'), 'rb'))
processed = compute_indicators(d, UNIVERSE)
first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
eval_index = first_df.index
n = len(eval_index)
wts = cross_weights(processed, UNIVERSE, eval_index)
split_idx = min(17520, n // 2)

dir_rets = run_directional(processed, wts, UNIVERSE, eval_index, n, 0.50)
sp_rets = run_spread(processed, UNIVERSE, eval_index, n, 0.40)
ens_best = 0.35*dir_rets + 0.65*sp_rets

idx_price = pd.DataFrame({s: d[s]['close'] for s in UNIVERSE}).mean(axis=1)

print(f"{'SMA':>6} {'Full':>7} {'Train':>7} {'Test':>7} {'T/Tr':>6} {'Ret':>9} {'MaxDD':>8}")
print("-" * 60)

for sma in range(360, 721, 24):
    macro = (idx_price > idx_price.rolling(sma).mean()).astype(float).fillna(0.0)
    macro_vals = macro.reindex(eval_index).fillna(0.0).values
    
    filt = np.where(macro_vals > 0.5, ens_best, 0.0)
    sh_full = fast_sharpe(filt)
    sh_train = fast_sharpe(filt[:split_idx])
    sh_test = fast_sharpe(filt[split_idx:])
    ratio = sh_test / sh_train if sh_train > 0 else 0
    ret = (np.prod(1+filt)-1)*100
    cum = np.cumprod(1+filt)
    dd = np.min((cum - np.maximum.accumulate(cum)) / (np.maximum.accumulate(cum)+1e-8)) * 100
    marker = " <<<" if sh_full >= 1.5 else ""
    print(f"{sma:>6} {sh_full:>7.3f} {sh_train:>7.3f} {sh_test:>7.3f} {ratio:>6.2f} {ret:>+8.1f}% {dd:>7.1f}%{marker}")
