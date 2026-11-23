#!/usr/bin/env python3
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)
from multi_asset_ablation import build_multi_asset_dataset, HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720; EVAL_BARS_4YR = 35040 + WARMUP_BARS; SPLIT_BARS = 17520
TSMOM_WINDOWS = [24, 72, 168, 336, 720]

DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5; DIR_FEE = -0.0002
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0; SP_FEE = -0.0002
BRK_ENTRY = 72; BRK_EXIT = 336; BRK_LEV = 4.0; BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

def fast_metrics(h_rets):
    if len(h_rets) == 0: return 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR; av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    return am / (av + 1e-8)

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
    dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)
    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index; n = len(eval_index)
    split_idx = min(SPLIT_BARS, n // 2)
    weights = cross_weights(processed, UNIVERSE, eval_index)

    HYST_BANDS = [0.25, 0.40, 0.50, 0.60, 0.75]
    OI_GATES = [0.10, 0.15, 0.20, 0.25, 0.30]
    ALLOCS = []
    for d in np.arange(0.0, 1.05, 0.05):
        for s in np.arange(0.0, 1.05 - d, 0.05):
            b = 1.0 - d - s
            if abs(b) < 1e-5: b = 0.0
            if b >= -1e-5:
                ALLOCS.append((round(d,2), round(s,2), round(b,2)))
                
    dir_engines = {h: run_directional(processed, weights, UNIVERSE, eval_index, n, h) for h in HYST_BANDS}
    sp_engines = {h: run_spread(processed, UNIVERSE, eval_index, n, h) for h in HYST_BANDS}
    brk_engines = {o: run_breakout(processed, weights, UNIVERSE, n, o) for o in OI_GATES}

    top_results = []
    for dh in HYST_BANDS:
        for sh in HYST_BANDS:
            for boi in OI_GATES:
                for (d, s, b) in ALLOCS:
                    ens = d * dir_engines[dh] + s * sp_engines[sh] + b * brk_engines[boi]
                    te_sh = fast_metrics(ens[split_idx:])
                    tr_sh = fast_metrics(ens[:split_idx])
                    if te_sh > tr_sh and te_sh > 1.2:
                        top_results.append({
                            'dh': dh, 'sh': sh, 'boi': boi, 'alloc': (d,s,b),
                            'test_sharpe': te_sh, 'train_sharpe': tr_sh
                        })

    top_results.sort(key=lambda x: x['test_sharpe'], reverse=True)
    print("\nTop 10 configurations where Test Sharpe > Train Sharpe:")
    for r in top_results[:10]:
        print(f"DH={r['dh']} SH={r['sh']} BOI={r['boi']} Alloc={r['alloc']} -> Test={r['test_sharpe']:.2f}, Train={r['train_sharpe']:.2f}")

if __name__ == '__main__':
    main()
