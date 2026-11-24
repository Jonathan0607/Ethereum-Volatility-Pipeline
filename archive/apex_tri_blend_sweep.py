#!/usr/bin/env python3
"""
===================================================================================
APEX TRI-BLEND PORTFOLIO OPTIMIZATION SWEEP
===================================================================================
Sweeps capital allocation weights across the three fundamental engines:
1. Directional Engine (5.0 bps Taker, True 8h Funding)
2. Spread Engine (2.0 bps Maker, True 8h Funding)
3. Funding Arb Engine (z > 2.5, 2.0 bps Maker, True 8h Funding)

Constraint: W_dir + W_spread + W_arb = 1.0 (Step size: 0.05)
Macro Filter: Daily 21d SMA applied ONLY to the Core (Dir + Spread).
===================================================================================
"""
import os, sys, pickle, warnings, time, itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

HOURS_PER_YEAR = 8760.0
UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
TSMOM_WINDOWS = [24, 72, 168, 336, 720]
MACRO_DAILY_SMA = 21

# Core signal weights (tuned mix from ablation champion)
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
DIR_HYST, SPREAD_HYST = 0.50, 0.40
LEVERAGE = 2.5

# Fee structure
DIR_TAKER_FEE = 0.0005
SPREAD_MAKER_FEE = 0.0002
ARB_MAKER_FEE = 0.0002

GARCH_LOOKBACK = 720
GARCH_REFIT = 168

ARB_Z_THRESHOLD = 2.5
ARB_LOOKBACK = 720
ARB_MAX_POS = 0.5
ARB_EXIT_MULT = 0.5
ARB_SCALE = 2.0

# ============================================================================
# UTILITIES
# ============================================================================

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    return (np.mean(h_rets) * HOURS_PER_YEAR) / (np.std(h_rets) * np.sqrt(HOURS_PER_YEAR) + 1e-8)

def fast_metrics(h_rets):
    if len(h_rets) == 0: return {'sharpe': 0, 'ret': 0, 'dd': 0}
    sh = fast_sharpe(h_rets)
    cum = np.cumprod(1.0 + h_rets)
    cummax = np.maximum.accumulate(cum)
    dd = np.min((cum - cummax) / (cummax + 1e-8)) * 100.0
    ret = (cum[-1] - 1.0) * 100.0
    return {'sharpe': sh, 'ret': ret, 'dd': dd}

# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_features(dataset, univ):
    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in univ}).mean(axis=1)
    daily_idx = idx_price.resample('1D').last()
    daily_sma = daily_idx.rolling(MACRO_DAILY_SMA).mean()
    daily_regime = (daily_idx > daily_sma).astype(float).shift(1).fillna(0.0)
    macro_hourly = daily_regime.reindex(idx_price.index, method='ffill').fillna(0.0)

    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        df['macro_signal'] = macro_hourly

        vol_z = (v - v.rolling(720, min_periods=24).mean()) / (v.rolling(720, min_periods=24).std() + 1e-8)
        ret_cum = c.pct_change(48).fillna(0.0)
        synthetic_oi = 5000000.0 * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
        ret_24 = c.pct_change(24).fillna(0.0)
        synthetic_fr = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)

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

def compute_garch_weights(processed, univ, eval_index):
    from arch import arch_model
    n = len(eval_index)
    garch_vol = {s: np.full(n, np.nan) for s in univ}

    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        rets = df['hourly_return'].values * 100 
        last_omega, last_alpha, last_beta = 0.01, 0.05, 0.90
        cond_var = np.var(rets[:GARCH_LOOKBACK]) if len(rets) >= GARCH_LOOKBACK else 1.0

        for t in range(n):
            if t < GARCH_LOOKBACK:
                garch_vol[s][t] = np.std(rets[:max(t+1, 24)])
                continue
            if t % GARCH_REFIT == 0:
                try:
                    window = rets[t-GARCH_LOOKBACK:t]
                    model = arch_model(window, vol='GARCH', p=1, q=1, mean='Zero', rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    last_omega = res.params.get('omega', 0.01)
                    last_alpha = res.params.get('alpha[1]', 0.05)
                    last_beta = res.params.get('beta[1]', 0.90)
                except:
                    pass
            cond_var = last_omega + last_alpha * (rets[t-1]**2) + last_beta * cond_var
            garch_vol[s][t] = np.sqrt(max(cond_var, 1e-8))

    inv_garch = {}
    for s in univ:
        gv = np.array(garch_vol[s])
        gv = np.where(np.isnan(gv) | (gv < 1e-8), 1.0, gv)
        inv_garch[s] = 1.0 / gv
    total = sum(inv_garch[s] for s in univ)
    total = np.where(total < 1e-8, 1.0, total)
    return {s: inv_garch[s] / total for s in univ}

# ============================================================================
# COMPONENT ENGINES (UNWEIGHTED)
# ============================================================================

def generate_directional_stream(processed, dataset, univ, eval_index, n, wts):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps_dir = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps_dir[s] = DIR_WT*df['tsmom'].values + DIR_WC*df['carry'].values + DIR_WO*df['oi_regime'].values

    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps_dir[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5

    port_dir = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values
        fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values
        w = wts[s]; comp = comps_dir[s]; x = xs[s]
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * LEVERAGE
            if abs(ideal - cur) >= DIR_HYST: cur = ideal
            tgt[t] = cur
            
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        p_ret = (ret * shifted) - (turn * DIR_TAKER_FEE)
        p_fund = np.where(funding_hours, -shifted * fr, 0.0)
        port_dir += p_ret + p_fund
        
    return port_dir

def generate_spread_stream(processed, dataset, univ, eval_index, n):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps_sp = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps_sp[s] = SP_WT*df['tsmom'].values + SP_WC*df['carry'].values + SP_WO*df['oi_regime'].values

    cur_pos = {s: 0.0 for s in univ}
    port_sp = np.zeros(n)
    for t in range(1, n):
        sc = sorted([(s, comps_sp[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        top_s = sc[0][0]; bot_s = sc[-1][0]
        conv = abs(sc[0][1] - sc[-1][1])
        ideal = {s: 0.0 for s in univ}
        ideal[top_s] = conv * 1.0; ideal[bot_s] = -conv * 1.0
        for s in univ:
            old = cur_pos[s]
            if abs(ideal[s] - cur_pos[s]) >= SPREAD_HYST: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port_sp[t] -= turn * SPREAD_MAKER_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port_sp[t] += ret * cur_pos[s]
            if funding_hours[t]:
                fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values[t]
                port_sp[t] += -cur_pos[s] * fr
                
    return port_sp

def generate_arb_stream(dataset, univ, eval_index, n):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    port_arb = np.zeros(n)

    for s in univ:
        df = dataset[s].iloc[WARMUP_BARS:]
        fr_raw = df['funding_rate'].values
        ret = df['close'].pct_change().fillna(0.0).values

        fr_shifted = pd.Series(fr_raw).shift(1).fillna(0.0)
        rolling_mean = fr_shifted.rolling(ARB_LOOKBACK, min_periods=72).mean().fillna(0.0).values
        rolling_std = fr_shifted.rolling(ARB_LOOKBACK, min_periods=72).std().fillna(0.0).values
        z_score = (fr_shifted.values - rolling_mean) / (rolling_std + 1e-10)

        cur = 0.0
        tgt = np.zeros(n)
        for t in range(ARB_LOOKBACK, n):
            z = z_score[t]
            if z > ARB_Z_THRESHOLD:
                ideal = -min((z - ARB_Z_THRESHOLD) / ARB_SCALE, ARB_MAX_POS)
            elif z < -ARB_Z_THRESHOLD:
                ideal = min((-z - ARB_Z_THRESHOLD) / ARB_SCALE, ARB_MAX_POS)
            elif abs(z) < ARB_Z_THRESHOLD * ARB_EXIT_MULT:
                ideal = 0.0
            else:
                ideal = cur
            if abs(ideal - cur) > 0.05:
                cur = ideal
            tgt[t] = cur

        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        
        price_pnl = ret * shifted / len(univ)
        fee_cost = turn * ARB_MAKER_FEE / len(univ)
        funding_pnl = np.where(funding_hours, -shifted * fr_raw / len(univ), 0.0)

        port_arb += price_pnl - fee_cost + funding_pnl
        
    return port_arb

# ============================================================================
# MAIN SWEEP
# ============================================================================
def main():
    print("=" * 115)
    print("APEX TRI-BLEND PORTFOLIO OPTIMIZATION SWEEP")
    print("=" * 115)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_features(dataset, UNIVERSE)
    eval_index = processed[UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    macro_regime = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values
    
    # 1. GARCH Weights
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'garch_weights_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            wts_garch = pickle.load(f)
    else:
        print("Computing GARCH(1,1) weights...")
        wts_garch = compute_garch_weights(processed, UNIVERSE, eval_index)
        with open(cache_path, 'wb') as f:
            pickle.dump(wts_garch, f)

    # 2. Generate Base Streams
    print("Generating pure Directional stream (5.0 bps Taker)...")
    stream_dir_raw = generate_directional_stream(processed, dataset, UNIVERSE, eval_index, n, wts_garch)
    
    print("Generating pure Spread stream (2.0 bps Maker)...")
    stream_sp_raw = generate_spread_stream(processed, dataset, UNIVERSE, eval_index, n)
    
    print("Generating pure Funding Arb stream (2.0 bps Maker)...")
    stream_arb = generate_arb_stream(dataset, UNIVERSE, eval_index, n)

    # 3. Apply Macro Filter to Core components
    stream_dir = stream_dir_raw * macro_regime
    stream_sp = stream_sp_raw * macro_regime

    # 4. Sweep Constraints: W_dir + W_sp + W_arb = 1.0 (steps of 0.05)
    print("\nSweeping portfolio weights...")
    steps = np.round(np.arange(0.0, 1.05, 0.05), 2)
    
    results = []
    
    for w_dir in steps:
        for w_sp in steps:
            w_arb = round(1.0 - w_dir - w_sp, 2)
            if w_arb < 0.0 or w_arb > 1.0:
                continue
                
            blended = (w_dir * stream_dir) + (w_sp * stream_sp) + (w_arb * stream_arb)
            
            m = fast_metrics(blended)
            tr = fast_sharpe(blended[:split_idx])
            te = fast_sharpe(blended[split_idx:])
            tt = te / tr if abs(tr) > 0.01 else 0.0
            
            results.append({
                'w_dir': w_dir,
                'w_sp': w_sp,
                'w_arb': w_arb,
                'full_sh': m['sharpe'],
                'train_sh': tr,
                'test_sh': te,
                'tt_ratio': tt,
                'ret': m['ret'],
                'dd': m['dd']
            })

    # Sort by Full Sharpe
    results.sort(key=lambda x: x['full_sh'], reverse=True)

    print("\n" + "=" * 115)
    print("TOP 10 TRI-BLEND PORTFOLIO WEIGHTS")
    print("=" * 115)
    print(f"{'Weights (Dir/Sp/Arb)':22s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 115)
    
    for i, r in enumerate(results[:10]):
        w_str = f"{r['w_dir']:.2f} / {r['w_sp']:.2f} / {r['w_arb']:.2f}"
        print(f"{w_str:22s} | {r['full_sh']:+6.2f} | {r['train_sh']:+6.2f} | {r['test_sh']:+6.2f} | {r['tt_ratio']:5.2f} | {r['ret']:+7.1f}% | {r['dd']:6.1f}%")
        
    print("\n" + "=" * 115)
    print("TOP 10 COMBINATIONS (SORTED BY TEST OOS SHARPE)")
    print("=" * 115)
    results.sort(key=lambda x: x['test_sh'], reverse=True)
    for i, r in enumerate(results[:10]):
        w_str = f"{r['w_dir']:.2f} / {r['w_sp']:.2f} / {r['w_arb']:.2f}"
        print(f"{w_str:22s} | {r['full_sh']:+6.2f} | {r['train_sh']:+6.2f} | {r['test_sh']:+6.2f} | {r['tt_ratio']:5.2f} | {r['ret']:+7.1f}% | {r['dd']:6.1f}%")

if __name__ == '__main__':
    main()
