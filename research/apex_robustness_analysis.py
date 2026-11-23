#!/usr/bin/env python3
"""
===================================================================================
APEX STRUCTURAL ROBUSTNESS ANALYSIS
===================================================================================
Honest assessment of what is real alpha vs. what is parameter snooping.

Tests:
1. SIGNAL ISOLATION: Strip to pure TSMOM only. Do Carry/OI actually add alpha,
   or are they just redundant price transforms adding overfitting degrees of freedom?
2. NAIVE PARAMETER TEST: Run with completely naive/default parameters that no one
   could accuse of being optimized. If alpha survives, the structure is real.
3. PARAMETER SENSITIVITY: Perturb each key parameter ±30-50%. If Sharpe swings
   wildly, that parameter is overfit. If stable, the signal is structural.
4. ROLLING WINDOW STABILITY: 6-month rolling Sharpe to detect regime concentration.
5. ASSET SUBSET ROBUSTNESS: Test all 2-asset and single-asset subsets.

Constraints:
- Zero look-ahead: Daily 21d SMA Close, shift(1) day lag
- Realistic fees: 5.0 bps taker (directional), 2.0 bps maker (spread)
- True 8-hour Binance funding debits (00/08/16 UTC)
===================================================================================
"""
import os, sys, pickle, warnings, time, itertools
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

FULL_UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
TSMOM_WINDOWS = [24, 72, 168, 336, 720]

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# UTILITIES
# ============================================================================

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

# ============================================================================
# GENERIC ENGINE: Parameterized for sweep
# ============================================================================

def compute_features(dataset, univ, macro_sma_days=21):
    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in univ}).mean(axis=1)
    daily_idx = idx_price.resample('1D').last()
    daily_sma = daily_idx.rolling(macro_sma_days).mean()
    daily_regime = (daily_idx > daily_sma).astype(float).shift(1).fillna(0.0)
    macro_regime_hourly = daily_regime.reindex(idx_price.index, method='ffill').fillna(0.0)
    
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        df['macro_signal'] = macro_regime_hourly
        
        # Synthetic OI
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

def run_strategy(processed, dataset, univ, eval_index, n, macro_regime,
                 dir_wt=0.30, dir_wc=0.50, dir_wo=0.20,
                 sp_wt=0.30, sp_wc=0.00, sp_wo=0.70,
                 dir_weight=0.35, spread_weight=0.65,
                 dir_hyst=0.50, spread_hyst=0.40,
                 leverage=2.5,
                 dir_fee_bps=5.0, sp_fee_bps=2.0):
    """Fully parameterized strategy engine."""
    dir_fee = dir_fee_bps / 10000.0
    sp_fee = sp_fee_bps / 10000.0
    funding_hours = eval_index.hour.isin([0, 8, 16])
    
    # ATR weights
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in univ}, index=eval_index)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0/len(univ))
    wts = {s: w[s].values for s in univ}
    
    # Directional composites
    comps_dir = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps_dir[s] = dir_wt*df['tsmom'].values + dir_wc*df['carry'].values + dir_wo*df['oi_regime'].values
    
    # Cross-sectional ranking
    xs = {s: np.ones(n) for s in univ}
    if len(univ) >= 3:
        for t in range(1, n):
            sc = sorted([(s, comps_dir[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
            xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5
    elif len(univ) == 2:
        for t in range(1, n):
            sc = sorted([(s, comps_dir[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
            xs[sc[0][0]][t] = 1.25; xs[sc[1][0]][t] = 0.75
    
    port_dir = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values; wv = wts[s]; comp = comps_dir[s]; x = xs[s]
        fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            w_val = wv[t] if not np.isnan(wv[t]) else 0.0
            c_val = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = c_val * w_val * x[t] * leverage
            if abs(ideal - cur) >= dir_hyst: cur = ideal
            tgt[t] = cur
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        p_ret = (ret * shifted) - (turn * dir_fee)
        p_fund = np.where(funding_hours, -shifted * fr, 0.0)
        port_dir += p_ret + p_fund

    # Spread (only when 2+ assets)
    port_sp = np.zeros(n)
    if len(univ) >= 2:
        comps_sp = {}
        for s in univ:
            df = processed[s].iloc[WARMUP_BARS:]
            comps_sp[s] = sp_wt*df['tsmom'].values + sp_wc*df['carry'].values + sp_wo*df['oi_regime'].values
        cur_pos = {s: 0.0 for s in univ}
        for t in range(1, n):
            sc = sorted([(s, comps_sp[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
            top_s = sc[0][0]; bot_s = sc[-1][0]
            conv = abs(sc[0][1] - sc[-1][1])
            ideal = {s: 0.0 for s in univ}
            ideal[top_s] = conv * 1.0; ideal[bot_s] = -conv * 1.0
            for s in univ:
                old = cur_pos[s]
                if abs(ideal[s] - cur_pos[s]) >= spread_hyst: cur_pos[s] = ideal[s]
                turn = abs(cur_pos[s] - old)
                if turn > 0.001: port_sp[t] -= turn * sp_fee
                ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
                port_sp[t] += ret * cur_pos[s]
                if funding_hours[t]:
                    fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values[t]
                    port_sp[t] += -cur_pos[s] * fr

    raw = dir_weight * port_dir + spread_weight * port_sp
    return raw * macro_regime

def main():
    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)
    
    processed = compute_features(dataset, FULL_UNIVERSE)
    eval_index = processed[FULL_UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    macro_regime = processed[FULL_UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values

    def eval_strat(**kwargs):
        p = run_strategy(processed, dataset, kwargs.pop('univ', FULL_UNIVERSE),
                         eval_index, n, macro_regime, **kwargs)
        sh, ret, dd = fast_metrics(p)
        tr = fast_sharpe(p[:split_idx])
        te = fast_sharpe(p[split_idx:])
        return sh, tr, te, te/tr if tr > 0.01 else 0.0, ret, dd, p

    # ===========================================================================
    # TEST 1: SIGNAL ISOLATION
    # ===========================================================================
    print("=" * 110)
    print("TEST 1: SIGNAL ISOLATION — Which signals carry real alpha vs. redundant noise?")
    print("=" * 110)
    print("If Carry and OI (which are just price/volume transforms) add no marginal alpha,")
    print("they are overfitting degrees of freedom and should be removed.\n")
    
    signal_configs = [
        ("TSMOM Only",        dict(dir_wt=1.0, dir_wc=0.0, dir_wo=0.0, sp_wt=1.0, sp_wc=0.0, sp_wo=0.0)),
        ("TSMOM + Carry",     dict(dir_wt=0.50, dir_wc=0.50, dir_wo=0.0, sp_wt=0.50, sp_wc=0.50, sp_wo=0.0)),
        ("TSMOM + OI",        dict(dir_wt=0.50, dir_wc=0.0, dir_wo=0.50, sp_wt=0.50, sp_wc=0.0, sp_wo=0.50)),
        ("Current Tuned Mix", dict(dir_wt=0.30, dir_wc=0.50, dir_wo=0.20, sp_wt=0.30, sp_wc=0.00, sp_wo=0.70)),
        ("Equal Weight All",  dict(dir_wt=0.33, dir_wc=0.33, dir_wo=0.34, sp_wt=0.33, sp_wc=0.33, sp_wo=0.34)),
    ]
    
    print(f"{'Config':22s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 75)
    for name, cfg in signal_configs:
        sh, tr, te, tt, ret, dd, _ = eval_strat(**cfg)
        print(f"{name:22s} | {sh:6.2f} | {tr:6.2f} | {te:6.2f} | {tt:5.2f} | {ret:+7.1f}% | {dd:6.1f}%")

    # ===========================================================================
    # TEST 2: NAIVE PARAMETER TEST
    # ===========================================================================
    print("\n" + "=" * 110)
    print("TEST 2: NAIVE PARAMETERS — Does alpha survive with un-optimized defaults?")
    print("=" * 110)
    print("If the strategy requires specific parameter values to work, it's overfit.")
    print("True structural alpha survives naive parameterization.\n")
    
    naive_configs = [
        ("Fully Naive (50/50, TSMOM only, no hyst)", 
         dict(dir_wt=1.0, dir_wc=0.0, dir_wo=0.0, sp_wt=1.0, sp_wc=0.0, sp_wo=0.0,
              dir_weight=0.50, spread_weight=0.50, dir_hyst=0.0, spread_hyst=0.0, leverage=2.0)),
        ("Naive with Hysteresis (50/50, TSMOM, hyst=0.3)",
         dict(dir_wt=1.0, dir_wc=0.0, dir_wo=0.0, sp_wt=1.0, sp_wc=0.0, sp_wo=0.0,
              dir_weight=0.50, spread_weight=0.50, dir_hyst=0.30, spread_hyst=0.30, leverage=2.0)),
        ("Naive All-Signals (50/50, equal signal wts)",
         dict(dir_wt=0.33, dir_wc=0.33, dir_wo=0.34, sp_wt=0.33, sp_wc=0.33, sp_wo=0.34,
              dir_weight=0.50, spread_weight=0.50, dir_hyst=0.30, spread_hyst=0.30, leverage=2.0)),
        ("Spread-Heavy Naive (30/70, TSMOM, hyst=0.3)",
         dict(dir_wt=1.0, dir_wc=0.0, dir_wo=0.0, sp_wt=1.0, sp_wc=0.0, sp_wo=0.0,
              dir_weight=0.30, spread_weight=0.70, dir_hyst=0.30, spread_hyst=0.30, leverage=2.0)),
        ("Current Tuned Champion",
         dict(dir_wt=0.30, dir_wc=0.50, dir_wo=0.20, sp_wt=0.30, sp_wc=0.00, sp_wo=0.70,
              dir_weight=0.35, spread_weight=0.65, dir_hyst=0.50, spread_hyst=0.40, leverage=2.5)),
    ]
    
    print(f"{'Config':48s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 100)
    for name, cfg in naive_configs:
        sh, tr, te, tt, ret, dd, _ = eval_strat(**cfg)
        print(f"{name:48s} | {sh:6.2f} | {tr:6.2f} | {te:6.2f} | {tt:5.2f} | {ret:+7.1f}% | {dd:6.1f}%")

    # ===========================================================================
    # TEST 3: PARAMETER SENSITIVITY
    # ===========================================================================
    print("\n" + "=" * 110)
    print("TEST 3: PARAMETER SENSITIVITY — How fragile are the tuned parameters?")
    print("=" * 110)
    print("If changing a parameter by 30% destroys performance, it's overfit.\n")
    
    base_params = dict(dir_wt=0.30, dir_wc=0.50, dir_wo=0.20, sp_wt=0.30, sp_wc=0.00, sp_wo=0.70,
                       dir_weight=0.35, spread_weight=0.65, dir_hyst=0.50, spread_hyst=0.40, leverage=2.5)
    
    sensitivity_tests = {
        'dir_weight': [0.15, 0.25, 0.35, 0.45, 0.55],
        'dir_hyst': [0.20, 0.35, 0.50, 0.65, 0.80],
        'spread_hyst': [0.15, 0.30, 0.40, 0.55, 0.70],
        'leverage': [1.5, 2.0, 2.5, 3.0, 3.5],
    }
    
    for param_name, values in sensitivity_tests.items():
        print(f"\n  Sweeping '{param_name}' (current = {base_params[param_name]}):")
        for val in values:
            cfg = base_params.copy()
            cfg[param_name] = val
            if param_name == 'dir_weight':
                cfg['spread_weight'] = 1.0 - val
            sh, tr, te, tt, ret, dd, _ = eval_strat(**cfg)
            marker = " ◄ current" if val == base_params[param_name] else ""
            print(f"    {param_name}={val:5.2f} → Sharpe={sh:.2f} | Train={tr:.2f} | Test={te:.2f} | T/T={tt:.2f} | Ret={ret:+.1f}% | DD={dd:.1f}%{marker}")

    # ===========================================================================
    # TEST 4: MACRO SMA SENSITIVITY (Most critical single parameter)
    # ===========================================================================
    print("\n" + "=" * 110)
    print("TEST 4: MACRO SMA PERIOD SENSITIVITY")
    print("=" * 110)
    print("The macro filter is the single largest alpha contributor. If it only works")
    print("at exactly 21 days, it's overfit. Testing 10d to 42d.\n")
    
    for sma_days in [10, 14, 18, 21, 25, 30, 35, 42]:
        proc_sma = compute_features(dataset, FULL_UNIVERSE, macro_sma_days=sma_days)
        macro_s = proc_sma[FULL_UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values
        p = run_strategy(proc_sma, dataset, FULL_UNIVERSE, eval_index, n, macro_s,
                         **base_params)
        sh, ret, dd = fast_metrics(p)
        tr = fast_sharpe(p[:split_idx])
        te = fast_sharpe(p[split_idx:])
        marker = " ◄ current" if sma_days == 21 else ""
        print(f"  SMA={sma_days:2d}d → Sharpe={sh:.2f} | Train={tr:.2f} | Test={te:.2f} | T/T={te/tr:.2f} | Ret={ret:+.1f}% | DD={dd:.1f}%{marker}")

    # ===========================================================================
    # TEST 5: ASSET SUBSET ROBUSTNESS
    # ===========================================================================
    print("\n" + "=" * 110)
    print("TEST 5: ASSET SUBSET ROBUSTNESS — Does it survive different asset combinations?")
    print("=" * 110)
    print("If alpha only exists with exactly SOL+AVAX+LINK, that's selection bias.\n")
    
    subsets = [
        ("SOL only", ['SOLUSDT']),
        ("AVAX only", ['AVAXUSDT']),
        ("LINK only", ['LINKUSDT']),
        ("SOL + AVAX", ['SOLUSDT', 'AVAXUSDT']),
        ("SOL + LINK", ['SOLUSDT', 'LINKUSDT']),
        ("AVAX + LINK", ['AVAXUSDT', 'LINKUSDT']),
        ("All 3 (SOL+AVAX+LINK)", FULL_UNIVERSE),
    ]
    
    print(f"{'Universe':26s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 80)
    for name, univ in subsets:
        proc_sub = compute_features(dataset, univ)
        eval_sub = proc_sub[univ[0]].iloc[WARMUP_BARS:]
        n_sub = len(eval_sub)
        macro_sub = eval_sub['macro_signal'].values
        # For single asset: 100% directional, 0% spread
        dw = 1.0 if len(univ) == 1 else 0.35
        sw = 0.0 if len(univ) == 1 else 0.65
        p = run_strategy(proc_sub, dataset, univ, eval_sub.index, n_sub, macro_sub,
                         dir_weight=dw, spread_weight=sw, **{k: base_params[k] for k in
                         ['dir_wt','dir_wc','dir_wo','sp_wt','sp_wc','sp_wo','dir_hyst','spread_hyst','leverage']})
        sh, ret, dd = fast_metrics(p)
        sp = min(17520, n_sub // 2)
        tr = fast_sharpe(p[:sp])
        te = fast_sharpe(p[sp:])
        print(f"{name:26s} | {sh:6.2f} | {tr:6.2f} | {te:6.2f} | {te/tr if tr > 0.01 else 0:5.2f} | {ret:+7.1f}% | {dd:6.1f}%")

    # ===========================================================================
    # TEST 6: ROLLING 6-MONTH SHARPE STABILITY
    # ===========================================================================
    print("\n" + "=" * 110)
    print("TEST 6: ROLLING 6-MONTH SHARPE — Checking for regime concentration")
    print("=" * 110)
    print("If alpha is concentrated in one specific period (e.g., 2023 bull), it's fragile.\n")
    
    sh_full, _, _, _, _, _, champion = eval_strat(**base_params)
    rs = pd.Series(champion, index=eval_index)
    window_hours = 4380  # ~6 months
    rolling_sharpes = []
    for start in range(0, n - window_hours, 720):  # step ~1 month
        end = start + window_hours
        chunk = champion[start:end]
        s = fast_sharpe(chunk)
        mid_date = eval_index[start + window_hours // 2]
        rolling_sharpes.append((mid_date, s))
        print(f"  {eval_index[start].strftime('%Y-%m')} to {eval_index[min(end,n)-1].strftime('%Y-%m')}: Sharpe = {s:+.2f}")
    
    pos_periods = sum(1 for _, s in rolling_sharpes if s > 0)
    total_periods = len(rolling_sharpes)
    print(f"\n  Positive Sharpe windows: {pos_periods}/{total_periods} ({pos_periods/total_periods*100:.0f}%)")

    # ===========================================================================
    # SUMMARY
    # ===========================================================================
    print("\n" + "=" * 110)
    print("STRUCTURAL ROBUSTNESS SUMMARY")
    print("=" * 110)

if __name__ == '__main__':
    main()
