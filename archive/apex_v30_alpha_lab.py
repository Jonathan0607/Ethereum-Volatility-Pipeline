#!/usr/bin/env python3
"""
===================================================================================
APEX v30: ALPHA LAB - DIVERGENCE & DYNAMIC LEVERAGE WFA
===================================================================================
Hunts for Alpha by comparing the True Dataset against the Synthetic Proxies.
Utilizes dynamic leverage scaling based on volatility regimes and divergence strength.
Optimizes hyperparameters dynamically using Walk-Forward Analysis.
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

# Fixed weights for the triple ensemble logic 
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
BRK_ENTRY = 72; BRK_EXIT = 336
PROFIT_LOCK_THRESHOLD = 0.20

# Fees
TAKER_FEE = -0.0010
MAKER_FEE = -0.0002

# Hyperparameter Search Grids
HYST_BANDS = [0.25, 0.40, 0.50, 0.60]
OI_GATES = [0.10, 0.20, 0.30]
DIV_MULTIPLIERS = [0.5, 1.0, 1.5, 2.0]

# Walk-Forward parameters
HOURS_PER_MONTH = 730
LOOKBACK_MONTHS = 6
LOOKBACK_BARS = HOURS_PER_MONTH * LOOKBACK_MONTHS
STEP_BARS = HOURS_PER_MONTH

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
        
        # True Open Interest & Funding Rate
        true_fr = df['funding_rate'].fillna(0.0)
        true_oi = df['open_interest'].fillna(method='ffill')
        
        # Synthetic Open Interest Proxy (from original logic)
        vol_z = (v - v.rolling(720, min_periods=24).mean()) / (v.rolling(720, min_periods=24).std() + 1e-8)
        ret_cum = c.pct_change(48).fillna(0.0)
        base_oi = true_oi.iloc[0] if not true_oi.empty and pd.notna(true_oi.iloc[0]) else 5000000.0
        synthetic_oi = base_oi * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
        
        # Synthetic Funding Rate Proxy (from original logic)
        ret_24 = c.pct_change(24).fillna(0.0)
        synthetic_fr = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)
        
        # Divergences (Normalized to Z-Scores)
        true_oi_z = (true_oi - true_oi.rolling(720).mean()) / (true_oi.rolling(720).std() + 1e-8)
        synth_oi_z = (synthetic_oi - synthetic_oi.rolling(720).mean()) / (synthetic_oi.rolling(720).std() + 1e-8)
        df['oi_divergence'] = (true_oi_z - synth_oi_z).fillna(0.0)
        
        true_fr_z = (true_fr - true_fr.rolling(720).mean()) / (true_fr.rolling(720).std() + 1e-8)
        synth_fr_z = (synthetic_fr - synthetic_fr.rolling(720).mean()) / (synthetic_fr.rolling(720).std() + 1e-8)
        df['fr_divergence'] = (true_fr_z - synth_fr_z).fillna(0.0)

        # Volatility Regime
        cs = c.shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['vol_regime'] = (atr / (atr.rolling(720).mean() + 1e-8)).fillna(1.0) # > 1 is high vol, < 1 is low vol

        # Dynamic Leverage Multiplier (Inverse to Volatility, capped 1x to 3x)
        # When vol is low (0.5), multiplier is 2.0 (scaled up). When vol is high (2.0), multiplier is 0.5
        df['dyn_leverage'] = np.clip(1.0 / df['vol_regime'], 1.0, 3.0).fillna(1.0)
        
        # Standard Indicators
        df[f'upper_{BRK_ENTRY}'] = h.rolling(BRK_ENTRY).max().shift(1)
        df[f'lower_{BRK_EXIT}'] = l.rolling(BRK_EXIT).min().shift(1)
        df[f'lower_72'] = l.rolling(72).min().shift(1)
        df['atr_pct'] = atr / (cs + 1e-8)
        df['inv_vol'] = 1.0 / df['atr_pct'].replace(0, np.nan).fillna(0.0)
        
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w)
            vw = c.pct_change().rolling(w).std() * np.sqrt(w)
            parts.append(np.tanh(rw / (vw + 1e-8)))
        df['tsmom'] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
        
        mu = true_fr.rolling(720, min_periods=72).mean()
        sig = true_fr.rolling(720, min_periods=72).std()
        df['carry'] = (-np.tanh((true_fr - mu) / (sig + 1e-8))).fillna(0.0)
        
        doi = true_oi.pct_change(48).fillna(0.0)
        df['doi_48'] = doi.shift(1).fillna(0.0)
        df['oi_regime'] = np.tanh(df['doi_48'] * 5.0).fillna(0.0)
        
        df['hourly_return'] = c.pct_change().fillna(0.0)
        processed[s] = df
    return processed

def cross_weights(processed, univ, eval_index):
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in univ}, index=eval_index)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {s: w[s].values for s in univ}

def run_directional(processed, wts, univ, eval_index, n, hyst, div_mult):
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        # Base direction is Momentum + Carry + OI
        base_comp = DIR_WT*df['tsmom'].values + DIR_WC*df['carry'].values + DIR_WO*df['oi_regime'].values
        # Add Funding Divergence: if real funding is lower than synthetic, long side has edge
        comps[s] = base_comp - (div_mult * df['fr_divergence'].values * 0.1)
    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values; w = wts[s]; comp = comps[s]; x = xs[s]
        dyn_lev = df['dyn_leverage'].values
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * 2.5 * dyn_lev[t] # Base lev 2.5 * dyn
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
        # Spread is market neutral, keep leverage fixed to 1x
        ideal[top_s] = conv * 1.0; ideal[bot_s] = -conv * 1.0
        for s in univ:
            old = cur_pos[s]
            if abs(ideal[s] - cur_pos[s]) >= hyst: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port[t] += turn * MAKER_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port[t] += ret * cur_pos[s]
    return port

def run_breakout(processed, wts, univ, n, oi_gate, div_mult):
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        cv = df['close'].values; ret = df['hourly_return'].values
        upper = df[f'upper_{BRK_ENTRY}'].values; lower_base = df[f'lower_{BRK_EXIT}'].values
        lower_fast = df['lower_72'].values
        doi = df['doi_48'].values; w = wts[s]
        oi_div = df['oi_divergence'].values
        dyn_lev = df['dyn_leverage'].values
        pos = 0.0; tgt = np.zeros(n)
        entry_price = 0.0; profit_lock_active = False
        for t in range(n):
            if np.isnan(cv[t]) or np.isnan(upper[t]): continue
            if pos > 0:
                open_profit = (cv[t] - entry_price) / entry_price
                if open_profit > PROFIT_LOCK_THRESHOLD: profit_lock_active = True
                exit_price = lower_fast[t] if profit_lock_active else lower_base[t]
                if cv[t] < exit_price: pos = 0.0; profit_lock_active = False
            elif cv[t] > upper[t] and (doi[t] >= oi_gate or oi_div[t] > (oi_gate * div_mult)):
                # Enter if True OI is expanding OR if True OI is massively diverging upwards from Synthetic OI
                pos = 1.0; entry_price = cv[t]; profit_lock_active = False
            
            # Dynamic leverage applied to breakout up to 4.0x
            tgt[t] = pos * (w[t] if not np.isnan(w[t]) else 0.0) * 4.0 * dyn_lev[t]
            
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * TAKER_FEE)
    return port

def main():
    print("=" * 80)
    print("APEX v30: ALPHA LAB - DIVERGENCE & DYNAMIC LEVERAGE WFA")
    print("=" * 80)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    if not os.path.exists(pkl_path):
        print(f"ERROR: {pkl_path} not found. Run true data builder first.")
        return
        
    print(f"\n[LOADING] True dataset from {pkl_path}")
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
    # Pre-compute Engine Variants for Walk-Forward Optimization
    # =========================================================================
    print("\n[STEP 1] Pre-computing WFA Engine Variants...")
    
    dir_engines = {}
    sp_engines = {}
    brk_engines = {}
    
    # 4 Hysteresis * 4 Div Multipliers = 16 Directional
    for dh in HYST_BANDS:
        for divm in DIV_MULTIPLIERS:
            dir_engines[(dh, divm)] = run_directional(processed, wts, UNIVERSE, eval_index, n, dh, divm)
            
    # 4 Hysteresis = 4 Spread
    for sh in HYST_BANDS:
        sp_engines[sh] = run_spread(processed, UNIVERSE, eval_index, n, sh)
        
    # 3 OI Gates * 4 Div Multipliers = 12 Breakout
    for boi in OI_GATES:
        for divm in DIV_MULTIPLIERS:
            brk_engines[(boi, divm)] = run_breakout(processed, wts, UNIVERSE, n, boi, divm)

    # =========================================================================
    # Walk-Forward Optimization
    # =========================================================================
    print(f"\n[STEP 2] Walk-Forward Optimization (6-Month Lookback, 1-Month Step)...")
    
    # Grid weights
    d_wt, s_wt, b_wt = 0.20, 0.55, 0.25
    
    wfa_rets = np.zeros(n)
    start_idx = LOOKBACK_BARS
    num_steps = (n - start_idx) // STEP_BARS

    # Warmup uses a robust default configuration
    def_dh, def_divm = 0.50, 1.0
    def_sh = 0.50
    def_boi = 0.20
    
    warmup_ens = (d_wt * dir_engines[(def_dh, def_divm)][:start_idx]) + \
                 (s_wt * sp_engines[def_sh][:start_idx]) + \
                 (b_wt * brk_engines[(def_boi, def_divm)][:start_idx])
    wfa_rets[:start_idx] = warmup_ens

    param_history = []
    
    total_perms = len(HYST_BANDS) * len(DIV_MULTIPLIERS) * len(HYST_BANDS) * len(OI_GATES) * len(DIV_MULTIPLIERS)
    
    for step in range(num_steps + 1):
        is_end = start_idx + step * STEP_BARS
        is_start = is_end - LOOKBACK_BARS
        oos_start = is_end
        oos_end = min(oos_start + STEP_BARS, n)
        if oos_start >= n: break

        best_is_sh = -999; best_params = None
        
        # Grid search over the pre-computed arrays for the In-Sample period
        for dh in HYST_BANDS:
            for divm_dir in DIV_MULTIPLIERS:
                d_arr = dir_engines[(dh, divm_dir)][is_start:is_end]
                for sh in HYST_BANDS:
                    s_arr = sp_engines[sh][is_start:is_end]
                    for boi in OI_GATES:
                        for divm_brk in DIV_MULTIPLIERS:
                            b_arr = brk_engines[(boi, divm_brk)][is_start:is_end]
                            
                            is_ens = (d_wt * d_arr) + (s_wt * s_arr) + (b_wt * b_arr)
                            sh_val = fast_sharpe(is_ens)
                            
                            if sh_val > best_is_sh:
                                best_is_sh = sh_val
                                best_params = (dh, divm_dir, sh, boi, divm_brk)

        opt_dh, opt_divm_dir, opt_sh, opt_boi, opt_divm_brk = best_params
        
        # Apply optimal params to Out-Of-Sample
        wfa_rets[oos_start:oos_end] = (d_wt * dir_engines[(opt_dh, opt_divm_dir)][oos_start:oos_end]) + \
                                      (s_wt * sp_engines[opt_sh][oos_start:oos_end]) + \
                                      (b_wt * brk_engines[(opt_boi, opt_divm_brk)][oos_start:oos_end])
        
        param_history.append({'date': eval_index[oos_start], 'sh': best_is_sh, 
                              'dh': opt_dh, 'div_dir': opt_divm_dir, 'sh_val': opt_sh, 
                              'boi': opt_boi, 'div_brk': opt_divm_brk})
                              
        if step % 6 == 0:
            print(f"    Step {step}/{num_steps} | {eval_index[oos_start].date()} | IS Sharpe: {best_is_sh:.2f} | "
                  f"Params: DH={opt_dh}(x{opt_divm_dir}) SH={opt_sh} BOI={opt_boi}(x{opt_divm_brk})")

    # =========================================================================
    # Evaluation
    # =========================================================================
    wfa_sh_full, wfa_net, wfa_dd = fast_metrics(wfa_rets)
    wfa_sh_train = fast_sharpe(wfa_rets[:split_idx])
    wfa_sh_test = fast_sharpe(wfa_rets[split_idx:])

    print("\n" + "=" * 80)
    print("V30 ALPHA LAB: WALK-FORWARD RESULTS")
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
             label=f'Apex v30 Alpha Lab (Sharpe: {wfa_sh_full:.2f}, Net: {wfa_net:+.1f}%)')
    ax1.plot(eval_index, bench_cum, color='#78909C', linewidth=1.5, linestyle=':', alpha=0.5,
             label='Buy & Hold Benchmark')
             
    ax1.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.set_title(f'Apex v30 Alpha Lab: Dynamic Leverage & Divergence Engineering', color='white', fontsize=14, fontweight='bold')
    ax1.grid(color='gray', linestyle=':', alpha=0.2)
    ax1.legend(loc='upper left', fontsize=11, framealpha=0.9, facecolor='#1E222D')
    
    ph_df = pd.DataFrame(param_history).set_index('date')
    ph_df = ph_df.reindex(eval_index).fillna(method='ffill')
    plot_idx = eval_index[start_idx:]
    ph_plot = ph_df.loc[plot_idx]
    
    # Plot Dynamic Leverage
    dyn_lev_mean = pd.DataFrame({s: processed[s]['dyn_leverage'] for s in UNIVERSE}).mean(axis=1).loc[plot_idx]
    ax2.plot(plot_idx, dyn_lev_mean, label='Average Dynamic Leverage', color='#FFAB00', linewidth=2)
    ax2.set_title('Regime-Based Dynamic Leverage Scaling', color='white', fontsize=11)
    ax2.set_ylim(0.5, 3.5)
    ax2.grid(color='gray', linestyle=':', alpha=0.2)
    ax2.legend(loc='upper left', fontsize=9)

    out_path = os.path.join(RESULTS_DIR, 'apex_v30_alpha_lab.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] Equity curve to {out_path}")

if __name__ == '__main__':
    main()
