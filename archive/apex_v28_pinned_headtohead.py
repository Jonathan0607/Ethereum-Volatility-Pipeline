#!/usr/bin/env python3
"""
===================================================================================
APEX v28: DEFINITIVE PINNED-DATA HEAD-TO-HEAD
===================================================================================
Loads the frozen dataset from data/pinned_dataset_4yr.pkl so that:
  1. The full 28,875-point static grid sweep finds the GLOBAL BEST on this exact data
  2. The Walk-Forward hyperparameter optimizer runs on the SAME exact data
  3. Both are compared head-to-head with zero data drift
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
SPLIT_BARS = 17520
TSMOM_WINDOWS = [24, 72, 168, 336, 720]

DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5; DIR_FEE = -0.0002
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0; SP_FEE = -0.0002
BRK_ENTRY = 72; BRK_EXIT = 336; BRK_LEV = 4.0; BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

HYST_BANDS = [0.25, 0.40, 0.50, 0.60, 0.75]
OI_GATES = [0.10, 0.15, 0.20, 0.25, 0.30]

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
                if open_profit > PROFIT_LOCK_THRESHOLD: profit_lock_active = True
                exit_price = lower_fast[t] if profit_lock_active else lower_base[t]
                if cv[t] < exit_price: pos = 0.0; profit_lock_active = False
            elif cv[t] > upper[t] and doi[t] >= oi_gate:
                pos = 1.0; entry_price = cv[t]; profit_lock_active = False
            tgt[t] = pos * (w[t] if not np.isnan(w[t]) else 0.0) * BRK_LEV
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * BRK_FEE)
    return port

def main():
    print("=" * 80)
    print("APEX v28: DEFINITIVE PINNED-DATA HEAD-TO-HEAD")
    print("=" * 80)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'pinned_dataset_4yr.pkl')
    print(f"\n[LOADING] Pinned dataset from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    split_idx = min(SPLIT_BARS, n // 2)
    wts = cross_weights(processed, UNIVERSE, eval_index)

    print(f"  Dataset: {eval_index[0]} to {eval_index[-1]} ({n} bars)")
    print(f"  Train/Test split at bar {split_idx} ({eval_index[split_idx]})")

    # =========================================================================
    # STEP 1: Pre-compute all 15 engine variants
    # =========================================================================
    print("\n[STEP 1] Pre-computing 15 Engine Variants...")
    dir_engines = {h: run_directional(processed, wts, UNIVERSE, eval_index, n, h) for h in HYST_BANDS}
    sp_engines = {h: run_spread(processed, UNIVERSE, eval_index, n, h) for h in HYST_BANDS}
    brk_engines = {o: run_breakout(processed, wts, UNIVERSE, n, o) for o in OI_GATES}

    # =========================================================================
    # STEP 2: Full 28,875-point Grid Sweep
    # =========================================================================
    print("\n[STEP 2] Running Full 28,875-Point Static Grid Sweep...")
    ALLOCS = []
    for d in np.arange(0.0, 1.05, 0.05):
        for s in np.arange(0.0, 1.05 - d, 0.05):
            b = 1.0 - d - s
            if abs(b) < 1e-5: b = 0.0
            if b >= -1e-5:
                ALLOCS.append((round(d,2), round(s,2), round(b,2)))

    best_static_sh = -999; best_static_cfg = None; best_static_ens = None
    count = 0; total = len(HYST_BANDS)**2 * len(OI_GATES) * len(ALLOCS)

    for dh in HYST_BANDS:
        for sh in HYST_BANDS:
            for boi in OI_GATES:
                d_arr = dir_engines[dh]; s_arr = sp_engines[sh]; b_arr = brk_engines[boi]
                for (d, s, b) in ALLOCS:
                    ens = d * d_arr + s * s_arr + b * b_arr
                    full_sh = fast_sharpe(ens)
                    if full_sh > best_static_sh:
                        best_static_sh = full_sh
                        tr_sh = fast_sharpe(ens[:split_idx])
                        te_sh = fast_sharpe(ens[split_idx:])
                        _, full_net, full_dd = fast_metrics(ens)
                        best_static_cfg = {
                            'dh': dh, 'sh': sh, 'boi': boi, 'd': d, 's': s, 'b': b,
                            'full_sharpe': full_sh, 'train_sharpe': tr_sh, 'test_sharpe': te_sh,
                            'full_net': full_net, 'full_dd': full_dd
                        }
                        best_static_ens = ens.copy()
                    count += 1
                    if count % 5000 == 0: print(f"  Processed {count}/{total}...")

    cfg = best_static_cfg
    print(f"\n  >>> GLOBAL BEST STATIC CONFIG:")
    print(f"      DH={cfg['dh']} SH={cfg['sh']} BOI={cfg['boi']} [{int(cfg['d']*100)}D/{int(cfg['s']*100)}S/{int(cfg['b']*100)}B]")
    print(f"      Full Sharpe: {cfg['full_sharpe']:.2f} | Train: {cfg['train_sharpe']:.2f} | Test: {cfg['test_sharpe']:.2f}")
    print(f"      Net: {cfg['full_net']:+.2f}% | MaxDD: {cfg['full_dd']:.2f}%")

    # =========================================================================
    # STEP 3: Walk-Forward using the SAME weights from the best static config
    # =========================================================================
    best_d = cfg['d']; best_s = cfg['s']; best_b = cfg['b']

    print(f"\n[STEP 3] Walk-Forward Hyperparameter Optimization...")
    print(f"    LOCKED Weights: {int(best_d*100)}D / {int(best_s*100)}S / {int(best_b*100)}B")

    wfa_rets = np.zeros(n)
    start_idx = LOOKBACK_BARS
    num_steps = (n - start_idx) // STEP_BARS

    # Backfill warmup with best static
    static_dh = cfg['dh']; static_sh_val = cfg['sh']; static_boi = cfg['boi']
    warmup_ens = (best_d * dir_engines[static_dh][:start_idx]) + \
                 (best_s * sp_engines[static_sh_val][:start_idx]) + \
                 (best_b * brk_engines[static_boi][:start_idx])
    wfa_rets[:start_idx] = warmup_ens

    param_history = []
    for step in range(num_steps + 1):
        is_end = start_idx + step * STEP_BARS
        is_start = is_end - LOOKBACK_BARS
        oos_start = is_end
        oos_end = min(oos_start + STEP_BARS, n)
        if oos_start >= n: break

        best_is_sh = -999; best_params = (static_dh, static_sh_val, static_boi)
        for dh in HYST_BANDS:
            for sh in HYST_BANDS:
                for boi in OI_GATES:
                    is_ens = (best_d * dir_engines[dh][is_start:is_end]) + \
                             (best_s * sp_engines[sh][is_start:is_end]) + \
                             (best_b * brk_engines[boi][is_start:is_end])
                    sh_val = fast_sharpe(is_ens)
                    if sh_val > best_is_sh:
                        best_is_sh = sh_val
                        best_params = (dh, sh, boi)

        opt_dh, opt_sh, opt_boi = best_params
        wfa_rets[oos_start:oos_end] = (best_d * dir_engines[opt_dh][oos_start:oos_end]) + \
                                      (best_s * sp_engines[opt_sh][oos_start:oos_end]) + \
                                      (best_b * brk_engines[opt_boi][oos_start:oos_end])
        param_history.append({'date': eval_index[oos_start], 'dh': opt_dh, 'sh': opt_sh, 'boi': opt_boi})
        if step % 6 == 0:
            print(f"    Step {step}/{num_steps} | {eval_index[oos_start].date()} | IS Sharpe: {best_is_sh:.2f} | DH={opt_dh}, SH={opt_sh}, BOI={opt_boi}")

    # =========================================================================
    # STEP 4: Head-to-Head
    # =========================================================================
    static_sh_full, static_net, static_dd = fast_metrics(best_static_ens)
    wfa_sh_full, wfa_net, wfa_dd = fast_metrics(wfa_rets)
    static_sh_train = fast_sharpe(best_static_ens[:split_idx])
    static_sh_test = fast_sharpe(best_static_ens[split_idx:])
    wfa_sh_train = fast_sharpe(wfa_rets[:split_idx])
    wfa_sh_test = fast_sharpe(wfa_rets[split_idx:])

    print("\n" + "=" * 80)
    print("DEFINITIVE HEAD-TO-HEAD (PINNED DATA, ZERO DRIFT)")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Static Global Best':<25} {'Walk-Forward Adaptive':<25}")
    print("-" * 80)
    print(f"{'Full 4-Year Sharpe':<30} {static_sh_full:<25.2f} {wfa_sh_full:<25.2f}")
    print(f"{'Train Sharpe (2022-2024)':<30} {static_sh_train:<25.2f} {wfa_sh_train:<25.2f}")
    print(f"{'Test Sharpe (2024-2026)':<30} {static_sh_test:<25.2f} {wfa_sh_test:<25.2f}")
    net_s = f"{static_net:+.2f}%"; net_w = f"{wfa_net:+.2f}%"
    dd_s = f"{static_dd:.2f}%"; dd_w = f"{wfa_dd:.2f}%"
    print(f"{'Net Return':<30} {net_s:<25} {net_w:<25}")
    print(f"{'Maximum Drawdown':<30} {dd_s:<25} {dd_w:<25}")
    cfg_str = f"DH={cfg['dh']} SH={cfg['sh']} BOI={cfg['boi']}"
    wt_str = f"{int(best_d*100)}D/{int(best_s*100)}S/{int(best_b*100)}B"
    print(f"{'Config':<30} {cfg_str:<25} {'6m Rolling Optimal':<25}")
    print(f"{'Weights':<30} {wt_str:<25} {(wt_str + ' (locked)'):<25}")
    print("=" * 80)
    print(f"\nWFA vs Static Delta: {wfa_sh_full - static_sh_full:+.2f} Sharpe")

    # =========================================================================
    # STEP 5: Visualization
    # =========================================================================
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})

    static_cum = np.cumprod(1 + best_static_ens)
    wfa_cum = np.cumprod(1 + wfa_rets)
    bench_h = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    bench_cum = np.cumprod(1 + bench_h.values)

    ax1.plot(eval_index, wfa_cum, color='#FF5252', linewidth=2.5,
             label=f'Walk-Forward Adaptive (Sharpe: {wfa_sh_full:.2f}, Net: {wfa_net:+.1f}%)')
    ax1.plot(eval_index, static_cum, color='#00E5FF', linewidth=1.8, alpha=0.8,
             label=f'Static Global Best (Sharpe: {static_sh_full:.2f}, Net: {static_net:+.1f}%)')
    ax1.plot(eval_index, bench_cum, color='#78909C', linewidth=1.0, linestyle=':', alpha=0.5,
             label='Buy & Hold Benchmark')
    ax1.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axvline(eval_index[start_idx], color='white', linestyle='--', linewidth=1.0, alpha=0.3, label='WFA Warmup End')
    ax1.set_title(f'Apex v28: Pinned-Data Head-to-Head\n'
                  f'Full Sharpe: {wfa_sh_full:.2f} (WFA) vs {static_sh_full:.2f} (Static)',
                  color='white', fontsize=13, fontweight='bold')
    ax1.grid(color='gray', linestyle=':', alpha=0.2)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9, facecolor='#1E222D')

    ph_df = pd.DataFrame(param_history).set_index('date')
    ph_df = ph_df.reindex(eval_index).fillna(method='ffill')
    plot_idx = eval_index[start_idx:]
    ph_plot = ph_df.loc[plot_idx]
    ax2.plot(ph_plot.index, ph_plot['dh'], label='Dir Hysteresis', color='#2962FF', linewidth=2)
    ax2.plot(ph_plot.index, ph_plot['sh'], label='Spread Hysteresis', color='#00C853', linewidth=2)
    ax2.plot(ph_plot.index, ph_plot['boi'], label='Breakout OI Gate', color='#FFAB00', linewidth=2)
    ax2.set_title('Dynamic Hyperparameter Adaptation', color='white', fontsize=11)
    ax2.set_ylim(0, 0.85)
    ax2.grid(color='gray', linestyle=':', alpha=0.2)
    ax2.legend(loc='upper right', ncol=3, fontsize=9)

    out_path = os.path.join(RESULTS_DIR, 'apex_v28_pinned_headtohead.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] {out_path}")

if __name__ == '__main__':
    main()
