#!/usr/bin/env python3
"""
===================================================================================
APEX CAUSAL FEATURE ABLATION STUDY
===================================================================================
Rigorous ablation on the fully lagged, fee-adjusted execution pipeline:
- Macro Filter: Daily 21d SMA Close (lagged 1 full day, evaluated at 00:00 UTC)
- Directional Fee: 5.0 bps Taker fee per turnover unit
- Spread Fee: 2.0 bps Maker fee per turnover unit
- Funding Model: True 8-hour Binance perpetual funding debits (00/08/16 UTC)
- Zero look-ahead bias across all signals and position shifts

Architectures Evaluated:
1. Pure Baseline:       Raw Signals + ATR Risk Parity (No Kalman, No GARCH)
2. GARCH Only:           Raw Signals + GARCH(1,1) Volatility Weighting (No Kalman)
3. Kalman Only (Heavy):  Kalman (Q/R=0.020) Signals + ATR Risk Parity (No GARCH)
4. Kalman Only (Light):  Kalman (Q/R=0.500) Signals + ATR Risk Parity (No GARCH)
5. Kalman Heavy + GARCH: Kalman (Q/R=0.020) + GARCH(1,1) (Original V44b)
6. Kalman Light + GARCH: Kalman (Q/R=0.500) + GARCH(1,1) (Phase-lag corrected)
7. Kalman Fast + GARCH:  Kalman (Q/R=1.000) + GARCH(1,1) (Minimal filtering)
===================================================================================
"""
import os, sys, pickle, warnings, time
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

# Weights
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
DIR_WEIGHT = 0.35
SPREAD_WEIGHT = 0.65
DIR_HYST = 0.50
SPREAD_HYST = 0.40

DIR_TAKER_FEE = 0.0005  # 5.0 bps
SPREAD_MAKER_FEE = 0.0002  # 2.0 bps
MACRO_DAILY_SMA = 21

GARCH_LOOKBACK = 720
GARCH_REFIT = 168

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

def kalman_filter_1d(signal, Q, R):
    n = len(signal)
    x_est = np.zeros(n)
    p_est = np.zeros(n)
    x_est[0] = signal[0] if not np.isnan(signal[0]) else 0.0
    p_est[0] = 1.0
    for t in range(1, n):
        x_pred = x_est[t-1]
        p_pred = p_est[t-1] + Q
        z = signal[t] if not np.isnan(signal[t]) else x_pred
        K = p_pred / (p_pred + R)
        x_est[t] = x_pred + K * (z - x_pred)
        p_est[t] = (1 - K) * p_pred
    return x_est

def compute_features(dataset, univ):
    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in univ}).mean(axis=1)
    
    # Strictly lagged daily macro regime
    daily_idx = idx_price.resample('1D').last()
    daily_sma = daily_idx.rolling(MACRO_DAILY_SMA).mean()
    daily_regime = (daily_idx > daily_sma).astype(float).shift(1).fillna(0.0)
    macro_regime_hourly = daily_regime.reindex(idx_price.index, method='ffill').fillna(0.0)
    
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        df['macro_signal'] = macro_regime_hourly
        
        # Synthetic OI proxy
        vol_z = (v - v.rolling(720, min_periods=24).mean()) / (v.rolling(720, min_periods=24).std() + 1e-8)
        ret_cum = c.pct_change(48).fillna(0.0)
        synthetic_oi = 5000000.0 * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
        
        # Synthetic Funding Rate proxy
        ret_24 = c.pct_change(24).fillna(0.0)
        synthetic_fr = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)
        
        # ATR Risk Parity
        cs = c.shift(1)
        tr = pd.concat([h-l, (h-cs).abs(), (l-cs).abs()], axis=1).max(axis=1)
        df['atr_pct'] = tr.rolling(14).mean() / (cs + 1e-8)
        df['inv_vol'] = 1.0 / df['atr_pct'].replace(0, np.nan).fillna(0.0)
        
        # Raw TSMOM
        parts = []
        for w in TSMOM_WINDOWS:
            rw = c.pct_change(w)
            vw = c.pct_change().rolling(w).std() * np.sqrt(w)
            parts.append(np.tanh(rw / (vw + 1e-8)))
        df['tsmom_raw'] = pd.concat(parts, axis=1).mean(axis=1).fillna(0.0)
        
        # Raw Carry
        mu = synthetic_fr.rolling(720, min_periods=72).mean()
        sig = synthetic_fr.rolling(720, min_periods=72).std()
        df['carry_raw'] = (-np.tanh((synthetic_fr - mu) / (sig + 1e-8))).fillna(0.0)
        
        # Raw OI Regime
        doi = synthetic_oi.pct_change(48).fillna(0.0)
        df['doi_48'] = doi.shift(1).fillna(0.0)
        df['oi_regime_raw'] = np.tanh(df['doi_48'] * 5.0).fillna(0.0)
        
        df['hourly_return'] = c.pct_change().fillna(0.0)
        processed[s] = df
    return processed

def compute_atr_weights(processed, univ, eval_index):
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in univ}, index=eval_index)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {s: w[s].values for s in univ}

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

def build_signal_dict(processed, univ, qr_ratio):
    signals = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        if qr_ratio is None:
            tsmom = df['tsmom_raw'].values
            carry = df['carry_raw'].values
            oi = df['oi_regime_raw'].values
        else:
            R = 0.05
            Q = qr_ratio * R
            tsmom = kalman_filter_1d(df['tsmom_raw'].values, Q, R)
            carry = kalman_filter_1d(df['carry_raw'].values, Q * 0.5, R * 0.5)
            oi = kalman_filter_1d(df['oi_regime_raw'].values, Q, R)
        signals[s] = {
            'dir_comp': DIR_WT*tsmom + DIR_WC*carry + DIR_WO*oi,
            'sp_comp': SP_WT*tsmom + SP_WC*carry + SP_WO*oi
        }
    return signals

def run_portfolio_simulation(processed, dataset, signals, wts, univ, eval_index, n, macro_regime):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    
    # Directional Engine (5.0 bps Taker Fee + 8h Funding)
    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, signals[s]['dir_comp'][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5
        
    port_dir = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values; w = wts[s]; comp = signals[s]['dir_comp']; x = xs[s]
        fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * 2.5
            if abs(ideal - cur) >= DIR_HYST: cur = ideal
            tgt[t] = cur
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        p_ret = (ret * shifted) - (turn * DIR_TAKER_FEE)
        p_fund = np.where(funding_hours, -shifted * fr, 0.0)
        port_dir += p_ret + p_fund

    # Spread Engine (2.0 bps Maker Fee + 8h Funding)
    cur_pos = {s: 0.0 for s in univ}
    port_sp = np.zeros(n)
    for t in range(1, n):
        sc = sorted([(s, signals[s]['sp_comp'][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
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

    raw_blend = DIR_WEIGHT * port_dir + SPREAD_WEIGHT * port_sp
    return raw_blend * macro_regime

def main():
    print("=" * 100)
    print("APEX CAUSAL FEATURE ABLATION STUDY")
    print("Pipeline: 5.0 bps Taker (Dir) | 2.0 bps Maker (Sp) | True 8h Funding | Daily 21d SMA Lagged Macro")
    print("=" * 100)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_features(dataset, UNIVERSE)
    eval_index = processed[UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    macro_regime = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values

    print(f"Dataset: {eval_index[0]} to {eval_index[-1]} ({n} bars)")
    print(f"In-Sample Train:      {split_idx} bars ({eval_index[0].strftime('%Y-%m-%d')} to {eval_index[split_idx-1].strftime('%Y-%m-%d')})")
    print(f"Out-of-Sample Test:   {n - split_idx} bars ({eval_index[split_idx].strftime('%Y-%m-%d')} to {eval_index[-1].strftime('%Y-%m-%d')})")

    # Compute Weight Types
    print("\nComputing ATR Risk Parity Weights...")
    wts_atr = compute_atr_weights(processed, UNIVERSE, eval_index)

    print("Computing GARCH(1,1) Conditional Volatility Weights...")
    t0 = time.time()
    wts_garch = compute_garch_weights(processed, UNIVERSE, eval_index)
    print(f"  GARCH computed in {time.time()-t0:.1f}s")

    # Architectures
    architectures = [
        ("The Pure Baseline", None, wts_atr, "Raw Signals + ATR Risk Parity"),
        ("GARCH Only", None, wts_garch, "Raw Signals + GARCH(1,1) Volatility"),
        ("Kalman Only (Heavy)", 0.020, wts_atr, "Kalman Q/R=0.020 (6.6h lag) + ATR"),
        ("Kalman Only (Light)", 0.500, wts_atr, "Kalman Q/R=0.500 (0.9h lag) + ATR"),
        ("Kalman Heavy + GARCH", 0.020, wts_garch, "Kalman Q/R=0.020 + GARCH(1,1)"),
        ("Kalman Light + GARCH", 0.500, wts_garch, "Kalman Q/R=0.500 + GARCH(1,1)"),
        ("Kalman Fast + GARCH", 1.000, wts_garch, "Kalman Q/R=1.000 (0.6h lag) + GARCH"),
    ]

    results = []
    curves = {}

    print("\n" + "=" * 100)
    print(f"{'Architecture':24s} | {'Full Sharpe':11s} | {'Train':6s} | {'Test':6s} | {'T/T Ratio':9s} | {'Net Return':10s} | {'Max DD':8s} | {'Calmar':6s}")
    print("-" * 100)

    for name, qr, wts, desc in architectures:
        sig_dict = build_signal_dict(processed, UNIVERSE, qr)
        p_rets = run_portfolio_simulation(processed, dataset, sig_dict, wts, UNIVERSE, eval_index, n, macro_regime)
        
        sh_full, ret_full, dd_full = fast_metrics(p_rets)
        sh_train = fast_sharpe(p_rets[:split_idx])
        sh_test = fast_sharpe(p_rets[split_idx:])
        tt_ratio = sh_test / sh_train if sh_train > 0 else 0.0
        calmar = (ret_full / abs(dd_full)) if abs(dd_full) > 0 else 0.0
        
        curves[name] = p_rets
        results.append({
            'name': name,
            'desc': desc,
            'qr': qr,
            'sh_full': sh_full,
            'sh_train': sh_train,
            'sh_test': sh_test,
            'tt_ratio': tt_ratio,
            'ret_full': ret_full,
            'dd_full': dd_full,
            'calmar': calmar,
            'series': pd.Series(p_rets, index=eval_index)
        })
        
        print(f"{name:24s} | {sh_full:11.2f} | {sh_train:6.2f} | {sh_test:6.2f} | {tt_ratio:9.2f} | {ret_full:+9.1f}% | {dd_full:7.1f}% | {calmar:6.2f}")

    print("=" * 100)

    # Detailed Yearly Breakdown
    print("\nPER-YEAR SHARPE RATIO BREAKDOWN:")
    print(f"{'Architecture':24s} | {'2022':7s} | {'2023':7s} | {'2024':7s} | {'2025':7s} | {'2026':7s}")
    print("-" * 75)
    for r in results:
        ser = r['series']
        yr_sharpes = []
        for yr in [2022, 2023, 2024, 2025, 2026]:
            chunk = ser[ser.index.year == yr].values
            yr_sharpes.append(f"{fast_sharpe(chunk):+7.2f}" if len(chunk) > 0 else "   N/A ")
        print(f"{r['name']:24s} | " + " | ".join(yr_sharpes))
    print("=" * 75)

    # Visualization
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1.5]})

    ax1 = axes[0]
    colors = ['#00E5FF', '#76FF03', '#FF5252', '#FFD600', '#E040FB', '#FF6D00', '#00B0FF']
    
    for (r, col) in zip(results, colors):
        cum = np.cumprod(1 + r['series'].values)
        ax1.plot(eval_index, cum, label=f"{r['name']} (Sharpe {r['sh_full']:.2f} | Test {r['sh_test']:.2f})",
                 color=col, linewidth=1.8 if 'Baseline' in r['name'] or 'GARCH Only' in r['name'] else 1.2,
                 alpha=1.0 if 'Baseline' in r['name'] or 'GARCH Only' in r['name'] else 0.7)

    # Bench index
    bench = pd.DataFrame({s: processed[s]['close'].pct_change().fillna(0.0) for s in UNIVERSE}).mean(axis=1).loc[eval_index]
    ax1.plot(eval_index, np.cumprod(1 + bench.values), color='#78909C', linestyle=':', label='Equal-Weight Index (Buy & Hold)', alpha=0.5)
    ax1.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.7, label='Train / Test Split')
    ax1.set_title('Strict Causal Feature Ablation: Cumulative Returns Under Real Friction', fontsize=14, fontweight='bold', color='white')
    ax1.set_ylabel('Cumulative Return', color='white')
    ax1.grid(color='gray', linestyle=':', alpha=0.2)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, facecolor='#1E222D')

    # Drawdown comparison for top models
    ax2 = axes[1]
    for r, col in zip([results[0], results[1], results[3], results[5]], ['#00E5FF', '#76FF03', '#FFD600', '#FF6D00']):
        cum = np.cumprod(1 + r['series'].values)
        dd = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum) * 100.0
        ax2.plot(eval_index, dd, label=f"{r['name']} (MaxDD: {r['dd_full']:.1f}%)", color=col, linewidth=1.4)
    ax2.axvline(eval_index[split_idx], color='#FF4081', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title('Underwater Drawdown Comparison (Top Architectures)', fontsize=12, color='white')
    ax2.set_ylabel('Drawdown (%)', color='white')
    ax2.set_ylim(-35, 2)
    ax2.grid(color='gray', linestyle=':', alpha=0.2)
    ax2.legend(loc='lower left', fontsize=8, framealpha=0.9, facecolor='#1E222D')

    out_plot = os.path.join(RESULTS_DIR, 'apex_causal_ablation.png')
    plt.tight_layout()
    plt.savefig(out_plot, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED PLOT] {out_plot}")

if __name__ == '__main__':
    main()
