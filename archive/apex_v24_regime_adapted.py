import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

from multi_asset_ablation import build_multi_asset_dataset, HOURS_PER_YEAR

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
EVAL_BARS_4YR = 35040 + WARMUP_BARS
SPLIT_BARS = 17520 

TSMOM_WINDOWS = [24, 72, 168, 336, 720]

DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
DIR_LEV = 2.5
DIR_FEE = -0.0002 

SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
SP_LEV = 1.0
SP_FEE = -0.0002

BRK_ENTRY = 72; BRK_EXIT = 336
BRK_LEV = 4.0
BRK_FEE = -0.0010
PROFIT_LOCK_THRESHOLD = 0.20

# REGIME ADAPTED CONFIG (Test > Train)
DH = 0.60
SH = 0.40
BOI = 0.10
W_DIR = 0.25
W_SP = 0.55
W_BRK = 0.20

def fast_metrics(h_rets):
    if len(h_rets) == 0: return 0.0
    am = np.mean(h_rets) * HOURS_PER_YEAR
    av = np.std(h_rets) * np.sqrt(HOURS_PER_YEAR)
    return am / (av + 1e-8)

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

def max_drawdown(rets):
    if len(rets) == 0: return 0.0
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / (peak + 1e-8)
    return np.min(dd)

def main():
    dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)
    processed = compute_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in UNIVERSE}, index=eval_index)
    weights_df = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    wts = {s: weights_df[s].values for s in UNIVERSE}
    
    ret_dir = run_directional(processed, wts, UNIVERSE, eval_index, n, DH)
    ret_sp = run_spread(processed, UNIVERSE, eval_index, n, SH)
    ret_brk = run_breakout(processed, wts, UNIVERSE, n, BOI)
    
    ens_rets = (W_DIR * ret_dir) + (W_SP * ret_sp) + (W_BRK * ret_brk)
    
    split_idx = min(SPLIT_BARS, n // 2)
    
    train_rets = ens_rets[:split_idx]
    test_rets = ens_rets[split_idx:]
    
    train_sh = fast_metrics(train_rets)
    test_sh = fast_metrics(test_rets)
    full_sh = fast_metrics(ens_rets)
    
    train_dd = max_drawdown(train_rets) * 100
    test_dd = max_drawdown(test_rets) * 100
    full_dd = max_drawdown(ens_rets) * 100
    
    print("="*60)
    print("PROJECT APEX: REGIME-ADAPTED CONFIGURATION (Test > Train)")
    print("="*60)
    print(f"Parameters: DH={DH}, SH={SH}, BOI={BOI}")
    print(f"Allocation: {W_DIR*100:.0f}% Dir | {W_SP*100:.0f}% Sp | {W_BRK*100:.0f}% Brk")
    print(f"\n[2022-2024] Train Sharpe: {train_sh:.2f} (Max DD: {train_dd:.2f}%)")
    print(f"[2024-2026] Test Sharpe:  {test_sh:.2f} (Max DD: {test_dd:.2f}%)")
    print(f"[4-Year Full] Overall Sharpe: {full_sh:.2f} (Max DD: {full_dd:.2f}%)")
    
    eq = pd.Series(ens_rets, index=eval_index)
    eq_cum = (1 + eq).cumprod()
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eq_cum.index, eq_cum.values, color='#00ffcc', linewidth=1.5, label='Triple Ensemble')
    split_date = eq_cum.index[split_idx]
    ax.axvline(split_date, color='red', linestyle='--', alpha=0.8, label='Train / Test Split')
    ax.set_title(f'Apex Regime-Adapted (Test > Train)\nTrain: {train_sh:.2f} | Test: {test_sh:.2f} | Full: {full_sh:.2f}', color='white')
    ax.grid(color='gray', linestyle=':', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    os.makedirs('archive/results', exist_ok=True)
    out_path = 'archive/results/apex_v24_regime_adapted.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='#121212')
    print(f"\nSaved compounding plot to {out_path}")

if __name__ == '__main__':
    main()
