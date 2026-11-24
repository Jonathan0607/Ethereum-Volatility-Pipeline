#!/usr/bin/env python3
"""
===================================================================================
APEX STRUCTURAL FACTORS ANALYSIS
===================================================================================
Testing three exogenous structural alpha factors on the GARCH(1,1) Core Engine:
1. BTC Lead-Lag (Information Flow)
2. Weekend / Liquidity Desert Filter
3. Hourly VPIN Proxy (Order Flow Imbalance)

Base Engine: GARCH(1,1) Core (TSMOM + Macro)
Fees: 5.0 bps Taker (Directional), 2.0 bps Maker (Spread) + True 8h Funding
===================================================================================
"""
import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import ccxt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

HOURS_PER_YEAR = 8760.0

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
TSMOM_WINDOWS = [24, 72, 168, 336, 720]
MACRO_DAILY_SMA = 21

# Core signal weights (tuned mix from ablation champion)
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
DIR_WEIGHT, SPREAD_WEIGHT = 0.35, 0.65
DIR_HYST, SPREAD_HYST = 0.50, 0.40
LEVERAGE = 2.5

# Fee structure
DIR_TAKER_FEE = 0.0005
SPREAD_MAKER_FEE = 0.0002

GARCH_LOOKBACK = 720
GARCH_REFIT = 168

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATA FETCHING
# ============================================================================
def fetch_btc_data(start_dt, end_dt):
    """Fetch BTCUSDT hourly data from Binance to match exact timeframe."""
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'btc_1h_4yr_cache.pkl')
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    print("Fetching BTC/USD hourly data from Kraken...")
    exchange = ccxt.kraken()
    since = int(start_dt.timestamp() * 1000)
    end = int(end_dt.timestamp() * 1000)
    
    all_ohlcv = []
    while since < end:
        try:
            ohlcv = exchange.fetch_ohlcv('BTC/USD', '1h', since=since, limit=720)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 3600000
            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching BTC data: {e}")
            time.sleep(1)
            
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.to_pickle(cache_path)
    return df

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
# ENGINE
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
        
        # FACTOR 3: VPIN
        h_minus_l = h - l
        # Avoid div by 0
        h_minus_l_safe = np.where(h_minus_l == 0, 1e-8, h_minus_l)
        buy_vol = v * (c - l) / h_minus_l_safe
        sell_vol = v * (h - c) / h_minus_l_safe
        imbalance = buy_vol - sell_vol
        
        vpin = imbalance.rolling(24).sum() / (v.rolling(24).sum() + 1e-8)
        vpin_z = (vpin - vpin.rolling(720).mean()) / (vpin.rolling(720).std() + 1e-8)
        df['vpin_z'] = vpin_z.shift(1).fillna(0.0)  # Causal shift

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

def run_core_engine(processed, dataset, univ, eval_index, n, macro_regime, wts, 
                    btc_df=None, filter_type=None):
    funding_hours = eval_index.hour.isin([0, 8, 16])

    # Pre-compute BTC signal if needed
    btc_signal = np.ones(n)
    if filter_type == 'btc' and btc_df is not None:
        btc_c = btc_df['close'].reindex(eval_index).fillna(method='ffill')
        # 4-hour TSMOM
        btc_rw = btc_c.pct_change(4)
        btc_vw = btc_c.pct_change().rolling(4).std() * np.sqrt(4)
        btc_tsmom = np.tanh(btc_rw / (btc_vw + 1e-8))
        btc_signal_raw = (btc_tsmom > 0.5).astype(float)
        btc_signal = btc_signal_raw.shift(1).fillna(1.0).values
        
    # Pre-compute Weekend/Liquidity filter
    liquidity_filter = np.ones(n)
    if filter_type == 'liquidity':
        is_weekend = np.isin(eval_index.dayofweek, [5, 6]).astype(float)
        
        # 12h volume percentile
        total_vol = pd.Series(sum(processed[s].iloc[WARMUP_BARS:]['volume'].values for s in univ))
        vol_12h = total_vol.rolling(12).sum()
        vol_30th_perc = vol_12h.rolling(720).quantile(0.3)
        low_vol = (vol_12h < vol_30th_perc).astype(float).values
        
        # Shift 1 to ensure causality
        low_liq = np.logical_or(is_weekend, low_vol)
        low_liq_shifted = pd.Series(low_liq).shift(1).fillna(0.0).values
        
        # 1 means OK to trade, 0 means block NEW entries
        liquidity_filter = 1.0 - low_liq_shifted

    comps_dir = {}
    vpin_z = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps_dir[s] = DIR_WT*df['tsmom'].values + DIR_WC*df['carry'].values + DIR_WO*df['oi_regime'].values
        vpin_z[s] = df['vpin_z'].values

    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps_dir[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5

    port_dir = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values
        fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values
        w = wts[s]; comp = comps_dir[s]; x = xs[s]; vpin = vpin_z[s]
        
        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            
            ideal = cv * wv * x[t] * LEVERAGE
            
            # Apply Filters
            if filter_type == 'btc':
                # Only go long if BTC 4h TSMOM > 0.5
                if ideal > 0 and btc_signal[t] == 0:
                    ideal = 0.0 
                    
            elif filter_type == 'liquidity':
                # Block NEW directional entries during low liquidity
                if liquidity_filter[t] == 0 and abs(ideal) > abs(cur):
                    ideal = cur
                    
            elif filter_type == 'vpin':
                # VPIN as leading indicator: block longs if vpin < 0, block shorts if vpin > 0
                if ideal > 0 and vpin[t] < 0:
                    ideal = 0.0
                elif ideal < 0 and vpin[t] > 0:
                    ideal = 0.0

            if abs(ideal - cur) >= DIR_HYST: cur = ideal
            tgt[t] = cur
            
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port_dir += (ret * shifted) - (turn * DIR_TAKER_FEE)
        port_dir += np.where(funding_hours, -shifted * fr, 0.0)

    # Spread Engine (Unchanged by these directional filters)
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
            if turn > 0.001:
                port_sp[t] -= turn * SPREAD_MAKER_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port_sp[t] += ret * cur_pos[s]
            if funding_hours[t]:
                fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values[t]
                port_sp[t] += -cur_pos[s] * fr

    raw_core = DIR_WEIGHT * port_dir + SPREAD_WEIGHT * port_sp
    return raw_core * macro_regime

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 100)
    print("APEX STRUCTURAL FACTORS ANALYSIS")
    print("=" * 100)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = compute_features(dataset, UNIVERSE)
    eval_index = processed[UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    macro_regime = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values
    
    # Fetch BTC data
    btc_df = fetch_btc_data(eval_index[0] - pd.Timedelta(days=5), eval_index[-1] + pd.Timedelta(days=1))

    # Get GARCH weights
    cache_path = os.path.join(PROJECT_ROOT, 'data', 'garch_weights_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            wts_garch = pickle.load(f)
    else:
        print("\nComputing GARCH(1,1) weights...")
        wts_garch = compute_garch_weights(processed, UNIVERSE, eval_index)
        with open(cache_path, 'wb') as f:
            pickle.dump(wts_garch, f)

    # Base Engine
    print("\nRunning Base GARCH Core Engine...")
    base_ret = run_core_engine(processed, dataset, UNIVERSE, eval_index, n, macro_regime, wts_garch, filter_type=None)
    
    # Factor 1
    print("Running Core + BTC Lead-Lag Filter...")
    btc_ret = run_core_engine(processed, dataset, UNIVERSE, eval_index, n, macro_regime, wts_garch, btc_df=btc_df, filter_type='btc')
    
    # Factor 2
    print("Running Core + Weekend/Liquidity Filter...")
    liq_ret = run_core_engine(processed, dataset, UNIVERSE, eval_index, n, macro_regime, wts_garch, filter_type='liquidity')
    
    # Factor 3
    print("Running Core + VPIN Proxy Filter...")
    vpin_ret = run_core_engine(processed, dataset, UNIVERSE, eval_index, n, macro_regime, wts_garch, filter_type='vpin')

    print("\n" + "=" * 100)
    print("RESULTS: EXOGENOUS STRUCTURAL FACTORS")
    print("=" * 100)

    print(f"{'Configuration':35s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 95)

    def report(name, ret):
        m = fast_metrics(ret)
        tr = fast_sharpe(ret[:split_idx])
        te = fast_sharpe(ret[split_idx:])
        tt = te/tr if abs(tr)>0.01 else 0
        print(f"{name:35s} | {m['sharpe']:+6.2f} | {tr:+6.2f} | {te:+6.2f} | {tt:5.2f} | {m['ret']:+7.1f}% | {m['dd']:6.1f}%")
        return ret

    rets = {
        'Base GARCH Core': base_ret,
        '+ BTC Lead-Lag': btc_ret,
        '+ Weekend/Liquidity Filter': liq_ret,
        '+ Hourly VPIN Proxy': vpin_ret
    }

    for name, r in rets.items():
        report(name, r)

if __name__ == '__main__':
    main()
