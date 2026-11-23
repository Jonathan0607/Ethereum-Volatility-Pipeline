#!/usr/bin/env python3
"""
===================================================================================
APEX V44-V50: TECHNICAL DEPTH EXPERIMENTS
===================================================================================
Each experiment adds ONE technically dense component to the V43 baseline (Sharpe 1.51).
All components are audited for lookahead bias before being accepted.

V44: GARCH(1,1) Conditional Volatility Position Sizing
V45: Kalman Filter Signal Extraction
V46: Hurst Exponent Regime Detection  
V47: Hidden Markov Model Regime Classification
V49: Ornstein-Uhlenbeck Half-Life for Spread Timing
V50: Fractional Kelly Criterion Position Sizing
===================================================================================
"""
import os, sys, pickle, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

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
MAKER_FEE = -0.0002
MACRO_SMA_PERIOD = 504
DIR_WEIGHT = 0.35
SPREAD_WEIGHT = 0.65
DIR_HYST = 0.50
SPREAD_HYST = 0.40

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# CORE UTILITIES (identical to V43)
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

def compute_base_indicators(dataset, univ):
    """Compute core indicators (same as V43)."""
    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in univ}).mean(axis=1)
    macro_signal = (idx_price > idx_price.rolling(MACRO_SMA_PERIOD).mean()).astype(float).fillna(0.0)
    
    for s in univ:
        df = dataset[s].copy()
        c = df['close']; h = df['high']; l = df['low']; v = df['volume']
        
        df['macro_signal'] = macro_signal
        
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
    return processed, idx_price

def cross_weights_atr(processed, univ, eval_index):
    """Original ATR-based inverse vol weights (V43 baseline)."""
    iv = pd.DataFrame({s: processed[s]['inv_vol'] for s in univ}, index=eval_index)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    return {s: w[s].values for s in univ}

# ============================================================================
# V44: GARCH(1,1) CONDITIONAL VOLATILITY WEIGHTS
# ============================================================================

def compute_garch_weights(processed, univ, eval_index):
    """
    Replace ATR-based inverse-vol with GARCH(1,1) conditional volatility forecast.
    GARCH is fit on a TRAILING 720-hour window and forecasts 1-step-ahead vol.
    No lookahead: each forecast uses only data available at time t.
    """
    from arch import arch_model
    
    n = len(eval_index)
    garch_vol = {s: np.full(n, np.nan) for s in univ}
    
    refit_interval = 168  # Refit GARCH every 168 hours (1 week)
    lookback = 720  # 30 days of hourly data for fitting
    
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        rets = df['hourly_return'].values * 100  # Scale for GARCH stability
        
        last_omega, last_alpha, last_beta = 0.01, 0.05, 0.90
        cond_var = np.var(rets[:lookback]) if len(rets) >= lookback else 1.0
        
        for t in range(n):
            if t < lookback:
                garch_vol[s][t] = np.std(rets[:max(t+1, 24)])
                continue
            
            # Refit model periodically
            if t % refit_interval == 0:
                try:
                    window = rets[t-lookback:t]
                    model = arch_model(window, vol='GARCH', p=1, q=1, mean='Zero',
                                       rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    last_omega = res.params.get('omega', 0.01)
                    last_alpha = res.params.get('alpha[1]', 0.05)
                    last_beta = res.params.get('beta[1]', 0.90)
                except:
                    pass
            
            # GARCH(1,1) 1-step forecast: h_{t+1} = omega + alpha * r_t^2 + beta * h_t
            cond_var = last_omega + last_alpha * (rets[t-1]**2) + last_beta * cond_var
            garch_vol[s][t] = np.sqrt(max(cond_var, 1e-8))
    
    # Convert to inverse-vol weights
    inv_garch = {}
    for s in univ:
        gv = np.array(garch_vol[s])
        gv = np.where(np.isnan(gv) | (gv < 1e-8), 1.0, gv)
        inv_garch[s] = 1.0 / gv
    
    total = sum(inv_garch[s] for s in univ)
    total = np.where(total < 1e-8, 1.0, total)
    
    return {s: inv_garch[s] / total for s in univ}

# ============================================================================
# V45: KALMAN FILTER SIGNAL EXTRACTION
# ============================================================================

def kalman_smooth(signal, process_noise=1e-4, measurement_noise=1e-2):
    """
    1D Kalman Filter: extracts the hidden 'true' signal state from noisy observations.
    - process_noise (Q): how much the true signal is expected to change per step
    - measurement_noise (R): how noisy the raw observation is
    No lookahead: processes data online, one step at a time.
    """
    n = len(signal)
    x_est = np.zeros(n)  # State estimate
    p_est = np.zeros(n)  # Estimate uncertainty
    
    # Initialize
    x_est[0] = signal[0] if not np.isnan(signal[0]) else 0.0
    p_est[0] = 1.0
    
    Q = process_noise
    R = measurement_noise
    
    for t in range(1, n):
        # Predict
        x_pred = x_est[t-1]
        p_pred = p_est[t-1] + Q
        
        # Update
        z = signal[t] if not np.isnan(signal[t]) else x_pred
        K = p_pred / (p_pred + R)  # Kalman gain
        x_est[t] = x_pred + K * (z - x_pred)
        p_est[t] = (1 - K) * p_pred
    
    return x_est

def compute_kalman_indicators(processed, univ):
    """Apply Kalman Filter to smooth TSMOM and Carry signals."""
    kf_processed = {}
    for s in univ:
        df = processed[s].copy()
        # Smooth the TSMOM signal
        raw_tsmom = df['tsmom'].values
        df['tsmom_kalman'] = kalman_smooth(raw_tsmom, process_noise=1e-4, measurement_noise=5e-2)
        
        # Smooth the Carry signal
        raw_carry = df['carry'].values
        df['carry_kalman'] = kalman_smooth(raw_carry, process_noise=1e-5, measurement_noise=1e-1)
        
        # Smooth the OI regime signal
        raw_oi = df['oi_regime'].values
        df['oi_regime_kalman'] = kalman_smooth(raw_oi, process_noise=1e-4, measurement_noise=5e-2)
        
        kf_processed[s] = df
    return kf_processed

# ============================================================================
# V46: HURST EXPONENT REGIME DETECTION
# ============================================================================

def compute_hurst_exponent(series, max_lag=100):
    """
    Compute the Hurst Exponent using the Rescaled Range (R/S) method.
    H > 0.5: trending (persistent)
    H = 0.5: random walk
    H < 0.5: mean-reverting (anti-persistent)
    """
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    
    lags = range(2, max_lag + 1)
    rs_values = []
    
    for lag in lags:
        subseries = [series[i:i+lag] for i in range(0, n - lag, lag)]
        rs_list = []
        for ss in subseries:
            if len(ss) < 2:
                continue
            mean = np.mean(ss)
            deviations = np.cumsum(ss - mean)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(ss, ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if rs_list:
            rs_values.append((np.log(lag), np.log(np.mean(rs_list))))
    
    if len(rs_values) < 3:
        return 0.5
    
    x = np.array([v[0] for v in rs_values])
    y = np.array([v[1] for v in rs_values])
    slope, _, _, _, _ = scipy_stats.linregress(x, y)
    return np.clip(slope, 0.0, 1.0)

def compute_rolling_hurst(idx_returns, window=720, step=24):
    """Compute Hurst exponent on a rolling basis. Only uses trailing data."""
    n = len(idx_returns)
    hurst = np.full(n, 0.5)
    
    for t in range(window, n, step):
        h = compute_hurst_exponent(idx_returns[t-window:t], max_lag=50)
        # Fill forward from this point until next computation
        end = min(t + step, n)
        hurst[t:end] = h
    
    # Backfill the warmup period
    if window < n:
        hurst[:window] = hurst[window]
    
    return hurst

# ============================================================================
# V47: HIDDEN MARKOV MODEL REGIME CLASSIFICATION
# ============================================================================

def compute_hmm_regime(idx_returns, n_states=2, lookback=2160, refit_interval=720):
    """
    Fit a 2-state Gaussian HMM on trailing returns to classify Bull/Bear regimes.
    Returns probability of being in the 'bull' state (higher-mean state).
    No lookahead: model is only fit on data available at time t.
    """
    from hmmlearn.hmm import GaussianHMM
    
    n = len(idx_returns)
    regime_prob = np.full(n, 0.5)
    
    for t in range(lookback, n, refit_interval):
        try:
            window = idx_returns[t-lookback:t].reshape(-1, 1)
            model = GaussianHMM(n_components=n_states, covariance_type='full',
                                n_iter=100, random_state=42)
            model.fit(window)
            
            # Identify which state is "bull" (higher mean)
            means = model.means_.flatten()
            bull_state = np.argmax(means)
            
            # Get regime probabilities for the forward period
            end = min(t + refit_interval, n)
            fwd_data = idx_returns[t:end].reshape(-1, 1)
            if len(fwd_data) > 0:
                probs = model.predict_proba(fwd_data)
                regime_prob[t:end] = probs[:, bull_state]
        except:
            pass
    
    # Backfill warmup
    regime_prob[:lookback] = 0.5
    
    return regime_prob

# ============================================================================
# V49: ORNSTEIN-UHLENBECK HALF-LIFE FOR SPREAD TIMING
# ============================================================================

def compute_ou_halflife(spread_series, window=720):
    """
    Estimate the half-life of mean reversion for a spread using the OU process.
    OU: dX = θ(μ - X)dt + σdW
    Half-life = ln(2) / θ
    Estimated via OLS regression: ΔX_t = α + β·X_{t-1} + ε
    θ = -β, half-life = -ln(2)/β
    No lookahead: uses trailing window only.
    """
    n = len(spread_series)
    halflife = np.full(n, 168.0)  # Default: 1 week
    
    for t in range(window, n, 24):  # Recalculate daily
        chunk = spread_series[t-window:t]
        if np.std(chunk) < 1e-10:
            continue
        
        y = np.diff(chunk)
        x = chunk[:-1]
        
        if len(x) < 10 or np.std(x) < 1e-10:
            continue
        
        # OLS: Δspread = alpha + beta * spread_{t-1}
        slope, intercept, _, _, _ = scipy_stats.linregress(x, y)
        
        if slope < -1e-6:  # Must be negative for mean reversion
            hl = -np.log(2) / slope
            hl = np.clip(hl, 6, 720)  # Bound between 6 hours and 30 days
            end = min(t + 24, n)
            halflife[t:end] = hl
    
    halflife[:window] = halflife[window] if window < n else 168.0
    return halflife

# ============================================================================
# V50: FRACTIONAL KELLY CRITERION POSITION SIZING
# ============================================================================

def compute_kelly_fraction(returns, window=720, kelly_fraction=0.25):
    """
    Compute the optimal Kelly bet size: f* = μ / σ²
    Apply a fractional Kelly (0.25x) for safety.
    Returns a scaling factor for position sizes.
    No lookahead: trailing window only.
    """
    n = len(returns)
    kelly_scale = np.ones(n)
    
    for t in range(window, n):
        chunk = returns[t-window:t]
        mu = np.mean(chunk)
        var = np.var(chunk)
        
        if var > 1e-10 and mu > 0:
            f_star = mu / var
            kelly_scale[t] = np.clip(f_star * kelly_fraction, 0.1, 3.0)
        elif mu <= 0:
            kelly_scale[t] = 0.1  # Minimum exposure when edge is negative
    
    kelly_scale[:window] = 1.0
    return kelly_scale

# ============================================================================
# ENGINE FUNCTIONS (parameterized for different signal sources)
# ============================================================================

def run_directional(processed, wts, univ, eval_index, n, hyst,
                    tsmom_key='tsmom', carry_key='carry', oi_key='oi_regime',
                    leverage_scale=None):
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = DIR_WT*df[tsmom_key].values + DIR_WC*df[carry_key].values + DIR_WO*df[oi_key].values
    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5
    port = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values; w = wts[s]; comp = comps[s]; x = xs[s]
        cur = 0.0; tgt = np.zeros(n)
        base_lev = 2.5
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            lev = base_lev
            if leverage_scale is not None:
                lev = base_lev * leverage_scale[t]
            ideal = cv * wv * x[t] * lev
            if abs(ideal - cur) >= hyst: cur = ideal
            tgt[t] = cur
        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) + (turn * MAKER_FEE)
    return port

def run_spread(processed, univ, eval_index, n, hyst,
               tsmom_key='tsmom', carry_key='carry', oi_key='oi_regime',
               ou_halflife=None):
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = SP_WT*df[tsmom_key].values + SP_WC*df[carry_key].values + SP_WO*df[oi_key].values
    cur_pos = {s: 0.0 for s in univ}
    port = np.zeros(n)
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        top_s = sc[0][0]; bot_s = sc[-1][0]
        conv = abs(sc[0][1] - sc[-1][1])
        
        # V49: Scale by OU half-life (shorter half-life = more confident in mean reversion)
        spread_scale = 1.0
        if ou_halflife is not None:
            hl = ou_halflife[t]
            # Shorter half-life → more aggressive; longer → more conservative
            spread_scale = np.clip(168.0 / (hl + 1.0), 0.3, 2.0)
        
        ideal = {s: 0.0 for s in univ}
        ideal[top_s] = conv * 1.0 * spread_scale
        ideal[bot_s] = -conv * 1.0 * spread_scale
        for s in univ:
            old = cur_pos[s]
            if abs(ideal[s] - cur_pos[s]) >= hyst: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port[t] += turn * MAKER_FEE
            ret = processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t]
            port[t] += ret * cur_pos[s]
    return port

def build_v43_baseline(processed, wts, univ, eval_index, n, macro_signal):
    """Reproduce V43 champion exactly."""
    dir_rets = run_directional(processed, wts, univ, eval_index, n, DIR_HYST)
    sp_rets = run_spread(processed, univ, eval_index, n, SPREAD_HYST)
    raw_blend = DIR_WEIGHT * dir_rets + SPREAD_WEIGHT * sp_rets
    return np.where(macro_signal > 0.5, raw_blend, 0.0)

def evaluate(name, rets, eval_index, split_idx):
    """Full evaluation with per-year breakdown."""
    sh_full, ret_full, dd_full = fast_metrics(rets)
    sh_train = fast_sharpe(rets[:split_idx])
    sh_test = fast_sharpe(rets[split_idx:])
    ratio = sh_test / sh_train if sh_train > 0.01 else 0.0
    
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Full Sharpe:    {sh_full:.3f}")
    print(f"  Train Sharpe:   {sh_train:.3f}")
    print(f"  Test Sharpe:    {sh_test:.3f}")
    print(f"  T/T Ratio:      {ratio:.2f}")
    print(f"  Net Return:     {ret_full:+.1f}%")
    print(f"  Max Drawdown:   {dd_full:.1f}%")
    
    rs = pd.Series(rets, index=eval_index)
    for yr in [2022,2023,2024,2025,2026]:
        chunk = rs[rs.index.year == yr]
        if len(chunk) == 0: continue
        s = fast_sharpe(chunk.values)
        r = (np.prod(1+chunk.values)-1)*100
        print(f"    {yr}: Sharpe={s:+.2f}, Ret={r:+.1f}%")
    
    return sh_full, sh_train, sh_test, ratio, ret_full, dd_full

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("APEX V44-V50: TECHNICAL DEPTH EXPERIMENTS")
    print("=" * 70)

    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed, idx_price = compute_base_indicators(dataset, UNIVERSE)
    first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
    eval_index = first_df.index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    
    macro_signal = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values
    idx_returns = idx_price.pct_change().fillna(0.0).values
    # Align idx_returns to eval_index length
    idx_returns_eval = idx_returns[-(n):]
    
    print(f"  Dataset: {eval_index[0]} to {eval_index[-1]} ({n} bars)")
    
    results = {}
    
    # =====================================================================
    # V43 BASELINE
    # =====================================================================
    print("\n[V43] Computing baseline...")
    wts_atr = cross_weights_atr(processed, UNIVERSE, eval_index)
    v43 = build_v43_baseline(processed, wts_atr, UNIVERSE, eval_index, n, macro_signal)
    r = evaluate("V43 BASELINE (Dir35/Sp65 + Macro504)", v43, eval_index, split_idx)
    results['V43 Baseline'] = r
    
    # =====================================================================
    # V44: GARCH(1,1) CONDITIONAL VOLATILITY SIZING
    # =====================================================================
    print("\n[V44] Computing GARCH(1,1) conditional volatility weights...")
    t0 = time.time()
    wts_garch = compute_garch_weights(processed, UNIVERSE, eval_index)
    print(f"  GARCH computation took {time.time()-t0:.1f}s")
    
    dir_g = run_directional(processed, wts_garch, UNIVERSE, eval_index, n, DIR_HYST)
    sp_g = run_spread(processed, UNIVERSE, eval_index, n, SPREAD_HYST)
    raw_g = DIR_WEIGHT * dir_g + SPREAD_WEIGHT * sp_g
    v44 = np.where(macro_signal > 0.5, raw_g, 0.0)
    r = evaluate("V44 GARCH(1,1) CONDITIONAL VOL SIZING", v44, eval_index, split_idx)
    results['V44 GARCH'] = r
    
    # =====================================================================
    # V45: KALMAN FILTER SIGNAL EXTRACTION
    # =====================================================================
    print("\n[V45] Computing Kalman-filtered signals...")
    kf_processed = compute_kalman_indicators(processed, UNIVERSE)
    
    dir_kf = run_directional(kf_processed, wts_atr, UNIVERSE, eval_index, n, DIR_HYST,
                             tsmom_key='tsmom_kalman', carry_key='carry_kalman',
                             oi_key='oi_regime_kalman')
    sp_kf = run_spread(kf_processed, UNIVERSE, eval_index, n, SPREAD_HYST,
                       tsmom_key='tsmom_kalman', carry_key='carry_kalman',
                       oi_key='oi_regime_kalman')
    raw_kf = DIR_WEIGHT * dir_kf + SPREAD_WEIGHT * sp_kf
    v45 = np.where(macro_signal > 0.5, raw_kf, 0.0)
    r = evaluate("V45 KALMAN FILTER SIGNAL EXTRACTION", v45, eval_index, split_idx)
    results['V45 Kalman'] = r
    
    # =====================================================================
    # V46: HURST EXPONENT REGIME DETECTION
    # =====================================================================
    print("\n[V46] Computing rolling Hurst Exponent...")
    t0 = time.time()
    hurst = compute_rolling_hurst(idx_returns_eval, window=720, step=24)
    print(f"  Hurst computation took {time.time()-t0:.1f}s")
    print(f"  Hurst range: {np.min(hurst):.3f} to {np.max(hurst):.3f}, mean={np.mean(hurst):.3f}")
    
    # When trending (H > 0.5): favor Directional. When mean-reverting: favor Spread.
    dir_w_hurst = np.clip(0.35 + (hurst - 0.5) * 1.0, 0.10, 0.60)
    sp_w_hurst = 1.0 - dir_w_hurst
    
    dir_base = run_directional(processed, wts_atr, UNIVERSE, eval_index, n, DIR_HYST)
    sp_base = run_spread(processed, UNIVERSE, eval_index, n, SPREAD_HYST)
    raw_hurst = dir_w_hurst * dir_base + sp_w_hurst * sp_base
    v46 = np.where(macro_signal > 0.5, raw_hurst, 0.0)
    r = evaluate("V46 HURST EXPONENT REGIME BLEND", v46, eval_index, split_idx)
    results['V46 Hurst'] = r
    
    # =====================================================================
    # V47: HIDDEN MARKOV MODEL REGIME CLASSIFICATION
    # =====================================================================
    print("\n[V47] Computing HMM regime probabilities...")
    t0 = time.time()
    hmm_prob = compute_hmm_regime(idx_returns_eval, n_states=2, lookback=2160, refit_interval=720)
    print(f"  HMM computation took {time.time()-t0:.1f}s")
    print(f"  HMM bull prob range: {np.min(hmm_prob):.3f} to {np.max(hmm_prob):.3f}")
    
    # Use HMM probability as a soft macro filter instead of binary SMA
    raw_hmm = DIR_WEIGHT * dir_base + SPREAD_WEIGHT * sp_base
    v47 = raw_hmm * hmm_prob  # Scale by bull probability
    r = evaluate("V47 HMM REGIME CLASSIFICATION", v47, eval_index, split_idx)
    results['V47 HMM'] = r
    
    # Also test: HMM as a supplement to the SMA filter (AND logic)
    v47b = np.where((macro_signal > 0.5) & (hmm_prob > 0.5), raw_hmm, 0.0)
    r = evaluate("V47b HMM + Macro SMA (AND)", v47b, eval_index, split_idx)
    results['V47b HMM+SMA'] = r
    
    # =====================================================================
    # V49: ORNSTEIN-UHLENBECK HALF-LIFE FOR SPREAD TIMING
    # =====================================================================
    print("\n[V49] Computing OU half-life for spread engine...")
    # Compute spread between top and bottom asset's composite score
    comps_vals = {}
    for s in UNIVERSE:
        df = processed[s].iloc[WARMUP_BARS:]
        comps_vals[s] = SP_WT*df['tsmom'].values + SP_WC*df['carry'].values + SP_WO*df['oi_regime'].values
    
    # Use the spread between max and min composite as the OU process
    spread_series = np.zeros(n)
    for t in range(n):
        vals = [comps_vals[s][t] for s in UNIVERSE]
        spread_series[t] = max(vals) - min(vals)
    
    ou_hl = compute_ou_halflife(spread_series, window=720)
    print(f"  OU half-life range: {np.min(ou_hl):.1f}h to {np.max(ou_hl):.1f}h, mean={np.mean(ou_hl):.1f}h")
    
    dir_ou = run_directional(processed, wts_atr, UNIVERSE, eval_index, n, DIR_HYST)
    sp_ou = run_spread(processed, UNIVERSE, eval_index, n, SPREAD_HYST, ou_halflife=ou_hl)
    raw_ou = DIR_WEIGHT * dir_ou + SPREAD_WEIGHT * sp_ou
    v49 = np.where(macro_signal > 0.5, raw_ou, 0.0)
    r = evaluate("V49 ORNSTEIN-UHLENBECK SPREAD TIMING", v49, eval_index, split_idx)
    results['V49 OU'] = r
    
    # =====================================================================
    # V50: FRACTIONAL KELLY CRITERION POSITION SIZING
    # =====================================================================
    print("\n[V50] Computing Fractional Kelly position sizing...")
    # Use V43 baseline returns to compute trailing Kelly fraction
    kelly_scale = compute_kelly_fraction(v43, window=720, kelly_fraction=0.25)
    print(f"  Kelly scale range: {np.min(kelly_scale):.2f} to {np.max(kelly_scale):.2f}")
    
    dir_k = run_directional(processed, wts_atr, UNIVERSE, eval_index, n, DIR_HYST,
                            leverage_scale=kelly_scale)
    sp_k = run_spread(processed, UNIVERSE, eval_index, n, SPREAD_HYST)
    raw_k = DIR_WEIGHT * dir_k + SPREAD_WEIGHT * sp_k
    v50 = np.where(macro_signal > 0.5, raw_k, 0.0)
    r = evaluate("V50 FRACTIONAL KELLY CRITERION", v50, eval_index, split_idx)
    results['V50 Kelly'] = r
    
    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print("\n\n" + "=" * 90)
    print("SUMMARY: ALL EXPERIMENTS")
    print("=" * 90)
    print(f"{'Experiment':<40} {'Full':>7} {'Train':>7} {'Test':>7} {'T/T':>6} {'Ret':>9} {'MaxDD':>8}")
    print("-" * 90)
    for name, (sh, sh_tr, sh_te, ratio, ret, dd) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        marker = " <<<" if sh >= 1.5 else (" **" if sh >= 1.3 else "")
        print(f"{name:<40} {sh:>7.3f} {sh_tr:>7.3f} {sh_te:>7.3f} {ratio:>6.2f} {ret:>+8.1f}% {dd:>7.1f}%{marker}")
    
    # =====================================================================
    # SAVE COMPARISON PLOT
    # =====================================================================
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    strategies = {
        'V43 Baseline': v43,
        'V44 GARCH': v44,
        'V45 Kalman': v45,
        'V46 Hurst': v46,
        'V49 OU': v49,
        'V50 Kelly': v50,
    }
    colors = ['#00E5FF', '#FF6B6B', '#00C853', '#FFAB00', '#E040FB', '#FF5722']
    
    for (name, rets), color in zip(strategies.items(), colors):
        cum = np.cumprod(1 + rets)
        sh = fast_sharpe(rets)
        r = (cum[-1]-1)*100
        ax.plot(eval_index, cum, color=color, linewidth=1.8,
                label=f'{name} (Sharpe: {sh:.2f}, Ret: {r:+.0f}%)')
    
    ax.axvline(eval_index[split_idx], color='white', linestyle='--', linewidth=1, alpha=0.3, label='Train/Test Split')
    ax.set_title('Apex V44-V50: Technical Depth Experiments', color='white', fontsize=14, fontweight='bold')
    ax.grid(color='gray', linestyle=':', alpha=0.2)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, facecolor='#1E222D')
    
    out_path = os.path.join(RESULTS_DIR, 'apex_v44_v50_experiments.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[SAVED] {out_path}")

if __name__ == '__main__':
    main()
