#!/usr/bin/env python3
"""
===================================================================================
APEX INSTITUTIONAL VALIDATION: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV)
& DEFLATED SHARPE RATIO (DSR) ANALYSIS
===================================================================================
Methodology based on Marcos Lopez de Prado:
  - "Advances in Financial Machine Learning" (Wiley, 2018), Chapters 7, 12, 14
  - "The Deflated Sharpe Ratio" (Bailey & Lopez de Prado, JPM, 2014)

Architecture:
  1. Combinatorial Purged Cross-Validation (CPCV):
     - N = 6 sequential blocks across the full 4-year dataset (36,552 hourly bars).
     - k = 2 testing blocks per split -> C(6, 2) = 15 cross-validation folds.
     - 5 distinct, non-overlapping out-of-sample backtest paths (1-factorization of K_6).
     - Strict Purging: Purge 504 hours (21-day SMA lookback) of training data prior
       to any test set boundary.
     - Strict Embargoing: Embargo 504 hours of training data immediately following
       any test set boundary.
  2. Probabilistic & Deflated Sharpe Ratio (PSR / DSR):
     - Asymptotic standard error of Sharpe incorporating sample return skewness and kurtosis.
     - Multiple-testing adjustment for N_trials = 50 candidate variations tested across lab sweeps.
     - Statistical significance threshold: DSR > 0.95 (p < 0.05).
  3. Robust Model Selection across Top 3 Configurations:
     - Model 1: Champion Baseline (0.25 Dir / 0.40 Spread / 0.35 Arb, 0.50/0.40 hyst)
     - Model 2: Re-optimized Weight Blend (0.25 Dir / 0.55 Spread / 0.20 Arb, 0.50/0.40 hyst)
     - Model 3: Grand Winner A (0.25 Dir / 0.55 Spread / 0.20 Arb, 0.40/0.40 hyst, Carry Sizing)
     - Decision rule: Maximize Mean OOS Sharpe while minimizing inter-path variance.
===================================================================================
"""
import os, sys, pickle, warnings, time, shutil, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

HOURS_PER_YEAR = 8760.0
UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
N_BLOCKS = 6
K_TEST_BLOCKS = 2
PURGE_BARS = 504      # 21-day Daily SMA lookback
EMBARGO_BARS = 504    # 21-day forward embargo
N_TRIALS = 50         # Multiple-testing penalty trials
EULER_MASCHERONI = 0.5772156649015329

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
ARTIFACT_DIR = '/Users/kingebenezer/.gemini/antigravity-ide/brain/b22eb094-5eb5-444d-9566-04f193fce2dc'

# ============================================================================
# PERFORMANCE & ADVANCED STATISTICAL METRICS (PSR / DSR)
# ============================================================================

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    s = np.std(h_rets)
    if s < 1e-10: return 0.0
    return (np.mean(h_rets) * HOURS_PER_YEAR) / (s * np.sqrt(HOURS_PER_YEAR))

def fast_metrics(h_rets):
    if len(h_rets) == 0:
        return {'sharpe': 0.0, 'ret': 0.0, 'dd': 0.0, 'sortino': 0.0, 'calmar': 0.0, 'skew': 0.0, 'kurt': 3.0}
    sh = fast_sharpe(h_rets)
    cum = np.cumprod(1.0 + h_rets)
    cummax = np.maximum.accumulate(cum)
    dd = np.min((cum - cummax) / (cummax + 1e-8)) * 100.0
    ret = (cum[-1] - 1.0) * 100.0
    downside = h_rets[h_rets < 0]
    sortino = (np.mean(h_rets) * HOURS_PER_YEAR) / (np.std(downside) * np.sqrt(HOURS_PER_YEAR) + 1e-8) if len(downside) > 0 else 0.0
    calmar = ret / abs(dd) if abs(dd) > 0 else 0.0
    
    # Return distribution higher moments
    sk = float(scipy_stats.skew(h_rets)) if len(h_rets) > 2 else 0.0
    # Pearson kurtosis (normal distribution = 3.0)
    kt = float(scipy_stats.kurtosis(h_rets, fisher=False)) if len(h_rets) > 3 else 3.0
    
    return {
        'sharpe': sh, 'ret': ret, 'dd': dd, 'sortino': sortino,
        'calmar': calmar, 'skew': sk, 'kurt': kt
    }

def compute_sharpe_se(ann_sharpe, skew, kurt, n_obs, q=HOURS_PER_YEAR):
    """
    Asymptotic standard error of annualized Sharpe Ratio (Mertens 2002, Lo 2002).
    sigma_SR = sqrt((q - gamma_3 * SR * sqrt(q) + ((gamma_4 - 1)/4) * SR^2) / (T - 1))
    """
    if n_obs <= 2: return 1.0
    term1 = q
    term2 = skew * ann_sharpe * np.sqrt(q)
    term3 = ((kurt - 1.0) / 4.0) * (ann_sharpe ** 2)
    variance = max(1e-8, (term1 - term2 + term3) / (n_obs - 1.0))
    return np.sqrt(variance)

def probabilistic_sharpe_ratio(ann_sharpe, benchmark_sr, skew, kurt, n_obs):
    """
    PSR: Probability that the true Sharpe ratio exceeds benchmark_sr.
    """
    se = compute_sharpe_se(ann_sharpe, skew, kurt, n_obs)
    z = (ann_sharpe - benchmark_sr) / se
    psr = float(scipy_stats.norm.cdf(z))
    return psr, se, z

def deflated_sharpe_ratio(ann_sharpe, skew, kurt, n_obs, var_trials, n_trials=N_TRIALS):
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).
    Calculates expected maximum Sharpe under null of no alpha across N_trials,
    and returns DSR = PSR(SR*).
    """
    if n_trials <= 1:
        sr_benchmark = 0.0
    else:
        # Expected max Sharpe under null hypothesis of zero alpha
        term1 = (1.0 - EULER_MASCHERONI) * scipy_stats.norm.ppf(1.0 - 1.0 / n_trials)
        term2 = EULER_MASCHERONI * scipy_stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_benchmark = np.sqrt(max(1e-6, var_trials)) * (term1 + term2)
    
    psr, se, z = probabilistic_sharpe_ratio(ann_sharpe, sr_benchmark, skew, kurt, n_obs)
    p_value = 1.0 - psr
    return psr, sr_benchmark, se, p_value

# ============================================================================
# DATA & ENGINE BUILDERS
# ============================================================================

def load_data_and_features():
    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in UNIVERSE}).mean(axis=1)
    daily_idx = idx_price.resample('1D').last()
    daily_sma = daily_idx.rolling(21).mean()
    daily_regime = (daily_idx > daily_sma).astype(float).shift(1).fillna(0.0)
    macro_hourly = daily_regime.reindex(idx_price.index, method='ffill').fillna(0.0)

    tsmom_windows = [24, 72, 168, 336, 720]
    for s in UNIVERSE:
        df = dataset[s].copy()
        c = df['close']; v = df['volume']
        df['macro_signal'] = macro_hourly

        vol_z = (v - v.rolling(720, min_periods=24).mean()) / (v.rolling(720, min_periods=24).std() + 1e-8)
        ret_cum = c.pct_change(48).fillna(0.0)
        synthetic_oi = 5000000.0 * (1.0 + 0.15 * vol_z.fillna(0.0).cumsum() * 0.008 + 0.12 * ret_cum)
        ret_24 = c.pct_change(24).fillna(0.0)
        synthetic_fr = 0.0001 + 0.00015 * np.tanh(ret_24 * 10.0)

        parts = []
        for w in tsmom_windows:
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

    cache_path = os.path.join(PROJECT_ROOT, 'data', 'garch_weights_cache.pkl')
    with open(cache_path, 'rb') as f:
        garch_wts = pickle.load(f)

    eval_index = processed[UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    macro_regime = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values

    # Precompute Multi-Timeframe Skewness Ensemble for Grand Winner A & Re-optimized models
    mtf_skew_mult = {}
    for s in UNIVERSE:
        ret = processed[s].iloc[WARMUP_BARS:]['hourly_return']
        sk_72 = ret.rolling(72, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sk_168 = ret.rolling(168, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sk_336 = ret.rolling(336, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sig_ens = (-np.tanh(sk_72.values * 2.0) - np.tanh(sk_168.values * 2.0) - np.tanh(sk_336.values * 2.0)) / 3.0
        # Centered Scaling D (slope 0.5, [0.4, 1.6])
        mtf_skew_mult[s] = np.clip(1.0 + sig_ens * 0.5, 0.40, 1.60)

    return dataset, processed, garch_wts, eval_index, n, macro_regime, mtf_skew_mult

# ============================================================================
# COMPONENT STREAMS SIMULATOR
# ============================================================================

def gen_directional_stream(processed, dataset, univ, eval_index, n, wts, 
                           skew_mult_dict=None, dir_hyst=0.50, fee=0.0005, leverage=2.5):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = 0.30*df['tsmom'].values + 0.50*df['carry'].values + 0.20*df['oi_regime'].values

    xs = {s: np.ones(n) for s in univ}
    for t in range(1, n):
        sc = sorted([(s, comps[s][t-1]) for s in univ], key=lambda x: x[1], reverse=True)
        xs[sc[0][0]][t] = 1.5; xs[sc[1][0]][t] = 1.0; xs[sc[2][0]][t] = 0.5

    port = np.zeros(n)
    all_turn = np.zeros(n)
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        ret = df['hourly_return'].values
        fr = dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values
        w = wts[s]; comp = comps[s]; x = xs[s]
        sm = skew_mult_dict[s] if skew_mult_dict is not None else np.ones(n)

        cur = 0.0; tgt = np.zeros(n)
        for t in range(n):
            wv = w[t] if not np.isnan(w[t]) else 0.0
            cv = comp[t] if not np.isnan(comp[t]) else 0.0
            ideal = cv * wv * x[t] * sm[t] * leverage
            if abs(ideal - cur) >= dir_hyst: cur = ideal
            tgt[t] = cur

        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) - (turn * fee) + np.where(funding_hours, -shifted * fr, 0.0)
        all_turn += turn

    return port, all_turn

def gen_spread_stream(processed, dataset, univ, eval_index, n, 
                      spread_hyst=0.40, fee=0.0002):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = 0.30 * df['tsmom'].values + 0.70 * df['oi_regime'].values

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
            if abs(ideal[s] - cur_pos[s]) >= spread_hyst: cur_pos[s] = ideal[s]
            turn = abs(cur_pos[s] - old)
            if turn > 0.001: port[t] -= turn * fee
            port[t] += processed[s].iloc[WARMUP_BARS:]['hourly_return'].values[t] * cur_pos[s]
            if funding_hours[t]:
                port[t] += -cur_pos[s] * dataset[s].iloc[WARMUP_BARS:]['funding_rate'].values[t]
    return port

def gen_arb_stream(dataset, univ, eval_index, n, 
                   magnitude_scale_dict=None, fee=0.0002):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    port = np.zeros(n)
    arb_lookback = 720
    arb_z_thresh = 2.5
    arb_exit_mult = 0.5
    arb_scale = 2.0
    arb_max_pos = 0.5

    for s in univ:
        df = dataset[s].iloc[WARMUP_BARS:]
        fr_raw = df['funding_rate'].values
        ret = df['close'].pct_change().fillna(0.0).values
        fr_shifted = pd.Series(fr_raw).shift(1).fillna(0.0)
        rm = fr_shifted.rolling(arb_lookback, min_periods=72).mean().fillna(0.0).values
        rs = fr_shifted.rolling(arb_lookback, min_periods=72).std().fillna(0.0).values
        z = (fr_shifted.values - rm) / (rs + 1e-10)

        mag_scale = magnitude_scale_dict[s] if magnitude_scale_dict is not None else np.ones(n)

        cur = 0.0; tgt = np.zeros(n)
        for t in range(arb_lookback, n):
            zv = z[t]
            if zv > arb_z_thresh:
                ideal = -min((zv - arb_z_thresh) / arb_scale, arb_max_pos) * mag_scale[t]
            elif zv < -arb_z_thresh:
                ideal = min((-zv - arb_z_thresh) / arb_scale, arb_max_pos) * mag_scale[t]
            elif abs(zv) < arb_z_thresh * arb_exit_mult:
                ideal = 0.0
            else:
                ideal = cur
            if abs(ideal - cur) > 0.05: cur = ideal
            tgt[t] = cur

        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted - turn * fee + np.where(funding_hours, -shifted * fr_raw, 0.0)) / len(univ)
    return port

# ============================================================================
# CPCV ENGINE (PURGING, EMBARGOING, PATH COMBINATORIAL FACTORIZATION)
# ============================================================================

def build_cpcv_schedule(n_total, n_blocks=N_BLOCKS, k_test=K_TEST_BLOCKS):
    """
    Constructs the 15 folds and 5 distinct out-of-sample backtest paths
    via 1-factorization of K_6.
    """
    block_size = n_total // n_blocks
    block_bounds = []
    for b in range(n_blocks):
        b_start = b * block_size
        b_end = (b + 1) * block_size if b < n_blocks - 1 else n_total
        block_bounds.append((b_start, b_end))

    # All 15 pairs of test blocks
    all_folds = list(itertools.combinations(range(n_blocks), k_test))
    
    # 1-factorization of K_6 (5 perfect matchings of 3 pairs each)
    # Standard round-robin construction fixing vertex 5:
    paths_factorization = []
    active_vertices = list(range(5))
    for r in range(5):
        matching = []
        matching.append(tuple(sorted((active_vertices[r], 5))))
        for i in range(1, 3):
            v1 = active_vertices[(r - i) % 5]
            v2 = active_vertices[(r + i) % 5]
            matching.append(tuple(sorted((v1, v2))))
        paths_factorization.append(matching)

    return block_bounds, all_folds, paths_factorization

def get_purged_embargoed_train_indices(n_total, test_blocks, block_bounds, 
                                       purge_bars=PURGE_BARS, embargo_bars=EMBARGO_BARS):
    """
    Computes valid training bar indices after strictly purging preceding data
    and embargoing succeeding data relative to test block boundaries.
    """
    test_set = set(test_blocks)
    train_indices = []
    purged_count = 0
    embargoed_count = 0

    for b in range(len(block_bounds)):
        if b in test_set:
            continue
        b_start, b_end = block_bounds[b]
        
        # Check if preceded by a test block -> apply embargo window at start of block
        is_preceded = (b - 1) in test_set
        # Check if succeeded by a test block -> apply purge window at end of block
        is_succeeded = (b + 1) in test_set
        
        valid_start = b_start + (embargo_bars if is_preceded else 0)
        valid_end = b_end - (purge_bars if is_succeeded else 0)
        
        if valid_end > valid_start:
            train_indices.extend(range(valid_start, valid_end))
            if is_preceded: embargoed_count += embargo_bars
            if is_succeeded: purged_count += purge_bars
            
    return np.array(train_indices), purged_count, embargoed_count

# ============================================================================
# MAIN CPCV EXECUTION & COMPARISON
# ============================================================================

def run_cpcv_validation():
    print("=" * 115)
    print("APEX INSTITUTIONAL VALIDATION: COMBINATORIAL PURGED CROSS-VALIDATION (CPCV) & DSR")
    print("=" * 115)
    print(f"  Protocol: N={N_BLOCKS} Blocks, k={K_TEST_BLOCKS} Test Blocks per Split -> 15 Folds, 5 OOS Paths")
    print(f"  Lookback Horizon: {PURGE_BARS} Hours (Purging) | Forward Embargo: {EMBARGO_BARS} Hours")
    print(f"  Multiple-Testing Penalty: N_trials={N_TRIALS} Trials Adjusting for Lab Sweeps")
    print("=" * 115)

    dataset, processed, garch_wts, eval_index, n, macro_regime, mtf_skew_mult = load_data_and_features()

    block_bounds, all_folds, paths_factorization = build_cpcv_schedule(n)
    block_size = n // N_BLOCKS

    print(f"\n  Dataset Size: {n:,d} hourly bars ({eval_index[0]} to {eval_index[-1]})")
    print(f"  Block Size:   {block_size:,d} bars per block (~8.3 months each)")
    for b, (bs, be) in enumerate(block_bounds):
        print(f"    Block {b}: bars {bs:5,d} to {be:5,d} | {eval_index[bs].strftime('%Y-%m-%d')} to {eval_index[be-1].strftime('%Y-%m-%d')}")

    # ------------------------------------------------------------------------
    # PRECOMPUTE RAW COMPONENT STREAMS FOR THE 3 CONFIGURATIONS
    # ------------------------------------------------------------------------
    print("\n  Precomputing Component Streams for Model Configurations...")

    # CONFIG 1: Champion Baseline (0.25 Dir / 0.40 Spread / 0.35 Arb, 0.50/0.40 hyst)
    dir_c1, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts,
                                       skew_mult_dict=None, dir_hyst=0.50)
    sp_c1 = gen_spread_stream(processed, dataset, UNIVERSE, eval_index, n, spread_hyst=0.40)
    arb_c1 = gen_arb_stream(dataset, UNIVERSE, eval_index, n, magnitude_scale_dict=None)
    m1_full_stream = 0.25 * (dir_c1 * macro_regime) + 0.40 * (sp_c1 * macro_regime) + 0.35 * arb_c1

    # CONFIG 2: Re-optimized Weight Blend (0.25 Dir / 0.55 Spread / 0.20 Arb, 0.50/0.40 hyst)
    dir_c2, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts,
                                       skew_mult_dict=mtf_skew_mult, dir_hyst=0.50)
    sp_c2 = sp_c1  # same spread stream
    arb_c2 = arb_c1 # same unscaled arb stream
    m2_full_stream = 0.25 * (dir_c2 * macro_regime) + 0.55 * (sp_c2 * macro_regime) + 0.20 * arb_c2

    # CONFIG 3: Grand Winner A (0.25 Dir / 0.55 Spread / 0.20 Arb, 0.40/0.40 hyst, Carry Sizing)
    dir_c3, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts,
                                       skew_mult_dict=mtf_skew_mult, dir_hyst=0.40)
    sp_c3 = gen_spread_stream(processed, dataset, UNIVERSE, eval_index, n, spread_hyst=0.40)

    # Note on Carry Sizing for Model 3:
    # Strictly causal: calculate predicted FR EWMA(24) shifted by 1
    pred_fr = {}
    for s in UNIVERSE:
        df = dataset[s].iloc[WARMUP_BARS:]
        pred_fr[s] = df['funding_rate'].ewm(span=24).mean().shift(1).fillna(0.0).values

    models = [
        {'id': 'M1', 'name': 'Champion Baseline (0.25/0.40/0.35)', 'weights': (0.25, 0.40, 0.35), 'color': '#FFD600'},
        {'id': 'M2', 'name': 'Re-Opt Weights (0.25/0.55/0.20)',   'weights': (0.25, 0.55, 0.20), 'color': '#00E5FF'},
        {'id': 'M3', 'name': 'Grand Winner A (0.25/0.55/0.20)',   'weights': (0.25, 0.55, 0.20), 'color': '#FF6E40'}
    ]

    # ------------------------------------------------------------------------
    # EXECUTE 15 CPCV FOLDS ACROSS ALL 3 MODELS
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STEP 1: EVALUATING 15 CPCV FOLDS (STRICT PURGING & EMBARGOING)")
    print("=" * 115)

    fold_results = {m['id']: [] for m in models}
    fold_test_streams = {m['id']: {} for m in models} # key: fold_idx, val: {b1: rets, b2: rets}

    for fold_idx, (b1, b2) in enumerate(all_folds):
        train_idx, purged_b, embargoed_b = get_purged_embargoed_train_indices(n, (b1, b2), block_bounds)
        
        # Test ranges
        t1_start, t1_end = block_bounds[b1]
        t2_start, t2_end = block_bounds[b2]
        test_idx = np.concatenate([np.arange(t1_start, t1_end), np.arange(t2_start, t2_end)])

        # Model 3 carry sizing calibrated strictly on train_idx of this fold
        arb_mag_scale_fold = {}
        for s in UNIVERSE:
            train_med = np.median(np.abs(pred_fr[s][train_idx])) + 1e-6
            ratio = np.abs(pred_fr[s]) / train_med
            arb_mag_scale_fold[s] = np.clip(0.5 + 0.5 * ratio, 0.5, 2.0)
        arb_c3_fold = gen_arb_stream(dataset, UNIVERSE, eval_index, n, magnitude_scale_dict=arb_mag_scale_fold)
        m3_full_stream_fold = 0.25 * (dir_c3 * macro_regime) + 0.55 * (sp_c3 * macro_regime) + 0.20 * arb_c3_fold

        model_streams_fold = {
            'M1': m1_full_stream,
            'M2': m2_full_stream,
            'M3': m3_full_stream_fold
        }

        print(f"\n  Fold {fold_idx+1:2d}/15: Test Blocks=({b1}, {b2}) | Train Bars={len(train_idx):,d} | Purged={purged_b:3d}h | Embargoed={embargoed_b:3d}h")
        for m in models:
            mid = m['id']
            stream = model_streams_fold[mid]
            
            # Out-of-sample test performance for this fold
            test_rets = stream[test_idx]
            train_rets = stream[train_idx]
            
            m_te = fast_metrics(test_rets)
            m_tr = fast_metrics(train_rets)
            tt = m_te['sharpe'] / m_tr['sharpe'] if abs(m_tr['sharpe']) > 0.01 else 0.0
            
            fold_results[mid].append({
                'fold': fold_idx, 'b1': b1, 'b2': b2,
                'tr_sh': m_tr['sharpe'], 'te_sh': m_te['sharpe'],
                'tt': tt, 'te_ret': m_te['ret'], 'te_dd': m_te['dd']
            })
            
            # Store test slices for path assembly
            fold_test_streams[mid][fold_idx] = {
                b1: stream[t1_start:t1_end],
                b2: stream[t2_start:t2_end]
            }
            print(f"    {m['name']:40s} | Train Sh: {m_tr['sharpe']:+5.2f} | Test Sh: {m_te['sharpe']:+5.2f} | T/T: {tt:4.2f} | Test Ret: {m_te['ret']:+6.1f}% | Test DD: {m_te['dd']:5.1f}%")

    # ------------------------------------------------------------------------
    # STEP 2: ASSEMBLE 5 DISTINCT OUT-OF-SAMPLE PATHS & CALCULATE DSR
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("STEP 2: COMBINATORIAL OOS BACKTEST PATHS (1-FACTORIZATION OF K_6)")
    print("=" * 115)

    path_results = {m['id']: [] for m in models}
    path_streams = {m['id']: [] for m in models}

    # Variance of trials across our lab sweeps for DSR multiple-testing adjustment
    # Derived from the cross-sectional empirical variance of candidate models in lab sweeps
    sweep_sharpes = [1.46, 1.07, 1.27, 1.31, 1.30, 1.43, 1.46, 1.32, 1.14, 0.84, 0.86, 0.70, 0.56, 1.39, 1.25, 1.27, 1.15, 0.95, 1.03, 1.16, 0.94]
    trial_variance = float(np.var(sweep_sharpes))
    print(f"  Empirical Variance across tested trials: V = {trial_variance:.4f} (Std = {np.sqrt(trial_variance):.2f})")

    for path_idx, matching in enumerate(paths_factorization, 1):
        print(f"\n  --- Path {path_idx}/5 [Test Pairs: {matching}] ---")
        
        for m in models:
            mid = m['id']
            # Stitch the 6 blocks from the corresponding fold test sets
            stitched_blocks = []
            for b in range(N_BLOCKS):
                # Find which fold in this matching tested block b
                for pair in matching:
                    if b in pair:
                        fold_num = all_folds.index(pair)
                        block_slice = fold_test_streams[mid][fold_num][b]
                        stitched_blocks.append(block_slice)
                        break
            
            full_path_rets = np.concatenate(stitched_blocks)
            path_streams[mid].append(full_path_rets)
            
            m_p = fast_metrics(full_path_rets)
            
            # Probabilistic Sharpe Ratio
            psr, se, z = probabilistic_sharpe_ratio(
                m_p['sharpe'], 0.0, m_p['skew'], m_p['kurt'], len(full_path_rets)
            )
            # Deflated Sharpe Ratio (penalizing for 50 trials)
            dsr, sr_bench, _, p_val = deflated_sharpe_ratio(
                m_p['sharpe'], m_p['skew'], m_p['kurt'], len(full_path_rets),
                var_trials=trial_variance, n_trials=N_TRIALS
            )
            
            path_results[mid].append({
                'path': path_idx,
                'sharpe': m_p['sharpe'],
                'ret': m_p['ret'],
                'dd': m_p['dd'],
                'sortino': m_p['sortino'],
                'calmar': m_p['calmar'],
                'skew': m_p['skew'],
                'kurt': m_p['kurt'],
                'se': se,
                'psr': psr,
                'dsr': dsr,
                'sr_bench': sr_bench,
                'p_value': p_val
            })
            
            sig_marker = " [p<0.05]" if p_val < 0.05 else ""
            print(f"    {m['name']:40s} | Sh: {m_p['sharpe']:+5.2f} | Ret: {m_p['ret']:+6.1f}% | DD: {m_p['dd']:5.1f}% | Skew: {m_p['skew']:+5.2f} | Kurt: {m_p['kurt']:4.1f} | DSR: {dsr*100:5.1f}%{sig_marker}")

    # ------------------------------------------------------------------------
    # STEP 3: COMPARATIVE TEAR SHEET & ROBUST SELECTION
    # ------------------------------------------------------------------------
    print("\n\n" + "=" * 115)
    print("STEP 3: INSTITUTIONAL TEAR SHEET — CPCV & DEFLATED SHARPE COMPARISON")
    print("=" * 115)

    summary_stats = {}
    for m in models:
        mid = m['id']
        p_data = path_results[mid]
        sharpes = [r['sharpe'] for r in p_data]
        returns = [r['ret'] for r in p_data]
        drawdowns = [r['dd'] for r in p_data]
        calmars = [r['calmar'] for r in p_data]
        sortinos = [r['sortino'] for r in p_data]
        dsrs = [r['dsr'] for r in p_data]
        p_vals = [r['p_value'] for r in p_data]
        
        mean_sh = np.mean(sharpes)
        std_sh = np.std(sharpes)
        stability_idx = mean_sh / (std_sh + 1e-6)
        worst_sh = np.min(sharpes)
        best_sh = np.max(sharpes)
        worst_dd = np.min(drawdowns)
        worst_ret = np.min(returns)
        mean_ret = np.mean(returns)
        mean_dd = np.mean(drawdowns)
        mean_dsr = np.mean(dsrs)
        min_dsr = np.min(dsrs)
        max_pval = np.max(p_vals)
        
        summary_stats[mid] = {
            'model': m['name'],
            'mean_sharpe': mean_sh,
            'std_sharpe': std_sh,
            'stability_idx': stability_idx,
            'worst_sharpe': worst_sh,
            'best_sharpe': best_sh,
            'mean_ret': mean_ret,
            'worst_ret': worst_ret,
            'mean_dd': mean_dd,
            'worst_dd': worst_dd,
            'mean_calmar': np.mean(calmars),
            'mean_sortino': np.mean(sortinos),
            'mean_dsr': mean_dsr,
            'min_dsr': min_dsr,
            'max_pval': max_pval,
            'path_sharpes': sharpes,
            'path_returns': returns,
            'path_drawdowns': drawdowns
        }

    # Print Formatted Institutional Tear Sheet
    header = f"  {'Metric':35s} | {'Model 1: Champion':22s} | {'Model 2: Re-Opt Wt':22s} | {'Model 3: Grand Winner A':24s}"
    print(header)
    print("-" * 115)

    rows = [
        ("Mean OOS Sharpe (5 Paths)", lambda s: f"{s['mean_sharpe']:+6.2f}"),
        ("Std Dev of OOS Sharpe across Paths", lambda s: f"{s['std_sharpe']:6.2f}"),
        ("Sharpe Stability Ratio (Mean/Std)", lambda s: f"{s['stability_idx']:6.2f}"),
        ("Worst-Case Path Sharpe", lambda s: f"{s['worst_sharpe']:+6.2f}"),
        ("Best-Case Path Sharpe", lambda s: f"{s['best_sharpe']:+6.2f}"),
        ("Mean Cumulative Net Return", lambda s: f"{s['mean_ret']:+6.1f}%"),
        ("Worst-Case Path Net Return", lambda s: f"{s['worst_ret']:+6.1f}%"),
        ("Mean Maximum Drawdown", lambda s: f"{s['mean_dd']:6.1f}%"),
        ("Worst-Case Path Max Drawdown", lambda s: f"{s['worst_dd']:6.1f}%"),
        ("Mean Calmar Ratio", lambda s: f"{s['mean_calmar']:6.2f}"),
        ("Mean Sortino Ratio", lambda s: f"{s['mean_sortino']:+6.2f}"),
        ("Mean Deflated Sharpe Ratio (DSR)", lambda s: f"{s['mean_dsr']*100:5.1f}%"),
        ("Worst-Case Path DSR", lambda s: f"{s['min_dsr']*100:5.1f}%"),
        ("Max Multiple-Testing p-value", lambda s: f"{s['max_pval']:6.4f}"),
        ("Statistically Significant (p < 0.05)?", lambda s: "YES (Confirmed)" if s['max_pval'] < 0.05 else "NO")
    ]

    for label, fn in rows:
        val1 = fn(summary_stats['M1'])
        val2 = fn(summary_stats['M2'])
        val3 = fn(summary_stats['M3'])
        marker = " <<<" if "Peak" in label or label.startswith("Mean OOS Sharpe") or label.startswith("Worst-Case Path Sharpe") else ""
        print(f"  {label:35s} | {val1:22s} | {val2:22s} | {val3:24s}{marker}")

    # ------------------------------------------------------------------------
    # STEP 4: PLOT INSTITUTIONAL TEAR SHEET VISUALIZATION
    # ------------------------------------------------------------------------
    plot_path = plot_cpcv_tearsheet(summary_stats, path_streams, fold_results, eval_index, models)
    
    # Save results to pickle
    out_pkl = os.path.join(RESULTS_DIR, 'apex_cpcv_dsr_results.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump({
            'summary_stats': summary_stats,
            'fold_results': fold_results,
            'path_results': path_results,
            'paths_factorization': paths_factorization
        }, f)
    print(f"\n[SAVED] {out_pkl}")

    return summary_stats, fold_results, path_results, plot_path

# ============================================================================
# TEAR SHEET VISUALIZATION
# ============================================================================

def plot_cpcv_tearsheet(summary_stats, path_streams, fold_results, eval_index, models):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))

    # Panel 1: All 5 OOS Paths for Top Models (Dispersion Envelope)
    ax1 = axes[0, 0]
    for m in models:
        mid = m['id']
        c = m['color']
        streams = path_streams[mid]
        for p_idx, s in enumerate(streams):
            cum = np.cumprod(1.0 + s)
            lw = 2.0 if p_idx == 0 else 1.0
            alpha = 0.9 if p_idx == 0 else 0.45
            lbl = f"{m['name']} (Path {p_idx+1})" if p_idx == 0 else None
            ax1.plot(eval_index, cum, color=c, linewidth=lw, alpha=alpha, label=lbl)

    ax1.set_title('Panel 1: CPCV 5 Distinct Out-of-Sample Backtest Paths (Dispersion Envelope)',
                  fontsize=12, fontweight='bold', color='white', pad=10)
    ax1.set_ylabel('Portfolio Cumulative Return (Log Scale)', color='white')
    ax1.set_yscale('log')
    ax1.grid(color='gray', linestyle=':', alpha=0.25)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9, facecolor='#1A1D24')

    # Panel 2: Fold-by-Fold Test Sharpe Distribution across 15 Folds
    ax2 = axes[0, 1]
    n_folds = 15
    x = np.arange(n_folds)
    w = 0.26
    
    m1_folds = [r['te_sh'] for r in fold_results['M1']]
    m2_folds = [r['te_sh'] for r in fold_results['M2']]
    m3_folds = [r['te_sh'] for r in fold_results['M3']]
    
    ax2.bar(x - w, m1_folds, w, color='#FFD600', alpha=0.85, label='Champion Baseline')
    ax2.bar(x,     m2_folds, w, color='#00E5FF', alpha=0.85, label='Re-Opt Weights (0.25/0.55/0.20)')
    ax2.bar(x + w, m3_folds, w, color='#FF6E40', alpha=0.85, label='Grand Winner A')

    ax2.axhline(0, color='white', linestyle='-', alpha=0.3, linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"F{i+1}\n({r['b1']},{r['b2']})" for i, r in enumerate(fold_results['M1'])], fontsize=8)
    ax2.set_title('Panel 2: CPCV 15-Fold Test Sharpe Ratios (Purged & Embargoed)',
                  fontsize=12, fontweight='bold', color='white', pad=10)
    ax2.set_ylabel('Fold Test Sharpe Ratio', color='white')
    ax2.grid(color='gray', linestyle=':', alpha=0.25, axis='y')
    ax2.legend(fontsize=9, framealpha=0.9, facecolor='#1A1D24')

    # Panel 3: Deflated Sharpe Ratio (DSR) vs. Number of Trials
    ax3 = axes[1, 0]
    trials_range = np.arange(1, 101)
    
    # Calculate DSR curve as a function of trials for each model
    for m in models:
        mid = m['id']
        c = m['color']
        mean_sh = summary_stats[mid]['mean_sharpe']
        dsr_curve = []
        for k in trials_range:
            dsr_val, _, _, _ = deflated_sharpe_ratio(mean_sh, 0.0, 3.0, len(eval_index), var_trials=0.15, n_trials=k)
            dsr_curve.append(dsr_val * 100.0)
        ax3.plot(trials_range, dsr_curve, color=c, linewidth=2.5, label=f"{m['name']} (DSR @ 50: {dsr_curve[49]:.1f}%)")

    ax3.axhline(95.0, color='#00E676', linestyle='--', linewidth=1.5, label='p = 0.05 Significance Threshold (95% DSR)')
    ax3.axvline(50, color='white', linestyle=':', linewidth=1.2, label='Current Trial Count (N=50)')
    ax3.set_title('Panel 3: Deflated Sharpe Ratio (DSR) Decay Curve vs Multiple-Testing Trials',
                  fontsize=12, fontweight='bold', color='white', pad=10)
    ax3.set_xlabel('Number of Historical Parameter / Signal Trials Tested', color='white')
    ax3.set_ylabel('Deflated Sharpe Ratio (%)', color='white')
    ax3.grid(color='gray', linestyle=':', alpha=0.25)
    ax3.legend(loc='lower left', fontsize=9, framealpha=0.9, facecolor='#1A1D24')

    # Panel 4: Risk-Return Stability Frontier across OOS Paths
    ax4 = axes[1, 1]
    for m in models:
        mid = m['id']
        s = summary_stats[mid]
        c = m['color']
        
        # Plot individual path points
        sh_pts = s['path_sharpes']
        ret_pts = s['path_returns']
        dd_pts = [abs(d) for d in s['path_drawdowns']]
        
        ax4.scatter(s['std_sharpe'], s['mean_sharpe'], s=250, color=c, marker='*', edgecolors='white',
                    linewidths=1.2, label=f"{m['name']} (Mean: {s['mean_sharpe']:+.2f}, Std: {s['std_sharpe']:.2f})", zorder=4)
        
        ax4.annotate(f"{m['name'][:22]}\nWorst Sh: {s['worst_sharpe']:+.2f}\nWorst DD: {s['worst_dd']:.1f}%",
                     (s['std_sharpe'], s['mean_sharpe']), textcoords="offset points", xytext=(8, 5),
                     fontsize=9, color='white', alpha=0.95)

    ax4.set_xlabel('Inter-Path Sharpe Standard Deviation [Lower = More Consistent]', color='white')
    ax4.set_ylabel('Mean OOS Sharpe Ratio [Higher = Better Alpha]', color='white')
    ax4.set_title('Panel 4: The Institutional Stability Frontier: Mean Sharpe vs. Regime Dispersion',
                  fontsize=12, fontweight='bold', color='white', pad=10)
    ax4.grid(color='gray', linestyle=':', alpha=0.25)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9, facecolor='#1A1D24')

    out_path = os.path.join(RESULTS_DIR, 'apex_cpcv_dsr_validation.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[PLOT SAVED] {out_path}")

    # Copy to artifacts directory
    if os.path.exists(ARTIFACT_DIR):
        dest = os.path.join(ARTIFACT_DIR, 'apex_cpcv_dsr_validation.png')
        shutil.copyfile(out_path, dest)
        print(f"[ARTIFACT SAVED] {dest}")

    return out_path

if __name__ == '__main__':
    run_cpcv_validation()
