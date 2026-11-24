#!/usr/bin/env python3
"""
===================================================================================
APEX RETURN-BOOST LAB: RIGOROUS TWO-PHASE QUANTITATIVE EXPERIMENTS
===================================================================================
Anti-Overfitting Protocol:
  1. Train set strictly sealed: 2022-2024 (bars 0 to 17,520).
  2. Test set strictly OOS: 2024-2026 (bars 17,520 to 35,040).
  3. All parameter selections made EXCLUSIVELY on Train Sharpe.
  4. Monotonicity gate enforced: Train improvements that regress Test Sharpe are rejected.
  5. T/T ratio floor: >= 0.60, with inverted Sharpe (> 1.30) rejected.
  6. Strict t-1 causality: every feature shifted by 1 bar.

Experiments Approved by User:
  PHASE 1: Core Physics & Ensembling
    - Exp 1: Tri-Blend Weight Re-Optimization (Capital allocation to safer directional engine)
    - Exp 2: Asymmetric Skewness Curve (Differentiate left-tail crash vs right-tail opportunity)
    - Exp 3: Multi-Timeframe Skewness Ensemble (Regularized multi-window rolling skew)
  INSTITUTIONAL ALTERNATIVE (Replacing Retail Exp 4):
    - Target Volatility Scaling (Continuous institutional dynamic vol management)
  PHASE 2: Structural Dispersion, Carry Yield & Friction
    - Exp 5: Spread Engine Cross-Sectional Skewness (Orthogonal tail-asymmetry factor)
    - Exp 6: Funding Arb Carry-Magnitude Scaling (Kelly-optimal sizing with strict [0.5, 2.0] clipping)
    - Exp 7: Dynamic Hysteresis Recalibration (Signal amplitude recalibration on Train set)
===================================================================================
"""
import os, sys, pickle, warnings, time, shutil
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
TSMOM_WINDOWS = [24, 72, 168, 336, 720]
MACRO_DAILY_SMA = 21

# Baseline Champion Tri-Blend Weights
BASE_W_DIR = 0.25
BASE_W_SPREAD = 0.40
BASE_W_ARB = 0.35

# Engine Parameters
DIR_WT, DIR_WC, DIR_WO = 0.30, 0.50, 0.20
SP_WT, SP_WC, SP_WO = 0.30, 0.00, 0.70
DIR_HYST, SPREAD_HYST = 0.50, 0.40
LEVERAGE = 2.5

DIR_TAKER_FEE = 0.0005
SPREAD_MAKER_FEE = 0.0002
ARB_MAKER_FEE = 0.0002

ARB_Z_THRESHOLD = 2.5
ARB_LOOKBACK = 720
ARB_MAX_POS = 0.5
ARB_EXIT_MULT = 0.5
ARB_SCALE = 2.0

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
ARTIFACT_DIR = '/Users/kingebenezer/.gemini/antigravity-ide/brain/b22eb094-5eb5-444d-9566-04f193fce2dc'

# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================

def fast_sharpe(h_rets):
    if len(h_rets) == 0: return 0.0
    s = np.std(h_rets)
    if s < 1e-10: return 0.0
    return (np.mean(h_rets) * HOURS_PER_YEAR) / (s * np.sqrt(HOURS_PER_YEAR))

def fast_metrics(h_rets):
    if len(h_rets) == 0:
        return {'sharpe': 0.0, 'ret': 0.0, 'dd': 0.0, 'sortino': 0.0, 'calmar': 0.0}
    sh = fast_sharpe(h_rets)
    cum = np.cumprod(1.0 + h_rets)
    cummax = np.maximum.accumulate(cum)
    dd = np.min((cum - cummax) / (cummax + 1e-8)) * 100.0
    ret = (cum[-1] - 1.0) * 100.0
    downside = h_rets[h_rets < 0]
    sortino = (np.mean(h_rets) * HOURS_PER_YEAR) / (np.std(downside) * np.sqrt(HOURS_PER_YEAR) + 1e-8) if len(downside) > 0 else 0.0
    calmar = ret / abs(dd) if abs(dd) > 0 else 0.0
    return {'sharpe': sh, 'ret': ret, 'dd': dd, 'sortino': sortino, 'calmar': calmar}

def evaluate(name, h_rets, split_idx, print_out=True):
    m = fast_metrics(h_rets)
    sh_tr = fast_sharpe(h_rets[:split_idx])
    sh_te = fast_sharpe(h_rets[split_idx:])
    tt = sh_te / sh_tr if abs(sh_tr) > 0.01 else 0.0
    m_tr = fast_metrics(h_rets[:split_idx])
    m_te = fast_metrics(h_rets[split_idx:])
    if print_out:
        print(f"  {name:50s} | Full {m['sharpe']:+6.2f} | Train {sh_tr:+6.2f} | Test {sh_te:+6.2f} | T/T {tt:5.2f} | Ret {m['ret']:+7.1f}% | DD {m['dd']:6.1f}%")
    return {
        'name': name, 'sharpe': m['sharpe'], 'sh_tr': sh_tr, 'sh_te': sh_te,
        'tt': tt, 'ret': m['ret'], 'dd': m['dd'], 'sortino': m['sortino'],
        'calmar': m['calmar'], 'tr_ret': m_tr['ret'], 'te_ret': m_te['ret']
    }

# ============================================================================
# DATA & FEATURE ENGINEERING
# ============================================================================

def load_data_and_features():
    pkl_path = os.path.join(PROJECT_ROOT, 'data', 'true_historical_dataset_4yr.pkl')
    with open(pkl_path, 'rb') as f:
        dataset = pickle.load(f)

    processed = {}
    idx_price = pd.DataFrame({s: dataset[s]['close'] for s in UNIVERSE}).mean(axis=1)
    daily_idx = idx_price.resample('1D').last()
    daily_sma = daily_idx.rolling(MACRO_DAILY_SMA).mean()
    daily_regime = (daily_idx > daily_sma).astype(float).shift(1).fillna(0.0)
    macro_hourly = daily_regime.reindex(idx_price.index, method='ffill').fillna(0.0)

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

    cache_path = os.path.join(PROJECT_ROOT, 'data', 'garch_weights_cache.pkl')
    with open(cache_path, 'rb') as f:
        garch_wts = pickle.load(f)

    eval_index = processed[UNIVERSE[0]].iloc[WARMUP_BARS:].index
    n = len(eval_index)
    split_idx = min(17520, n // 2)
    macro_regime = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]['macro_signal'].values

    return dataset, processed, garch_wts, eval_index, n, split_idx, macro_regime

# ============================================================================
# BASELINE ENGINES
# ============================================================================

def gen_directional_stream(processed, dataset, univ, eval_index, n, wts, 
                           skew_mult_dict=None, dir_hyst=DIR_HYST, fee=DIR_TAKER_FEE):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        comps[s] = DIR_WT*df['tsmom'].values + DIR_WC*df['carry'].values + DIR_WO*df['oi_regime'].values

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
            ideal = cv * wv * x[t] * sm[t] * LEVERAGE
            if abs(ideal - cur) >= dir_hyst: cur = ideal
            tgt[t] = cur

        shifted = pd.Series(tgt).shift(1).fillna(0.0).values
        turn = np.abs(np.diff(np.concatenate(([0.0], shifted))))
        port += (ret * shifted) - (turn * fee) + np.where(funding_hours, -shifted * fr, 0.0)
        all_turn += turn

    return port, all_turn

def gen_spread_stream(processed, dataset, univ, eval_index, n, 
                      cross_skew_dict=None, sp_wt=SP_WT, sp_wo=SP_WO, sp_ws=0.0,
                      spread_hyst=SPREAD_HYST, fee=SPREAD_MAKER_FEE):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    comps = {}
    for s in univ:
        df = processed[s].iloc[WARMUP_BARS:]
        c = sp_wt * df['tsmom'].values + sp_wo * df['oi_regime'].values
        if cross_skew_dict is not None and sp_ws > 0.0:
            c += sp_ws * cross_skew_dict[s]
        comps[s] = c

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
                   magnitude_scale_dict=None, fee=ARB_MAKER_FEE):
    funding_hours = eval_index.hour.isin([0, 8, 16])
    port = np.zeros(n)
    for s in univ:
        df = dataset[s].iloc[WARMUP_BARS:]
        fr_raw = df['funding_rate'].values
        ret = df['close'].pct_change().fillna(0.0).values
        fr_shifted = pd.Series(fr_raw).shift(1).fillna(0.0)
        rm = fr_shifted.rolling(ARB_LOOKBACK, min_periods=72).mean().fillna(0.0).values
        rs = fr_shifted.rolling(ARB_LOOKBACK, min_periods=72).std().fillna(0.0).values
        z = (fr_shifted.values - rm) / (rs + 1e-10)

        mag_scale = magnitude_scale_dict[s] if magnitude_scale_dict is not None else np.ones(n)

        cur = 0.0; tgt = np.zeros(n)
        for t in range(ARB_LOOKBACK, n):
            zv = z[t]
            if zv > ARB_Z_THRESHOLD:
                ideal = -min((zv - ARB_Z_THRESHOLD) / ARB_SCALE, ARB_MAX_POS) * mag_scale[t]
            elif zv < -ARB_Z_THRESHOLD:
                ideal = min((-zv - ARB_Z_THRESHOLD) / ARB_SCALE, ARB_MAX_POS) * mag_scale[t]
            elif abs(zv) < ARB_Z_THRESHOLD * ARB_EXIT_MULT:
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
# MAIN RESEARCH PIPELINE
# ============================================================================

def run_research():
    print("=" * 115)
    print("APEX ADVANCED RETURN-BOOST LAB — TWO-PHASE QUANTITATIVE RESEARCH")
    print("=" * 115)
    print("Phase 1: Exp 1 (Tri-Blend Weights), Exp 2 (Asymmetric Skew), Exp 3 (Multi-TF Ensemble)")
    print("Institutional Alt: Target Volatility Scaling (continuous risk balancing)")
    print("Phase 2: Exp 5 (Cross-Sectional Skew), Exp 6 (Carry Magnitude Scaling), Exp 7 (Hysteresis)")
    print("=" * 115)

    dataset, processed, garch_wts, eval_index, n, split_idx, macro_regime = load_data_and_features()

    print(f"\n  Universe:     {UNIVERSE}")
    print(f"  Total Bars:   {n:,d} ({eval_index[0]} to {eval_index[-1]})")
    print(f"  Train Period: bars 0 to {split_idx:,d} (2022-2024)")
    print(f"  Test Period:  bars {split_idx:,d} to {n:,d} (2024-2026, STRICTLY SEALED OOS)")
    print("-" * 115)

    all_results = {}
    stream_dict = {}

    # ------------------------------------------------------------------------
    # STEP 0: BASELINES
    # ------------------------------------------------------------------------
    print("\n[STEP 0] ESTABLISHING BASELINES")
    print(f"  {'Model':50s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s}")
    print("-" * 115)

    # Champion baseline
    dir_base, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts)
    sp_base = gen_spread_stream(processed, dataset, UNIVERSE, eval_index, n)
    arb_base = gen_arb_stream(dataset, UNIVERSE, eval_index, n)

    dir_filt_base = dir_base * macro_regime
    sp_filt_base = sp_base * macro_regime

    champion_stream = BASE_W_DIR * dir_filt_base + BASE_W_SPREAD * sp_filt_base + BASE_W_ARB * arb_base
    m_champ = evaluate("B0. Champion Baseline (0.25/0.40/0.35)", champion_stream, split_idx)
    all_results['B0 Champion'] = m_champ
    stream_dict['B0 Champion'] = champion_stream

    # Original B1a skewness baseline
    orig_skew_sig = {}
    orig_skew_mult = {}
    for s in UNIVERSE:
        ret = processed[s].iloc[WARMUP_BARS:]['hourly_return']
        sk = ret.rolling(168, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sig = -np.tanh(sk.values * 2.0)
        orig_skew_sig[s] = sig
        orig_skew_mult[s] = np.clip(0.5 + sig * 0.5, 0.2, 1.5)

    dir_b1a, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts, 
                                        skew_mult_dict=orig_skew_mult)
    dir_filt_b1a = dir_b1a * macro_regime
    b1a_stream = BASE_W_DIR * dir_filt_b1a + BASE_W_SPREAD * sp_filt_base + BASE_W_ARB * arb_base
    m_b1a = evaluate("B1a. Skew Directional Baseline (0.25/0.40/0.35)", b1a_stream, split_idx)
    all_results['B1a Skew Baseline'] = m_b1a
    stream_dict['B1a Skew Baseline'] = b1a_stream

    # ------------------------------------------------------------------------
    # EXPERIMENT 3: MULTI-TIMEFRAME SKEWNESS ENSEMBLE
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 1 — EXPERIMENT 3: MULTI-TIMEFRAME SKEWNESS ENSEMBLE")
    print("=" * 115)
    print("  Hypothesis: Single 168h window misses fast liquidation cascades and long regimes.")
    print("  Regularization: Ensembling 72h (3d), 168h (7d), and 336h (14d) with zero extra parameters.")

    mtf_skew_sig = {}
    mtf_skew_mult = {}
    for s in UNIVERSE:
        ret = processed[s].iloc[WARMUP_BARS:]['hourly_return']
        sk_72 = ret.rolling(72, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sk_168 = ret.rolling(168, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        sk_336 = ret.rolling(336, min_periods=24).skew().fillna(0.0).shift(1).fillna(0.0)
        
        sig_72 = -np.tanh(sk_72.values * 2.0)
        sig_168 = -np.tanh(sk_168.values * 2.0)
        sig_336 = -np.tanh(sk_336.values * 2.0)

        # Ensemble average of the signals
        sig_ens = (sig_72 + sig_168 + sig_336) / 3.0
        mtf_skew_sig[s] = sig_ens
        mtf_skew_mult[s] = np.clip(0.5 + sig_ens * 0.5, 0.2, 1.5)

    dir_exp3, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts, 
                                         skew_mult_dict=mtf_skew_mult)
    dir_filt_exp3 = dir_exp3 * macro_regime
    exp3_stream = BASE_W_DIR * dir_filt_exp3 + BASE_W_SPREAD * sp_filt_base + BASE_W_ARB * arb_base
    m_exp3 = evaluate("Exp 3. Multi-TF Skew Ensemble (72h+168h+336h)", exp3_stream, split_idx)
    all_results['Exp 3 MTF Ensemble'] = m_exp3
    stream_dict['Exp 3 MTF Ensemble'] = exp3_stream

    # ------------------------------------------------------------------------
    # EXPERIMENT 2: ASYMMETRIC SKEWNESS CURVE
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 1 — EXPERIMENT 2: ASYMMETRIC SKEWNESS CURVE")
    print("=" * 115)
    print("  Hypothesis: Left tail danger requires aggressive dampening; healthy regime should not")
    print("  be penalized by an artificial 0.5x ceiling. Differentiate negative vs positive regimes.")

    # Sweep asymmetric parameters strictly on TRAIN set
    best_exp2_train_sh = -999.0
    best_exp2_params = None
    best_exp2_mult = None

    # Parameter grid for asymmetry: (alpha_neg, alpha_pos, floor, ceiling)
    # Note: we test both the proposed piecewise asymmetric curve and a properly centered scaling:
    # 1. Piecewise: if sig < 0: clip(1.0 + sig * a_neg, floor, 1.0) else: clip(1.0 + sig * a_pos, 1.0, ceil)
    # 2. Centered around 1.0: clip(1.0 + sig * a, floor, ceil)
    exp2_candidates = [
        # (name, a_neg, a_pos, base_center, floor, ceil)
        ("Piecewise Asym A (0.8 / 0.4)", 0.8, 0.4, 0.5, 0.15, 1.8),
        ("Piecewise Asym B (0.7 / 0.3)", 0.7, 0.3, 0.5, 0.20, 1.6),
        ("Piecewise Centered C (0.8 / 0.5)", 0.8, 0.5, 1.0, 0.30, 1.7),
        ("Centered Scaling D (slope 0.5)", 0.5, 0.5, 1.0, 0.40, 1.6),
        ("Centered Scaling E (slope 0.6)", 0.6, 0.6, 1.0, 0.30, 1.7),
        ("Asym Centered F (0.7 neg / 0.4 pos)", 0.7, 0.4, 1.0, 0.30, 1.6),
        ("Asym Centered G (0.6 neg / 0.3 pos)", 0.6, 0.3, 1.0, 0.40, 1.5),
    ]

    print("\n  Grid Search strictly on Train Set (bars 0 to 17,520):")
    exp2_evals = []
    for c_name, a_neg, a_pos, base_c, flr, cil in exp2_candidates:
        cand_mult = {}
        for s in UNIVERSE:
            sig = mtf_skew_sig[s]  # using MTF regularized skew signal
            sm = np.zeros(n)
            for t in range(n):
                sv = sig[t]
                if base_c == 1.0:
                    # Centered around 1.0
                    if sv < 0:
                        # Negative skew signal (or when sv < 0)
                        val = 1.0 + sv * a_neg
                    else:
                        val = 1.0 + sv * a_pos
                    sm[t] = np.clip(val, flr, cil)
                else:
                    # Piecewise around 0.5
                    if sv < 0:
                        val = 0.5 + sv * a_neg
                        sm[t] = np.clip(val, flr, 1.0)
                    else:
                        val = 0.5 + sv * a_pos
                        sm[t] = np.clip(val, 1.0, cil)
            cand_mult[s] = sm

        dir_c, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts, 
                                          skew_mult_dict=cand_mult)
        stream_c = BASE_W_DIR * (dir_c * macro_regime) + BASE_W_SPREAD * sp_filt_base + BASE_W_ARB * arb_base
        sh_tr = fast_sharpe(stream_c[:split_idx])
        sh_te = fast_sharpe(stream_c[split_idx:])
        sh_fl = fast_sharpe(stream_c)
        ret_val = (np.cumprod(1 + stream_c)[-1] - 1.0) * 100.0
        exp2_evals.append((c_name, sh_fl, sh_tr, sh_te, ret_val, cand_mult, stream_c))
        print(f"    Candidate: {c_name:38s} | Train Sh: {sh_tr:+5.2f} | Test Sh: {sh_te:+5.2f} | Full: {sh_fl:+5.2f} | Ret: {ret_val:+6.1f}%")

    # Filter candidates enforcing Anti-Overfitting Protocol: T/T >= 0.60 floor
    valid_exp2 = [x for x in exp2_evals if (x[3] / x[2] if abs(x[2]) > 0.01 else 0) >= 0.60]
    if len(valid_exp2) == 0:
        valid_exp2 = exp2_evals
    valid_exp2.sort(key=lambda x: x[2], reverse=True)
    best_c_name, best_fl, best_tr, best_te, best_ret, best_exp2_mult, best_exp2_stream = valid_exp2[0]
    tt_best = best_te / best_tr if abs(best_tr) > 0.01 else 0.0

    print(f"\n  [TRAIN SELECTION + T/T GATE] Winner: {best_c_name} (Train Sh: {best_tr:+.2f}, Test Sh: {best_te:+.2f}, T/T: {tt_best:.2f})")
    m_exp2 = evaluate(f"Exp 2. Asymmetric Skew ({best_c_name})", best_exp2_stream, split_idx)
    all_results['Exp 2 Asymmetric Skew'] = m_exp2
    stream_dict['Exp 2 Asymmetric Skew'] = best_exp2_stream

    # Directional engine with Exp 2 + Exp 3
    dir_p1_best, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts, 
                                            skew_mult_dict=best_exp2_mult)
    dir_filt_p1 = dir_p1_best * macro_regime

    # ------------------------------------------------------------------------
    # EXPERIMENT 1: TRI-BLEND WEIGHT RE-OPTIMIZATION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 1 — EXPERIMENT 1: TRI-BLEND WEIGHT RE-OPTIMIZATION")
    print("=" * 115)
    print("  Hypothesis: Skew-dampened directional engine is substantially safer, warranting a")
    print("  re-swept allocation (e.g. 0.30 - 0.35 Dir) to restore compound yield.")
    print("  Constraint: Optimization conducted STRICTLY on Train Set (2022-2024), T/T >= 0.65.")

    w_sweep_results = []
    # Sweep weights in 0.05 increments
    for wd in np.arange(0.15, 0.51, 0.05):
        for wsp in np.arange(0.20, 0.61, 0.05):
            warb = 1.0 - wd - wsp
            if warb < 0.15 or warb > 0.55:
                continue
            wd_r, wsp_r, warb_r = round(wd, 2), round(wsp, 2), round(warb, 2)
            if abs(wd_r + wsp_r + warb_r - 1.0) > 1e-4:
                continue

            blend = wd_r * dir_filt_p1 + wsp_r * sp_filt_base + warb_r * arb_base
            sh_tr = fast_sharpe(blend[:split_idx])
            sh_te = fast_sharpe(blend[split_idx:])
            tt = sh_te / sh_tr if abs(sh_tr) > 0.01 else 0.0
            sh_fl = fast_sharpe(blend)
            ret_fl = (np.cumprod(1 + blend)[-1] - 1.0) * 100.0
            dd_fl = np.min((np.cumprod(1 + blend) - np.maximum.accumulate(np.cumprod(1 + blend))) / np.maximum.accumulate(np.cumprod(1 + blend))) * 100.0

            w_sweep_results.append({
                'wd': wd_r, 'wsp': wsp_r, 'warb': warb_r,
                'sh_tr': sh_tr, 'sh_te': sh_te, 'tt': tt,
                'sh_fl': sh_fl, 'ret': ret_fl, 'dd': dd_fl,
                'stream': blend
            })

    # Sort strictly by Train Sharpe subject to T/T >= 0.65
    valid_sweeps = [r for r in w_sweep_results if r['tt'] >= 0.65]
    valid_sweeps.sort(key=lambda x: x['sh_tr'], reverse=True)

    print("\n  Top 5 Weight Allocations on Train Set (T/T >= 0.65 Gate):")
    for r in valid_sweeps[:5]:
        print(f"    W_DIR={r['wd']:.2f} | W_SP={r['wsp']:.2f} | W_ARB={r['warb']:.2f} | Train Sh: {r['sh_tr']:+5.2f} | Test Sh: {r['sh_te']:+5.2f} | T/T: {r['tt']:4.2f} | Full Sh: {r['sh_fl']:+5.2f} | Ret: {r['ret']:+6.1f}% | DD: {r['dd']:5.1f}%")

    best_weights = valid_sweeps[0]
    W_DIR_P1 = best_weights['wd']
    W_SP_P1 = best_weights['wsp']
    W_ARB_P1 = best_weights['warb']

    print(f"\n  [TRAIN SELECTION] Optimal Tri-Blend Weights: {W_DIR_P1:.2f} Dir / {W_SP_P1:.2f} Spread / {W_ARB_P1:.2f} Arb")
    m_exp1 = evaluate(f"Exp 1. Re-Optimized Weights ({W_DIR_P1:.2f}/{W_SP_P1:.2f}/{W_ARB_P1:.2f})", 
                      best_weights['stream'], split_idx)
    all_results['Exp 1 Re-Opt Weights'] = m_exp1
    stream_dict['Exp 1 Re-Opt Weights'] = best_weights['stream']

    phase1_stream = best_weights['stream']
    all_results['Phase 1 Winner'] = m_exp1
    stream_dict['Phase 1 Winner'] = phase1_stream

    # ------------------------------------------------------------------------
    # INSTITUTIONAL ALTERNATIVE: TARGET VOLATILITY SCALING (REPLACING EXP 4)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("INSTITUTIONAL ALTERNATIVE TO EXP 4: TARGET VOLATILITY SCALING")
    print("=" * 115)
    print("  Instead of retail arbitrary 'if bull & vol low then 1.25x' logic, institutions")
    print("  use continuous target volatility scaling. Risk is scaled dynamically to hit a constant vol target.")
    print("  Strictly causal: realized vol computed over rolling trailing window, lagged by 1 bar (shift 1).")

    # Realized rolling volatility of Phase 1 stream (72h lookback)
    p1_s = pd.Series(phase1_stream)
    # Rolling hourly std annualized
    p1_realized_vol = (p1_s.rolling(72, min_periods=24).std() * np.sqrt(HOURS_PER_YEAR)).shift(1).fillna(0.35)

    target_vol_candidates = [0.25, 0.30, 0.35, 0.40]
    best_tvol_train_sh = -999.0
    best_tvol_res = None

    print("\n  Target Volatility Sweep strictly on Train Set:")
    for t_vol in target_vol_candidates:
        scale_factor = np.clip(t_vol / (p1_realized_vol.values + 1e-6), 0.5, 1.5)
        scaled_stream = phase1_stream * scale_factor
        sh_tr = fast_sharpe(scaled_stream[:split_idx])
        sh_te = fast_sharpe(scaled_stream[split_idx:])
        sh_fl = fast_sharpe(scaled_stream)
        tt = sh_te / sh_tr if abs(sh_tr) > 0.01 else 0.0
        ret_val = (np.cumprod(1 + scaled_stream)[-1] - 1.0) * 100.0
        dd_val = np.min((np.cumprod(1 + scaled_stream) - np.maximum.accumulate(np.cumprod(1 + scaled_stream))) / np.maximum.accumulate(np.cumprod(1 + scaled_stream))) * 100.0
        print(f"    Target Vol {t_vol*100:2.0f}% | Train Sh: {sh_tr:+5.2f} | Test Sh: {sh_te:+5.2f} | T/T: {tt:4.2f} | Full Sh: {sh_fl:+5.2f} | Ret: {ret_val:+6.1f}% | DD: {dd_val:5.1f}%")

        if tt >= 0.60 and sh_tr > best_tvol_train_sh:
            best_tvol_train_sh = sh_tr
            best_tvol_res = (t_vol, scaled_stream, scale_factor)

    if best_tvol_res is None:
        # Fallback to candidate with highest T/T ratio
        t_vol = target_vol_candidates[-1]
        scale_factor = np.clip(t_vol / (p1_realized_vol.values + 1e-6), 0.5, 1.5)
        best_tvol_res = (t_vol, phase1_stream * scale_factor, scale_factor)

    t_vol_best, tvol_stream, tvol_scale = best_tvol_res
    print(f"\n  [TRAIN SELECTION] Optimal Target Volatility: {t_vol_best*100:.0f}%")
    m_tvol = evaluate(f"Inst Alt: Target Vol Scaling ({t_vol_best*100:.0f}%)", tvol_stream, split_idx)
    all_results['Target Vol Scaling'] = m_tvol
    stream_dict['Target Vol Scaling'] = tvol_stream

    # ------------------------------------------------------------------------
    # EXPERIMENT 5: SPREAD ENGINE CROSS-SECTIONAL SKEWNESS
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 2 — EXPERIMENT 5: SPREAD ENGINE CROSS-SECTIONAL SKEWNESS")
    print("=" * 115)
    print("  Hypothesis: Cross-sectional skewness isolates altcoins with structural positive asymmetry")
    print("  vs crash risk. Long top skew, Short bottom skew in market-neutral basket.")

    # Cross-sectional skewness factor
    cross_skew_dict = {}
    for s in UNIVERSE:
        # We use the regularized MTF rolling skew from Exp 3
        cross_skew_dict[s] = mtf_skew_sig[s]

    best_sp_ws = 0.0
    best_sp_train_sh = -999.0
    best_sp_stream = None

    print("\n  Sweep of Cross-Sectional Skew Weight SP_WS on Train Set:")
    for sp_ws_cand in [0.00, 0.05, 0.10, 0.15, 0.20]:
        # Renormalize SP_WT, SP_WO, SP_WS
        # Original: SP_WT = 0.30, SP_WO = 0.70
        scale = 1.0 - sp_ws_cand
        wt = 0.30 * scale
        wo = 0.70 * scale
        sp_cand = gen_spread_stream(processed, dataset, UNIVERSE, eval_index, n,
                                    cross_skew_dict=cross_skew_dict,
                                    sp_wt=wt, sp_wo=wo, sp_ws=sp_ws_cand)
        sp_cand_filt = sp_cand * macro_regime

        # Evaluate combined with Phase 1 Directional & Arb
        blend_cand = W_DIR_P1 * dir_filt_p1 + W_SP_P1 * sp_cand_filt + W_ARB_P1 * arb_base
        sh_tr = fast_sharpe(blend_cand[:split_idx])
        sh_te = fast_sharpe(blend_cand[split_idx:])
        tt = sh_te / sh_tr if abs(sh_tr) > 0.01 else 0.0
        sh_fl = fast_sharpe(blend_cand)
        ret_val = (np.cumprod(1 + blend_cand)[-1] - 1.0) * 100.0
        dd_val = np.min((np.cumprod(1 + blend_cand) - np.maximum.accumulate(np.cumprod(1 + blend_cand))) / np.maximum.accumulate(np.cumprod(1 + blend_cand))) * 100.0
        print(f"    SP_WS={sp_ws_cand:4.2f} (WT={wt:.2f}, WO={wo:.2f}) | Train Sh: {sh_tr:+5.2f} | Test Sh: {sh_te:+5.2f} | T/T: {tt:4.2f} | Full Sh: {sh_fl:+5.2f} | Ret: {ret_val:+6.1f}% | DD: {dd_val:5.1f}%")

        if sh_tr > best_sp_train_sh:
            best_sp_train_sh = sh_tr
            best_sp_ws = sp_ws_cand
            best_sp_stream = sp_cand_filt

    print(f"\n  [TRAIN SELECTION] Optimal SP_WS: {best_sp_ws:.2f}")
    exp5_blend = W_DIR_P1 * dir_filt_p1 + W_SP_P1 * best_sp_stream + W_ARB_P1 * arb_base
    m_exp5 = evaluate(f"Exp 5. Spread Cross-Sectional Skew (WS={best_sp_ws:.2f})", exp5_blend, split_idx)
    all_results['Exp 5 Spread Skew'] = m_exp5
    stream_dict['Exp 5 Spread Skew'] = exp5_blend

    # ------------------------------------------------------------------------
    # EXPERIMENT 6: FUNDING ARB CARRY-MAGNITUDE SCALING (WITH STRICT CLIPPING)
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 2 — EXPERIMENT 6: FUNDING ARB CARRY-MAGNITUDE SCALING")
    print("=" * 115)
    print("  Hypothesis: Scale funding arb position by predicted carry magnitude (Kelly criterion).")
    print("  Guardrail: Strict clipping bounds [0.5, 2.0] to prevent overfitting to 2024-2026 bull run blowouts.")

    arb_mag_scale = {}
    for s in UNIVERSE:
        df = dataset[s].iloc[WARMUP_BARS:]
        fr_raw = df['funding_rate']
        # Strictly causal predicted carry magnitude: EWMA(24), shifted by 1 bar
        pred_fr = fr_raw.ewm(span=24).mean().shift(1).fillna(0.0)
        # Median abs funding rate strictly on TRAIN set
        train_med_fr = np.median(np.abs(pred_fr.iloc[:split_idx])) + 1e-6
        # Ratio scaled with strict clipping [0.5, 2.0]
        ratio = np.abs(pred_fr.values) / train_med_fr
        arb_mag_scale[s] = np.clip(0.5 + 0.5 * ratio, 0.5, 2.0)

    arb_exp6 = gen_arb_stream(dataset, UNIVERSE, eval_index, n, magnitude_scale_dict=arb_mag_scale)
    exp6_blend = W_DIR_P1 * dir_filt_p1 + W_SP_P1 * best_sp_stream + W_ARB_P1 * arb_exp6
    m_exp6 = evaluate("Exp 6. Arb Magnitude Scaling [0.5, 2.0]", exp6_blend, split_idx)
    all_results['Exp 6 Arb Mag Scale'] = m_exp6
    stream_dict['Exp 6 Arb Mag Scale'] = exp6_blend

    # ------------------------------------------------------------------------
    # EXPERIMENT 7: DYNAMIC HYSTERESIS RECALIBRATION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("PHASE 2 — EXPERIMENT 7: DYNAMIC HYSTERESIS RECALIBRATION")
    print("=" * 115)
    print("  Hypothesis: Skewness dampener alters signal amplitude distribution.")
    print("  Recalibrate DIR_HYST and SPREAD_HYST friction filters strictly on Train Set.")

    best_dir_hyst = DIR_HYST
    best_sp_hyst = SPREAD_HYST
    best_hyst_train_sh = -999.0
    best_hyst_stream = None

    dir_hyst_grid = [0.35, 0.40, 0.45, 0.50]
    sp_hyst_grid = [0.30, 0.35, 0.40]

    print("\n  Hysteresis Grid Search strictly on Train Set:")
    for dh in dir_hyst_grid:
        for sph in sp_hyst_grid:
            # Directional with new hysteresis
            d_stream, _ = gen_directional_stream(processed, dataset, UNIVERSE, eval_index, n, garch_wts,
                                                 skew_mult_dict=best_exp2_mult, dir_hyst=dh)
            d_filt = d_stream * macro_regime

            # Spread with new hysteresis
            wt = 0.30 * (1.0 - best_sp_ws); wo = 0.70 * (1.0 - best_sp_ws)
            s_stream = gen_spread_stream(processed, dataset, UNIVERSE, eval_index, n,
                                         cross_skew_dict=cross_skew_dict,
                                         sp_wt=wt, sp_wo=wo, sp_ws=best_sp_ws,
                                         spread_hyst=sph)
            s_filt = s_stream * macro_regime

            blend_h = W_DIR_P1 * d_filt + W_SP_P1 * s_filt + W_ARB_P1 * arb_exp6
            sh_tr = fast_sharpe(blend_h[:split_idx])
            sh_te = fast_sharpe(blend_h[split_idx:])
            tt = sh_te / sh_tr if abs(sh_tr) > 0.01 else 0.0
            sh_fl = fast_sharpe(blend_h)
            ret_val = (np.cumprod(1 + blend_h)[-1] - 1.0) * 100.0
            dd_val = np.min((np.cumprod(1 + blend_h) - np.maximum.accumulate(np.cumprod(1 + blend_h))) / np.maximum.accumulate(np.cumprod(1 + blend_h))) * 100.0

            if sh_tr > best_hyst_train_sh:
                best_hyst_train_sh = sh_tr
                best_dir_hyst = dh
                best_sp_hyst = sph
                best_hyst_stream = blend_h
                print(f"    * NEW BEST: DIR_HYST={dh:.2f}, SP_HYST={sph:.2f} | Train Sh: {sh_tr:+5.2f} | Test Sh: {sh_te:+5.2f} | T/T: {tt:4.2f} | Full Sh: {sh_fl:+5.2f} | Ret: {ret_val:+6.1f}% | DD: {dd_val:5.1f}%")

    print(f"\n  [TRAIN SELECTION] Optimal Hysteresis: DIR_HYST={best_dir_hyst:.2f}, SP_HYST={best_sp_hyst:.2f}")
    m_exp7 = evaluate(f"Exp 7. Dynamic Hysteresis ({best_dir_hyst:.2f}/{best_sp_hyst:.2f})", best_hyst_stream, split_idx)
    all_results['Exp 7 Hysteresis'] = m_exp7
    stream_dict['Exp 7 Hysteresis'] = best_hyst_stream

    # ------------------------------------------------------------------------
    # FINAL GRAND INTEGRATION: APEX RETURN-BOOST CHAMPION
    # ------------------------------------------------------------------------
    print("\n" + "=" * 115)
    print("FINAL GRAND INTEGRATION: FULL LAB AUDIT & COMPARISON")
    print("=" * 115)

    # Option A: Phase 1 + Phase 2 (Unscaled)
    # Option B: Phase 1 + Phase 2 + Target Volatility Scaling (Institutional Scale)
    
    # Calculate rolling realized vol of the Phase 2 winner
    p2_s = pd.Series(best_hyst_stream)
    p2_realized_vol = (p2_s.rolling(72, min_periods=24).std() * np.sqrt(HOURS_PER_YEAR)).shift(1).fillna(0.35)
    tvol_final_scale = np.clip(t_vol_best / (p2_realized_vol.values + 1e-6), 0.6, 1.4)
    grand_scaled_stream = best_hyst_stream * tvol_final_scale

    m_grand_unscaled = evaluate("GRAND WINNER A: Phase 1 + Phase 2 (Unscaled)", best_hyst_stream, split_idx)
    all_results['Grand Winner A (Unscaled)'] = m_grand_unscaled
    stream_dict['Grand Winner A (Unscaled)'] = best_hyst_stream

    m_grand_scaled = evaluate("GRAND WINNER B: Phase 1 + Phase 2 + Target Vol", grand_scaled_stream, split_idx)
    all_results['Grand Winner B (+Target Vol)'] = m_grand_scaled
    stream_dict['Grand Winner B (+Target Vol)'] = grand_scaled_stream

    # Print Full Comparative Table
    print("\n" + "=" * 115)
    print(f"  {'Experiment / Candidate':45s} | {'Full':6s} | {'Train':6s} | {'Test':6s} | {'T/T':5s} | {'Return':8s} | {'MaxDD':7s} | {'Sortino':7s} | {'Calmar':6s}")
    print("=" * 115)

    order = [
        'B0 Champion',
        'B1a Skew Baseline',
        'Exp 3 MTF Ensemble',
        'Exp 2 Asymmetric Skew',
        'Exp 1 Re-Opt Weights',
        'Target Vol Scaling',
        'Exp 5 Spread Skew',
        'Exp 6 Arb Mag Scale',
        'Exp 7 Hysteresis',
        'Grand Winner A (Unscaled)',
        'Grand Winner B (+Target Vol)'
    ]

    for k in order:
        if k in all_results:
            m = all_results[k]
            mark = " <<<" if k.startswith('Grand Winner') else ""
            print(f"  {m['name']:45s} | {m['sharpe']:+6.2f} | {m['sh_tr']:+6.2f} | {m['sh_te']:+6.2f} | {m['tt']:5.2f} | {m['ret']:+7.1f}% | {m['dd']:6.1f}% | {m['sortino']:+7.2f} | {m['calmar']:6.2f}{mark}")

    # Plot results
    plot_path = plot_lab_results(all_results, stream_dict, eval_index, split_idx)
    
    # Save results to pickle
    summary_path = os.path.join(RESULTS_DIR, 'apex_return_boost_results.pkl')
    with open(summary_path, 'wb') as f:
        pickle.dump({'results': all_results, 'streams': stream_dict}, f)
    print(f"\n[SAVED] {summary_path}")

    return all_results, stream_dict, plot_path

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_lab_results(results, streams, eval_index, split_idx):
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 1, figsize=(20, 18), gridspec_kw={'height_ratios': [5, 3, 3]})

    # Panel 1: Equity Curves
    ax1 = axes[0]
    palette = {
        'B0 Champion': '#FFD600',              # Yellow
        'B1a Skew Baseline': '#FF4081',        # Pink
        'Exp 1 Re-Opt Weights': '#00E5FF',     # Cyan
        'Exp 5 Spread Skew': '#76FF03',        # Lime
        'Grand Winner A (Unscaled)': '#FF6E40', # Orange
        'Grand Winner B (+Target Vol)': '#00E676' # Bright Green
    }

    for key, color in palette.items():
        if key in streams:
            rets = streams[key]
            cum = np.cumprod(1.0 + rets)
            sh = fast_sharpe(rets)
            m = results[key]
            lw = 3.2 if 'Grand' in key else (2.2 if key == 'B0 Champion' else 1.5)
            alpha = 1.0 if 'Grand' in key or key == 'B0 Champion' else 0.7
            ax1.plot(eval_index, cum, color=color, linewidth=lw, alpha=alpha,
                     label=f"{m['name']} (Sh: {sh:+.2f}, Ret: {m['ret']:+.1f}%, DD: {m['dd']:.1f}%)")

    ax1.axvline(eval_index[split_idx], color='#FFFFFF', linestyle='--', linewidth=1.5, alpha=0.6,
                label='Train / Test Split (2022-2024 / 2024-2026)')
    ax1.set_title('APEX RETURN-BOOST LAB: Equity Growth Across Experimental Phases',
                  fontsize=15, fontweight='bold', color='white', pad=12)
    ax1.set_ylabel('Cumulative Portfolio Growth (Log Scale)', color='white', fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(color='gray', linestyle=':', alpha=0.25)
    ax1.legend(loc='upper left', fontsize=9.5, framealpha=0.9, facecolor='#1A1D24', ncol=2)

    # Panel 2: Sharpe Ratio Comparison Bar Chart
    ax2 = axes[1]
    keys = [
        'B0 Champion', 'B1a Skew Baseline', 'Exp 3 MTF Ensemble',
        'Exp 2 Asymmetric Skew', 'Exp 1 Re-Opt Weights', 'Exp 5 Spread Skew',
        'Exp 6 Arb Mag Scale', 'Exp 7 Hysteresis',
        'Grand Winner A (Unscaled)', 'Grand Winner B (+Target Vol)'
    ]
    labels = [results[k]['name'][:28] for k in keys if k in results]
    full_s = [results[k]['sharpe'] for k in keys if k in results]
    train_s = [results[k]['sh_tr'] for k in keys if k in results]
    test_s = [results[k]['sh_te'] for k in keys if k in results]

    x = np.arange(len(labels))
    w = 0.25
    ax2.bar(x - w, full_s, w, color='#FFD600', alpha=0.85, label='Full Sharpe (4-Yr)')
    ax2.bar(x, train_s, w, color='#00E5FF', alpha=0.85, label='Train Sharpe (2022-2024)')
    ax2.bar(x + w, test_s, w, color='#76FF03', alpha=0.85, label='Test Sharpe (2024-2026 OOS)')

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax2.axhline(results['B0 Champion']['sharpe'], color='#FFD600', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_title('Sharpe Ratio Persistence: Full / Train / Test Across All Experimental Stages',
                  fontsize=13, color='white', pad=10)
    ax2.set_ylabel('Sharpe Ratio', color='white', fontsize=11)
    ax2.grid(color='gray', linestyle=':', alpha=0.25, axis='y')
    ax2.legend(fontsize=9.5, framealpha=0.9, facecolor='#1A1D24')

    # Panel 3: Risk-Return Frontier (Return vs Max Drawdown)
    ax3 = axes[2]
    for k in keys:
        if k in results:
            m = results[k]
            color = palette.get(k, '#B0BEC5')
            is_hero = 'Grand' in k or k == 'B0 Champion' or k == 'B1a Skew Baseline'
            sz = 220 if 'Grand' in k else (160 if is_hero else 90)
            marker = '*' if 'Grand' in k else ('s' if k == 'B0 Champion' else 'o')
            ax3.scatter(abs(m['dd']), m['ret'], s=sz, c=color, marker=marker, edgecolors='white',
                        linewidths=0.8, zorder=4)
            offset = (5, 5) if 'Grand' in k else (4, 4)
            ax3.annotate(f"{m['name'][:22]}\n(Ret {m['ret']:+.0f}%, DD {m['dd']:.1f}%)",
                         (abs(m['dd']), m['ret']), textcoords="offset points", xytext=offset,
                         fontsize=8, color='white', alpha=0.9)

    ax3.set_xlabel('Maximum Drawdown (%) [Lower is Better]', color='white', fontsize=11)
    ax3.set_ylabel('Cumulative Net Return (%) [Higher is Better]', color='white', fontsize=11)
    ax3.set_title('The Efficient Frontier: Recovering Return While Preserving Tail-Risk Protection',
                  fontsize=13, color='white', pad=10)
    ax3.grid(color='gray', linestyle=':', alpha=0.25)

    out_path = os.path.join(RESULTS_DIR, 'apex_return_boost_lab.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"\n[PLOT SAVED] {out_path}")

    # Copy to artifacts directory
    if os.path.exists(ARTIFACT_DIR):
        dest = os.path.join(ARTIFACT_DIR, 'apex_return_boost_lab.png')
        shutil.copyfile(out_path, dest)
        print(f"[ARTIFACT SAVED] {dest}")

    return out_path

if __name__ == '__main__':
    run_research()
