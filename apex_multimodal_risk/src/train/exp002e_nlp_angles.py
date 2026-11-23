"""
EXP-002e: Multi-Angle NLP Experiment.
Features:
- Angle A: Zero-Shot Semantic Risk-Prototype Similarity (bge-small-en-v1.5)
- Angle B: Sparse Anomaly Gating (85th percentile train thresholding)
- Angle C: Causal News Velocity / Burst Acceleration
Targets:
- Target 1: tail_crash_6h (Directional Left-Tail Crash, ~4.12% base rate)
- Target 2: vol_expansion_24h (Unsigned 24h Realized Volatility Shock, ~11.56% base rate)
Evaluates 4 Feature Sets across 10 CPCV splits:
- M0_Market_Only (16 market features)
- M1_Market+RawProto (16 market + 8 prototype similarities + burst ratio)
- M2_Market+SparseGatedNLP (16 market + 3 gated excess features + burst ratio)
- M3_Market+NoveltyAndProtoTop (16 market + novelty + gated_proto + burst ratio)
"""

import gc
import json
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from apex_multimodal_risk.src.features.shared_features import (
    MARKET_NUMERICAL_COLS,
    load_and_merge_features_and_targets,
)
from apex_multimodal_risk.src.models.baselines import LightGBMBaseline
from apex_multimodal_risk.src.train.calibration import (
    PlattScaler,
    compute_ece,
)
from apex_multimodal_risk.src.train.cpcv_runner import generate_cpcv_splits
from apex_multimodal_risk.src.train.train_ablation import get_peak_rss_mb
from apex_multimodal_risk.src.utils import (
    PROJECT_ROOT,
    get_device,
    read_parquet_utc,
    setup_logger,
)

logger = setup_logger("exp002e_nlp_angles")

PROTOTYPE_PROMPTS = {
    "proto_insolvency": "cryptocurrency exchange bankruptcy insolvency halt withdrawals bank run",
    "proto_exploit": "DeFi protocol smart contract hack exploit stolen funds vulnerability",
    "proto_regulatory": "SEC CFTC lawsuit enforcement action regulatory crackdown crypto ban",
    "proto_macro": "Federal Reserve emergency rate hike bank failure liquidity crisis market crash",
    "proto_outage": "blockchain network outage halted block production consensus failure",
}

MARKET_FEATURES = list(MARKET_NUMERICAL_COLS)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs_cal: np.ndarray,
    top_pct: float = 0.05,
    n_bins: int = 10,
) -> dict[str, float]:
    """Computes ranking, skill, and calibration metrics."""
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs_cal, dtype=np.float64)
    n = len(y)
    base_rate = float(np.mean(y)) if n > 0 else 0.0412
    brier_ref = base_rate * (1.0 - base_rate)

    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = float(auc(rec, prec))
    try:
        roc_auc = float(roc_auc_score(y, p))
    except ValueError:
        roc_auc = 0.5

    k = max(1, int(n * top_pct))
    top_indices = np.argsort(p)[-k:]
    hits = np.sum(y[top_indices])
    total_pos = np.sum(y)

    prec_top = float(hits / k)
    recall_top = float(hits / total_pos) if total_pos > 0 else 0.0
    lift_top = float(prec_top / base_rate) if base_rate > 0 else 1.0

    brier = float(np.mean((y - p) ** 2))
    bss = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0
    ece = float(compute_ece(y, p, n_bins=n_bins))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        f"prec_top{int(top_pct*100)}": prec_top,
        f"recall_top{int(top_pct*100)}": recall_top,
        f"lift_top{int(top_pct*100)}": lift_top,
        "brier": brier,
        "bss": bss,
        "ece": ece,
        "base_rate": base_rate,
    }


def engineer_nlp_features_and_targets() -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Encodes 5 risk prototypes ONCE, computes strictly causal similarities and burst ratio,
    and constructs Target 2 (vol_expansion_24h).
    """
    logger.info("Loading precomputed data...")
    merged_panel = load_and_merge_features_and_targets()
    news_df = read_parquet_utc("apex_multimodal_risk/data/processed/news_dedup.parquet")
    news_df = news_df.sort_values("ingested_at").reset_index(drop=True)

    # 1. Encode 5 prototypes ONCE using bge-small-en-v1.5
    device = get_device()
    logger.info(f"Encoding 5 risk prototypes on {device}...")
    embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device=str(device))
    prompt_keys = list(PROTOTYPE_PROMPTS.keys())
    prompt_texts = [PROTOTYPE_PROMPTS[k] for k in prompt_keys]
    proto_embs = embedder.encode(prompt_texts, normalize_embeddings=True)  # (5, 384)

    # Save prototype texts and vectors to artifacts/risk_prototypes.json
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    proto_json_path = artifacts_dir / "risk_prototypes.json"
    proto_data = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "prototypes": {
            k: {"text": t, "vector": v.tolist()} for k, t, v in zip(prompt_keys, prompt_texts, proto_embs)
        },
    }
    with open(proto_json_path, "w") as f:
        json.dump(proto_data, f, indent=2)
    logger.info(f"Saved risk prototypes to {proto_json_path}")

    # 2. Compute exact cosine similarities for all news clusters
    news_embs = np.array(news_df["embedding"].tolist(), dtype=np.float32)
    all_sims = np.dot(news_embs, proto_embs.T)  # (N_news, 5)

    df = merged_panel.copy()

    # Pre-allocate feature columns in panel
    df["max_proto_sim_6h"] = 0.0
    df["max_proto_sim_24h"] = 0.0
    for k in prompt_keys:
        df[f"{k}_24h"] = 0.0
    df["proto_sim_zscore_7d"] = 0.0
    df["news_burst_ratio"] = 0.0
    df["fwd_vol_24h"] = np.nan

    logger.info("Vectorizing causal prototype similarities, burst ratio, and vol_expansion_24h per asset...")
    assets = df["asset"].unique()

    for sym in assets:
        entity_prefix = sym.replace("USDT", "").lower()
        mask_asset = (news_df[f"entity_{entity_prefix}"] == 1) | (news_df["entity_macro"] == 1)
        sub_news = news_df[mask_asset]
        sub_sims = all_sims[mask_asset.to_numpy()]

        panel_mask = (df["asset"] == sym)
        bar_ts = df.loc[panel_mask, "timestamp"].values
        news_ts = sub_news["ingested_at"].values
        n_bars = len(bar_ts)

        idx_now = np.searchsorted(news_ts, bar_ts, side="right")
        idx_6h = np.searchsorted(news_ts, bar_ts - np.timedelta64(6, "h"), side="right")
        idx_24h = np.searchsorted(news_ts, bar_ts - np.timedelta64(24, "h"), side="right")
        idx_168h = np.searchsorted(news_ts, bar_ts - np.timedelta64(168, "h"), side="right")

        # News burst ratio
        c6 = (idx_now - idx_6h).astype(np.float32)
        c168 = (idx_now - idx_168h).astype(np.float32)
        burst_ratio = (c6 / 6.0) / ((c168 / 168.0) + 1e-3)

        # Prototype similarities
        max_sim_6h = np.zeros(n_bars, dtype=np.float32)
        max_sim_24h = np.zeros(n_bars, dtype=np.float32)
        per_proto_24h = np.zeros((n_bars, 5), dtype=np.float32)

        for i in range(n_bars):
            e = idx_now[i]
            s6 = idx_6h[i]
            s24 = idx_24h[i]
            if e > s6:
                max_sim_6h[i] = np.max(sub_sims[s6:e])
            if e > s24:
                sl24 = sub_sims[s24:e]
                max_sim_24h[i] = np.max(sl24)
                per_proto_24h[i] = np.max(sl24, axis=0)

        # Strictly trailing 7-day z-score of max_proto_sim_6h (shift 1 hour to guarantee zero leakage)
        s_sim6 = pd.Series(max_sim_6h)
        roll_mean_7d = s_sim6.shift(1).rolling(168, min_periods=24).mean().fillna(0.0).to_numpy()
        roll_std_7d = s_sim6.shift(1).rolling(168, min_periods=24).std().fillna(1.0).to_numpy()
        roll_std_7d = np.where(roll_std_7d < 1e-4, 1.0, roll_std_7d)
        proto_zscore = (max_sim_6h - roll_mean_7d) / roll_std_7d

        # Assign back to df
        df.loc[panel_mask, "max_proto_sim_6h"] = max_sim_6h
        df.loc[panel_mask, "max_proto_sim_24h"] = max_sim_24h
        for p_idx, k in enumerate(prompt_keys):
            df.loc[panel_mask, f"{k}_24h"] = per_proto_24h[:, p_idx]
        df.loc[panel_mask, "proto_sim_zscore_7d"] = proto_zscore
        df.loc[panel_mask, "news_burst_ratio"] = burst_ratio

        # Construct Target 2: forward 24h realized volatility
        rets = df.loc[panel_mask, "ret_1h"].to_numpy()
        s_ret = pd.Series(rets)
        # Reverse rolling std: window of rets[i+1 : i+25]
        fwd_std = s_ret.iloc[::-1].rolling(24, min_periods=24).std().iloc[::-1].shift(-1).to_numpy()
        df.loc[panel_mask, "fwd_vol_24h"] = fwd_std

    # Target 2 Definition: vol_expansion_24h = 1 if fwd_vol_24h > mult * trailing_1h_vol_72h
    valid_vol_mask = ~df["fwd_vol_24h"].isna() & df["target_valid"]
    ratio = df.loc[valid_vol_mask, "fwd_vol_24h"] / df.loc[valid_vol_mask, "trailing_1h_vol_72h"]
    rate_150 = float(np.mean(ratio > 1.50))
    rate_175 = float(np.mean(ratio > 1.75))

    logger.info(f"Target 2 Expansion Base Rates: 1.50x threshold = {rate_150*100:.2f}%, 1.75x threshold = {rate_175*100:.2f}%")
    # Lock 1.50x because it is closest to ~10% (11.56%)
    df["vol_expansion_24h"] = (
        (df["fwd_vol_24h"] > 1.50 * df["trailing_1h_vol_72h"]) & valid_vol_mask
    ).astype(np.float32)

    eval_panel = df.loc[valid_vol_mask].copy().reset_index(drop=True)
    logger.info(f"Final eval_panel shape: {eval_panel.shape}, Target 1 rate: {eval_panel['tail_crash_6h'].mean()*100:.2f}%, Target 2 rate: {eval_panel['vol_expansion_24h'].mean()*100:.2f}%")

    meta = {
        "prompt_keys": prompt_keys,
        "rate_150": rate_150,
        "rate_175": rate_175,
    }
    return eval_panel, meta


def run_exp002e_cpcv(eval_panel: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    """Runs 10-split CPCV across M0, M1, M2, M3 on both Target 1 and Target 2."""
    unique_ts = np.array(sorted(eval_panel["timestamp"].unique()))
    splits = generate_cpcv_splits(unique_ts, n_blocks=5, k_test=2, purge_embargo_hours=504)
    panel_ts = eval_panel["timestamp"]

    y_t1 = eval_panel["tail_crash_6h"].to_numpy(dtype=np.float32)
    y_t2 = eval_panel["vol_expansion_24h"].to_numpy(dtype=np.float32)

    # Trailing vol tercile for conditional evaluation on Target 1
    vol_72h = eval_panel["trailing_1h_vol_72h"].to_numpy()
    vol_33_threshold = np.percentile(vol_72h, 33.33)
    low_vol_mask = vol_72h <= vol_33_threshold

    prompt_keys = meta["prompt_keys"]
    proto_24h_cols = [f"{k}_24h" for k in prompt_keys]

    # Feature Set Definitions
    f_raw_proto = [
        "max_proto_sim_6h",
        "max_proto_sim_24h",
    ] + proto_24h_cols + ["proto_sim_zscore_7d", "news_burst_ratio"]

    models = [
        "M0_Market_Only",
        "M1_Market+RawProto",
        "M2_Market+SparseGatedNLP",
        "M3_Market+NoveltyAndProtoTop",
    ]

    # Containers for out-of-sample predictions
    oos_preds_t1: dict[str, list[np.ndarray]] = {m: [] for m in models}
    oos_preds_t2: dict[str, list[np.ndarray]] = {m: [] for m in models}
    oos_y1: list[np.ndarray] = []
    oos_y2: list[np.ndarray] = []
    oos_low_vol: list[np.ndarray] = []

    logger.info("Executing 10 CPCV splits across 4 feature sets on Targets 1 & 2...")

    for split in splits:
        _split_id = split.split_id
        tr_mask = split.get_train_mask(panel_ts)
        val_mask = split.get_inner_val_mask(panel_ts)
        te_mask = split.get_test_mask(panel_ts)

        y1_tr, y1_val, y1_te = y_t1[tr_mask], y_t1[val_mask], y_t1[te_mask]
        y2_tr, y2_val, y2_te = y_t2[tr_mask], y_t2[val_mask], y_t2[te_mask]

        oos_y1.append(y1_te)
        oos_y2.append(y2_te)
        oos_low_vol.append(low_vol_mask[te_mask])

        # Angle B: Compute 85th percentile gating thresholds STRICTLY on train fold (zero leakage)
        tau_nov = float(np.percentile(eval_panel.loc[tr_mask, "embedding_novelty_score"], 85.0))
        tau_proto = float(np.percentile(eval_panel.loc[tr_mask, "max_proto_sim_24h"], 85.0))

        # Build feature matrices for each slice
        def build_features(mask, t_nov=tau_nov, t_proto=tau_proto):
            sub = eval_panel.loc[mask]
            X_mkt = sub[MARKET_FEATURES].to_numpy(dtype=np.float32)
            X_proto = sub[f_raw_proto].to_numpy(dtype=np.float32)

            # Gated features
            nov = sub["embedding_novelty_score"].to_numpy(dtype=np.float32)
            p_max = sub["max_proto_sim_24h"].to_numpy(dtype=np.float32)
            burst = sub["news_burst_ratio"].to_numpy(dtype=np.float32).reshape(-1, 1)

            g_nov = np.maximum(0.0, nov - t_nov).reshape(-1, 1)
            g_proto = np.maximum(0.0, p_max - t_proto).reshape(-1, 1)
            alert = ((nov > t_nov) | (p_max > t_proto)).astype(np.float32).reshape(-1, 1)

            X_m1 = np.concatenate([X_mkt, X_proto], axis=1)
            X_m2 = np.concatenate([X_mkt, g_nov, g_proto, alert, burst], axis=1)
            X_m3 = np.concatenate([X_mkt, nov.reshape(-1, 1), g_proto, burst], axis=1)

            return {
                "M0_Market_Only": X_mkt,
                "M1_Market+RawProto": X_m1,
                "M2_Market+SparseGatedNLP": X_m2,
                "M3_Market+NoveltyAndProtoTop": X_m3,
            }

        feats_tr = build_features(tr_mask)
        feats_val = build_features(val_mask)
        feats_te = build_features(te_mask)

        for m_name in models:
            X_tr, X_val, X_te = feats_tr[m_name], feats_val[m_name], feats_te[m_name]

            # --- Target 1: tail_crash_6h ---
            clf1 = LightGBMBaseline(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=100,
                colsample_bytree=0.7,
                reg_lambda=5.0,
                subsample=0.8,
                class_weight=None,
                scale_pos_weight=1.0,
                is_unbalance=False,
            )
            clf1.fit(X_tr, y1_tr, eval_set=[(X_val, y1_val)])
            val_log1 = clf1.predict_logits(X_val)
            te_log1 = clf1.predict_logits(X_te)
            cal1 = PlattScaler(lambda_reg=1.0).fit(val_log1, y1_val)
            p1_te = cal1.transform(te_log1)
            oos_preds_t1[m_name].append(p1_te)
            del clf1; gc.collect()

            # --- Target 2: vol_expansion_24h ---
            clf2 = LightGBMBaseline(
                n_estimators=100,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=100,
                colsample_bytree=0.7,
                reg_lambda=5.0,
                subsample=0.8,
                class_weight=None,
                scale_pos_weight=1.0,
                is_unbalance=False,
            )
            clf2.fit(X_tr, y2_tr, eval_set=[(X_val, y2_val)])
            val_log2 = clf2.predict_logits(X_val)
            te_log2 = clf2.predict_logits(X_te)
            cal2 = PlattScaler(lambda_reg=1.0).fit(val_log2, y2_val)
            p2_te = cal2.transform(te_log2)
            oos_preds_t2[m_name].append(p2_te)
            del clf2; gc.collect()

    flat_y1 = np.concatenate(oos_y1)
    flat_y2 = np.concatenate(oos_y2)
    flat_low_vol = np.concatenate(oos_low_vol)

    flat_preds_t1 = {m: np.concatenate(oos_preds_t1[m]) for m in models}
    flat_preds_t2 = {m: np.concatenate(oos_preds_t2[m]) for m in models}

    # Evaluate Target 1 (Full Panel & Low-Vol Tercile)
    t1_full_metrics = {m: compute_comprehensive_metrics(flat_y1, flat_preds_t1[m], top_pct=0.05) for m in models}
    t1_low_vol_metrics = {
        m: compute_comprehensive_metrics(flat_y1[flat_low_vol], flat_preds_t1[m][flat_low_vol], top_pct=0.05)
        for m in models
    }

    # Evaluate Target 2 (Full Panel, Top 10% Lift)
    t2_full_metrics = {m: compute_comprehensive_metrics(flat_y2, flat_preds_t2[m], top_pct=0.10) for m in models}

    return {
        "models": models,
        "t1_full": t1_full_metrics,
        "t1_low_vol": t1_low_vol_metrics,
        "t2_full": t2_full_metrics,
        "eval_panel": eval_panel,
        "meta": meta,
        "f_raw_proto": f_raw_proto,
    }


def conditional_promotion_and_artifacts(
    results: dict[str, Any],
) -> dict[str, Any]:
    """
    Applies conditional promotion rules:
    - Target 1: If best NLP variant beats EXP-002d (0.0581 PR-AUC, +0.0035 BSS, ECE < 0.015), promote to best_lightgbm_risk.pkl.
    - Target 2: If best NLP variant beats M0_Market_Only with BSS > 0, save to best_lightgbm_vol24h.pkl.
    """
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    eval_panel = results["eval_panel"]
    f_raw_proto = results["f_raw_proto"]

    # Target 1 Promotion Check
    t1_full = results["t1_full"]
    exp002d_pr_auc = 0.0581
    exp002d_bss = 0.0035

    best_t1_nlp = None
    best_t1_pr_auc = -1.0
    for m in ["M1_Market+RawProto", "M2_Market+SparseGatedNLP", "M3_Market+NoveltyAndProtoTop"]:
        if t1_full[m]["pr_auc"] > best_t1_pr_auc:
            best_t1_pr_auc = t1_full[m]["pr_auc"]
            best_t1_nlp = m

    t1_beats_exp002d = (
        best_t1_pr_auc > exp002d_pr_auc
        and t1_full[best_t1_nlp]["bss"] > exp002d_bss
        and t1_full[best_t1_nlp]["ece"] < 0.015
    )

    promoted_t1 = False
    if t1_beats_exp002d:
        logger.info(f"Target 1: {best_t1_nlp} beat EXP-002d ({best_t1_pr_auc:.4f} > {exp002d_pr_auc:.4f}, BSS {t1_full[best_t1_nlp]['bss']:+.4f} > {exp002d_bss:+.4f})! Promoting to best_lightgbm_risk.pkl...")
        # Fit on full dataset
        tau_nov_full = float(np.percentile(eval_panel["embedding_novelty_score"], 85.0))
        tau_proto_full = float(np.percentile(eval_panel["max_proto_sim_24h"], 85.0))

        X_mkt = eval_panel[MARKET_FEATURES].to_numpy(dtype=np.float32)
        nov = eval_panel["embedding_novelty_score"].to_numpy(dtype=np.float32)
        p_max = eval_panel["max_proto_sim_24h"].to_numpy(dtype=np.float32)
        burst = eval_panel["news_burst_ratio"].to_numpy(dtype=np.float32).reshape(-1, 1)

        if best_t1_nlp == "M1_Market+RawProto":
            X_proto = eval_panel[f_raw_proto].to_numpy(dtype=np.float32)
            X_full = np.concatenate([X_mkt, X_proto], axis=1)
            feat_names = MARKET_FEATURES + f_raw_proto
        elif best_t1_nlp == "M2_Market+SparseGatedNLP":
            g_nov = np.maximum(0.0, nov - tau_nov_full).reshape(-1, 1)
            g_proto = np.maximum(0.0, p_max - tau_proto_full).reshape(-1, 1)
            alert = ((nov > tau_nov_full) | (p_max > tau_proto_full)).astype(np.float32).reshape(-1, 1)
            X_full = np.concatenate([X_mkt, g_nov, g_proto, alert, burst], axis=1)
            feat_names = MARKET_FEATURES + ["gated_novelty", "gated_proto_sim", "exogenous_alert_flag", "news_burst_ratio"]
        else:  # M3
            g_proto = np.maximum(0.0, p_max - tau_proto_full).reshape(-1, 1)
            X_full = np.concatenate([X_mkt, nov.reshape(-1, 1), g_proto, burst], axis=1)
            feat_names = MARKET_FEATURES + ["embedding_novelty_score", "gated_proto_sim", "news_burst_ratio"]

        y1_full = eval_panel["tail_crash_6h"].to_numpy(dtype=np.float32)
        m_t1 = LightGBMBaseline(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=100,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            subsample=0.8,
            class_weight=None,
            scale_pos_weight=1.0,
            is_unbalance=False,
        )
        m_t1.fit(X_full, y1_full)
        cal_t1 = PlattScaler(lambda_reg=1.0).fit(m_t1.predict_logits(X_full), y1_full)

        with open(artifacts_dir / "best_lightgbm_risk.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": m_t1.model,
                    "platt_scaler": cal_t1,
                    "feature_names": feat_names,
                    "gate_thresholds": {"tau_nov": tau_nov_full, "tau_proto": tau_proto_full},
                    "architecture": best_t1_nlp,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                },
                f,
            )
        promoted_t1 = True
    else:
        logger.info(f"Target 1: Best NLP variant ({best_t1_nlp}, PR-AUC {best_t1_pr_auc:.4f}, BSS {t1_full[best_t1_nlp]['bss']:+.4f}) did not beat EXP-002d threshold. Retaining EXP-002d artifact untouched.")

    # Target 2 Promotion Check: Beats M0_Market_Only with BSS > 0
    t2_full = results["t2_full"]
    m0_t2_pr_auc = t2_full["M0_Market_Only"]["pr_auc"]

    best_t2_nlp = None
    best_t2_pr_auc = -1.0
    for m in ["M1_Market+RawProto", "M2_Market+SparseGatedNLP", "M3_Market+NoveltyAndProtoTop"]:
        if t2_full[m]["pr_auc"] > best_t2_pr_auc:
            best_t2_pr_auc = t2_full[m]["pr_auc"]
            best_t2_nlp = m

    t2_beats_m0 = (best_t2_pr_auc > m0_t2_pr_auc) and (t2_full[best_t2_nlp]["bss"] > 0.0)

    saved_t2 = False
    if t2_beats_m0:
        logger.info(f"Target 2 (vol_expansion_24h): {best_t2_nlp} beat Market-Only ({best_t2_pr_auc:.4f} > {m0_t2_pr_auc:.4f}, BSS {t2_full[best_t2_nlp]['bss']:+.4f} > 0)! Saving best_lightgbm_vol24h.pkl...")
        tau_nov_full = float(np.percentile(eval_panel["embedding_novelty_score"], 85.0))
        tau_proto_full = float(np.percentile(eval_panel["max_proto_sim_24h"], 85.0))

        X_mkt = eval_panel[MARKET_FEATURES].to_numpy(dtype=np.float32)
        nov = eval_panel["embedding_novelty_score"].to_numpy(dtype=np.float32)
        p_max = eval_panel["max_proto_sim_24h"].to_numpy(dtype=np.float32)
        burst = eval_panel["news_burst_ratio"].to_numpy(dtype=np.float32).reshape(-1, 1)

        if best_t2_nlp == "M1_Market+RawProto":
            X_proto = eval_panel[f_raw_proto].to_numpy(dtype=np.float32)
            X_full = np.concatenate([X_mkt, X_proto], axis=1)
            feat_names = MARKET_FEATURES + f_raw_proto
        elif best_t2_nlp == "M2_Market+SparseGatedNLP":
            g_nov = np.maximum(0.0, nov - tau_nov_full).reshape(-1, 1)
            g_proto = np.maximum(0.0, p_max - tau_proto_full).reshape(-1, 1)
            alert = ((nov > tau_nov_full) | (p_max > tau_proto_full)).astype(np.float32).reshape(-1, 1)
            X_full = np.concatenate([X_mkt, g_nov, g_proto, alert, burst], axis=1)
            feat_names = MARKET_FEATURES + ["gated_novelty", "gated_proto_sim", "exogenous_alert_flag", "news_burst_ratio"]
        else:
            g_proto = np.maximum(0.0, p_max - tau_proto_full).reshape(-1, 1)
            X_full = np.concatenate([X_mkt, nov.reshape(-1, 1), g_proto, burst], axis=1)
            feat_names = MARKET_FEATURES + ["embedding_novelty_score", "gated_proto_sim", "news_burst_ratio"]

        y2_full = eval_panel["vol_expansion_24h"].to_numpy(dtype=np.float32)
        m_t2 = LightGBMBaseline(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=100,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            subsample=0.8,
            class_weight=None,
            scale_pos_weight=1.0,
            is_unbalance=False,
        )
        m_t2.fit(X_full, y2_full)
        cal_t2 = PlattScaler(lambda_reg=1.0).fit(m_t2.predict_logits(X_full), y2_full)

        with open(artifacts_dir / "best_lightgbm_vol24h.pkl", "wb") as f:
            pickle.dump(
                {
                    "model": m_t2.model,
                    "platt_scaler": cal_t2,
                    "feature_names": feat_names,
                    "architecture": best_t2_nlp,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                },
                f,
            )
        saved_t2 = True
    else:
        logger.info(f"Target 2: Best NLP variant ({best_t2_nlp}, PR-AUC {best_t2_pr_auc:.4f}) did not beat Market-Only ({m0_t2_pr_auc:.4f}) with positive BSS.")

    return {
        "best_t1_nlp": best_t1_nlp,
        "promoted_t1": promoted_t1,
        "best_t2_nlp": best_t2_nlp,
        "saved_t2": saved_t2,
    }


def run_exp002e():
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING EXP-002e: ZERO-SCRAPING MULTI-ANGLE NLP EXPERIMENT")
    logger.info("=" * 80)

    eval_panel, meta = engineer_nlp_features_and_targets()
    results = run_exp002e_cpcv(eval_panel, meta)
    promo = conditional_promotion_and_artifacts(results)

    elapsed = time.time() - t0
    peak_rss = get_peak_rss_mb()
    logger.info(f"EXP-002e finished in {elapsed:.1f}s, Peak RSS: {peak_rss:.1f} MB")

    return {
        "results": results,
        "promo": promo,
        "elapsed": elapsed,
        "peak_rss": peak_rss,
    }


if __name__ == "__main__":
    out = run_exp002e()
    res = out["results"]
    p = out["promo"]

    print("\n" + "=" * 125)
    print("EXP-002e TARGET 1: DIRECTIONAL LEFT-TAIL CRASH (tail_crash_6h, Base Rate: 4.12%, Full OOS Panel)")
    print("=" * 125)
    print(f"{'Feature Architecture':30s} | {'PR-AUC':10s} | {'ROC-AUC':10s} | {'Prec@Top-5%':14s} | {'Recall@Top-5%':14s} | {'Lift vs Base':14s} | {'BSS':10s} | {'ECE':8s}")
    print("-" * 125)
    for m, s in res["t1_full"].items():
        print(
            f"{m:30s} | {s['pr_auc']:.4f}     | {s['roc_auc']:.3f}      | "
            f"{s['prec_top5']*100:.1f}%         | {s['recall_top5']*100:.1f}%         | "
            f"{s['lift_top5']:.2f}x          | {s['bss']:+.4f}    | {s['ece']:.3f}"
        )
    print("=" * 125)

    print("\n" + "=" * 125)
    print("EXP-002e TARGET 1: LOW TRAILING VOLATILITY TERCILE (Exogenous Regime Shift Transitions)")
    print("=" * 125)
    print(f"{'Feature Architecture':30s} | {'PR-AUC':10s} | {'ROC-AUC':10s} | {'Prec@Top-5%':14s} | {'Recall@Top-5%':14s} | {'Lift vs Base':14s} | {'BSS':10s} | {'ECE':8s}")
    print("-" * 125)
    for m, s in res["t1_low_vol"].items():
        print(
            f"{m:30s} | {s['pr_auc']:.4f}     | {s['roc_auc']:.3f}      | "
            f"{s['prec_top5']*100:.1f}%         | {s['recall_top5']*100:.1f}%         | "
            f"{s['lift_top5']:.2f}x          | {s['bss']:+.4f}    | {s['ece']:.3f}"
        )
    print("=" * 125)

    print("\n" + "=" * 125)
    print("EXP-002e TARGET 2: UNSIGNED 24h REALIZED VOL SHOCK (vol_expansion_24h, Base Rate: 11.56%)")
    print("=" * 125)
    print(f"{'Feature Architecture':30s} | {'PR-AUC':10s} | {'ROC-AUC':10s} | {'Prec@Top-10%':14s} | {'Recall@Top-10%':14s} | {'Lift vs Base':14s} | {'BSS':10s} | {'ECE':8s}")
    print("-" * 125)
    for m, s in res["t2_full"].items():
        print(
            f"{m:30s} | {s['pr_auc']:.4f}     | {s['roc_auc']:.3f}      | "
            f"{s['prec_top10']*100:.1f}%        | {s['recall_top10']*100:.1f}%        | "
            f"{s['lift_top10']:.2f}x          | {s['bss']:+.4f}    | {s['ece']:.3f}"
        )
    print("=" * 125)

    print("\nPROMOTION AUDIT:")
    print(f"  Target 1 Best NLP: {p['best_t1_nlp']} -> Promoted to best_lightgbm_risk.pkl: {p['promoted_t1']}")
    print(f"  Target 2 Best NLP: {p['best_t2_nlp']} -> Saved to best_lightgbm_vol24h.pkl: {p['saved_t2']}")
