"""
Comprehensive Diagnostic Audit Module for Sprint 2.5 (EXP-002c).
Executes:
1. Hidden Metadata Leakage Audit:
   - Evaluates confirmation_count causality (retrospective vs causal).
   - Verifies zero post-hoc columns (vote_counts, total_votes, market_direction) in feature matrices.
   - Executes 1-split News-Shuffle Placebo Test to confirm NLP lift collapse.
2. Conditional Attribution & Market-Only vs. Multimodal Proof:
   - Benchmarks LightGBM_Market_Only vs LightGBM_Market+News vs iTransformer_Only vs Multimodal_iTransformer_NLP.
   - Evaluates PR-AUC, Precision@Top-5%, Recall@Top-5% (Lift vs base rate), Brier Skill Score (BSS), and ECE.
   - Evaluates across Full OOS panel (with per-split breakdown), News-Active vs Quiet slices, and Low-Vol tercile.
   - Logs top 10 LightGBM feature importances and learned ReZero gate per fold.
3. Historical Shock Case Studies & 3-Way Stress-Tested Overlay:
   - Hourly lead-lag analysis around FTX (Nov 2022), SEC suits (Jun 2023), and Aug 2024 crash.
   - 3-Way APEX v45 Downstream Overlay with 25 bps one-way stress slippage on scale turnover.
"""

import gc
import os
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from apex_multimodal_risk.src.features.shared_features import (
    MARKET_NUMERICAL_COLS,
    CPCVStandardScaler,
    load_and_merge_features_and_targets,
)
from apex_multimodal_risk.src.models.baselines import (
    LIGHTGBM_FEATURES,
    LightGBMBaseline,
)
from apex_multimodal_risk.src.models.itransformer import iTransformerOnlyModel
from apex_multimodal_risk.src.models.multimodal_fusion import (
    MultimodalCrossAttentionFusion,
)
from apex_multimodal_risk.src.train.calibration import (
    PlattScaler,
    compute_ece,
)
from apex_multimodal_risk.src.train.cpcv_runner import generate_cpcv_splits
from apex_multimodal_risk.src.train.train_ablation import (
    build_lazy_news_index,
    build_lookback_tensor,
    eval_itransformer_batched,
    eval_multimodal_batched,
    get_peak_rss_mb,
    prepare_tabular_data,
)
from apex_multimodal_risk.src.utils import (
    PROJECT_ROOT,
    get_device,
    read_parquet_utc,
    setup_logger,
)

logger = setup_logger("audit_diagnostics")

LIGHTGBM_MARKET_ONLY_FEATURES = [
    "ret_1h",
    "ret_6h",
    "ret_24h",
    "ret_72h",
    "realized_vol_24h",
    "realized_vol_72h",
    "realized_vol_168h",
    "garch_cond_vol",
    "skew_72h",
    "skew_168h",
    "skew_336h",
    "funding_zscore",
    "oi_delta_48h",
    "oi_zscore",
    "cross_dispersion_24h",
    "cross_rank_24h",
]


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs_cal: np.ndarray,
    top_pct: float = 0.05,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Computes comprehensive ranking and skill metrics:
    - PR-AUC
    - ROC-AUC
    - Precision@Top-5%
    - Recall@Top-5%
    - Lift@Top-5% (Precision / base_rate)
    - Brier score
    - Brier Skill Score (BSS = 1 - Brier / Brier_ref)
    - ECE (10 bins)
    """
    from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probs_cal, dtype=np.float64)
    n = len(y)
    base_rate = float(np.mean(y)) if n > 0 else 0.0412
    brier_ref = base_rate * (1.0 - base_rate)

    # PR-AUC & ROC-AUC
    prec, rec, _ = precision_recall_curve(y, p)
    pr_auc = float(auc(rec, prec))
    try:
        roc_auc = float(roc_auc_score(y, p))
    except ValueError:
        roc_auc = 0.5

    # Top 5% ranking metrics
    k = max(1, int(n * top_pct))
    top_indices = np.argsort(p)[-k:]
    hits = np.sum(y[top_indices])
    total_pos = np.sum(y)

    prec_top5 = float(hits / k)
    recall_top5 = float(hits / total_pos) if total_pos > 0 else 0.0
    lift_top5 = float(prec_top5 / base_rate) if base_rate > 0 else 1.0

    # Calibration & Skill
    brier = float(np.mean((y - p) ** 2))
    bss = float(1.0 - (brier / brier_ref)) if brier_ref > 0 else 0.0
    ece = float(compute_ece(y, p, n_bins=n_bins))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "prec_top5": prec_top5,
        "recall_top5": recall_top5,
        "lift_top5": lift_top5,
        "brier": brier,
        "bss": bss,
        "ece": ece,
        "base_rate": base_rate,
    }


def audit_metadata_and_causality() -> dict[str, Any]:
    """
    Part 1: Audits confirmation_count causality, verifies absence of post-hoc columns,
    and runs a 1-split News-Shuffle Placebo Test.
    """
    logger.info("=" * 80)
    logger.info("PART 1: HIDDEN METADATA & CAUSALITY AUDIT")
    logger.info("=" * 80)

    # 1. Inspect confirmation_count
    news_dedup = pd.read_parquet(PROJECT_ROOT / "apex_multimodal_risk/data/processed/news_dedup.parquet")
    conf_counts = news_dedup["confirmation_count"].value_counts().to_dict()
    is_retrospective = any(c > 1 for c in conf_counts)

    logger.info(f"confirmation_count audit: unique counts = {conf_counts}")
    if is_retrospective:
        logger.warning(
            "FLAGGED: confirmation_count in noise_filter.py was accumulated across rolling 6h window "
            "mutating primary cluster records retrospectively when duplicate stories arrived later!"
        )
    else:
        logger.info("confirmation_count is strictly 1 at ingestion.")

    # 2. Check post-hoc columns in feature definitions
    forbidden_cols = ["vote_counts", "total_votes", "market_direction", "vote_counts_json"]
    entered_lgbm = [c for c in forbidden_cols if c in LIGHTGBM_FEATURES]
    entered_market = [c for c in forbidden_cols if c in MARKET_NUMERICAL_COLS]

    logger.info(f"Post-hoc columns in LIGHTGBM_FEATURES: {entered_lgbm} (Clean: {len(entered_lgbm) == 0})")
    logger.info(f"Post-hoc columns in MARKET_NUMERICAL_COLS: {entered_market} (Clean: {len(entered_market) == 0})")

    # 3. 1-Split News-Shuffle Placebo Test on Split 0
    logger.info("\n--- Running 1-Split News-Shuffle Placebo Test on CPCV Split 0 ---")
    merged_panel = load_and_merge_features_and_targets()
    panel_df = prepare_tabular_data(merged_panel, news_dedup)
    eval_panel = panel_df.loc[panel_df["target_valid"].to_numpy()].copy().reset_index(drop=True)
    unique_ts = np.array(sorted(eval_panel["timestamp"].unique()))
    splits = generate_cpcv_splits(unique_ts, n_blocks=5, k_test=2, purge_embargo_hours=504)
    split_0 = splits[0]

    ts_panel = eval_panel["timestamp"]
    tr_mask = split_0.get_train_mask(ts_panel)
    val_mask = split_0.get_inner_val_mask(ts_panel)
    te_mask = split_0.get_test_mask(ts_panel)
    y_te = eval_panel.loc[te_mask, "tail_crash_6h"].to_numpy(dtype=np.float32)

    # a. Market-Only
    m_market = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
    m_market.fit(
        eval_panel.loc[tr_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy(),
        eval_panel.loc[tr_mask, "tail_crash_6h"].to_numpy(),
        eval_set=[(eval_panel.loc[val_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy(), eval_panel.loc[val_mask, "tail_crash_6h"].to_numpy())],
    )
    val_log_mkt = m_market.predict_logits(eval_panel.loc[val_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy())
    te_log_mkt = m_market.predict_logits(eval_panel.loc[te_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy())
    cal_mkt = PlattScaler().fit(val_log_mkt, eval_panel.loc[val_mask, "tail_crash_6h"].to_numpy())
    probs_mkt = cal_mkt.transform(te_log_mkt)
    res_mkt = compute_comprehensive_metrics(y_te, probs_mkt)

    # b. Market + News (Normal)
    m_news = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
    m_news.fit(
        eval_panel.loc[tr_mask, LIGHTGBM_FEATURES].to_numpy(),
        eval_panel.loc[tr_mask, "tail_crash_6h"].to_numpy(),
        eval_set=[(eval_panel.loc[val_mask, LIGHTGBM_FEATURES].to_numpy(), eval_panel.loc[val_mask, "tail_crash_6h"].to_numpy())],
    )
    val_log_news = m_news.predict_logits(eval_panel.loc[val_mask, LIGHTGBM_FEATURES].to_numpy())
    te_log_news = m_news.predict_logits(eval_panel.loc[te_mask, LIGHTGBM_FEATURES].to_numpy())
    cal_news = PlattScaler().fit(val_log_news, eval_panel.loc[val_mask, "tail_crash_6h"].to_numpy())
    probs_news = cal_news.transform(te_log_news)
    res_news = compute_comprehensive_metrics(y_te, probs_news)

    # c. Market + News (Shuffled Placebo: randomly permute news features)
    news_cols = [c for c in LIGHTGBM_FEATURES if c not in LIGHTGBM_MARKET_ONLY_FEATURES]
    shuffled_panel = eval_panel.copy()
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(shuffled_panel))
    for c in news_cols:
        shuffled_panel[c] = shuffled_panel[c].to_numpy()[shuffled_idx]

    m_placebo = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
    m_placebo.fit(
        shuffled_panel.loc[tr_mask, LIGHTGBM_FEATURES].to_numpy(),
        shuffled_panel.loc[tr_mask, "tail_crash_6h"].to_numpy(),
        eval_set=[(shuffled_panel.loc[val_mask, LIGHTGBM_FEATURES].to_numpy(), shuffled_panel.loc[val_mask, "tail_crash_6h"].to_numpy())],
    )
    val_log_plac = m_placebo.predict_logits(shuffled_panel.loc[val_mask, LIGHTGBM_FEATURES].to_numpy())
    te_log_plac = m_placebo.predict_logits(shuffled_panel.loc[te_mask, LIGHTGBM_FEATURES].to_numpy())
    cal_plac = PlattScaler().fit(val_log_plac, shuffled_panel.loc[val_mask, "tail_crash_6h"].to_numpy())
    probs_plac = cal_plac.transform(te_log_plac)
    res_plac = compute_comprehensive_metrics(y_te, probs_plac)

    logger.info(f"Split 0 | Market_Only:          PR-AUC = {res_mkt['pr_auc']:.4f}, Lift@Top-5% = {res_mkt['lift_top5']:.2f}x, BSS = {res_mkt['bss']:.4f}")
    logger.info(f"Split 0 | Market+News (True):   PR-AUC = {res_news['pr_auc']:.4f}, Lift@Top-5% = {res_news['lift_top5']:.2f}x, BSS = {res_news['bss']:.4f}")
    logger.info(f"Split 0 | News-Shuffle Placebo: PR-AUC = {res_plac['pr_auc']:.4f}, Lift@Top-5% = {res_plac['lift_top5']:.2f}x, BSS = {res_plac['bss']:.4f}")

    lift_true = res_news["pr_auc"] - res_mkt["pr_auc"]
    lift_placebo = res_plac["pr_auc"] - res_mkt["pr_auc"]
    logger.info(f"True News Marginal Lift:    {lift_true:+.4f}")
    logger.info(f"Placebo News Marginal Lift: {lift_placebo:+.4f} (Collapses to zero/negative: {lift_placebo <= 0.001})")

    return {
        "is_retrospective": is_retrospective,
        "post_hoc_clean": len(entered_lgbm) == 0 and len(entered_market) == 0,
        "res_market_only": res_mkt,
        "res_market_news": res_news,
        "res_placebo": res_plac,
        "lift_true": lift_true,
        "lift_placebo": lift_placebo,
    }


def execute_full_conditional_attribution() -> dict[str, Any]:
    """
    Part 2: Runs 10-split CPCV across 4 models:
    - LightGBM_Market_Only
    - LightGBM_Market+News
    - iTransformer_Only
    - Multimodal_iTransformer_NLP
    Reports metrics across:
    - Full OOS panel (with per-split breakdown)
    - News-Active slice (>=1 material news cluster in past 24h) vs Quiet slice
    - Low Trailing Volatility tercile
    Extracts top 10 LightGBM feature importances and ReZero gates per fold.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: CONDITIONAL ATTRIBUTION & MARKET-ONLY VS MULTIMODAL PROOF")
    logger.info("=" * 80)

    device = get_device()
    cache_dir = str(PROJECT_ROOT / "apex_multimodal_risk/data/processed/text_encoder_cpcv_cache")

    # Load data
    merged_panel = load_and_merge_features_and_targets()
    news_df = read_parquet_utc("apex_multimodal_risk/data/processed/news_dedup.parquet")
    panel_df = prepare_tabular_data(merged_panel, news_df)
    valid_mask = panel_df["target_valid"].to_numpy()
    eval_panel = panel_df.loc[valid_mask].copy().reset_index(drop=True)
    unique_ts = np.array(sorted(eval_panel["timestamp"].unique()))

    splits = generate_cpcv_splits(unique_ts, n_blocks=5, k_test=2, purge_embargo_hours=504)
    panel_timestamps = eval_panel["timestamp"]
    y_clf_all = eval_panel["tail_crash_6h"].to_numpy(dtype=np.float32)
    y_var_all = eval_panel["fwd_min_ret_6h"].to_numpy(dtype=np.float32)
    vol_72h = eval_panel["trailing_1h_vol_72h"].to_numpy()

    # Precompute lookback & news structure
    lookback_tensor = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)
    _, cluster_indices, cluster_recency, cluster_pad_mask = build_lazy_news_index(
        eval_panel, news_df, max_clusters=16
    )
    bge_embs = np.array(news_df["embedding"].tolist(), dtype=np.float32)
    tier_num = (news_df["source_tier"] == "tier_1").astype(np.float32).to_numpy()
    conf_num = np.log1p(news_df["confirmation_count"].to_numpy(dtype=np.float32))
    meta_arr = np.stack([tier_num, conf_num], axis=1)

    prob_cols = [
        "p_non_event_noise",
        "p_regulatory_action",
        "p_protocol_exploit",
        "p_macro_shock",
        "p_network_outage",
        "p_exchange_insolvency",
    ]

    models = [
        "LightGBM_Market_Only",
        "LightGBM_Market+News",
        "iTransformer_Only",
        "Multimodal_iTransformer_NLP",
    ]

    split_metrics: dict[str, list[dict[str, float]]] = {m: [] for m in models}
    all_oos_probs: dict[str, list[np.ndarray]] = {m: [] for m in models}
    all_oos_y: list[np.ndarray] = []
    all_oos_ts: list[pd.Series] = []
    all_oos_active_mask: list[np.ndarray] = []
    all_oos_vol: list[np.ndarray] = []

    rezero_gates = []
    lgbm_feature_importances: list[np.ndarray] = []

    for split in splits:
        split_id = split.split_id
        logger.info(f"--- CPCV Split {split_id}/9 ---")

        tr_mask = split.get_train_mask(panel_timestamps)
        val_mask = split.get_inner_val_mask(panel_timestamps)
        te_mask = split.get_test_mask(panel_timestamps)

        scaler = CPCVStandardScaler(feature_cols=MARKET_NUMERICAL_COLS)
        scaler.fit(eval_panel.loc[tr_mask])

        y_tr, y_val, y_te = y_clf_all[tr_mask], y_clf_all[val_mask], y_clf_all[te_mask]
        v_tr, _v_val, _v_te = y_var_all[tr_mask], y_var_all[val_mask], y_var_all[te_mask]

        all_oos_y.append(y_te)
        all_oos_ts.append(panel_timestamps[te_mask])
        all_oos_active_mask.append((eval_panel.loc[te_mask, "news_count_24h"] > 0).to_numpy())
        all_oos_vol.append(vol_72h[te_mask])

        # 1. LightGBM Market-Only
        X_mkt_tr = eval_panel.loc[tr_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy()
        X_mkt_val = eval_panel.loc[val_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy()
        X_mkt_te = eval_panel.loc[te_mask, LIGHTGBM_MARKET_ONLY_FEATURES].to_numpy()

        m_mkt = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
        m_mkt.fit(X_mkt_tr, y_tr, eval_set=[(X_mkt_val, y_val)])
        val_logits_mkt = m_mkt.predict_logits(X_mkt_val)
        te_logits_mkt = m_mkt.predict_logits(X_mkt_te)
        cal_mkt = PlattScaler().fit(val_logits_mkt, y_val)
        p_te_mkt = cal_mkt.transform(te_logits_mkt)

        res_mkt = compute_comprehensive_metrics(y_te, p_te_mkt)
        split_metrics["LightGBM_Market_Only"].append(res_mkt)
        all_oos_probs["LightGBM_Market_Only"].append(p_te_mkt)
        del m_mkt; gc.collect()

        # 2. LightGBM Market+News
        X_tab_tr = eval_panel.loc[tr_mask, LIGHTGBM_FEATURES].to_numpy()
        X_tab_val = eval_panel.loc[val_mask, LIGHTGBM_FEATURES].to_numpy()
        X_tab_te = eval_panel.loc[te_mask, LIGHTGBM_FEATURES].to_numpy()

        m_tab = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
        m_tab.fit(X_tab_tr, y_tr, eval_set=[(X_tab_val, y_val)])
        val_logits_tab = m_tab.predict_logits(X_tab_val)
        te_logits_tab = m_tab.predict_logits(X_tab_te)
        cal_tab = PlattScaler().fit(val_logits_tab, y_val)
        p_te_tab = cal_tab.transform(te_logits_tab)

        res_tab = compute_comprehensive_metrics(y_te, p_te_tab)
        split_metrics["LightGBM_Market+News"].append(res_tab)
        all_oos_probs["LightGBM_Market+News"].append(p_te_tab)
        lgbm_feature_importances.append(m_tab.model.feature_importances_)
        del m_tab; gc.collect()

        # 3. iTransformer_Only
        m_vec = torch.tensor(scaler.means_, dtype=torch.float32).view(1, 1, -1)
        s_vec = torch.tensor(scaler.stds_, dtype=torch.float32).view(1, 1, -1)
        t_lb_tr = (torch.tensor(lookback_tensor[tr_mask], dtype=torch.float32) - m_vec) / s_vec
        t_lb_val = (torch.tensor(lookback_tensor[val_mask], dtype=torch.float32) - m_vec) / s_vec
        t_lb_te = (torch.tensor(lookback_tensor[te_mask], dtype=torch.float32) - m_vec) / s_vec

        t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
        t_v_tr = torch.tensor(v_tr, dtype=torch.float32)

        m3 = iTransformerOnlyModel(
            lookback_len=48, num_variates=len(MARKET_NUMERICAL_COLS), d_model=64, nhead=4, num_layers=2, dim_feedforward=128
        ).to(device)
        opt_m3 = torch.optim.AdamW(m3.parameters(), lr=1e-3, weight_decay=1e-4)
        ds_m3 = TensorDataset(t_lb_tr, t_y_tr, t_v_tr)
        loader_m3 = DataLoader(ds_m3, batch_size=128, shuffle=True, num_workers=0)

        m3.train()
        for _ in range(3):
            for b_lb, b_y, b_v in loader_m3:
                b_lb, b_y, b_v = b_lb.to(device), b_y.to(device), b_v.to(device)
                opt_m3.zero_grad()
                out3 = m3(b_lb)
                loss3, _ = m3.compute_loss(out3, b_y, b_v, pos_weight=23.0)
                loss3.backward()
                opt_m3.step()

        val_log_m3, _, _ = eval_itransformer_batched(m3, t_lb_val, device, batch_size=128)
        te_log_m3, _, _ = eval_itransformer_batched(m3, t_lb_te, device, batch_size=128)
        cal_m3 = PlattScaler().fit(val_log_m3, y_val)
        p_te_m3 = cal_m3.transform(te_log_m3)

        res_m3 = compute_comprehensive_metrics(y_te, p_te_m3)
        split_metrics["iTransformer_Only"].append(res_m3)
        all_oos_probs["iTransformer_Only"].append(p_te_m3)
        del m3, opt_m3, loader_m3, ds_m3; gc.collect()

        # 4. Multimodal_iTransformer_NLP
        cached_split_file = os.path.join(cache_dir, f"split_{split_id}.parquet")
        cached_df = pd.read_parquet(cached_split_file, columns=prob_cols)
        fold_probs = cached_df[prob_cols].to_numpy(dtype=np.float32)
        full_news_feats = np.concatenate([bge_embs, fold_probs, meta_arr], axis=1)
        full_news_matrix = torch.from_numpy(full_news_feats).float()

        t_idx_tr = torch.tensor(cluster_indices[tr_mask], dtype=torch.int32)
        t_idx_val = torch.tensor(cluster_indices[val_mask], dtype=torch.int32)
        t_idx_te = torch.tensor(cluster_indices[te_mask], dtype=torch.int32)
        t_rec_tr = torch.tensor(cluster_recency[tr_mask], dtype=torch.float32)
        t_rec_val = torch.tensor(cluster_recency[val_mask], dtype=torch.float32)
        t_rec_te = torch.tensor(cluster_recency[te_mask], dtype=torch.float32)
        t_mask_tr = torch.tensor(cluster_pad_mask[tr_mask], dtype=torch.bool)
        t_mask_val = torch.tensor(cluster_pad_mask[val_mask], dtype=torch.bool)
        t_mask_te = torch.tensor(cluster_pad_mask[te_mask], dtype=torch.bool)

        m4 = MultimodalCrossAttentionFusion(
            lookback_len=48, num_variates=len(MARKET_NUMERICAL_COLS), news_feat_dim=394, max_news_clusters=16,
            d_model=64, nhead=4, num_itransformer_layers=2, dim_feedforward=128, dropout=0.1, news_dropout=0.35, noise_threshold=0.80
        ).to(device)
        opt_m4 = torch.optim.AdamW(m4.parameters(), lr=1e-3, weight_decay=1e-3)
        ds_m4 = TensorDataset(t_lb_tr, t_idx_tr, t_rec_tr, t_mask_tr, t_y_tr, t_v_tr)
        loader_m4 = DataLoader(ds_m4, batch_size=128, shuffle=True, num_workers=0)

        m4.train()
        for _ in range(3):
            for b_lb, b_idx, b_rec, b_m, b_y, b_v in loader_m4:
                b_lb = b_lb.to(device)
                b_m = b_m.to(device)
                b_y, b_v = b_y.to(device), b_v.to(device)
                b_static = full_news_matrix[b_idx.long()].to(device)
                b_rec_dev = b_rec.to(device).unsqueeze(-1)
                b_act_dev = (~b_m).float().unsqueeze(-1)
                b_news = torch.cat([b_static, b_rec_dev, b_act_dev], dim=-1)

                opt_m4.zero_grad()
                out4 = m4(b_lb, b_news, b_m)
                loss4, _ = m4.compute_loss(out4, b_y, b_v, pos_weight=23.0)
                loss4.backward()
                opt_m4.step()

        val_log_m4, _, _ = eval_multimodal_batched(m4, t_lb_val, t_idx_val, t_rec_val, t_mask_val, full_news_matrix, device, batch_size=128)
        te_log_m4, _, _ = eval_multimodal_batched(m4, t_lb_te, t_idx_te, t_rec_te, t_mask_te, full_news_matrix, device, batch_size=128)
        cal_m4 = PlattScaler().fit(val_log_m4, y_val)
        p_te_m4 = cal_m4.transform(te_log_m4)

        gate_val = float(torch.tanh(m4.alpha).item())
        rezero_gates.append(gate_val)

        res_m4 = compute_comprehensive_metrics(y_te, p_te_m4)
        split_metrics["Multimodal_iTransformer_NLP"].append(res_m4)
        all_oos_probs["Multimodal_iTransformer_NLP"].append(p_te_m4)

        del m4, opt_m4, loader_m4, ds_m4, t_lb_tr, t_lb_val, t_lb_te, full_news_matrix; gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # Aggregate flat arrays
    flat_y = np.concatenate(all_oos_y)
    flat_ts = pd.concat(all_oos_ts).reset_index(drop=True)
    flat_active = np.concatenate(all_oos_active_mask)
    flat_vol = np.concatenate(all_oos_vol)
    vol_tercile_33 = np.percentile(flat_vol, 33.33)
    flat_low_vol = flat_vol <= vol_tercile_33

    flat_probs = {m: np.concatenate(all_oos_probs[m]) for m in models}

    # Slice Evaluations
    slice_masks = {
        "Full OOS Panel": np.ones(len(flat_y), dtype=bool),
        "News-Active Slice": flat_active,
        "Quiet Slice": ~flat_active,
        "Low Trailing Vol Tercile": flat_low_vol,
    }

    slice_results: dict[str, dict[str, dict[str, float]]] = {}
    for s_name, s_mask in slice_masks.items():
        slice_results[s_name] = {}
        for m in models:
            slice_results[s_name][m] = compute_comprehensive_metrics(flat_y[s_mask], flat_probs[m][s_mask])

    # Top 10 LightGBM feature importances
    avg_feat_imp = np.mean(lgbm_feature_importances, axis=0)
    top_10_indices = np.argsort(avg_feat_imp)[-10:][::-1]
    top_10_features = [(LIGHTGBM_FEATURES[idx], float(avg_feat_imp[idx])) for idx in top_10_indices]

    return {
        "models": models,
        "split_metrics": split_metrics,
        "slice_results": slice_results,
        "flat_probs": flat_probs,
        "flat_y": flat_y,
        "flat_ts": flat_ts,
        "eval_panel": eval_panel,
        "top_10_features": top_10_features,
        "rezero_gates": rezero_gates,
    }


def execute_shock_case_studies_and_slippage_overlay(
    audit_data: dict[str, Any]
) -> dict[str, Any]:
    """
    Part 3: Hourly lead-lag shock case studies (FTX, SEC, Aug 2024) and 3-way stress overlay
    with 25 bps one-way slippage.
    """
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: HISTORICAL SHOCK CASE STUDIES & 3-WAY STRESS OVERLAY")
    logger.info("=" * 80)

    flat_ts = audit_data["flat_ts"]
    flat_y = audit_data["flat_y"]
    flat_probs = audit_data["flat_probs"]

    df_preds = pd.DataFrame({
        "timestamp": flat_ts,
        "tail_crash_6h": flat_y,
        "prob_mkt": flat_probs["LightGBM_Market_Only"],
        "prob_tab_news": flat_probs["LightGBM_Market+News"],
        "prob_multimodal": flat_probs["Multimodal_iTransformer_NLP"],
    })

    # Group by timestamp (mean across assets)
    hourly_preds = df_preds.groupby("timestamp").mean().sort_index()

    # Shock windows
    shock_windows = {
        "FTX Collapse (Nov 2022)": ("2022-11-06 00:00:00+00:00", "2022-11-12 23:00:00+00:00"),
        "SEC Lawsuits (Jun 2023)": ("2023-06-04 00:00:00+00:00", "2023-06-10 23:00:00+00:00"),
        "Aug 2024 Flash Crash": ("2024-08-02 00:00:00+00:00", "2024-08-07 23:00:00+00:00"),
    }

    shock_case_studies = {}
    news_df = read_parquet_utc("apex_multimodal_risk/data/processed/news_dedup.parquet")

    for s_name, (t_start, t_end) in shock_windows.items():
        sub_preds = hourly_preds.loc[t_start:t_end].copy()
        sub_news = news_df[(news_df["ingested_at"] >= t_start) & (news_df["ingested_at"] <= t_end)]

        # Max crash risk hour
        peak_mkt_hour = sub_preds["prob_mkt"].idxmax()
        peak_multi_hour = sub_preds["prob_multimodal"].idxmax()
        lead_hours = (peak_mkt_hour - peak_multi_hour).total_seconds() / 3600.0

        shock_case_studies[s_name] = {
            "peak_mkt_hour": str(peak_mkt_hour),
            "peak_mkt_prob": float(sub_preds.loc[peak_mkt_hour, "prob_mkt"]),
            "peak_multi_hour": str(peak_multi_hour),
            "peak_multi_prob": float(sub_preds.loc[peak_multi_hour, "prob_multimodal"]),
            "lead_hours_multimodal_vs_market": float(lead_hours),
            "news_count_in_window": len(sub_news),
            "sample_headlines": sub_news["title"].head(3).tolist(),
        }

    # 3-Way Stress-Tested APEX Overlay with 25 bps slippage
    logger.info("Executing 3-Way Stress-Tested APEX v45 Overlay with 25 bps Slippage...")
    sys.path.insert(0, str(PROJECT_ROOT / "archive"))
    sys.path.insert(0, str(PROJECT_ROOT))
    from archive.apex_v45_champion import (
        UNIVERSE,
        W_ARB,
        W_DIR,
        W_SPREAD,
        fast_metrics,
        load_data_and_features,
        run_directional_engine,
        run_funding_arb_engine,
        run_spread_engine,
    )

    dataset, processed, garch_wts, eval_index, n, _split_idx, macro_regime, mtf_skew_mult, arb_mag_scale = (
        load_data_and_features()
    )

    dir_stream, _ = run_directional_engine(processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult)
    spread_stream, _ = run_spread_engine(processed, dataset, UNIVERSE, eval_index, n)
    arb_stream, _ = run_funding_arb_engine(dataset, UNIVERSE, eval_index, n, arb_mag_scale)

    dir_filtered = dir_stream * macro_regime
    spread_filtered = spread_stream * macro_regime
    unhedged_stream = W_DIR * dir_filtered + W_SPREAD * spread_filtered + W_ARB * arb_stream

    overlap_mask = eval_index.isin(flat_ts)
    if np.sum(overlap_mask) == 0:
        overlap_mask = np.ones(len(eval_index), dtype=bool)

    def evaluate_circuit_breaker_with_slippage(probs_series: pd.Series, tau: float = 0.097, slippage_bps: float = 25.0):
        aligned_p = probs_series.reindex(eval_index).fillna(0.0412).values
        scale = np.where(aligned_p <= tau, 1.0, np.clip(1.0 - 4.0 * (aligned_p - tau), 0.25, 1.0))

        # Slippage drag: slippage_bps * |scale_t - scale_{t-1}| * W_DIR
        scale_shift = np.roll(scale, 1)
        scale_shift[0] = 1.0
        scale_turnover = np.abs(scale - scale_shift)
        slippage_drag = (slippage_bps / 10_000.0) * scale_turnover * W_DIR

        dampened_dir = dir_filtered * scale
        net_stream = W_DIR * dampened_dir + W_SPREAD * spread_filtered + W_ARB * arb_stream - slippage_drag

        sub_net = net_stream[overlap_mask]
        m = fast_metrics(sub_net)

        # 95% CVaR (Expected Shortfall)
        q5 = np.percentile(sub_net, 5.0)
        cvar95 = float(np.mean(sub_net[sub_net <= q5]))

        fired_pct = float(np.mean(scale[overlap_mask] < 1.0) * 100.0)

        return {
            "sharpe": float(m["sharpe"]),
            "dd": float(m["dd"]),
            "calmar": float(m["calmar"]),
            "cvar95": cvar95,
            "fired_pct": fired_pct,
            "total_slippage_drag_bps": float(np.sum(slippage_drag[overlap_mask]) * 10_000.0),
        }

    # Baseline Unhedged metrics
    sub_unhedged = unhedged_stream[overlap_mask]
    m_unh = fast_metrics(sub_unhedged)
    q5_unh = np.percentile(sub_unhedged, 5.0)
    cvar95_unh = float(np.mean(sub_unhedged[sub_unhedged <= q5_unh]))

    unhedged_res = {
        "sharpe": float(m_unh["sharpe"]),
        "dd": float(m_unh["dd"]),
        "calmar": float(m_unh["calmar"]),
        "cvar95": cvar95_unh,
        "fired_pct": 0.0,
        "total_slippage_drag_bps": 0.0,
    }

    # Market-Only Overlay
    p_mkt_series = hourly_preds["prob_mkt"]
    mkt_overlay_res = evaluate_circuit_breaker_with_slippage(p_mkt_series, tau=0.097, slippage_bps=25.0)

    # Multimodal Overlay
    p_multi_series = hourly_preds["prob_multimodal"]
    multi_overlay_res = evaluate_circuit_breaker_with_slippage(p_multi_series, tau=0.103, slippage_bps=25.0)

    # Tabular Market+News Overlay
    p_tab_series = hourly_preds["prob_tab_news"]
    tab_overlay_res = evaluate_circuit_breaker_with_slippage(p_tab_series, tau=0.097, slippage_bps=25.0)

    return {
        "shock_case_studies": shock_case_studies,
        "unhedged": unhedged_res,
        "market_only": mkt_overlay_res,
        "multimodal": multi_overlay_res,
        "tabular_news": tab_overlay_res,
    }


def run_full_diagnostic_audit() -> dict[str, Any]:
    t0 = time.time()
    logger.info("=" * 80)
    logger.info("STARTING SPRINT 2.5 READ-ONLY DIAGNOSTIC AUDIT (EXP-002c)")
    logger.info("=" * 80)

    # 1. Metadata & Causality Audit
    part1_res = audit_metadata_and_causality()

    # 2. Conditional Attribution across 10 CPCV splits & slices
    part2_res = execute_full_conditional_attribution()

    # 3. Shock case studies & 3-way stress overlay
    part3_res = execute_shock_case_studies_and_slippage_overlay(part2_res)

    elapsed = time.time() - t0
    logger.info(f"Full Diagnostic Audit completed in {elapsed:.1f}s, Peak RSS: {get_peak_rss_mb():.1f} MB")

    return {
        "part1": part1_res,
        "part2": part2_res,
        "part3": part3_res,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    results = run_full_diagnostic_audit()

    print("\n" + "=" * 125)
    print("EXP-002c CONDITIONAL ATTRIBUTION TABLE: FULL OOS PANEL (10 CPCV SPLITS)")
    print("=" * 125)
    print(f"{'Model Architecture':30s} | {'PR-AUC':10s} | {'ROC-AUC':10s} | {'Prec@Top-5%':14s} | {'Recall@Top-5%':14s} | {'Lift vs Prior':14s} | {'BSS':10s} | {'ECE':8s}")
    print("-" * 125)
    for m, stats in results["part2"]["slice_results"]["Full OOS Panel"].items():
        print(
            f"{m:30s} | {stats['pr_auc']:.4f}     | {stats['roc_auc']:.3f}      | "
            f"{stats['prec_top5']*100:.1f}%         | {stats['recall_top5']*100:.1f}%         | "
            f"{stats['lift_top5']:.2f}x          | {stats['bss']:+.4f}    | {stats['ece']:.3f}"
        )
    print("=" * 125)

    print("\n" + "=" * 125)
    print("EXP-002c SLICE BREAKDOWN: NEWS-ACTIVE VS QUIET VS LOW-VOLATILITY TERCILE")
    print("=" * 125)
    for s_name in ["News-Active Slice", "Quiet Slice", "Low Trailing Vol Tercile"]:
        print(f"\n--- {s_name.upper()} ---")
        print(f"{'Model Architecture':30s} | {'PR-AUC':10s} | {'Prec@Top-5%':14s} | {'Recall@Top-5%':14s} | {'Lift vs Prior':14s} | {'BSS':10s} | {'ECE':8s}")
        print("-" * 115)
        for m, stats in results["part2"]["slice_results"][s_name].items():
            print(
                f"{m:30s} | {stats['pr_auc']:.4f}     | {stats['prec_top5']*100:.1f}%         | "
                f"{stats['recall_top5']*100:.1f}%         | {stats['lift_top5']:.2f}x          | "
                f"{stats['bss']:+.4f}    | {stats['ece']:.3f}"
            )
    print("=" * 125)

    print("\n" + "=" * 120)
    print("HISTORICAL SHOCK CASE STUDIES: LEAD-LAG TIMING AROUND MAJOR LIQUIDATION CRISES")
    print("=" * 120)
    for s_name, data in results["part3"]["shock_case_studies"].items():
        print(f"\n>>> {s_name} (News Count: {data['news_count_in_window']})")
        print(f"  Peak Market-Only Risk: {data['peak_mkt_hour']} (p = {data['peak_mkt_prob']:.3f})")
        print(f"  Peak Multimodal Risk:  {data['peak_multi_hour']} (p = {data['peak_multi_prob']:.3f})")
        print(f"  Lead Time (Multimodal vs Market-Only): {data['lead_hours_multimodal_vs_market']:+.1f} hours")
        print(f"  Sample Headlines: {data['sample_headlines']}")
    print("=" * 120)

    print("\n" + "=" * 110)
    print("3-WAY STRESS-TESTED APEX V45 DOWNSTREAM OVERLAY (25 BPS ONE-WAY SLIPPAGE)")
    print("=" * 110)
    print(f"{'Downstream Strategy':30s} | {'Net Sharpe':12s} | {'Max DD':10s} | {'95% CVaR':12s} | {'Calmar':10s} | {'Circuit Fired %':15s}")
    print("-" * 110)
    p3 = results["part3"]
    strats = [
        ("Unhedged APEX v45 Baseline", p3["unhedged"]),
        ("Market-Only Circuit Breaker", p3["market_only"]),
        ("LightGBM Market+News Overlay", p3["tabular_news"]),
        ("Multimodal Cross-Attention Overlay", p3["multimodal"]),
    ]
    for name, s in strats:
        print(
            f"{name:30s} | {s['sharpe']:<12.2f} | {s['dd']:<10.1f}% | {s['cvar95']*100:<10.3f}% | {s['calmar']:<10.2f} | {s['fired_pct']:<15.1f}%"
        )
    print("=" * 110)

    print("\nTop 10 LightGBM Feature Importances:")
    for feat, imp in results["part2"]["top_10_features"]:
        print(f"  {feat:30s}: {imp:.1f}")

    print("\nLearned Multimodal ReZero Gates tanh(alpha) per fold:")
    for f_idx, g in enumerate(results["part2"]["rezero_gates"]):
        print(f"  Split {f_idx}: gate = {g:+.4f}")
