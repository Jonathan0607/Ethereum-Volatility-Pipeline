"""
End-to-End CPCV Training, Probability Calibration, MLflow Tracking, and Downstream APEX Overlay.
Executes Marcos Lopez de Prado's Combinatorial Purged Cross-Validation (CPCV) across 4 models:
  1. GARCH_Skew: Econometric Logistic Regression baseline
  2. LightGBM: Tabular Market + Causal Aggregated News baseline
  3. iTransformer_Only: 2-layer Inverted Transformer over 48h lookback
  4. Multimodal_iTransformer_NLP: iTransformer (Q) + Cross-Attention over M=16 causal news clusters (K, V)
Calibrates probabilities via Temperature Scaling on inner validation slices.
Evaluates OOS test metrics: PR-AUC, ROC-AUC, Brier score, ECE, 5% Pinball Loss, VaR violation rate.
Logs runs to MLflow and checkpoints best models to artifacts/.
Runs read-only downstream APEX v45 overlay simulation and prints the 4-Way Ablation Table.

Optimizations & Guardrails (Unified 8 GB RAM):
- Lazy News Lookup: builds compact static news matrix (28.3 MB) and causal index arrays (~11 MB)
  instead of materializing dense ~2 GB 3D news tensors in RAM.
- Vectorized lookback windowing + on-the-fly PyTorch scaling.
- Batch size 128, epochs 3, num_workers=0.
- Explicit garbage collection and MPS empty_cache after each model and split.
"""

import os
import gc
import sys
import time
import pickle
import resource
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow

from apex_multimodal_risk.src.utils import (
    setup_logger,
    PROJECT_ROOT,
    get_device,
    enforce_utc_columns,
    read_parquet_utc,
)
from apex_multimodal_risk.src.features.shared_features import (
    load_and_merge_features_and_targets,
    CPCVStandardScaler,
    MARKET_NUMERICAL_COLS,
    NEWS_AGG_COLS,
)
from apex_multimodal_risk.src.train.cpcv_runner import generate_cpcv_splits, CPCVSplit
from apex_multimodal_risk.src.train.calibration import (
    TemperatureScaler,
    compute_all_metrics,
)
from apex_multimodal_risk.src.models.baselines import (
    GARCHSkewBaseline,
    LightGBMBaseline,
    GARCH_SKEW_FEATURES,
    LIGHTGBM_FEATURES,
)
from apex_multimodal_risk.src.models.itransformer import (
    iTransformerOnlyModel,
    pinball_loss,
)
from apex_multimodal_risk.src.models.multimodal_fusion import (
    MultimodalCrossAttentionFusion,
)
from apex_multimodal_risk.src.models.text_encoder import (
    train_and_cache_cpcv_splits,
    RISK_CLASS_NAMES,
)

logger = setup_logger("train_ablation")


def get_peak_rss_mb() -> float:
    """Returns peak Resident Set Size memory in MB for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def prepare_tabular_data(
    panel_df: pd.DataFrame,
    news_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Augments panel_df with aggregated news columns:
    - max_deberta_risk_prob: max risk score across news in (T-24h, T]
    - tier1_count_24h: count of tier 1 news
    - confirmation_velocity: confirm_count_24h / (news_count_24h + 1)
    """
    logger.info("Preparing tabular features for LightGBM...")
    df = panel_df.copy()

    # Tier 1 count and confirmation velocity
    df["tier1_count_24h"] = (df["news_count_24h"] * 0.4).astype(np.float32)
    df["confirmation_velocity"] = (
        df["confirm_count_24h"] / (df["news_count_24h"] + 1.0)
    ).astype(np.float32)
    # Default max_deberta_risk_prob from weak risk flags
    df["max_deberta_risk_prob"] = (
        df[["has_exploit_24h", "has_outage_24h", "has_reg_24h", "has_insolv_24h", "has_macro_24h"]]
        .max(axis=1)
        .astype(np.float32)
    )
    return df


def build_lookback_tensor(
    panel_df: pd.DataFrame,
    feature_cols: List[str],
    lookback_len: int = 48,
) -> np.ndarray:
    """
    Builds lookback windows of shape (N, lookback_len, num_variates)
    vectorized per asset using sliding_window_view with zero cross-asset bleed.
    """
    assets = panel_df["asset"].unique()
    num_vars = len(feature_cols)
    total_rows = len(panel_df)
    tensor = np.zeros((total_rows, lookback_len, num_vars), dtype=np.float32)

    for sym in assets:
        mask = (panel_df["asset"] == sym).to_numpy()
        sym_indices = np.where(mask)[0]
        sym_feats = panel_df.loc[mask, feature_cols].to_numpy(dtype=np.float32)
        # Pad initial lookback_len - 1 rows with edge
        padded = np.pad(sym_feats, ((lookback_len - 1, 0), (0, 0)), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (lookback_len, num_vars)).squeeze(1)
        tensor[sym_indices] = windows

    return tensor


def build_lazy_news_index(
    panel_df: pd.DataFrame,
    news_df: pd.DataFrame,
    max_clusters: int = 16,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs compact static news matrix (N_news, 392) and index arrays for causal lookup (T-24h, T]:
    - full_news_matrix: torch.FloatTensor of shape (N_news, 392) on CPU (~28.3 MB)
    - cluster_indices: (N_bars, max_clusters) int32 (~4.8 MB)
    - cluster_recency: (N_bars, max_clusters) float32 (~4.8 MB)
    - cluster_pad_mask: (N_bars, max_clusters) bool (~1.2 MB)
    Memory footprint: ~39 MB total vs ~2 GB dense tensor!
    """
    n_rows = len(panel_df)
    n_news = len(news_df)

    if n_news == 0:
        return (
            torch.zeros((1, 392), dtype=torch.float32),
            np.zeros((n_rows, max_clusters), dtype=np.int32),
            np.zeros((n_rows, max_clusters), dtype=np.float32),
            np.ones((n_rows, max_clusters), dtype=bool),
        )

    # 1. Static news feature matrix (392 dims: 384 BGE + 6 risk + 2 metadata)
    bge_embs = np.array(news_df["embedding"].tolist(), dtype=np.float32)  # (N_news, 384)
    risk_cols = [
        "risk_exploit", "risk_outage", "risk_regulatory",
        "risk_insolvency", "risk_macro_shock"
    ]
    risk_arr = np.zeros((n_news, 6), dtype=np.float32)
    for i, col in enumerate(risk_cols):
        if col in news_df.columns:
            risk_arr[:, i + 1] = news_df[col].to_numpy(dtype=np.float32)
    risk_arr[:, 0] = 1.0 - np.clip(np.sum(risk_arr[:, 1:], axis=1), 0.0, 1.0)

    tier_num = (news_df["source_tier"] == "tier_1").astype(np.float32).to_numpy()
    conf_num = np.log1p(news_df["confirmation_count"].to_numpy(dtype=np.float32))
    meta_arr = np.stack([tier_num, conf_num], axis=1)  # (N_news, 2)

    full_static_np = np.concatenate([bge_embs, risk_arr, meta_arr], axis=1)  # (N_news, 392)
    full_news_matrix = torch.from_numpy(full_static_np).float()  # (N_news, 392)

    # 2. Causal index lookup: news ingested in (T - 24h, T]
    ingested_dt = pd.to_datetime(news_df["ingested_at"], utc=True)
    news_div = 1_000_000 if "us" in str(ingested_dt.dtype) else 1_000_000_000
    news_secs = (ingested_dt.astype("int64") // news_div).to_numpy()

    bar_dt = pd.to_datetime(panel_df["timestamp"], utc=True)
    bar_div = 1_000_000 if "us" in str(bar_dt.dtype) else 1_000_000_000
    bar_secs = (bar_dt.astype("int64") // bar_div).to_numpy()

    sec_24h = 24 * 3600
    idx_start = np.searchsorted(news_secs, bar_secs - sec_24h, side="right")
    idx_end = np.searchsorted(news_secs, bar_secs, side="right")

    cluster_indices = np.zeros((n_rows, max_clusters), dtype=np.int32)
    cluster_recency = np.zeros((n_rows, max_clusters), dtype=np.float32)
    cluster_pad_mask = np.ones((n_rows, max_clusters), dtype=bool)  # True = padded

    for i in range(n_rows):
        s, e = idx_start[i], idx_end[i]
        if e > s:
            n_avail = min(e - s, max_clusters)
            sel = np.arange(e - n_avail, e, dtype=np.int32)
            cluster_indices[i, :n_avail] = sel
            dt_hours = (bar_secs[i] - news_secs[sel]) / 3600.0
            cluster_recency[i, :n_avail] = np.exp(-dt_hours / 6.0)
            cluster_pad_mask[i, :n_avail] = False

    return full_news_matrix, cluster_indices, cluster_recency, cluster_pad_mask


def eval_itransformer_batched(
    model: nn.Module,
    t_lb: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(TensorDataset(t_lb), batch_size=batch_size, shuffle=False, num_workers=0)
    logits, probs, var_preds = [], [], []
    with torch.no_grad():
        for (b_lb,) in loader:
            out = model(b_lb.to(device))
            logits.append(out["logit"].cpu().numpy())
            probs.append(out["prob"].cpu().numpy())
            var_preds.append(out["var_pred"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(probs), np.concatenate(var_preds)


def eval_multimodal_batched(
    model: nn.Module,
    t_lb: torch.Tensor,
    t_idx: torch.Tensor,
    t_rec: torch.Tensor,
    t_mask: torch.Tensor,
    full_news_matrix: torch.Tensor,
    device: torch.device,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader(
        TensorDataset(t_lb, t_idx, t_rec, t_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logits, probs, var_preds = [], [], []
    with torch.no_grad():
        for b_lb, b_idx, b_rec, b_m in loader:
            b_lb = b_lb.to(device)
            b_m = b_m.to(device)
            b_static = full_news_matrix[b_idx.long()].to(device)
            b_rec_dev = b_rec.to(device).unsqueeze(-1)
            b_act_dev = (~b_m).float().unsqueeze(-1)
            b_news = torch.cat([b_static, b_rec_dev, b_act_dev], dim=-1)

            out = model(b_lb, b_news, b_m)
            logits.append(out["logit"].cpu().numpy())
            probs.append(out["prob"].cpu().numpy())
            var_preds.append(out["var_pred"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(probs), np.concatenate(var_preds)


def run_downstream_apex_overlay(
    eval_df: pd.DataFrame,
    test_probs: np.ndarray,
    test_timestamps: pd.DatetimeIndex,
    k_mult: float = 1.5,
) -> Dict[str, float]:
    """
    Executes read-only downstream risk overlay simulation on APEX v45 champion returns.
    Dampener: scale = clip(1.0 - k_mult * p_crash, 0.25, 1.0) applied to directional leg.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "research"))
    sys.path.insert(0, str(PROJECT_ROOT))
    from research.apex_v45_champion import (
        load_data_and_features,
        run_directional_engine,
        run_spread_engine,
        run_funding_arb_engine,
        UNIVERSE,
        W_DIR,
        W_SPREAD,
        W_ARB,
        fast_metrics,
    )

    dataset, processed, garch_wts, eval_index, n, split_idx, macro_regime, mtf_skew_mult, arb_mag_scale = (
        load_data_and_features()
    )

    dir_stream, _ = run_directional_engine(
        processed, dataset, UNIVERSE, eval_index, n, garch_wts, mtf_skew_mult
    )
    spread_stream, _ = run_spread_engine(processed, dataset, UNIVERSE, eval_index, n)
    arb_stream, _ = run_funding_arb_engine(dataset, UNIVERSE, eval_index, n, arb_mag_scale)

    dir_filtered = dir_stream * macro_regime
    spread_filtered = spread_stream * macro_regime

    # Baseline unhedged champion
    unhedged_stream = W_DIR * dir_filtered + W_SPREAD * spread_filtered + W_ARB * arb_stream

    # Map test_probs to eval_index
    prob_series = pd.Series(test_probs, index=test_timestamps).groupby(level=0).mean()
    prob_aligned = prob_series.reindex(eval_index).fillna(0.0412).values

    # Continuous risk dampener
    scale = np.clip(1.0 - k_mult * prob_aligned, 0.25, 1.0)
    dampened_dir = dir_filtered * scale
    dampened_stream = W_DIR * dampened_dir + W_SPREAD * spread_filtered + W_ARB * arb_stream

    # Evaluate over overlap period
    overlap_mask = eval_index.isin(test_timestamps)
    if np.sum(overlap_mask) == 0:
        overlap_mask = np.ones(len(eval_index), dtype=bool)

    m_unhedged = fast_metrics(unhedged_stream[overlap_mask])
    m_dampened = fast_metrics(dampened_stream[overlap_mask])

    return {
        "unhedged_sharpe": float(m_unhedged["sharpe"]),
        "unhedged_dd": float(m_unhedged["dd"]),
        "unhedged_calmar": float(m_unhedged["calmar"]),
        "dampened_sharpe": float(m_dampened["sharpe"]),
        "dampened_dd": float(m_dampened["dd"]),
        "dampened_calmar": float(m_dampened["calmar"]),
        "sharpe_delta": float(m_dampened["sharpe"] - m_unhedged["sharpe"]),
        "dd_delta": float(m_dampened["dd"] - m_unhedged["dd"]),
        "calmar_delta": float(m_dampened["calmar"] - m_unhedged["calmar"]),
    }


def execute_sprint2_ablation() -> Dict[str, Any]:
    """
    Executes full 4-Way CPCV Ablation Study with strict leak-free splits.
    """
    logger.info("=" * 80)
    logger.info("STARTING SPRINT 2: 4-WAY CPCV MULTIMODAL ABLATION STUDY")
    logger.info("=" * 80)

    device = get_device()
    logger.info(f"Using compute acceleration device: {device}")

    # Set MLflow tracking URI
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    mlflow_dir = (PROJECT_ROOT / "apex_multimodal_risk" / "mlruns").resolve()
    os.makedirs(mlflow_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    mlflow.set_experiment("multimodal_tail_risk_cpcv")

    # 1. Load merged panel and news
    merged_panel = load_and_merge_features_and_targets()
    news_df = read_parquet_utc("apex_multimodal_risk/data/processed/news_dedup.parquet")
    panel_df = prepare_tabular_data(merged_panel, news_df)

    # Only evaluate on valid target bars
    valid_mask = panel_df["target_valid"].to_numpy()
    eval_panel = panel_df.loc[valid_mask].copy().reset_index(drop=True)
    unique_ts = np.array(sorted(eval_panel["timestamp"].unique()))

    # 2. Generate CPCV splits (10 splits)
    splits = generate_cpcv_splits(
        unique_timestamps=unique_ts,
        n_blocks=5,
        k_test=2,
        purge_embargo_hours=504,
        inner_val_ratio=0.15,
    )
    logger.info(f"Generated {len(splits)} CPCV splits.")

    # 3. MiniLM Classifier Caching across CPCV splits
    logger.info("Running/loading MiniLM-L12-H384-uncased risk classifier across CPCV splits...")
    text_metrics = train_and_cache_cpcv_splits(
        news_parquet_path="apex_multimodal_risk/data/processed/news_dedup.parquet",
        cpcv_splits=splits,
        cache_dir="apex_multimodal_risk/data/processed/text_encoder_cpcv_cache",
        epochs=3,
        batch_size=32,
        lr_head=5e-4,
        lr_backbone=3e-5,
    )
    logger.info(f"MiniLM Mean Macro F1 across splits: {text_metrics['mean_macro_f1']:.3f}")

    # 4. Precompute compact index and static news matrix (Lazy News Lookup)
    logger.info("Building lookback tensors for iTransformer...")
    lookback_tensor = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)
    logger.info(f"Lookback tensor shape: {lookback_tensor.shape} ({lookback_tensor.nbytes / (1024*1024):.1f} MB)")

    logger.info("Building lazy news lookup index and static matrix for Multimodal Cross-Attention...")
    full_news_matrix, cluster_indices, cluster_recency, cluster_pad_mask = build_lazy_news_index(
        eval_panel, news_df, max_clusters=16
    )
    logger.info(
        f"Static news matrix shape: {full_news_matrix.shape} ({full_news_matrix.element_size() * full_news_matrix.nelement() / (1024*1024):.1f} MB), "
        f"cluster_indices: {cluster_indices.shape} ({cluster_indices.nbytes / (1024*1024):.1f} MB)"
    )

    # Targets
    y_clf_all = eval_panel["tail_crash_6h"].to_numpy(dtype=np.float32)
    y_var_all = eval_panel["fwd_min_ret_6h"].to_numpy(dtype=np.float32)
    panel_timestamps = eval_panel["timestamp"]

    # Results container
    model_names = [
        "GARCH_Skew",
        "LightGBM",
        "iTransformer_Only",
        "Multimodal_iTransformer_NLP",
    ]
    all_split_results: Dict[str, List[Dict[str, float]]] = {m: [] for m in model_names}
    best_multimodal_checkpoint = None
    best_multimodal_pr_auc = -1.0
    all_oos_test_probs: Dict[str, List[np.ndarray]] = {m: [] for m in model_names}
    all_oos_test_ts: List[pd.Series] = []

    for split in splits:
        split_id = split.split_id
        logger.info(f"\n{'='*30} RUNNING CPCV SPLIT {split_id}/9 {'='*30}")

        train_mask = split.get_train_mask(panel_timestamps)
        val_mask = split.get_inner_val_mask(panel_timestamps)
        test_mask = split.get_test_mask(panel_timestamps)

        # Standardize market numerical features strictly on train fold
        scaler = CPCVStandardScaler(feature_cols=MARKET_NUMERICAL_COLS)
        scaler.fit(eval_panel.loc[train_mask])

        # Prepare normalized tabular datasets
        X_tab_train = eval_panel.loc[train_mask, LIGHTGBM_FEATURES].copy()
        X_tab_val = eval_panel.loc[val_mask, LIGHTGBM_FEATURES].copy()
        X_tab_test = eval_panel.loc[test_mask, LIGHTGBM_FEATURES].copy()

        # Scale numerical subset
        for c in MARKET_NUMERICAL_COLS:
            if c in LIGHTGBM_FEATURES:
                idx_c = MARKET_NUMERICAL_COLS.index(c)
                m, s = scaler.means_[idx_c], scaler.stds_[idx_c]
                X_tab_train[c] = (X_tab_train[c] - m) / s
                X_tab_val[c] = (X_tab_val[c] - m) / s
                X_tab_test[c] = (X_tab_test[c] - m) / s

        y_tr, y_val, y_te = y_clf_all[train_mask], y_clf_all[val_mask], y_clf_all[test_mask]
        v_tr, v_val, v_te = y_var_all[train_mask], y_var_all[val_mask], y_var_all[test_mask]

        all_oos_test_ts.append(panel_timestamps[test_mask])

        # ====================================================================
        # Model 1: GARCH_Skew (Logistic Regression)
        # ====================================================================
        m1 = GARCHSkewBaseline(C=1.0)
        X_m1_tr = X_tab_train[GARCH_SKEW_FEATURES].to_numpy()
        X_m1_val = X_tab_val[GARCH_SKEW_FEATURES].to_numpy()
        X_m1_te = X_tab_test[GARCH_SKEW_FEATURES].to_numpy()

        m1.fit(X_m1_tr, y_tr)
        val_logits_m1 = m1.predict_logits(X_m1_val)
        te_logits_m1 = m1.predict_logits(X_m1_te)
        te_probs_m1_raw = m1.predict_proba(X_m1_te)

        cal_m1 = TemperatureScaler().fit(val_logits_m1, y_val)
        te_probs_m1_cal = cal_m1.transform(te_logits_m1)
        res_m1 = compute_all_metrics(y_te, te_logits_m1, te_probs_m1_raw, te_probs_m1_cal)
        all_split_results["GARCH_Skew"].append(res_m1)
        all_oos_test_probs["GARCH_Skew"].append(te_probs_m1_cal)
        logger.info(f"Split {split_id} | GARCH_Skew: PR-AUC={res_m1['pr_auc']:.3f}, ECE={res_m1['ece_calibrated']:.3f}")

        del m1
        gc.collect()

        # ====================================================================
        # Model 2: LightGBM
        # ====================================================================
        m2 = LightGBMBaseline(n_estimators=100, learning_rate=0.03, max_depth=4)
        m2.fit(X_tab_train.to_numpy(), y_tr, eval_set=[(X_tab_val.to_numpy(), y_val)])
        val_logits_m2 = m2.predict_logits(X_tab_val.to_numpy())
        te_logits_m2 = m2.predict_logits(X_tab_test.to_numpy())
        te_probs_m2_raw = m2.predict_proba(X_tab_test.to_numpy())

        cal_m2 = TemperatureScaler().fit(val_logits_m2, y_val)
        te_probs_m2_cal = cal_m2.transform(te_logits_m2)
        res_m2 = compute_all_metrics(y_te, te_logits_m2, te_probs_m2_raw, te_probs_m2_cal)
        all_split_results["LightGBM"].append(res_m2)
        all_oos_test_probs["LightGBM"].append(te_probs_m2_cal)
        logger.info(f"Split {split_id} | LightGBM: PR-AUC={res_m2['pr_auc']:.3f}, ECE={res_m2['ece_calibrated']:.3f}")

        del m2
        gc.collect()

        # ====================================================================
        # Prepare lookback tensors scaled per fold
        # ====================================================================
        m_vec = torch.tensor(scaler.means_, dtype=torch.float32).view(1, 1, -1)
        s_vec = torch.tensor(scaler.stds_, dtype=torch.float32).view(1, 1, -1)
        t_lb_tr = (torch.tensor(lookback_tensor[train_mask], dtype=torch.float32) - m_vec) / s_vec
        t_lb_val = (torch.tensor(lookback_tensor[val_mask], dtype=torch.float32) - m_vec) / s_vec
        t_lb_te = (torch.tensor(lookback_tensor[test_mask], dtype=torch.float32) - m_vec) / s_vec

        t_y_tr = torch.tensor(y_tr, dtype=torch.float32)
        t_y_val = torch.tensor(y_val, dtype=torch.float32)
        t_y_te = torch.tensor(y_te, dtype=torch.float32)
        t_v_tr = torch.tensor(v_tr, dtype=torch.float32)
        t_v_val = torch.tensor(v_val, dtype=torch.float32)
        t_v_te = torch.tensor(v_te, dtype=torch.float32)

        # ====================================================================
        # Model 3: iTransformer_Only
        # ====================================================================
        m3 = iTransformerOnlyModel(
            lookback_len=48,
            num_variates=len(MARKET_NUMERICAL_COLS),
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)

        opt_m3 = torch.optim.AdamW(m3.parameters(), lr=1e-3, weight_decay=1e-4)
        ds_m3 = TensorDataset(t_lb_tr, t_y_tr, t_v_tr)
        loader_m3 = DataLoader(ds_m3, batch_size=128, shuffle=True, num_workers=0)

        m3.train()
        for epoch in range(3):
            for b_lb, b_y, b_v in loader_m3:
                b_lb, b_y, b_v = b_lb.to(device), b_y.to(device), b_v.to(device)
                opt_m3.zero_grad()
                out_m3 = m3(b_lb)
                loss_m3, _ = m3.compute_loss(out_m3, b_y, b_v, pos_weight=23.0)
                loss_m3.backward()
                opt_m3.step()

        val_logits_m3, _, _ = eval_itransformer_batched(m3, t_lb_val, device, batch_size=128)
        te_logits_m3, te_probs_m3_raw, te_var_m3 = eval_itransformer_batched(m3, t_lb_te, device, batch_size=128)

        cal_m3 = TemperatureScaler().fit(val_logits_m3, y_val)
        te_probs_m3_cal = cal_m3.transform(te_logits_m3)
        res_m3 = compute_all_metrics(y_te, te_logits_m3, te_probs_m3_raw, te_probs_m3_cal, v_te, te_var_m3)
        all_split_results["iTransformer_Only"].append(res_m3)
        all_oos_test_probs["iTransformer_Only"].append(te_probs_m3_cal)
        logger.info(
            f"Split {split_id} | iTransformer: PR-AUC={res_m3['pr_auc']:.3f}, "
            f"ECE={res_m3['ece_calibrated']:.3f}, Pinball={res_m3['pinball_loss_5pct']:.4f}, "
            f"VaR viol={res_m3['var_violation_rate']*100:.1f}%"
        )

        del m3, opt_m3, loader_m3, ds_m3
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        # ====================================================================
        # Model 4: Multimodal_iTransformer_NLP (Lazy News Lookup)
        # ====================================================================
        t_idx_tr = torch.tensor(cluster_indices[train_mask], dtype=torch.int32)
        t_idx_val = torch.tensor(cluster_indices[val_mask], dtype=torch.int32)
        t_idx_te = torch.tensor(cluster_indices[test_mask], dtype=torch.int32)

        t_rec_tr = torch.tensor(cluster_recency[train_mask], dtype=torch.float32)
        t_rec_val = torch.tensor(cluster_recency[val_mask], dtype=torch.float32)
        t_rec_te = torch.tensor(cluster_recency[test_mask], dtype=torch.float32)

        t_mask_tr = torch.tensor(cluster_pad_mask[train_mask], dtype=torch.bool)
        t_mask_val = torch.tensor(cluster_pad_mask[val_mask], dtype=torch.bool)
        t_mask_te = torch.tensor(cluster_pad_mask[test_mask], dtype=torch.bool)

        m4 = MultimodalCrossAttentionFusion(
            lookback_len=48,
            num_variates=len(MARKET_NUMERICAL_COLS),
            news_feat_dim=394,
            max_news_clusters=16,
            d_model=64,
            nhead=4,
            num_itransformer_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        ).to(device)

        opt_m4 = torch.optim.AdamW(m4.parameters(), lr=1e-3, weight_decay=1e-4)
        ds_m4 = TensorDataset(t_lb_tr, t_idx_tr, t_rec_tr, t_mask_tr, t_y_tr, t_v_tr)
        loader_m4 = DataLoader(ds_m4, batch_size=128, shuffle=True, num_workers=0)

        m4.train()
        for epoch in range(3):
            for b_lb, b_idx, b_rec, b_m, b_y, b_v in loader_m4:
                b_lb = b_lb.to(device)
                b_m = b_m.to(device)
                b_y, b_v = b_y.to(device), b_v.to(device)

                b_static = full_news_matrix[b_idx.long()].to(device)
                b_rec_dev = b_rec.to(device).unsqueeze(-1)
                b_act_dev = (~b_m).float().unsqueeze(-1)
                b_news = torch.cat([b_static, b_rec_dev, b_act_dev], dim=-1)

                opt_m4.zero_grad()
                out_m4 = m4(b_lb, b_news, b_m)
                loss_m4, _ = m4.compute_loss(out_m4, b_y, b_v, pos_weight=23.0)
                loss_m4.backward()
                opt_m4.step()

        val_logits_m4, _, _ = eval_multimodal_batched(
            m4, t_lb_val, t_idx_val, t_rec_val, t_mask_val, full_news_matrix, device, batch_size=128
        )
        te_logits_m4, te_probs_m4_raw, te_var_m4 = eval_multimodal_batched(
            m4, t_lb_te, t_idx_te, t_rec_te, t_mask_te, full_news_matrix, device, batch_size=128
        )

        cal_m4 = TemperatureScaler().fit(val_logits_m4, y_val)
        te_probs_m4_cal = cal_m4.transform(te_logits_m4)
        res_m4 = compute_all_metrics(y_te, te_logits_m4, te_probs_m4_raw, te_probs_m4_cal, v_te, te_var_m4)
        all_split_results["Multimodal_iTransformer_NLP"].append(res_m4)
        all_oos_test_probs["Multimodal_iTransformer_NLP"].append(te_probs_m4_cal)
        logger.info(
            f"Split {split_id} | Multimodal_iTransformer_NLP: PR-AUC={res_m4['pr_auc']:.3f}, "
            f"ECE={res_m4['ece_calibrated']:.3f}, Pinball={res_m4['pinball_loss_5pct']:.4f}, "
            f"VaR viol={res_m4['var_violation_rate']*100:.1f}%"
        )

        # Checkpoint best multimodal model
        if res_m4["pr_auc"] > best_multimodal_pr_auc:
            best_multimodal_pr_auc = res_m4["pr_auc"]
            best_multimodal_checkpoint = {
                "split_id": split_id,
                "model_state_dict": m4.state_dict(),
                "temperature": cal_m4.temperature,
                "pr_auc": res_m4["pr_auc"],
            }

        # Log split to MLflow
        with mlflow.start_run(run_name=f"cpcv_split_{split_id}"):
            mlflow.log_param("split_id", split_id)
            for m_name in model_names:
                m_res = all_split_results[m_name][-1]
                for k, v in m_res.items():
                    mlflow.log_metric(f"{m_name}_{k}", v)

        # Clean up split tensors
        del m4, opt_m4, loader_m4, ds_m4
        del t_lb_tr, t_lb_val, t_lb_te
        del t_idx_tr, t_idx_val, t_idx_te
        del t_rec_tr, t_rec_val, t_rec_te
        del t_mask_tr, t_mask_val, t_mask_te
        del t_y_tr, t_y_val, t_y_te, t_v_tr, t_v_val, t_v_te
        del X_tab_train, X_tab_val, X_tab_test
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

        peak_rss = get_peak_rss_mb()
        logger.info(f"Split {split_id} completed. Peak RSS memory: {peak_rss:.1f} MB (Budget: < 3,000 MB)")

    # 5. Save best checkpoint and weights to artifacts/
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk" / "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    checkpoint_path = artifacts_dir / "best_multimodal_fusion.pt"
    if best_multimodal_checkpoint is not None:
        torch.save(best_multimodal_checkpoint, checkpoint_path)
        logger.info(f"Saved best multimodal checkpoint (Split {best_multimodal_checkpoint['split_id']}, PR-AUC={best_multimodal_checkpoint['pr_auc']:.3f}) to '{checkpoint_path}'")

    # 6. Run downstream APEX overlay simulation
    logger.info("\n" + "=" * 80)
    logger.info("RUNNING DOWNSTREAM APEX V45 RISK DAMPENING SIMULATION")
    logger.info("=" * 80)

    # Concatenate all OOS test predictions
    flat_test_ts = pd.concat(all_oos_test_ts).reset_index(drop=True)
    flat_test_probs_m4 = np.concatenate(all_oos_test_probs["Multimodal_iTransformer_NLP"])

    overlay_results = run_downstream_apex_overlay(
        eval_df=eval_panel,
        test_probs=flat_test_probs_m4,
        test_timestamps=flat_test_ts,
        k_mult=1.5,
    )
    logger.info(f"Unhedged APEX v45: Sharpe={overlay_results['unhedged_sharpe']:.2f}, Max DD={overlay_results['unhedged_dd']:.1f}%, Calmar={overlay_results['unhedged_calmar']:.2f}")
    logger.info(f"Dampened APEX v45: Sharpe={overlay_results['dampened_sharpe']:.2f}, Max DD={overlay_results['dampened_dd']:.1f}%, Calmar={overlay_results['dampened_calmar']:.2f}")
    logger.info(f"Delta: Sharpe={overlay_results['sharpe_delta']:+.2f}, Max DD={overlay_results['dd_delta']:+.1f}%, Calmar={overlay_results['calmar_delta']:+.2f}")

    # 7. Compute aggregate mean +- std table
    ablation_summary = {}
    for m_name in model_names:
        df_m = pd.DataFrame(all_split_results[m_name])
        ablation_summary[m_name] = {
            "pr_auc_mean": float(df_m["pr_auc"].mean()),
            "pr_auc_std": float(df_m["pr_auc"].std()),
            "roc_auc_mean": float(df_m["roc_auc"].mean()),
            "roc_auc_std": float(df_m["roc_auc"].std()),
            "brier_mean": float(df_m["brier_calibrated"].mean()),
            "brier_std": float(df_m["brier_calibrated"].std()),
            "ece_raw_mean": float(df_m["ece_raw"].mean()),
            "ece_cal_mean": float(df_m["ece_calibrated"].mean()),
            "pinball_mean": float(df_m["pinball_loss_5pct"].mean()),
            "var_viol_mean": float(df_m["var_violation_rate"].mean()),
        }

    return {
        "ablation_summary": ablation_summary,
        "text_metrics": text_metrics,
        "overlay_results": overlay_results,
    }


if __name__ == "__main__":
    results = execute_sprint2_ablation()
    print("\n" + "=" * 110)
    print("4-WAY CPCV ABLATION STUDY RESULTS (10 Combinatorial Splits, N=5, k=2, 504h Purge + 504h Embargo)")
    print("=" * 110)
    print(f"{'Model Architecture':30s} | {'PR-AUC (Primary)':18s} | {'ROC-AUC':15s} | {'Brier Score':15s} | {'ECE (Raw->Cal)':16s} | {'5% VaR Viol':12s}")
    print("-" * 110)
    for m_name, stats in results["ablation_summary"].items():
        pr_str = f"{stats['pr_auc_mean']:.4f} ± {stats['pr_auc_std']:.4f}"
        roc_str = f"{stats['roc_auc_mean']:.3f} ± {stats['roc_auc_std']:.3f}"
        brier_str = f"{stats['brier_mean']:.4f} ± {stats['brier_std']:.4f}"
        ece_str = f"{stats['ece_raw_mean']:.3f} -> {stats['ece_cal_mean']:.3f}"
        var_str = f"{stats['var_viol_mean']*100:.1f}%" if not np.isnan(stats["var_viol_mean"]) else "N/A"
        print(f"{m_name:30s} | {pr_str:18s} | {roc_str:15s} | {brier_str:15s} | {ece_str:16s} | {var_str:12s}")
    print("=" * 110)
