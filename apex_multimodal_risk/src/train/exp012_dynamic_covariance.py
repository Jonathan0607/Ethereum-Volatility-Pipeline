#!/usr/bin/env python3
"""
===================================================================================
EXP-012: Dynamic Covariance & Latent-Factor Beta Forecasting
===================================================================================
Benchmarks 4 strictly causal risk models across 10 expanding quarterly walk-forward
folds (168h embargo) over 22,440 out-of-sample hours (2022-11-01 to 2025-05-23):
  - R0 (Current Baseline): Trailing 720h (30d) Rolling OLS BTC Beta + Sample Covariance
  - R1 (Adaptive Statistical Shrinkage): Multi-horizon EWMA Beta (48h/168h/720h) with
       Vasicek Bayesian Shrinkage + Ledoit-Wolf Shrinkage Covariance
  - R2 (LightGBM Dynamic Beta Forecaster): Regularized LightGBM Huber model predicting
       forward 48h/168h BTC beta from causal microstructure/dispersion/absorption features
  - R3 (Compact PyTorch Cross-Asset Graph Attention Risk Network — GraphBetaCovNet):
       < 10,000 parameters, 2-layer Graph message passing over correlation adjacency,
       predicting beta residuals and guaranteed SPD 2-factor covariance matrix via QLIKE loss

Evaluates:
  1. Statistical Risk Gate: Beta RMSE/MAE/R^2, Crisis-Window RMSE, Covariance QLIKE/Frobenius,
     and Global Minimum Variance (GMV) Portfolio Realized OOS Volatility.
  2. Downstream Alpha Impact on EXP-008d 48h Total-Return LightGBM factor (7-tranche overlapping).
  3. Honest Vol-Matched APEX v45 Blends (90% APEX / 10% ML) at 0, 10, 25 bps slippage.
===================================================================================
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

# Dynamic sys.path insertion for local imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "research") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "research"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.covariance import LedoitWolf

import research.apex_v45_champion as v45
from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
    HOURS_PER_YEAR,
    QUARTERLY_TEST_STARTS,
    TEST_END_FINAL,
    apply_honest_vol_targeting,
    compute_metrics,
    run_walkforward_models_all,
)
from apex_multimodal_risk.src.train.exp008d_path_a_audit import (
    engineer_features_and_targets as load_base_panel,
)
from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("exp012_dynamic_covariance")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Set deterministic seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# =================================================================================
# 1. Feature Engineering for Risk Forecasting
# =================================================================================

RISK_FEATURE_COLS = [
    "beta_24h",
    "beta_72h",
    "beta_168h",
    "beta_720h",
    "vol_ratio_btc",
    "corr_btc_72h",
    "corr_btc_168h",
    "vol_expansion_24_168",
    "fr_zscore_30d",
    "skew_168h",
    "downside_ratio_72h",
    "cross_dispersion",
    "pc1_absorption",
]


def build_risk_features_and_targets(
    df_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes causal risk features, forward realized betas, and forward realized covariance.
    Operates on the 15-coin panel.
    """
    logger.info("Building strictly causal risk features and forward realized targets...")
    timestamps = sorted(df_panel["timestamp"].unique())
    assets = sorted(df_panel["asset"].unique())
    n_ts = len(timestamps)
    n_assets = len(assets)
    btc_idx = assets.index("BTCUSDT")

    # Pivot return matrix: shape (n_ts, n_assets)
    ret_df = df_panel.pivot(index="timestamp", columns="asset", values="ret_1h").reindex(
        index=timestamps, columns=assets
    ).fillna(0.0)
    ret_mat = ret_df.values

    # 1. Multi-horizon rolling BTC covariances and variances
    beta_mats: dict[int, np.ndarray] = {}
    corr_mats: dict[int, np.ndarray] = {}
    for w in [24, 72, 168, 720]:
        cov_btc = ret_df.rolling(w, min_periods=max(12, w // 4)).cov(ret_df["BTCUSDT"])
        var_btc = ret_df["BTCUSDT"].rolling(w, min_periods=max(12, w // 4)).var()
        b_mat = (cov_btc.div(var_btc + 1e-8, axis=0)).clip(0.2, 3.0).bfill().values
        b_mat[:, btc_idx] = 1.0
        beta_mats[w] = b_mat

        if w in [72, 168]:
            c_mat = ret_df.rolling(w, min_periods=max(12, w // 4)).corr(ret_df["BTCUSDT"]).clip(-1.0, 1.0).bfill().values
            corr_mats[w] = c_mat

    # Volatility of each asset vs BTC
    vol_168_df = ret_df.rolling(168, min_periods=24).std().bfill()
    vol_btc_168 = vol_168_df["BTCUSDT"].values[:, None]
    vol_ratio_btc_mat = np.clip(vol_168_df.values / (vol_btc_168 + 1e-6), 0.2, 5.0)

    vol_24_df = ret_df.rolling(24, min_periods=12).std().bfill()
    vol_expansion_mat = np.clip(vol_24_df.values / (vol_168_df.values + 1e-6), 0.2, 5.0)

    # Cross-sectional dispersion of 1h returns
    cross_disp_series = ret_df.std(axis=1).values[:, None]
    cross_disp_mat = np.repeat(cross_disp_series, n_assets, axis=1)

    # PC1 Absorption Ratio from trailing 168h correlation matrix
    # Compute causal rolling PCA every 24h and forward-fill for speed
    logger.info("Computing causal rolling PC1 absorption ratio...")
    pc1_absorption_arr = np.full(n_ts, 0.50)
    step = 24
    for t_i in range(168, n_ts, step):
        sub_r = ret_mat[t_i - 168 : t_i]
        std_r = np.std(sub_r, axis=0, ddof=1) + 1e-8
        stdized = (sub_r - np.mean(sub_r, axis=0)) / std_r
        corr_168 = (stdized.T @ stdized) / (len(sub_r) - 1)
        eigvals = np.linalg.eigvalsh(corr_168)
        eigvals = np.sort(eigvals)[::-1]
        abs_ratio = float(eigvals[0] / np.sum(eigvals)) if np.sum(eigvals) > 0 else 0.50
        pc1_absorption_arr[t_i : min(t_i + step, n_ts)] = abs_ratio
    pc1_absorption_mat = np.repeat(pc1_absorption_arr[:, None], n_assets, axis=1)

    # 2. Forward Realized Targets
    logger.info("Computing forward realized 48h and 168h BTC betas and 168h covariance...")
    fwd_beta_48 = np.full((n_ts, n_assets), np.nan)
    fwd_beta_168 = np.full((n_ts, n_assets), np.nan)
    fwd_cov_168 = np.full((n_ts, n_assets, n_assets), np.nan)

    for t_i in range(n_ts):
        if t_i + 48 <= n_ts:
            r_fwd_48 = ret_mat[t_i : t_i + 48]
            btc_fwd_48 = r_fwd_48[:, btc_idx]
            var_btc_48 = np.var(btc_fwd_48) + 1e-8
            cov_48 = np.mean(
                (r_fwd_48 - np.mean(r_fwd_48, axis=0)) * (btc_fwd_48[:, None] - np.mean(btc_fwd_48)),
                axis=0,
            )
            b48 = np.clip(cov_48 / var_btc_48, 0.2, 3.0)
            b48[btc_idx] = 1.0
            fwd_beta_48[t_i] = b48

        if t_i + 168 <= n_ts:
            r_fwd_168 = ret_mat[t_i : t_i + 168]
            btc_fwd_168 = r_fwd_168[:, btc_idx]
            var_btc_168 = np.var(btc_fwd_168) + 1e-8
            cov_168 = np.mean(
                (r_fwd_168 - np.mean(r_fwd_168, axis=0)) * (btc_fwd_168[:, None] - np.mean(btc_fwd_168)),
                axis=0,
            )
            b168 = np.clip(cov_168 / var_btc_168, 0.2, 3.0)
            b168[btc_idx] = 1.0
            fwd_beta_168[t_i] = b168

            # Full 15x15 covariance
            c_mat_168 = np.cov(r_fwd_168, rowvar=False)
            fwd_cov_168[t_i] = c_mat_168

    # Assign columns back to df_panel
    ts_map = {ts: i for i, ts in enumerate(timestamps)}
    asset_map = {a: i for i, a in enumerate(assets)}

    t_indices = [ts_map[t] for t in df_panel["timestamp"]]
    a_indices = [asset_map[a] for a in df_panel["asset"]]

    df_panel["beta_24h"] = beta_mats[24][t_indices, a_indices]
    df_panel["beta_72h"] = beta_mats[72][t_indices, a_indices]
    df_panel["beta_168h"] = beta_mats[168][t_indices, a_indices]
    df_panel["beta_720h"] = beta_mats[720][t_indices, a_indices]
    df_panel["vol_ratio_btc"] = vol_ratio_btc_mat[t_indices, a_indices]
    df_panel["corr_btc_72h"] = corr_mats[72][t_indices, a_indices]
    df_panel["corr_btc_168h"] = corr_mats[168][t_indices, a_indices]
    df_panel["vol_expansion_24_168"] = vol_expansion_mat[t_indices, a_indices]
    df_panel["cross_dispersion"] = cross_disp_mat[t_indices, a_indices]
    df_panel["pc1_absorption"] = pc1_absorption_mat[t_indices, a_indices]

    df_panel["target_beta_48h"] = fwd_beta_48[t_indices, a_indices]
    df_panel["target_beta_168h"] = fwd_beta_168[t_indices, a_indices]

    ts_index = pd.DatetimeIndex(timestamps)
    return df_panel, assets, ts_index, ret_mat, fwd_beta_48, fwd_beta_168, fwd_cov_168


# =================================================================================
# 2. PyTorch Graph Attention Risk Network (GraphBetaCovNet)
# =================================================================================

class GraphBetaCovNet(nn.Module):
    """
    Compact Cross-Asset Graph Attention Risk Network (< 10,000 parameters).
    Ingests 15-node feature matrix X_t (15 x F) and dynamic correlation adjacency A_t (15 x 15).
    Produces:
      1. Forward BTC Beta estimate (beta_hat = beta_168h + delta_beta)
      2. Factor loadings B_t (15 x 2) and idiosyncratic variance d_t^2 (15) for guaranteed SPD covariance.
    """

    def __init__(self, in_features: int = 13, d_model: int = 16, n_factors: int = 2):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model
        self.n_factors = n_factors

        # Input projection
        self.proj_in = nn.Linear(in_features, d_model)
        self.ln1 = nn.LayerNorm(d_model)

        # Layer 1 Graph Message Passing
        self.w_msg1 = nn.Linear(d_model, d_model, bias=False)
        self.w_self1 = nn.Linear(d_model, d_model, bias=True)
        self.ln2 = nn.LayerNorm(d_model)

        # Layer 2 Graph Message Passing with Residual
        self.w_msg2 = nn.Linear(d_model, d_model, bias=False)
        self.w_self2 = nn.Linear(d_model, d_model, bias=True)
        self.ln3 = nn.LayerNorm(d_model)

        # Beta prediction head (predicts residual delta around trailing 168h beta)
        self.beta_head = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        # Factor covariance head: B_t (15 x 2) + d_t (15)
        self.factor_head = nn.Linear(d_model, n_factors)
        self.idio_head = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        base_beta_168: torch.Tensor,
        btc_idx: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (batch, 15, in_features)
        adj: (batch, 15, 15) row-normalized adjacency
        base_beta_168: (batch, 15) trailing 168h beta
        Returns:
          beta_hat: (batch, 15)
          B: (batch, 15, 2)
          d: (batch, 15) positive idiosyncratic std
        """
        batch_size, n_assets, _ = x.shape

        h0 = self.ln1(self.proj_in(x))  # (B, 15, d_model)

        # Layer 1
        m1 = torch.bmm(adj, self.w_msg1(h0))  # (B, 15, d_model)
        s1 = self.w_self1(h0)
        h1 = self.ln2(F.gelu(m1 + s1) + h0)

        # Layer 2
        m2 = torch.bmm(adj, self.w_msg2(h1))
        s2 = self.w_self2(h1)
        h2 = self.ln3(F.gelu(m2 + s2) + h1)

        # Beta head
        delta_beta = self.beta_head(h2).squeeze(-1)  # (B, 15)
        beta_hat = base_beta_168 + delta_beta
        # BTC is strictly 1.0
        beta_hat = torch.clamp(beta_hat, 0.2, 3.0)
        beta_hat[:, btc_idx] = 1.0

        # Factor covariance head
        B = self.factor_head(h2)  # (B, 15, 2)
        d = F.softplus(self.idio_head(h2).squeeze(-1)) + 1e-4  # (B, 15)

        return beta_hat, B, d


def compute_woodbury_cov_and_loss(
    B: torch.Tensor,
    d: torch.Tensor,
    Sigma_true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Woodbury-inverted SPD covariance matrix and QLIKE loss:
      Sigma_hat = B @ B.T + diag(d^2)
      L_cov = (1/N) * [ tr(Sigma_hat^-1 @ Sigma_true) + logdet(Sigma_hat) ]
    """
    batch_size, n_assets, n_factors = B.shape
    d_sq = d**2  # (batch, 15)
    d_inv = 1.0 / (d_sq + 1e-8)  # (batch, 15)

    # D_inv @ B: (batch, 15, 2)
    D_inv_B = d_inv.unsqueeze(-1) * B

    # M = I_2 + B.T @ D_inv @ B: (batch, 2, 2)
    I_k = torch.eye(n_factors, device=B.device).unsqueeze(0).expand(batch_size, -1, -1)
    M = I_k + torch.bmm(B.transpose(1, 2), D_inv_B)

    # Invert 2x2 matrix analytically for maximum speed & stability
    det_M = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
    det_M = torch.clamp(det_M, min=1e-8)
    M_inv = torch.empty_like(M)
    M_inv[:, 0, 0] = M[:, 1, 1] / det_M
    M_inv[:, 1, 1] = M[:, 0, 0] / det_M
    M_inv[:, 0, 1] = -M[:, 0, 1] / det_M
    M_inv[:, 1, 0] = -M[:, 1, 0] / det_M

    # logdet(Sigma_hat) = sum(log(d^2)) + logdet(M)
    logdet_Sigma = torch.sum(torch.log(d_sq + 1e-8), dim=1) + torch.log(det_M)

    # tr(Sigma_hat^-1 @ Sigma_true):
    # Sigma_hat^-1 = D_inv - D_inv_B @ M_inv @ D_inv_B.T
    # tr(D_inv @ Sigma_true) = sum_i d_inv_i * Sigma_true_ii
    tr_d_inv_sigma = torch.sum(d_inv * torch.diagonal(Sigma_true, dim1=1, dim2=2), dim=1)

    # Second term: tr(M_inv @ D_inv_B.T @ Sigma_true @ D_inv_B)
    temp = torch.bmm(Sigma_true, D_inv_B)  # (B, 15, 2)
    temp2 = torch.bmm(D_inv_B.transpose(1, 2), temp)  # (B, 2, 2)
    tr_lowrank = torch.diagonal(torch.bmm(M_inv, temp2), dim1=1, dim2=2).sum(dim=1)

    tr_inv_sigma = tr_d_inv_sigma - tr_lowrank

    # Reconstruct full Sigma_hat for downstream return
    Sigma_hat = torch.bmm(B, B.transpose(1, 2)) + torch.diag_embed(d_sq)

    qlike_loss = torch.mean((tr_inv_sigma + logdet_Sigma) / n_assets)
    return Sigma_hat, qlike_loss


# =================================================================================
# 3. Model Training & OOS Walk-Forward Execution (R0, R1, R2, R3)
# =================================================================================

def run_risk_walkforward(
    df_panel: pd.DataFrame,
    assets: list[str],
    ts_index: pd.DatetimeIndex,
    ret_mat: np.ndarray,
    fwd_beta_48: np.ndarray,
    fwd_beta_168: np.ndarray,
    fwd_cov_168: np.ndarray,
) -> dict[str, Any]:
    """
    Executes the 10-quarter expanding walk-forward risk models:
      R0: 30d Rolling OLS Beta + Sample Covariance
      R1: Adaptive EWMA Beta (Vasicek Shrinkage) + Ledoit-Wolf Covariance
      R2: LightGBM Dynamic Beta Forecaster
      R3: GraphBetaCovNet (PyTorch GNN)
    """
    logger.info("Executing 10-Quarter Walk-Forward Risk Model Benchmarking (R0, R1, R2, R3)...")
    torch.set_num_threads(1)
    n_ts, n_assets = ret_mat.shape
    btc_idx = assets.index("BTCUSDT")

    # Arrays to store OOS predictions across all n_ts
    beta_hat_r0 = np.full((n_ts, n_assets), np.nan)
    cov_hat_r0 = np.full((n_ts, n_assets, n_assets), np.nan)

    beta_hat_r1 = np.full((n_ts, n_assets), np.nan)
    cov_hat_r1 = np.full((n_ts, n_assets, n_assets), np.nan)

    beta_hat_r2 = np.full((n_ts, n_assets), np.nan)

    beta_hat_r3 = np.full((n_ts, n_assets), np.nan)
    factors_r3 = np.full((n_ts, n_assets, 2), np.nan)
    cov_hat_r3 = np.full((n_ts, n_assets, n_assets), np.nan)

    # Also a feature-shuffle placebo for GNN
    beta_hat_placebo = np.full((n_ts, n_assets), np.nan)

    # Pre-extract matrices for speed
    feat_panel_map: dict[str, np.ndarray] = {}
    for col in RISK_FEATURE_COLS:
        feat_panel_map[col] = df_panel.pivot(index="timestamp", columns="asset", values=col).reindex(
            index=ts_index, columns=assets
        ).fillna(0.0).values

    # Precompute R0 and R1 directly across time
    logger.info("Computing R0 (30d OLS) and R1 (Vasicek EWMA + Ledoit-Wolf) signals...")

    # R0: 720h rolling OLS beta
    beta_hat_r0 = feat_panel_map["beta_720h"].copy()
    beta_hat_r0[:, btc_idx] = 1.0

    # R1: Multi-horizon EWMA beta + Vasicek shrinkage
    b_24 = feat_panel_map["beta_24h"]
    b_72 = feat_panel_map["beta_72h"]
    b_168 = feat_panel_map["beta_168h"]
    b_720 = feat_panel_map["beta_720h"]
    b_ewma_raw = 0.15 * b_24 + 0.25 * b_72 + 0.35 * b_168 + 0.25 * b_720
    # Vasicek shrinkage toward cross-sectional mean (clipped to 1.0)
    cs_var = np.var(b_ewma_raw, axis=1, keepdims=True) + 1e-4
    # Estimation variance proxy (higher for short horizons)
    est_var = 0.05
    w_vasicek = cs_var / (cs_var + est_var)
    beta_hat_r1 = np.clip(w_vasicek * b_ewma_raw + (1.0 - w_vasicek) * 1.0, 0.2, 3.0)
    beta_hat_r1[:, btc_idx] = 1.0

    # Execute quarterly expanding folds for LightGBM (R2) and GraphBetaCovNet (R3)
    oos_start_ts = pd.Timestamp(QUARTERLY_TEST_STARTS[0])
    final_end_ts = pd.Timestamp(TEST_END_FINAL)

    for q_idx in range(len(QUARTERLY_TEST_STARTS)):
        t_test_start = pd.Timestamp(QUARTERLY_TEST_STARTS[q_idx])
        t_test_end = (
            pd.Timestamp(QUARTERLY_TEST_STARTS[q_idx + 1])
            if q_idx + 1 < len(QUARTERLY_TEST_STARTS)
            else final_end_ts
        )
        t_train_end = t_test_start - timedelta(hours=168)  # 168h embargo

        test_idx = np.where(
            (ts_index >= t_test_start)
            & ((ts_index < t_test_end) if q_idx + 1 < len(QUARTERLY_TEST_STARTS) else (ts_index <= t_test_end))
        )[0]
        train_idx = np.where(ts_index + timedelta(hours=168) <= t_train_end)[0]

        logger.info(
            f"Fold {q_idx + 1}/{len(QUARTERLY_TEST_STARTS)}: Train [{ts_index[train_idx[0]].strftime('%Y-%m-%d')} -> {ts_index[train_idx[-1]].strftime('%Y-%m-%d')}] ({len(train_idx)}h), "
            f"Test [{t_test_start.strftime('%Y-%m-%d')} -> {t_test_end.strftime('%Y-%m-%d')}] ({len(test_idx)}h)"
        )

        # -------------------------------------------------------------
        # R2: Train LightGBM Beta Forecaster
        # -------------------------------------------------------------
        # Sample training data daily at 00:00 UTC to maintain efficiency
        daily_train_idx = [i for i in train_idx if ts_index[i].hour == 0]

        # Flatten features and targets across assets
        X_tr_list, y_tr_list = [], []
        for ti in daily_train_idx:
            if not np.isnan(fwd_beta_48[ti, 0]):
                for ai in range(n_assets):
                    feat_vec = [feat_panel_map[c][ti, ai] for c in RISK_FEATURE_COLS]
                    X_tr_list.append(feat_vec)
                    y_tr_list.append(fwd_beta_48[ti, ai])

        X_tr = np.array(X_tr_list)
        y_tr = np.array(y_tr_list)

        lgb_beta_model = lgb.LGBMRegressor(
            objective="huber",
            n_estimators=75,
            learning_rate=0.04,
            num_leaves=12,
            max_depth=4,
            min_child_samples=50,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42,
            verbose=-1,
        )
        lgb_beta_model.fit(X_tr, y_tr)

        # Vectorized R2 test prediction across all test hours
        X_te_all = np.array(
            [[feat_panel_map[c][ti, ai] for c in RISK_FEATURE_COLS] for ti in test_idx for ai in range(n_assets)]
        )
        preds_all = lgb_beta_model.predict(X_te_all).reshape(len(test_idx), n_assets)
        preds_all = np.clip(preds_all, 0.2, 3.0)
        preds_all[:, btc_idx] = 1.0
        beta_hat_r2[test_idx] = preds_all

        # -------------------------------------------------------------
        # R3: Train Compact PyTorch GraphBetaCovNet
        # -------------------------------------------------------------
        # Construct graph tensors for daily timestamps
        device = torch.device("cpu")
        gnn_model = GraphBetaCovNet(in_features=len(RISK_FEATURE_COLS), d_model=16, n_factors=2).to(device)
        optimizer = torch.optim.AdamW(gnn_model.parameters(), lr=0.003, weight_decay=0.01)

        # Prepare dataset of (X_t, A_t, base_beta_t, y_beta_t, y_cov_t)
        def build_graph_samples(indices: list[int]):
            X_list, A_list, base_b_list, y_b_list, y_cov_list = [], [], [], [], []
            for ti in indices:
                if ti >= 168 and not np.isnan(fwd_beta_168[ti, 0]) and not np.isnan(fwd_cov_168[ti, 0, 0]):
                    # Node features: (15, F)
                    x_t = np.array([[feat_panel_map[c][ti, ai] for c in RISK_FEATURE_COLS] for ai in range(n_assets)], dtype=np.float32)
                    # Adjacency: trailing 168h correlation
                    sub_r = ret_mat[ti - 168 : ti]
                    corr_168 = np.corrcoef(sub_r, rowvar=False)
                    corr_168 = np.nan_to_num(corr_168, nan=0.0)
                    adj_t = np.abs(corr_168)
                    np.fill_diagonal(adj_t, 0.0)
                    row_sums = adj_t.sum(axis=1, keepdims=True) + 1e-6
                    adj_t = adj_t / row_sums

                    X_list.append(x_t)
                    A_list.append(adj_t.astype(np.float32))
                    base_b_list.append(feat_panel_map["beta_168h"][ti].astype(np.float32))
                    y_b_list.append(fwd_beta_168[ti].astype(np.float32))
                    y_cov_list.append(fwd_cov_168[ti].astype(np.float32))
            return (
                torch.tensor(np.array(X_list)),
                torch.tensor(np.array(A_list)),
                torch.tensor(np.array(base_b_list)),
                torch.tensor(np.array(y_b_list)),
                torch.tensor(np.array(y_cov_list)),
            )

        X_train_g, A_train_g, b_train_g, yb_train_g, ycov_train_g = build_graph_samples(daily_train_idx)
        n_samples = len(X_train_g)
        # 80/20 train/val split for early stopping
        n_val = max(5, int(n_samples * 0.20))
        n_tr = n_samples - n_val

        X_tr_t = X_train_g[:n_tr].to(device)
        A_tr_t = A_train_g[:n_tr].to(device)
        b_tr_t = b_train_g[:n_tr].to(device)
        yb_tr_t = yb_train_g[:n_tr].to(device)
        ycov_tr_t = ycov_train_g[:n_tr].to(device)

        X_va_t = X_train_g[n_tr:].to(device)
        A_va_t = A_train_g[n_tr:].to(device)
        b_va_t = b_train_g[n_tr:].to(device)
        yb_va_t = yb_train_g[n_tr:].to(device)
        ycov_va_t = ycov_train_g[n_tr:].to(device)

        # Train with early stopping (25 epochs max)
        best_val_loss = float("inf")
        best_state = None
        batch_size = 32

        for epoch in range(25):
            gnn_model.train()
            perm = torch.randperm(n_tr)
            for i in range(0, n_tr, batch_size):
                b_idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                b_hat, B_hat, d_hat = gnn_model(X_tr_t[b_idx], A_tr_t[b_idx], b_tr_t[b_idx], btc_idx=btc_idx)
                loss_beta = F.smooth_l1_loss(b_hat, yb_tr_t[b_idx], beta=0.2)
                _, loss_cov = compute_woodbury_cov_and_loss(B_hat, d_hat, ycov_tr_t[b_idx])
                loss_total = loss_beta + 0.10 * loss_cov
                loss_total.backward()
                nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
                optimizer.step()

            # Validation
            gnn_model.eval()
            with torch.no_grad():
                b_hat_v, B_hat_v, d_hat_v = gnn_model(X_va_t, A_va_t, b_va_t, btc_idx=btc_idx)
                l_b_v = F.smooth_l1_loss(b_hat_v, yb_va_t, beta=0.2)
                _, l_c_v = compute_woodbury_cov_and_loss(B_hat_v, d_hat_v, ycov_va_t)
                val_loss = float(l_b_v + 0.10 * l_c_v)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in gnn_model.state_dict().items()}

        if best_state is not None:
            gnn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        gnn_model.eval()

        # Generate OOS test predictions for R3 (and covariance estimates)
        logger.info(f"Generating R3 OOS predictions for {len(test_idx)} hours...")
        with torch.no_grad():
            for ti in test_idx:
                if ti >= 168:
                    x_t = torch.tensor(
                        np.array([[feat_panel_map[c][ti, ai] for c in RISK_FEATURE_COLS] for ai in range(n_assets)], dtype=np.float32)
                    ).unsqueeze(0).to(device)

                    sub_r = ret_mat[ti - 168 : ti]
                    corr_168 = np.corrcoef(sub_r, rowvar=False)
                    corr_168 = np.nan_to_num(corr_168, nan=0.0)
                    adj_t = np.abs(corr_168)
                    np.fill_diagonal(adj_t, 0.0)
                    row_sums = adj_t.sum(axis=1, keepdims=True) + 1e-6
                    adj_t = adj_t / row_sums
                    adj_tensor = torch.tensor(adj_t.astype(np.float32)).unsqueeze(0).to(device)

                    base_b_t = torch.tensor(feat_panel_map["beta_168h"][ti].astype(np.float32)).unsqueeze(0).to(device)

                    b_hat, B_hat, d_hat = gnn_model(x_t, adj_tensor, base_b_t, btc_idx=btc_idx)
                    Sigma_hat_t, _ = compute_woodbury_cov_and_loss(
                        B_hat, d_hat, torch.eye(n_assets, device=device).unsqueeze(0)
                    )

                    beta_hat_r3[ti] = b_hat.squeeze(0).cpu().numpy()
                    factors_r3[ti] = B_hat.squeeze(0).cpu().numpy()
                    cov_hat_r3[ti] = Sigma_hat_t.squeeze(0).cpu().numpy()

                    # Shuffle placebo (permutes node features across assets)
                    x_t_shuffled = x_t[:, torch.randperm(n_assets)]
                    b_hat_pl, _, _ = gnn_model(x_t_shuffled, adj_tensor, base_b_t, btc_idx=btc_idx)
                    beta_hat_placebo[ti] = b_hat_pl.squeeze(0).cpu().numpy()

    # Precompute sample covariances and Ledoit-Wolf covariances across test_idx every 24h
    logger.info("Computing R0 (Sample Cov) and R1 (Ledoit-Wolf Cov) across test timestamps...")
    lw = LedoitWolf()
    for ti in range(n_ts):
        if ts_index[ti] >= oos_start_ts and ts_index[ti] <= final_end_ts:
            if ti >= 720 and ts_index[ti].hour == 0:
                sub_r_30d = ret_mat[ti - 720 : ti]
                # R0: Sample Covariance
                cov_s = np.cov(sub_r_30d, rowvar=False)
                cov_hat_r0[ti : min(ti + 24, n_ts)] = cov_s
                # R1: Ledoit-Wolf Shrinkage Covariance
                lw.fit(sub_r_30d)
                cov_hat_r1[ti : min(ti + 24, n_ts)] = lw.covariance_
            elif ti >= 720 and np.isnan(cov_hat_r0[ti, 0, 0]):
                cov_hat_r0[ti] = cov_hat_r0[ti - 1]
                cov_hat_r1[ti] = cov_hat_r1[ti - 1]

    return {
        "beta_r0": beta_hat_r0,
        "cov_r0": cov_hat_r0,
        "beta_r1": beta_hat_r1,
        "cov_r1": cov_hat_r1,
        "beta_r2": beta_hat_r2,
        "beta_r3": beta_hat_r3,
        "factors_r3": factors_r3,
        "cov_r3": cov_hat_r3,
        "beta_placebo": beta_hat_placebo,
        "ts_index": ts_index,
        "assets": assets,
        "ret_mat": ret_mat,
        "fwd_beta_48": fwd_beta_48,
        "fwd_beta_168": fwd_beta_168,
        "fwd_cov_168": fwd_cov_168,
    }


# =================================================================================
# 4. Mandatory Statistical Gate Evaluation
# =================================================================================

def evaluate_statistical_risk_gates(risk_data: dict[str, Any]) -> dict[str, Any]:
    """
    Evaluates:
      1. Forward 48h & 168h Beta Forecast: RMSE, MAE, OOS R^2, and Crisis-Window RMSE.
      2. Forward 168h Covariance: Gaussian NLL / QLIKE, Frobenius error, and Realized GMV Volatility.
      3. Feature-Shuffle Placebo.
    """
    logger.info("Computing Statistical Risk Gate Metrics across OOS 22,440 hours...")
    ts_index = risk_data["ts_index"]
    oos_mask = (ts_index >= pd.Timestamp(QUARTERLY_TEST_STARTS[0])) & (ts_index <= pd.Timestamp(TEST_END_FINAL))
    oos_indices = np.where(oos_mask)[0]

    fwd_b48 = risk_data["fwd_beta_48"]
    fwd_b168 = risk_data["fwd_beta_168"]
    fwd_c168 = risk_data["fwd_cov_168"]
    ret_mat = risk_data["ret_mat"]
    n_assets = len(risk_data["assets"])

    # Define Crisis Windows
    ftx_mask = oos_mask & (ts_index >= pd.Timestamp("2022-11-06 00:00:00+00:00")) & (ts_index <= pd.Timestamp("2022-11-20 23:00:00+00:00"))
    yen_mask = oos_mask & (ts_index >= pd.Timestamp("2024-08-02 00:00:00+00:00")) & (ts_index <= pd.Timestamp("2024-08-10 23:00:00+00:00"))
    crisis_mask = ftx_mask | yen_mask
    crisis_idx = np.where(crisis_mask)[0]

    # Baseline historic mean for R^2
    hist_mean_b48 = np.nanmean(fwd_b48[oos_indices])
    hist_mean_b168 = np.nanmean(fwd_b168[oos_indices])

    beta_models = {
        "R0 (30d OLS Baseline)": risk_data["beta_r0"],
        "R1 (Vasicek Shrinkage)": risk_data["beta_r1"],
        "R2 (LightGBM Beta)": risk_data["beta_r2"],
        "R3a (GraphBetaCovNet)": risk_data["beta_r3"],
        "Placebo (Feature Shuffle)": risk_data["beta_placebo"],
    }

    beta_metrics: dict[str, dict[str, float]] = {}
    for name, pred_b in beta_models.items():
        # Valid 48h evaluation mask
        valid_48 = oos_indices[~np.isnan(fwd_b48[oos_indices, 0]) & ~np.isnan(pred_b[oos_indices, 0])]
        err_48 = pred_b[valid_48] - fwd_b48[valid_48]
        rmse_48 = float(np.sqrt(np.mean(err_48**2)))
        mae_48 = float(np.mean(np.abs(err_48)))
        ss_tot_48 = np.sum((fwd_b48[valid_48] - hist_mean_b48) ** 2)
        r2_48 = float(1.0 - (np.sum(err_48**2) / (ss_tot_48 + 1e-8)))

        # Crisis RMSE (48h)
        val_crisis_48 = np.intersect1d(crisis_idx, valid_48)
        rmse_crisis_48 = float(np.sqrt(np.mean((pred_b[val_crisis_48] - fwd_b48[val_crisis_48]) ** 2)))

        # Valid 168h evaluation mask
        valid_168 = oos_indices[~np.isnan(fwd_b168[oos_indices, 0]) & ~np.isnan(pred_b[oos_indices, 0])]
        err_168 = pred_b[valid_168] - fwd_b168[valid_168]
        rmse_168 = float(np.sqrt(np.mean(err_168**2)))
        mae_168 = float(np.mean(np.abs(err_168)))
        ss_tot_168 = np.sum((fwd_b168[valid_168] - hist_mean_b168) ** 2)
        r2_168 = float(1.0 - (np.sum(err_168**2) / (ss_tot_168 + 1e-8)))

        # Crisis RMSE (168h)
        val_crisis_168 = np.intersect1d(crisis_idx, valid_168)
        rmse_crisis_168 = float(np.sqrt(np.mean((pred_b[val_crisis_168] - fwd_b168[val_crisis_168]) ** 2)))

        beta_metrics[name] = {
            "rmse_48": rmse_48,
            "mae_48": mae_48,
            "r2_48": r2_48,
            "crisis_rmse_48": rmse_crisis_48,
            "rmse_168": rmse_168,
            "mae_168": mae_168,
            "r2_168": r2_168,
            "crisis_rmse_168": rmse_crisis_168,
        }

    # Covariance Matrix Evaluation (R0, R1, R3)
    cov_models = {
        "R0 (Sample Covariance)": risk_data["cov_r0"],
        "R1 (Ledoit-Wolf Covariance)": risk_data["cov_r1"],
        "R3 (GraphBetaCovNet SPD)": risk_data["cov_r3"],
    }

    cov_metrics: dict[str, dict[str, float]] = {}
    daily_oos = [i for i in oos_indices if ts_index[i].hour == 0 and not np.isnan(fwd_c168[i, 0, 0])]

    for name, cov_arr in cov_models.items():
        qlike_list = []
        frob_list = []
        gmv_vols = []

        for ti in daily_oos:
            sigma_true = fwd_c168[ti]
            sigma_pred = cov_arr[ti]
            if np.isnan(sigma_pred[0, 0]):
                continue

            # Ensure SPD via eigenvalue regularization if needed
            w_eig, v_eig = np.linalg.eigh(sigma_pred)
            w_eig = np.maximum(w_eig, 1e-6)
            sigma_pred_spd = v_eig @ np.diag(w_eig) @ v_eig.T

            # QLIKE loss: (1/N) * [ tr(Sigma_pred^-1 @ Sigma_true) - logdet(Sigma_pred^-1 @ Sigma_true) - N ]
            try:
                inv_pred = np.linalg.inv(sigma_pred_spd)
                prod = inv_pred @ sigma_true
                tr_val = np.trace(prod)
                sign, logdet_val = np.linalg.slogdet(prod)
                if sign > 0:
                    qlike = (tr_val - logdet_val - n_assets) / n_assets
                    if not np.isnan(qlike) and qlike < 100.0:
                        qlike_list.append(qlike)
            except Exception:
                pass

            # Frobenius error
            frob_err = np.linalg.norm(sigma_pred_spd - sigma_true, "fro") / (np.linalg.norm(sigma_true, "fro") + 1e-8)
            frob_list.append(frob_err)

            # Global Minimum Variance (GMV) Portfolio
            try:
                ones = np.ones(n_assets)
                inv_pred = np.linalg.inv(sigma_pred_spd)
                w_gmv = inv_pred @ ones / (ones.T @ inv_pred @ ones)
                # Cap extreme leverage weights
                w_gmv = np.clip(w_gmv, -0.5, 0.5)
                w_gmv = w_gmv / np.sum(w_gmv)
                # Forward 168h realized return series
                fwd_returns = ret_mat[ti : ti + 168] @ w_gmv
                gmv_vol = np.std(fwd_returns) * np.sqrt(HOURS_PER_YEAR) * 100.0
                gmv_vols.append(gmv_vol)
            except Exception:
                pass

        cov_metrics[name] = {
            "qlike": float(np.mean(qlike_list)) if qlike_list else np.nan,
            "frobenius": float(np.mean(frob_list)) if frob_list else np.nan,
            "gmv_vol": float(np.mean(gmv_vols)) if gmv_vols else np.nan,
        }

    return {"beta_metrics": beta_metrics, "cov_metrics": cov_metrics}


# =================================================================================
# 5. Multi-Factor Nullspace Projection & Downstream Alpha Engine
# =================================================================================

def project_multi_factor_neutral(
    w0: np.ndarray,
    constraint_matrix: np.ndarray,
    gross_cap: float = 1.0,
    max_weight: float = 0.15,
) -> np.ndarray:
    """
    Closed-form nullspace projection onto C^T w = 0:
      P = I - C (C^T C)^-1 C^T
      w_proj = P @ w0
    Guarantees dollar-neutrality and beta/factor-neutrality down to machine precision (< 1e-12).
    Enforces gross exposure cap and individual position boundaries.
    """
    n = len(w0)
    C = constraint_matrix  # Shape (n, K)
    K = C.shape[1]

    # Orthogonal projection matrix
    CtC = C.T @ C
    CtC_inv = np.linalg.inv(CtC + np.eye(K) * 1e-12)
    P = np.eye(n) - C @ CtC_inv @ C.T
    w_proj = P @ w0

    # L1 normalization to gross_cap
    sum_abs = np.sum(np.abs(w_proj))
    if sum_abs < 1e-12:
        return np.zeros(n)
    w_scaled = w_proj * (gross_cap / sum_abs)

    # Position clipping with scalar shrinkage
    max_w = np.max(np.abs(w_scaled))
    if max_w > max_weight:
        w_scaled = w_scaled * (max_weight / max_w)

    return w_scaled


def simulate_downstream_sleeve(
    pred_mat: np.ndarray,
    ret_1h_mat: np.ndarray,
    fr_mat: np.ndarray,
    vol_mat: np.ndarray,
    constraint_matrices: list[np.ndarray],
    ts_index: pd.DatetimeIndex,
    true_beta_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Simulates the 7-tranche overlapping portfolio with the specified causal risk constraint matrix C_t.
    Holds EXP-008d alpha predictions strictly fixed.
    """
    n_ts, n_assets = pred_mat.shape
    sub_weights = [np.zeros((n_ts, n_assets)) for _ in range(7)]

    for k in range(7):
        current_longs: set[int] = set()
        current_shorts: set[int] = set()
        current_w = np.zeros(n_assets)

        for t in range(n_ts):
            if (t - 24 * k) >= 0 and (t - 24 * k) % 168 == 0:
                scores = pred_mat[t]
                vols = np.maximum(vol_mat[t], 1e-4)
                ranked_indices = np.argsort(scores)

                top_6 = set(ranked_indices[-6:])
                bot_6 = set(ranked_indices[:6])

                if len(current_longs) > 0 and len(current_shorts) > 0:
                    new_longs = {i for i in current_longs if i in top_6}
                    new_shorts = {i for i in current_shorts if i in bot_6}

                    for idx in reversed(ranked_indices):
                        if len(new_longs) == 4:
                            break
                        if idx not in new_shorts:
                            new_longs.add(idx)

                    for idx in ranked_indices:
                        if len(new_shorts) == 4:
                            break
                        if idx not in new_longs:
                            new_shorts.add(idx)
                else:
                    new_shorts = set(ranked_indices[:4])
                    new_longs = set(ranked_indices[-4:])

                current_longs = new_longs
                current_shorts = new_shorts

                w0 = np.zeros(n_assets)
                inv_vols = 1.0 / vols
                long_inv = [inv_vols[i] for i in current_longs]
                short_inv = [inv_vols[i] for i in current_shorts]

                for idx in current_longs:
                    w0[idx] = +0.50 * (inv_vols[idx] / sum(long_inv))
                for idx in current_shorts:
                    w0[idx] = -0.50 * (inv_vols[idx] / sum(short_inv))

                # Multi-factor nullspace projection
                C_t = constraint_matrices[t]
                current_w = project_multi_factor_neutral(w0, C_t, gross_cap=1.0, max_weight=0.15)

            sub_weights[k][t] = current_w

    weights = np.mean(sub_weights, axis=0)
    pos_shifted = pd.DataFrame(weights).shift(1).fillna(0.0).values
    turnover = np.sum(
        np.abs(np.diff(np.concatenate([np.zeros((1, n_assets)), pos_shifted], axis=0), axis=0)),
        axis=1,
    )

    funding_hours = ts_index.hour.isin([0, 8, 16])
    fund_pnl = np.where(funding_hours, np.sum(-pos_shifted * fr_mat, axis=1), 0.0)
    price_pnl = np.sum(pos_shifted * ret_1h_mat, axis=1)
    gross_pnl = price_pnl + fund_pnl

    r0 = gross_pnl - turnover * (2.0 / 10000.0)
    r10 = gross_pnl - turnover * (10.0 / 10000.0)
    r25 = gross_pnl - turnover * (25.0 / 10000.0)

    # Realized rolling 7d beta leakage vs ground truth realized beta
    net_beta = np.sum(pos_shifted * true_beta_mat, axis=1)
    rolling_7d_leakage = pd.Series(net_beta).rolling(168, min_periods=24).mean().abs().values

    m = {
        "m0": compute_metrics(r0, turnover, 0.0, ts_index),
        "m10": compute_metrics(r10, turnover, 10.0, ts_index),
        "m25": compute_metrics(r25, turnover, 25.0, ts_index),
        "mean_net_beta": float(np.mean(net_beta)),
        "std_net_beta": float(np.std(net_beta)),
        "p95_leakage": float(np.nanpercentile(rolling_7d_leakage, 95)),
    }
    return r10, turnover, net_beta, m


# =================================================================================
# 6. Main Experiment Execution
# =================================================================================

def run_exp012() -> None:
    """Executes EXP-012 full forensic pipeline."""
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("STARTING EXP-012: Dynamic Covariance & Latent-Factor Beta Forecasting")
    logger.info("=" * 80)

    # 1. Load data and walkforward models from EXP-008d
    df_panel = load_base_panel()
    df_panel, _ = run_walkforward_models_all(df_panel)

    # 2. Build risk features and targets
    df_panel, assets, ts_index, ret_mat, fwd_beta_48, fwd_beta_168, fwd_cov_168 = build_risk_features_and_targets(df_panel)

    # 3. Run risk walkforward models (R0, R1, R2, R3)
    risk_results = run_risk_walkforward(df_panel, assets, ts_index, ret_mat, fwd_beta_48, fwd_beta_168, fwd_cov_168)

    # 4. Evaluate Statistical Risk Gates
    stat_gates = evaluate_statistical_risk_gates(risk_results)

    # Print Statistical Gate Table
    print("\n" + "=" * 140)
    print("EXP-012 STAGE 1: STATISTICAL RISK-FORECASTING BENCHMARK (OOS 22,440 BARS)")
    print("=" * 140)
    print(
        f"{'Risk Model Candidate':<28} | {'48h RMSE':<10} | {'48h MAE':<9} | {'48h R^2 (%)':<11} | {'168h RMSE':<10} | {'168h MAE':<9} | {'168h R^2 (%)':<12} | {'Crisis RMSE (FTX/Aug24)':<24}"
    )
    print("-" * 140)
    for model_name, bm in stat_gates["beta_metrics"].items():
        print(
            f"{model_name:<28} | {bm['rmse_48']:<10.4f} | {bm['mae_48']:<9.4f} | {bm['r2_48']*100:<11.2f}% | {bm['rmse_168']:<10.4f} | {bm['mae_168']:<9.4f} | {bm['r2_168']*100:<12.2f}% | {bm['crisis_rmse_48']:<24.4f}"
        )
    print("=" * 140)

    print("\n" + "=" * 90)
    print("FORWARD 168h COVARIANCE MATRIX FORECAST ACCURACY & GMV PORTFOLIO TEST")
    print("=" * 90)
    print(f"{'Covariance Model':<30} | {'Gaussian NLL (QLIKE)':<22} | {'Frobenius Error':<16} | {'Realized GMV Vol (%)':<20}")
    print("-" * 90)
    for cov_name, cm in stat_gates["cov_metrics"].items():
        print(f"{cov_name:<30} | {cm['qlike']:<22.4f} | {cm['frobenius']:<16.4f} | {cm['gmv_vol']:<20.2f}%")
    print("=" * 90)

    # 5. Downstream Alpha Impact Evaluation
    logger.info("Executing Downstream Portfolio Simulation with EXP-008d Fixed Predictions...")
    oos_mask = (df_panel["timestamp"] >= QUARTERLY_TEST_STARTS[0]) & (df_panel["timestamp"] <= TEST_END_FINAL)
    oos_panel = df_panel[oos_mask].copy()

    timestamps = sorted(oos_panel["timestamp"].unique())
    n_ts = len(timestamps)
    n_assets = len(assets)
    ts_map = {ts: i for i, ts in enumerate(timestamps)}
    asset_map = {a: i for i, a in enumerate(assets)}

    pred_total_48h = np.zeros((n_ts, n_assets))
    ret_1h_mat = np.zeros((n_ts, n_assets))
    fr_mat = np.zeros((n_ts, n_assets))
    vol_mat = np.zeros((n_ts, n_assets))

    for row in oos_panel.itertuples():
        ti = ts_map[row.timestamp]
        ai = asset_map[row.asset]
        pred_total_48h[ti, ai] = getattr(row, "pred_total_48h", 0.0)
        ret_1h_mat[ti, ai] = row.ret_1h
        fr_mat[ti, ai] = row.funding_rate
        vol_mat[ti, ai] = row.trailing_vol_168h

    oos_ts_index = pd.DatetimeIndex(timestamps)
    oos_indices = np.where(
        (ts_index >= pd.Timestamp(QUARTERLY_TEST_STARTS[0])) & (ts_index <= pd.Timestamp(TEST_END_FINAL))
    )[0]
    true_beta_mat = risk_results["fwd_beta_48"][oos_indices]

    # Build constraint matrices C_t for each candidate
    ones = np.ones((n_assets, 1))

    # R0: [1, beta_OLS_30d]
    C_R0 = [np.hstack([ones, risk_results["beta_r0"][idx, :, None]]) for idx in oos_indices]

    # R1: [1, beta_EWMA_Vasicek]
    C_R1 = [np.hstack([ones, risk_results["beta_r1"][idx, :, None]]) for idx in oos_indices]

    # R2: [1, beta_LGBM]
    C_R2 = [np.hstack([ones, risk_results["beta_r2"][idx, :, None]]) for idx in oos_indices]

    # R3a: [1, beta_GNN]
    C_R3a = [np.hstack([ones, risk_results["beta_r3"][idx, :, None]]) for idx in oos_indices]

    # R3b: [1, B_GNN_1, B_GNN_2] (2-Factor Latent Nullspace)
    C_R3b = [np.hstack([ones, risk_results["factors_r3"][idx]]) for idx in oos_indices]

    downstream_candidates = [
        ("R0 (EXP-008d Baseline 30d OLS)", C_R0),
        ("R1 (EWMA + Vasicek Shrinkage)", C_R1),
        ("R2 (LightGBM Dynamic Beta)", C_R2),
        ("R3a (GraphBetaCovNet Beta)", C_R3a),
        ("R3b (GraphBetaCovNet 2-Factor)", C_R3b),
    ]

    # Load APEX baseline return series
    logger.info("Loading audited Frozen APEX v45 baseline...")
    dataset_v45, processed_v45, garch_wts_v45, eval_index_v45, n_v45, _, macro_regime_v45, mtf_skew_v45, arb_scale_v45 = (
        v45.load_data_and_features()
    )
    dir_s, _ = v45.run_directional_engine(
        processed_v45, dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45, garch_wts_v45, mtf_skew_v45
    )
    sp_s, _ = v45.run_spread_engine(processed_v45, dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45)
    arb_s, _ = v45.run_funding_arb_engine(dataset_v45, v45.UNIVERSE, eval_index_v45, n_v45, arb_scale_v45)
    champ_full = 0.25 * dir_s * macro_regime_v45 + 0.55 * sp_s * macro_regime_v45 + 0.20 * arb_s

    df_champ_full = pd.DataFrame({"r_apex": champ_full}, index=eval_index_v45)
    slice_22440 = df_champ_full.loc[QUARTERLY_TEST_STARTS[0] : TEST_END_FINAL]
    r_apex_oos = slice_22440["r_apex"].values
    r_apex_honest_vm, _, _ = apply_honest_vol_targeting(
        r_apex_oos, slippage_bps=10.0, underlying_turnover=np.zeros(len(r_apex_oos)), target_vol=0.1844
    )

    downstream_results = {}
    for name, C_list in downstream_candidates:
        r10, turn, net_beta, m = simulate_downstream_sleeve(
            pred_total_48h, ret_1h_mat, fr_mat, vol_mat, C_list, oos_ts_index, true_beta_mat
        )
        # Compute blend: 90% APEX / 10% ML
        r_blend_10bps = 0.90 * r_apex_oos + 0.10 * r10
        r_blend_honest_vm, _, _ = apply_honest_vol_targeting(
            r_blend_10bps, slippage_bps=10.0, underlying_turnover=0.10 * turn, target_vol=0.1844
        )
        m_blend = compute_metrics(r_blend_honest_vm, turn * 0.10, 10.0, oos_ts_index)

        # Correlation with APEX
        rho = float(np.corrcoef(r10, r_apex_oos)[0, 1])

        downstream_results[name] = {
            "sleeve_m": m,
            "blend_m": m_blend,
            "rho": rho,
        }

    # Print Downstream Comparison Table
    print("\n" + "=" * 135)
    print("EXP-012 STAGE 2: DOWNSTREAM IMPACT ON PROMOTED EXP-008d ALPHA SLEEVE (10 BPS INSTITUTIONAL SLIPPAGE)")
    print("=" * 135)
    print(
        f"{'Risk Projector':<32} | {'95% Beta Leak':<14} | {'CAGR (%)':<9} | {'Net Sharpe':<11} | {'Calmar':<8} | {'MaxDD (%)':<10} | {'Turnover':<10} | {'rho(APEX)':<10} | {'90/10 Blend Sharpe':<18}"
    )
    print("-" * 135)
    for name, res in downstream_results.items():
        sm = res["sleeve_m"]["m10"]
        leak = res["sleeve_m"]["p95_leakage"]
        bm = res["blend_m"]
        rho = res["rho"]
        print(
            f"{name:<32} | {leak:<14.4f} | {sm['cagr']:<9.2f}% | {sm['sharpe']:<11.2f} | {sm['calmar']:<8.2f} | {sm['max_dd']:<10.1f}% | {sm['ann_turnover']:<10.1f}x | {rho:<10.4f} | {bm['sharpe']:<18.2f}"
        )
    print("=" * 135)

    print("\n" + "=" * 105)
    print("HONEST VOL-MATCHED 90% APEX / 10% ML BLEND: YEAR-BY-YEAR SHARPE COMPARISON (10 BPS SLIPPAGE)")
    print("=" * 105)
    print(f"{'Risk Projector':<32} | {'Full OOS':<10} | {'2022 (FTX)':<12} | {'2023':<10} | {'2024':<10} | {'2025':<10} | {'MaxDD (%)':<10}")
    print("-" * 105)
    # Print baseline APEX
    base_m = compute_metrics(r_apex_honest_vm, np.zeros(len(r_apex_honest_vm)), 10.0, oos_ts_index)
    print(
        f"{'100% Frozen APEX v45 (Baseline)':<32} | {base_m['sharpe']:<10.2f} | {base_m['sr_years'][2022]:<12.2f} | {base_m['sr_years'][2023]:<10.2f} | {base_m['sr_years'][2024]:<10.2f} | {base_m['sr_years'][2025]:<10.2f} | {base_m['max_dd']:<10.1f}%"
    )
    for name, res in downstream_results.items():
        bm = res["blend_m"]
        print(
            f"{name:<32} | {bm['sharpe']:<10.2f} | {bm['sr_years'][2022]:<12.2f} | {bm['sr_years'][2023]:<10.2f} | {bm['sr_years'][2024]:<10.2f} | {bm['sr_years'][2025]:<10.2f} | {bm['max_dd']:<10.1f}%"
        )
    print("=" * 105)

    elapsed = time.time() - start_time
    logger.info(f"EXP-012 execution completed in {elapsed:.1f}s ({elapsed/60.0:.2f} mins).")


if __name__ == "__main__":
    run_exp012()
