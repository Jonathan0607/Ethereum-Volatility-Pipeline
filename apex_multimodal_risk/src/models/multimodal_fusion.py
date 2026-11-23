"""
Unified Multimodal Fusion Architecture: Multimodal_iTransformer_NLP.
Combines:
  1. Market Query (Q): Market-state representation from 2-layer Inverted Transformer.
  2. News Keys & Values (K, V): Up to M = 16 causal news clusters in (T-24h, T]
     fusing BGE-small embeddings (384-dim) + fold-OOS DeBERTa-v3 6-class probabilities (6-dim)
     + metadata (novelty, source tier, confirmation count, causal recency decay).
  3. Multi-Head Cross-Attention (num_heads=4) with NaN-safe padding masks.
  4. Dual Head: Binary tail_crash_6h logit (pos_weight BCE) + 5% VaR (Pinball loss, alpha=0.05).
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from apex_multimodal_risk.src.utils import setup_logger
from apex_multimodal_risk.src.models.itransformer import InvertedTransformerBackbone, pinball_loss

logger = setup_logger("multimodal_fusion")


class MultimodalCrossAttentionFusion(nn.Module):
    """
    Model 4: Multimodal iTransformer + Cross-Attention NLP Fusion.
    """

    def __init__(
        self,
        lookback_len: int = 48,
        num_variates: int = 16,
        news_feat_dim: int = 394,  # 384 BGE + 6 DeBERTa probs + 4 metadata
        max_news_clusters: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_itransformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_news_clusters = max_news_clusters

        # 1. Market Backbone (iTransformer)
        self.market_backbone = InvertedTransformerBackbone(
            lookback_len=lookback_len,
            num_variates=num_variates,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_itransformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # 2. News Projection
        self.news_proj = nn.Sequential(
            nn.Linear(news_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # 3. Cross-Attention: Market Query Q, News Keys K and Values V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 4. Fusion Feed-Forward Network
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 5. Dual Head
        self.clf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x_market: torch.Tensor,
        x_news: torch.Tensor,
        news_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          x_market: (B, lookback_len, num_variates)
          x_news: (B, max_news_clusters, news_feat_dim)
          news_pad_mask: (B, max_news_clusters) bool tensor (True for padded)
        """
        # 1. Market representation as Query Q: (B, 1, d_model)
        h_market, _ = self.market_backbone(x_market)
        q = h_market.unsqueeze(1)

        # 2. News representation as K, V: (B, M, d_model)
        k = self.news_proj(x_news)
        v = k

        # 3. NaN-safe cross-attention
        all_padded = news_pad_mask.all(dim=1, keepdim=True)  # (B, 1)
        safe_pad_mask = news_pad_mask.clone()
        safe_pad_mask[:, 0] = False  # prevent all-True mask NaN

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=safe_pad_mask,
        )

        # Zero out attention where all keys were originally padded
        attn_out = torch.where(all_padded.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)

        # Residual + Norm
        fused = self.norm1(q + self.dropout1(attn_out))
        # MLP + Norm
        out = self.norm2(fused + self.dropout2(self.mlp(fused)))
        h_fused = out.squeeze(1)  # (B, d_model)

        # Dual Head Predictions
        logit = self.clf_head(h_fused).squeeze(-1)
        var_pred = self.var_head(h_fused).squeeze(-1)

        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "var_pred": var_pred,
            "h_fused": h_fused,
            "attn_weights": attn_weights,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        y_clf: torch.Tensor,
        y_var: torch.Tensor,
        pos_weight: float = 23.0,
        lambda_var: float = 10.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pos_wt_tensor = torch.tensor([pos_weight], device=y_clf.device)
        bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_wt_tensor)
        loss_bce = bce_fn(outputs["logit"], y_clf)
        loss_pinball = pinball_loss(y_var, outputs["var_pred"], alpha=0.05)
        total_loss = loss_bce + lambda_var * loss_pinball
        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_bce": float(loss_bce.item()),
            "loss_pinball": float(loss_pinball.item()),
        }
