"""
Unified Multimodal Fusion Architecture: Multimodal_iTransformer_NLP.
Combines:
  1. Market Query (Q): Market-state representation from 2-layer Inverted Transformer.
  2. News Keys & Values (K, V): Up to M = 16 causal news clusters in (T-24h, T]
     fusing BGE-small embeddings (384-dim) + fold-OOS MiniLM 6-class probabilities (6-dim)
     + metadata (source tier, confirmation count, causal recency decay, active flag).
  3. Materiality Masking: Masks out news slots where p(non_event_noise) > 0.80,
     with clean zero-attention fallback defaulting strictly to market state.
  4. Low-Rank Bottleneck + Dropout: News projection down to d_model=64 with LayerNorm + Dropout(0.35).
  5. Zero-Initialized Gated Residual (Flamingo/ReZero style):
     h_fused = h_market + tanh(alpha) * attn_out, where alpha is initialized to 0.0.
     Guarantees the model starts at iTransformer_Only accuracy and only integrates
     news attention where it genuinely lowers validation loss.
  6. Dual Head: Binary tail_crash_6h logit (pos_weight BCE) + 5% VaR (Pinball loss, alpha=0.05).
"""


import torch
from torch import nn

from apex_multimodal_risk.src.models.itransformer import (
    InvertedTransformerBackbone,
    pinball_loss,
)
from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("multimodal_fusion")


class MultimodalCrossAttentionFusion(nn.Module):
    """
    Model 4: Regularized Multimodal iTransformer + Cross-Attention NLP Fusion.
    Equipped with Materiality Masking, Low-Rank Bottleneck, and Zero-Initialized Gated Residual.
    """

    def __init__(
        self,
        lookback_len: int = 48,
        num_variates: int = 16,
        news_feat_dim: int = 394,  # 384 BGE + 6 MiniLM probs + 4 metadata
        max_news_clusters: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_itransformer_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        news_dropout: float = 0.35,
        noise_threshold: float = 0.80,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_news_clusters = max_news_clusters
        self.noise_threshold = noise_threshold

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

        # 2. Low-Rank News Bottleneck Projection + High Dropout
        self.news_proj = nn.Sequential(
            nn.Linear(news_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(news_dropout),
        )

        # 3. Multi-Head Cross-Attention: Market Query Q, News Keys K and Values V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=news_dropout,
            batch_first=True,
        )

        # 4. Zero-Initialized Gated Residual Parameter (Flamingo / ReZero style)
        # Initializes at 0.0 -> tanh(0.0) = 0.0, guaranteeing h_fused = h_market initially
        self.alpha = nn.Parameter(torch.zeros(1))

        # 5. Dual Prediction Heads
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
    ) -> dict[str, torch.Tensor]:
        """
        Args:
          x_market: (B, lookback_len, num_variates)
          x_news: (B, max_news_clusters, news_feat_dim)
                  dim 384 is p(non_event_noise) from MiniLM
          news_pad_mask: (B, max_news_clusters) bool tensor (True for padded)
        """
        # 1. Market representation as Query Q: (B, 1, d_model)
        h_market, _ = self.market_backbone(x_market)
        q = h_market.unsqueeze(1)

        # 2. News representation as K, V: (B, M, d_model)
        k = self.news_proj(x_news)
        v = k

        # 3. Materiality Masking:
        # Mask out slots where p(non_event_noise) > 0.80 or already padded
        p_noise = x_news[:, :, 384]
        is_noise = p_noise > self.noise_threshold
        combined_pad_mask = news_pad_mask | is_noise

        all_padded = combined_pad_mask.all(dim=1, keepdim=True)  # (B, 1)
        safe_pad_mask = combined_pad_mask.clone()
        safe_pad_mask[:, 0] = False  # Prevent NaN in PyTorch MHA when all keys are masked

        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=safe_pad_mask,
        )

        # Fallback zero-attention pass when no material news fired in (T-24h, T]
        attn_out = torch.where(all_padded.unsqueeze(-1), torch.zeros_like(attn_out), attn_out)
        attn_out_squeezed = attn_out.squeeze(1)  # (B, d_model)

        # 4. Zero-Initialized Gated Residual (Flamingo/ReZero style)
        gate = torch.tanh(self.alpha)
        h_fused = h_market + gate * attn_out_squeezed  # (B, d_model)

        # 5. Dual Head Predictions
        logit = self.clf_head(h_fused).squeeze(-1)
        var_pred = self.var_head(h_fused).squeeze(-1)

        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "var_pred": var_pred,
            "h_fused": h_fused,
            "attn_weights": attn_weights,
            "gate": gate,
            "alpha": self.alpha,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        y_clf: torch.Tensor,
        y_var: torch.Tensor,
        pos_weight: float = 23.0,
        lambda_var: float = 10.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
