"""
Inverted Transformer (iTransformer) Architecture for Tail Risk Forecasting.
Paper: "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (Liu et al., ICLR 2024).
Each token represents the 48-hour temporal trajectory of a single feature/variate.
Applies cross-variate self-attention across feature channels to output a market-state representation.
Dual Head:
  1. Binary tail_crash_6h classification logit (pos_weight BCE)
  2. 5% Value-at-Risk (VaR) prediction on forward 6h minimum return (Pinball loss, alpha=0.05)
"""


import torch
import torch.nn.functional as F
from torch import nn

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("itransformer")


def pinball_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """
    Quantile (Pinball) Loss for Value-at-Risk forecasting at confidence level alpha.
    L_alpha(y, y_hat) = max(alpha * (y - y_hat), (alpha - 1) * (y - y_hat))
    """
    diff = y_true.view(-1) - y_pred.view(-1)
    loss = torch.max(alpha * diff, (alpha - 1.0) * diff)
    return loss.mean()


class InvertedTransformerBackbone(nn.Module):
    """
    Inverted Transformer encoder:
    Transforms lookback trajectories of V variates into a market-state representation.
    """

    def __init__(
        self,
        lookback_len: int = 48,
        num_variates: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lookback_len = lookback_len
        self.num_variates = num_variates
        self.d_model = d_model

        # Project temporal sequence of length L into token embedding d_model
        self.variate_embed = nn.Sequential(
            nn.Linear(lookback_len, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Attention pooling across variates
        self.pool_attn = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Tensor of shape (batch_size, lookback_len, num_variates)
        Returns:
          h_market: (batch_size, d_model) pooled market state
          variate_tokens: (batch_size, num_variates, d_model) transformed variates
        """
        # Transpose to (batch_size, num_variates, lookback_len)
        x_trans = x.transpose(1, 2)
        # Embed each variate trajectory: (batch_size, num_variates, d_model)
        tokens = self.variate_embed(x_trans)
        # Cross-variate self-attention
        encoded = self.transformer_encoder(tokens)
        encoded = self.norm(encoded)

        # Attentive pooling
        attn_weights = F.softmax(self.pool_attn(encoded), dim=1)  # (batch, num_variates, 1)
        h_market = (encoded * attn_weights).sum(dim=1)  # (batch, d_model)

        return h_market, encoded


class iTransformerOnlyModel(nn.Module):
    """
    Model 3: iTransformer_Only architecture with Dual Head.
    """

    def __init__(
        self,
        lookback_len: int = 48,
        num_variates: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = InvertedTransformerBackbone(
            lookback_len=lookback_len,
            num_variates=num_variates,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # Dual Head
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

    def forward(self, x_market: torch.Tensor) -> dict[str, torch.Tensor]:
        h_market, _ = self.backbone(x_market)
        logit = self.clf_head(h_market).squeeze(-1)
        var_pred = self.var_head(h_market).squeeze(-1)
        return {
            "logit": logit,
            "prob": torch.sigmoid(logit),
            "var_pred": var_pred,
            "h_market": h_market,
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
