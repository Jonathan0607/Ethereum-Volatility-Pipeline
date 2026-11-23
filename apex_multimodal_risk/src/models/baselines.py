"""
Baseline Models for Tail Risk Forecasting.
- Model 1 (GARCH_Skew): L2-regularized Logistic Regression on GARCH vol + MTF skewness.
- Model 2 (LightGBM): Gradient-boosted decision trees on full tabular market features
  plus aggregated 24h causal news features.
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("baselines")

GARCH_SKEW_FEATURES = [
    "garch_cond_vol",
    "skew_72h",
    "skew_168h",
    "skew_336h",
]

LIGHTGBM_FEATURES = [
    # Market features
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
    # News features
    "news_count_24h",
    "news_count_24h_asset",
    "news_count_24h_macro",
    "confirm_count_24h",
    "embedding_novelty_score",
    "has_exploit_24h",
    "has_outage_24h",
    "has_reg_24h",
    "has_insolv_24h",
    "has_macro_24h",
    "max_deberta_risk_prob",
    "tier1_count_24h",
    "confirmation_velocity",
]


class GARCHSkewBaseline:
    """
    Model 1: Econometric Baseline.
    Regularized Logistic Regression on GARCH volatility and multi-timeframe return skewness.
    """

    def __init__(self, C: float = 1.0, class_weight: str = "balanced"):
        self.C = C
        self.class_weight = class_weight
        self.model = LogisticRegression(
            C=C,
            class_weight=class_weight,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
        )
        self.feature_cols = GARCH_SKEW_FEATURES

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GARCHSkewBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X)


class LightGBMBaseline:
    """
    Model 2: Tabular Market + Causal Aggregated News Baseline.
    Gradient-boosted decision trees with depth regularization and balanced class weighting.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        num_leaves: int = 15,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.feature_cols = LIGHTGBM_FEATURES

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> "LightGBMBaseline":
        callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)] if eval_set else None
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks,
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict_logits(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict(X, raw_score=True)
        return raw
