"""
Delayed-Feedback Adaptive Conformal Inference (ACI) Module.
Wraps iTransformer 5% Value-at-Risk (VaR) output with strictly causal 6-hour delayed feedback:
- Rolling non-conformity residual buffer W = 500 hours per asset.
- Realized outcome y_{T-6} only updates the residual buffer and alpha_t at bar T.
- Prevents lookahead bias via strict delay queue and CausalViolationError enforcement.
"""

from collections import deque
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from apex_multimodal_risk.src.utils import PROJECT_ROOT, setup_logger

logger = setup_logger("conformal")


class CausalViolationError(Exception):
    """Raised when feedback is delivered before the strict 6-hour horizon has elapsed."""


def kupiec_pof_test(
    y_true: np.ndarray,
    var_pred: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """
    Computes Kupiec's Proportion of Failures (POF) Likelihood Ratio Test.
    H0: True violation rate == alpha (e.g. 0.05).
    Returns (p_value, empirical_rate, violation_count).
    p_value > 0.05 indicates model cannot be rejected at 95% confidence.
    """
    y = np.asarray(y_true, dtype=np.float64)
    v = np.asarray(var_pred, dtype=np.float64)
    violations = (y < v).astype(np.int32)
    x = int(np.sum(violations))
    n = len(violations)
    if n == 0 or x == 0 or x == n:
        return 0.0, float(x / max(1, n)), x

    p_hat = float(x / n)
    lr = 2.0 * (x * np.log(p_hat / alpha) + (n - x) * np.log((1.0 - p_hat) / (1.0 - alpha)))
    p_val = float(1.0 - stats.chi2.cdf(lr, df=1))
    return p_val, p_hat, x


def christoffersen_tests(
    y_true: np.ndarray,
    var_pred: np.ndarray,
    alpha: float = 0.05,
    step: int = 1,
) -> dict[str, Any]:
    """
    Computes Kupiec Unconditional Coverage (LR_uc), Christoffersen Independence (LR_ind),
    and Christoffersen Conditional Coverage (LR_cc) Likelihood Ratio Tests.
    Also computes Mean Realized Shortfall on violation bars.
    If step > 1 (e.g. step=6 for 6h non-overlapping sampling), sub-samples y_true and var_pred.
    """
    y = np.asarray(y_true, dtype=np.float64).ravel()
    v = np.asarray(var_pred, dtype=np.float64).ravel()
    if step > 1:
        y = y[::step]
        v = v[::step]
    hits = (y < v).astype(np.int32)
    T = len(hits)
    N = int(np.sum(hits))
    p = N / T if T > 0 else 0.0

    # 1. Kupiec LR_uc
    if N == 0:
        LR_uc = -2.0 * np.log((1.0 - alpha) ** T + 1e-15)
    elif N == T:
        LR_uc = -2.0 * np.log(alpha ** T + 1e-15)
    else:
        ll_null = (T - N) * np.log(1.0 - alpha) + N * np.log(alpha)
        ll_alt = (T - N) * np.log(1.0 - p) + N * np.log(p)
        LR_uc = -2.0 * (ll_null - ll_alt)
    p_uc = float(1.0 - stats.chi2.cdf(max(0.0, LR_uc), df=1))

    # 2. Christoffersen Independence LR_ind
    t00 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 0)))
    t01 = int(np.sum((hits[:-1] == 0) & (hits[1:] == 1)))
    t10 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 0)))
    t11 = int(np.sum((hits[:-1] == 1) & (hits[1:] == 1)))

    pi01 = t01 / (t00 + t01) if (t00 + t01) > 0 else 0.0
    pi11 = t11 / (t10 + t11) if (t10 + t11) > 0 else 0.0
    pi = (t01 + t11) / (t00 + t01 + t10 + t11) if (t00 + t01 + t10 + t11) > 0 else 0.0

    ll_0 = (t00 + t10) * np.log(max(1e-12, 1.0 - pi)) + (t01 + t11) * np.log(max(1e-12, pi))
    ll_1 = 0.0
    if t00 > 0: ll_1 += t00 * np.log(max(1e-12, 1.0 - pi01))
    if t01 > 0: ll_1 += t01 * np.log(max(1e-12, pi01))
    if t10 > 0: ll_1 += t10 * np.log(max(1e-12, 1.0 - pi11))
    if t11 > 0: ll_1 += t11 * np.log(max(1e-12, pi11))

    LR_ind = -2.0 * (ll_0 - ll_1)
    p_ind = float(1.0 - stats.chi2.cdf(max(0.0, LR_ind), df=1))

    # 3. Conditional Coverage LR_cc
    LR_cc = LR_uc + LR_ind
    p_cc = float(1.0 - stats.chi2.cdf(max(0.0, LR_cc), df=2))

    # 4. Shortfall metrics
    violation_returns = y[hits == 1]
    mean_shortfall = float(np.mean(violation_returns)) if len(violation_returns) > 0 else 0.0
    mean_breach = float(np.mean(v[hits == 1] - violation_returns)) if len(violation_returns) > 0 else 0.0

    return {
        "step": step,
        "T": T,
        "violations": N,
        "violation_rate": p,
        "coverage": 1.0 - p,
        "LR_uc": float(LR_uc),
        "p_uc": p_uc,
        "LR_ind": float(LR_ind),
        "p_ind": p_ind,
        "LR_cc": float(LR_cc),
        "p_cc": p_cc,
        "pi01": float(pi01),
        "pi11": float(pi11),
        "mean_realized_shortfall": mean_shortfall,
        "mean_breach_magnitude": mean_breach,
    }


class DelayedFeedbackACI:
    """
    Single-asset Delayed-Feedback Adaptive Conformal Inference Wrapper.
    Strictly causal 6-hour delayed feedback:
        err_{T-6} = 1.0 if y_realized_{T-6} < aci_var_{T-6} else 0.0
        alpha_t = clip(alpha_{t-1} + gamma * (0.05 - err_{T-6}), 0.005, 0.25)
        aci_var_t = raw_var_t - quantile(residual_buffer, 1 - alpha_t)
    where non-conformity score s = raw_var - y_realized.
    """

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        gamma: float = 0.01,
        window_size: int = 500,
        delay_hours: int = 6,
        alpha_min: float = 0.005,
        alpha_max: float = 0.25,
        asset: str | None = None,
    ):
        self.nominal_alpha = nominal_alpha
        self.alpha_t = nominal_alpha
        self.gamma = gamma
        self.window_size = window_size
        self.delay_hours = delay_hours
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.asset = asset

        self.residual_buffer: deque = deque(maxlen=window_size)
        # Pending queue maps bar_timestamp -> (raw_var, aci_var)
        self.pending_queue: dict[pd.Timestamp, tuple[float, float]] = {}

    def predict(
        self,
        raw_var: float,
        bar_time: pd.Timestamp | None = None,
    ) -> tuple[float, float]:
        """
        Computes conformal-adjusted VaR bound for current bar T:
            aci_var_t = raw_var - quantile(residual_buffer, 1 - alpha_t)
        Returns (aci_var_t, current_alpha_t).
        """
        if len(self.residual_buffer) >= 20:
            n_buf = len(self.residual_buffer)
            # Conservative finite-sample conformal quantile: ceil((1 - alpha_t) * (n_buf + 1)) / n_buf
            q_level = float(np.clip(np.ceil((1.0 - self.alpha_t) * (n_buf + 1)) / n_buf, 0.01, 1.0))
            q_adj = float(np.quantile(list(self.residual_buffer), q_level, method="higher"))
            aci_var = float(raw_var - q_adj)
        else:
            # Warm-up fallback
            aci_var = float(raw_var)

        if bar_time is not None:
            self.pending_queue[bar_time] = (float(raw_var), float(aci_var))

        return aci_var, self.alpha_t

    def step_feedback(
        self,
        target_bar_time: pd.Timestamp,
        current_time: pd.Timestamp,
        realized_y: float,
    ) -> dict[str, float]:
        """
        Receives feedback for bar target_bar_time at current_time.
        Enforces strict causal delay: (current_time - target_bar_time) >= delay_hours.
        Updates rolling residual buffer and alpha_t.
        """
        elapsed_hours = (current_time - target_bar_time).total_seconds() / 3600.0
        if elapsed_hours < self.delay_hours - 1e-6:
            raise CausalViolationError(
                f"Causal violation: cannot ingest feedback for bar {target_bar_time} "
                f"at {current_time} ({elapsed_hours:.2f}h < {self.delay_hours}h delay requirement)!"
            )

        if target_bar_time not in self.pending_queue:
            # If not in pending queue (e.g. warm-up), use raw_var == realized_y
            raw_var, aci_var = realized_y, realized_y
        else:
            raw_var, aci_var = self.pending_queue.pop(target_bar_time)

        # Check empirical violation on the delayed bar
        err = 1.0 if realized_y < aci_var else 0.0

        # Update adaptive alpha_t
        self.alpha_t = float(
            np.clip(
                self.alpha_t + self.gamma * (self.nominal_alpha - err),
                self.alpha_min,
                self.alpha_max,
            )
        )

        # Non-conformity score: shortfall beyond predicted raw VaR
        non_conformity = float(raw_var - realized_y)
        self.residual_buffer.append(non_conformity)

        return {
            "alpha_t": self.alpha_t,
            "err": err,
            "non_conformity": non_conformity,
            "buffer_len": len(self.residual_buffer),
        }


class MultiAssetDelayedFeedbackACI:
    """Multi-asset manager for Delayed-Feedback ACI across SOL, AVAX, LINK."""

    def __init__(
        self,
        nominal_alpha: float = 0.05,
        gamma: float = 0.01,
        window_size: int = 500,
        delay_hours: int = 6,
    ):
        self.nominal_alpha = nominal_alpha
        self.gamma = gamma
        self.window_size = window_size
        self.delay_hours = delay_hours
        self.engines: dict[str, DelayedFeedbackACI] = {}

    def get_engine(self, asset: str) -> DelayedFeedbackACI:
        if asset not in self.engines:
            self.engines[asset] = DelayedFeedbackACI(
                nominal_alpha=self.nominal_alpha,
                gamma=self.gamma,
                window_size=self.window_size,
                delay_hours=self.delay_hours,
                asset=asset,
            )
        return self.engines[asset]

    def predict(
        self,
        asset: str,
        raw_var: float,
        bar_time: pd.Timestamp | None = None,
    ) -> tuple[float, float]:
        return self.get_engine(asset).predict(raw_var, bar_time)

    def step_feedback(
        self,
        asset: str,
        target_bar_time: pd.Timestamp,
        current_time: pd.Timestamp,
        realized_y: float,
    ) -> dict[str, float]:
        return self.get_engine(asset).step_feedback(target_bar_time, current_time, realized_y)


def run_conformal_validation(gamma: float = 0.01) -> dict[str, Any]:
    """
    Evaluates Delayed-Feedback ACI on out-of-sample data across SOL, AVAX, LINK.
    Reuses chronological OOS fold predictions (Split 9 test fold or chronological walk-forward split).
    Reports overall empirical violation rate, Kupiec POF p-value, interval width,
    and stress coverage during FTX (Nov 2022) and Aug 2024 crash.
    """
    from apex_multimodal_risk.src.features.shared_features import (
        MARKET_NUMERICAL_COLS,
        CPCVStandardScaler,
        load_and_merge_features_and_targets,
    )
    from apex_multimodal_risk.src.models.itransformer import iTransformerOnlyModel
    from apex_multimodal_risk.src.train.train_ablation import (
        build_lookback_tensor,
        eval_itransformer_batched,
    )
    from apex_multimodal_risk.src.utils import get_device

    logger.info(f"Running ACI out-of-sample validation pass (gamma={gamma})...")
    df = load_and_merge_features_and_targets()
    eval_panel = df.loc[df["target_valid"]].copy().reset_index(drop=True)

    # Use chronological walk-forward test split: first 60% train, last 40% OOS test
    unique_ts = np.array(sorted(eval_panel["timestamp"].unique()))
    split_idx = int(len(unique_ts) * 0.60)
    split_ts = unique_ts[split_idx]

    tr_mask = eval_panel["timestamp"] < split_ts
    te_mask = eval_panel["timestamp"] >= split_ts

    scaler = CPCVStandardScaler(feature_cols=MARKET_NUMERICAL_COLS)
    scaler.fit(eval_panel.loc[tr_mask])

    lookback_tensor = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)
    m_vec = np.array(scaler.means_, dtype=np.float32).reshape(1, 1, -1)
    s_vec = np.array(scaler.stds_, dtype=np.float32).reshape(1, 1, -1)

    # Load promoted iTransformer checkpoint
    itrans_ckpt_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/best_itransformer_var.pt"
    device = get_device()
    m3 = iTransformerOnlyModel(
        lookback_len=48,
        num_variates=len(MARKET_NUMERICAL_COLS),
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    ).to(device)

    import torch
    ckpt = torch.load(itrans_ckpt_path, map_location=device, weights_only=False)
    m3.load_state_dict(ckpt["model_state_dict"])
    m3.eval()

    # Predict raw VaR on test set
    t_lb_te = (torch.tensor(lookback_tensor[te_mask], dtype=torch.float32) - torch.tensor(m_vec)) / torch.tensor(s_vec)
    _, _, raw_var_te = eval_itransformer_batched(m3, t_lb_te, device, batch_size=128)

    test_df = eval_panel.loc[te_mask].copy().reset_index(drop=True)
    test_df["raw_var"] = raw_var_te
    test_df = test_df.sort_values("timestamp").reset_index(drop=True)

    # Run Delayed-Feedback ACI simulation per asset
    aci_manager = MultiAssetDelayedFeedbackACI(
        nominal_alpha=0.05,
        gamma=gamma,
        window_size=500,
        delay_hours=6,
    )

    n_test = len(test_df)
    aci_var_arr = np.zeros(n_test, dtype=np.float32)
    alpha_arr = np.zeros(n_test, dtype=np.float32)

    # Pre-build lookup for O(1) delayed feedback retrieval
    lookup_y = {(row.asset, row.timestamp): float(row.fwd_min_ret_6h) for row in test_df.itertuples()}

    # Process sequentially by timestamp
    for i in range(n_test):
        row = test_df.iloc[i]
        sym = row["asset"]
        t = row["timestamp"]
        r_var = row["raw_var"]

        # Step feedback for bar t - 6h if present
        t_delayed = t - timedelta(hours=6)
        if (sym, t_delayed) in lookup_y:
            aci_manager.step_feedback(sym, t_delayed, t, lookup_y[(sym, t_delayed)])

        # Predict current bar
        v_adj, a_curr = aci_manager.predict(sym, r_var, t)
        aci_var_arr[i] = v_adj
        alpha_arr[i] = a_curr

    test_df["aci_var"] = aci_var_arr
    test_df["alpha_t"] = alpha_arr

    y_realized = test_df["fwd_min_ret_6h"].to_numpy()
    raw_var = test_df["raw_var"].to_numpy()
    aci_var = test_df["aci_var"].to_numpy()

    # Kupiec POF tests
    p_raw, rate_raw, viol_raw = kupiec_pof_test(y_realized, raw_var, alpha=0.05)
    p_aci, rate_aci, viol_aci = kupiec_pof_test(y_realized, aci_var, alpha=0.05)

    # Average VaR level (interval width)
    mean_raw_var = float(np.mean(raw_var))
    mean_aci_var = float(np.mean(aci_var))

    # Stress Window Evaluation
    stress_windows = {
        "FTX Collapse (Nov 2022)": ("2022-11-06 00:00:00+00:00", "2022-11-12 23:00:00+00:00"),
        "Aug 2024 Flash Crash": ("2024-08-02 00:00:00+00:00", "2024-08-07 23:00:00+00:00"),
    }
    stress_results = {}
    for w_name, (t_start, t_end) in stress_windows.items():
        w_mask = (test_df["timestamp"] >= t_start) & (test_df["timestamp"] <= t_end)
        sub = test_df.loc[w_mask]
        if len(sub) > 0:
            y_sub = sub["fwd_min_ret_6h"].to_numpy()
            v_raw_sub = sub["raw_var"].to_numpy()
            v_aci_sub = sub["aci_var"].to_numpy()
            cov_raw = float(np.mean(y_sub >= v_raw_sub) * 100.0)
            cov_aci = float(np.mean(y_sub >= v_aci_sub) * 100.0)
            stress_results[w_name] = {
                "bars": len(sub),
                "coverage_raw_pct": cov_raw,
                "coverage_aci_pct": cov_aci,
            }

    return {
        "gamma": gamma,
        "n_test_bars": n_test,
        "static_var": {
            "rate": rate_raw,
            "violations": viol_raw,
            "kupiec_p_value": p_raw,
            "mean_var": mean_raw_var,
        },
        "delayed_feedback_aci": {
            "rate": rate_aci,
            "violations": viol_aci,
            "kupiec_p_value": p_aci,
            "mean_var": mean_aci_var,
        },
        "stress_windows": stress_results,
    }


if __name__ == "__main__":
    for g in [0.005, 0.01]:
        res = run_conformal_validation(gamma=g)
        print(f"\n=== ACI OUT-OF-SAMPLE VALIDATION (gamma = {g}) ===")
        print(f"  Test Bars: {res['n_test_bars']:,}")
        print(f"  Static 5% VaR:    Rate = {res['static_var']['rate']*100:.2f}%, Kupiec p-val = {res['static_var']['kupiec_p_value']:.4f}, Mean VaR = {res['static_var']['mean_var']:.4f}")
        print(f"  Delayed ACI VaR:  Rate = {res['delayed_feedback_aci']['rate']*100:.2f}%, Kupiec p-val = {res['delayed_feedback_aci']['kupiec_p_value']:.4f}, Mean VaR = {res['delayed_feedback_aci']['mean_var']:.4f}")
        print("  Stress Window Coverage:")
        for w_name, s_data in res["stress_windows"].items():
            print(f"    {w_name}: Static Cov = {s_data['coverage_raw_pct']:.1f}%, ACI Cov = {s_data['coverage_aci_pct']:.1f}%")
