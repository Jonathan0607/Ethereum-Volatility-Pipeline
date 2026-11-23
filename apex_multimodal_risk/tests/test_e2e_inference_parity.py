"""
Audit C: End-to-End Inference Parity Test.
Compares 200 real historical bars between the offline PyTorch + LightGBM stack
and the ONNX serving microservice stack.
Asserts:
- Max abs diff in p_crash_cal < 5e-3 (mean < 5e-4)
- raw_var_5pct Pearson correlation >= 0.998
"""

import json
import pickle

import numpy as np
import pytest
import torch
from apex_multimodal_risk.src.features.shared_features import (
    MARKET_NUMERICAL_COLS,
    load_and_merge_features_and_targets,
)
from apex_multimodal_risk.src.models.itransformer import iTransformerOnlyModel
from apex_multimodal_risk.src.serving.app import app
from apex_multimodal_risk.src.train.train_ablation import build_lookback_tensor
from apex_multimodal_risk.src.utils import PROJECT_ROOT
from fastapi.testclient import TestClient
from sentence_transformers import SentenceTransformer


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_e2e_inference_parity(client):
    """
    Score 200 real historical bars through both offline Python stack and ONNX serving stack.
    Asserts strict numerical parity between offline models and INT8 ONNX microservice.
    """
    # 1. Load data
    df = load_and_merge_features_and_targets()
    eval_panel = df.loc[df["target_valid"]].copy().reset_index(drop=True)
    lookback_tensor = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)

    # 2. Select 200 real historical bars (100 news, 100 calm)
    news_bars = eval_panel[eval_panel["news_count_24h"] > 0].index.to_numpy()
    calm_bars = eval_panel[eval_panel["news_count_24h"] == 0].index.to_numpy()
    selected_indices = np.concatenate([news_bars[:100], calm_bars[:100]])
    selected_indices.sort()
    assert len(selected_indices) == 200

    # 3. Load offline models & weights
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts"
    with open(artifacts_dir / "best_lightgbm_risk.pkl", "rb") as f:
        lgb_bundle = pickle.load(f)
    lgb_model = lgb_bundle["model"]
    platt_scaler = lgb_bundle["platt_scaler"]
    tau_nov = lgb_bundle["gate_thresholds"]["tau_nov"]
    tau_proto = lgb_bundle["gate_thresholds"]["tau_proto"]

    st_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    with open(artifacts_dir / "risk_prototypes.json") as f:
        proto_data = json.load(f)
    proto_keys = list(proto_data["prototypes"].keys())
    proto_matrix = np.array([proto_data["prototypes"][k]["vector"] for k in proto_keys], dtype=np.float32)

    itr_ckpt = torch.load(artifacts_dir / "best_itransformer_var.pt", map_location="cpu", weights_only=False)
    itr_model = iTransformerOnlyModel(lookback_len=48, num_variates=16, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    itr_model.load_state_dict(itr_ckpt["model_state_dict"])
    itr_model.eval()

    scaler_means = np.array(itr_ckpt["scaler_means"], dtype=np.float32).reshape(1, 1, 16)
    scaler_stds = np.array(itr_ckpt["scaler_stds"], dtype=np.float32).reshape(1, 1, 16)

    sample_headlines = [
        "SEC files lawsuit against crypto exchange alleging unregistered securities",
        "Major smart contract exploit drains 50M liquidity pool",
        "Federal Reserve announces emergency rate hike",
        "Solana validators report consensus stall on mainnet",
    ]

    py_p_cal = []
    py_var = []
    onnx_p_cal = []
    onnx_var = []

    for idx_num, idx in enumerate(selected_indices):
        bar_row = eval_panel.iloc[idx]
        w_market = lookback_tensor[idx]  # (48, 16)
        has_news = bool(bar_row["news_count_24h"] > 0)
        headline = sample_headlines[idx_num % len(sample_headlines)] if has_news else None

        # --- Offline Python Stack ---
        if headline:
            emb = st_model.encode([headline], normalize_embeddings=True)[0]
            sims = np.dot(proto_matrix, emb)
            p_max = float(np.max(sims))
            nov = float(1.0 - p_max)
            g_nov = max(0.0, nov - tau_nov)
            g_proto = max(0.0, p_max - tau_proto)
            alert = 1.0 if (nov > tau_nov or p_max > tau_proto) else 0.0
            burst = 1.0
        else:
            nov, p_max, g_nov, g_proto, alert, burst = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        w_norm = (w_market.reshape(1, 48, 16) - scaler_means) / (scaler_stds + 1e-8)
        with torch.no_grad():
            itr_out = itr_model(torch.tensor(w_norm, dtype=torch.float32))
            v_py = float(itr_out["var_pred"].item())

        x_feat = np.zeros((1, 20), dtype=np.float32)
        x_feat[0, :16] = w_market[-1]
        x_feat[0, 16] = g_nov
        x_feat[0, 17] = g_proto
        x_feat[0, 18] = alert
        x_feat[0, 19] = burst

        logit_py = lgb_model.predict(x_feat, raw_score=True)[0]
        p_cal_py = float(platt_scaler.transform(np.array([logit_py]))[0])

        py_p_cal.append(p_cal_py)
        py_var.append(v_py)

        # --- Serving ONNX Stack ---
        payload = {
            "asset": bar_row["asset"],
            "market_window": w_market.tolist(),
            "headline": headline,
            "timeout_ms": 5000.0,
        }
        resp = client.post("/v1/score", json=payload)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert not data["degraded_fallback"]
        onnx_p_cal.append(data["p_crash_cal"])
        onnx_var.append(data["raw_var_5pct"])

    py_p_cal = np.array(py_p_cal)
    onnx_p_cal = np.array(onnx_p_cal)
    py_var = np.array(py_var)
    onnx_var = np.array(onnx_var)

    p_diff = np.abs(py_p_cal - onnx_p_cal)
    corr_v = float(np.corrcoef(py_var, onnx_var)[0, 1])
    max_p_diff = float(np.max(p_diff))
    mean_p_diff = float(np.mean(p_diff))

    print("\nAudit C Results on 200 Real Bars:")
    print(f"Max abs diff p_crash_cal: {max_p_diff:.6f} (target < 0.005)")
    print(f"Mean abs diff p_crash_cal: {mean_p_diff:.6f} (target < 0.0005)")
    print(f"raw_var_5pct Correlation: {corr_v:.6f} (target >= 0.998)")

    assert max_p_diff < 5e-3, f"Max p_crash_cal diff too large: {max_p_diff}"
    assert mean_p_diff < 5e-4, f"Mean p_crash_cal diff too large: {mean_p_diff}"
    assert corr_v >= 0.998, f"raw_var_5pct correlation below threshold: {corr_v}"
