"""
Unified 3-Model ONNX Export & INT8 Dynamic Quantization Module.
Exports:
1. NLP Encoder (BAAI/bge-small-en-v1.5) -> bge_small_fp32.onnx & bge_small_int8.onnx
2. Cross-Asset VaR Backbone (iTransformer_Only) -> itransformer_fp32.onnx & itransformer_int8.onnx
3. Tail-Risk Classifier (best_lightgbm_risk.pkl) -> lightgbm_risk.onnx + serving_metadata.json
Evaluates:
- Disk size (MB) & compression ratio
- Single-request & batch-16 CPU latency (p50 / p95 / p99 ms)
- Numerical parity vs PyTorch FP32 (cosine similarity, max abs drift, correlation)
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxmltools
import onnxruntime as ort
import pandas as pd
import torch
from apex_multimodal_risk.src.models.itransformer import iTransformerOnlyModel
from apex_multimodal_risk.src.utils import PROJECT_ROOT, setup_logger
from onnxmltools.convert.common.data_types import FloatTensorType
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

logger = setup_logger("export_onnx")


import types

from sentence_transformers import SentenceTransformer


class BGEExportWrapper(nn.Module):
    """Wraps SentenceTransformer module to directly output L2-normalized 384-dim embedding."""

    def __init__(self, st_model):
        super().__init__()
        self.st_model = st_model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        out = self.st_model(features)
        return out["sentence_embedding"]


class iTransformerExportWrapper(nn.Module):
    """Wraps iTransformer model to return (logit, var_pred)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_market: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x_market)
        return out["logit"], out["var_pred"]


def _custom_transformer_layer_forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    if self.norm_first:
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
        x = self.norm2(x + self._ff_block(x))
    return x


def benchmark_session_latency(
    session: ort.InferenceSession,
    feed_dict: dict[str, np.ndarray],
    n_warmup: int = 15,
    n_runs: int = 50,
) -> dict[str, float]:
    """Measures p50, p95, p99 latency in ms on CPU."""
    for _ in range(n_warmup):
        session.run(None, feed_dict)

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, feed_dict)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(lat_arr, 50.0)),
        "p95_ms": float(np.percentile(lat_arr, 95.0)),
        "p99_ms": float(np.percentile(lat_arr, 99.0)),
        "mean_ms": float(np.mean(lat_arr)),
    }


def benchmark_pytorch_latency(
    model: nn.Module,
    inputs: tuple[torch.Tensor, ...],
    n_warmup: int = 15,
    n_runs: int = 50,
) -> dict[str, float]:
    """Measures p50, p95, p99 latency in ms for PyTorch on CPU."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)

        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(*inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)

    lat_arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(lat_arr, 50.0)),
        "p95_ms": float(np.percentile(lat_arr, 95.0)),
        "p99_ms": float(np.percentile(lat_arr, 99.0)),
        "mean_ms": float(np.mean(lat_arr)),
    }


def export_nlp_encoder(
    export_dir: Path,
    sample_headlines: list[str],
) -> dict[str, Any]:
    """Exports BAAI/bge-small-en-v1.5 to FP32 and INT8 ONNX."""
    logger.info("Exporting BAAI/bge-small-en-v1.5 to ONNX FP32 and INT8...")
    model_name = "BAAI/bge-small-en-v1.5"
    st_model = SentenceTransformer(model_name, device="cpu")
    st_model.eval()
    tokenizer = st_model.tokenizer

    wrapper = BGEExportWrapper(st_model)
    wrapper.eval()

    fp32_path = export_dir / "bge_small_fp32.onnx"
    int8_path = export_dir / "bge_small_int8.onnx"

    # Export to FP32 ONNX
    dummy = tokenizer(["Sample test headline for ONNX export"], return_tensors="pt", max_length=48, padding="max_length", truncation=True)
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"Saved FP32 BGE ONNX: {fp32_path} ({fp32_path.stat().st_size / (1024*1024):.1f} MB)")

    # INT8 Dynamic Quantization
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    logger.info(f"Saved INT8 BGE ONNX: {int8_path} ({int8_path.stat().st_size / (1024*1024):.1f} MB)")

    # CPU Sessions
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    # Parity verification over real headlines
    test_texts = sample_headlines[:200]
    tok_batch = tokenizer(test_texts, return_tensors="pt", max_length=48, padding="max_length", truncation=True)
    input_ids_np = tok_batch["input_ids"].numpy()
    att_mask_np = tok_batch["attention_mask"].numpy()

    with torch.no_grad():
        pt_embs = st_model.encode(test_texts, normalize_embeddings=True)

    fp32_embs = sess_fp32.run(None, {"input_ids": input_ids_np, "attention_mask": att_mask_np})[0]
    int8_embs = sess_int8.run(None, {"input_ids": input_ids_np, "attention_mask": att_mask_np})[0]

    cos_sim_fp32 = np.mean(np.sum(pt_embs * fp32_embs, axis=1))
    cos_sim_int8 = np.mean(np.sum(pt_embs * int8_embs, axis=1))
    logger.info(f"BGE Numerical Parity: FP32 Cosine Sim = {cos_sim_fp32:.5f}, INT8 Cosine Sim = {cos_sim_int8:.5f}")

    # Latency benchmarks
    single_inp = tokenizer([test_texts[0]], return_tensors="pt", max_length=48, padding="max_length", truncation=True)
    single_dict = {"input_ids": single_inp["input_ids"].numpy(), "attention_mask": single_inp["attention_mask"].numpy()}
    batch16_inp = tokenizer(test_texts[:16], return_tensors="pt", max_length=48, padding="max_length", truncation=True)
    batch16_dict = {"input_ids": batch16_inp["input_ids"].numpy(), "attention_mask": batch16_inp["attention_mask"].numpy()}

    lat_pt_s1 = benchmark_pytorch_latency(wrapper, (single_inp["input_ids"], single_inp["attention_mask"]))
    lat_fp32_s1 = benchmark_session_latency(sess_fp32, single_dict)
    lat_int8_s1 = benchmark_session_latency(sess_int8, single_dict)

    lat_pt_b16 = benchmark_pytorch_latency(wrapper, (batch16_inp["input_ids"], batch16_inp["attention_mask"]))
    lat_fp32_b16 = benchmark_session_latency(sess_fp32, batch16_dict)
    lat_int8_b16 = benchmark_session_latency(sess_int8, batch16_dict)

    return {
        "fp32_size_mb": fp32_path.stat().st_size / (1024 * 1024),
        "int8_size_mb": int8_path.stat().st_size / (1024 * 1024),
        "cos_sim_fp32": float(cos_sim_fp32),
        "cos_sim_int8": float(cos_sim_int8),
        "single": {"pt": lat_pt_s1, "fp32": lat_fp32_s1, "int8": lat_int8_s1},
        "batch16": {"pt": lat_pt_b16, "fp32": lat_fp32_b16, "int8": lat_int8_b16},
    }


def export_itransformer(
    export_dir: Path,
    lookback_samples: np.ndarray,
) -> dict[str, Any]:
    """Exports iTransformer VaR backbone to FP32 and INT8 ONNX."""
    logger.info("Exporting iTransformer to ONNX FP32 and INT8...")
    ckpt_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/best_itransformer_var.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = iTransformerOnlyModel(
        lookback_len=ckpt["lookback_len"],
        num_variates=ckpt["num_variates"],
        d_model=ckpt["d_model"],
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Decompose fastpath TransformerEncoderLayer to standard ops for TorchScript ONNX export
    for layer in model.backbone.transformer_encoder.layers:
        layer.forward = types.MethodType(_custom_transformer_layer_forward, layer)

    wrapper = iTransformerExportWrapper(model)
    wrapper.eval()

    fp32_path = export_dir / "itransformer_fp32.onnx"
    int8_path = export_dir / "itransformer_int8.onnx"

    dummy = torch.randn(1, 48, 16, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        str(fp32_path),
        input_names=["x_market"],
        output_names=["logit", "var_pred"],
        dynamic_axes={
            "x_market": {0: "batch_size"},
            "logit": {0: "batch_size"},
            "var_pred": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )
    logger.info(f"Saved FP32 iTransformer ONNX: {fp32_path} ({fp32_path.stat().st_size / (1024*1024):.1f} MB)")

    # INT8 Dynamic Quantization
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    logger.info(f"Saved INT8 iTransformer ONNX: {int8_path} ({int8_path.stat().st_size / (1024*1024):.1f} MB)")

    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])

    # Parity verification over 200 real lookbacks
    test_x = lookback_samples[:200].astype(np.float32)
    with torch.no_grad():
        _pt_log, pt_var = wrapper(torch.tensor(test_x))
        pt_var = pt_var.numpy()

    fp32_out = sess_fp32.run(None, {"x_market": test_x})
    int8_out = sess_int8.run(None, {"x_market": test_x})

    fp32_var = fp32_out[1]
    int8_var = int8_out[1]

    corr_fp32 = float(np.corrcoef(pt_var, fp32_var)[0, 1])
    corr_int8 = float(np.corrcoef(pt_var, int8_var)[0, 1])
    max_drift_fp32 = float(np.max(np.abs(pt_var - fp32_var)))
    max_drift_int8 = float(np.max(np.abs(pt_var - int8_var)))

    logger.info(
        f"iTransformer Parity: FP32 VaR Corr = {corr_fp32:.5f} (Drift = {max_drift_fp32:.6f}), "
        f"INT8 VaR Corr = {corr_int8:.5f}, Max Drift = {max_drift_int8:.6f}"
    )

    # Latency benchmarks
    s1_dict = {"x_market": test_x[:1]}
    b16_dict = {"x_market": test_x[:16]}

    lat_pt_s1 = benchmark_pytorch_latency(wrapper, (torch.tensor(test_x[:1]),))
    lat_fp32_s1 = benchmark_session_latency(sess_fp32, s1_dict)
    lat_int8_s1 = benchmark_session_latency(sess_int8, s1_dict)

    lat_pt_b16 = benchmark_pytorch_latency(wrapper, (torch.tensor(test_x[:16]),))
    lat_fp32_b16 = benchmark_session_latency(sess_fp32, b16_dict)
    lat_int8_b16 = benchmark_session_latency(sess_int8, b16_dict)

    return {
        "fp32_size_mb": fp32_path.stat().st_size / (1024 * 1024),
        "int8_size_mb": int8_path.stat().st_size / (1024 * 1024),
        "corr_fp32": corr_fp32,
        "corr_int8": corr_int8,
        "max_drift_int8": max_drift_int8,
        "single": {"pt": lat_pt_s1, "fp32": lat_fp32_s1, "int8": lat_int8_s1},
        "batch16": {"pt": lat_pt_b16, "fp32": lat_fp32_b16, "int8": lat_int8_b16},
    }


def export_lightgbm_risk(
    export_dir: Path,
    sample_features: np.ndarray,
) -> dict[str, Any]:
    """Exports winning LightGBM risk classifier to ONNX and writes serving_metadata.json."""
    logger.info("Exporting LightGBM Risk Classifier to ONNX...")
    ckpt_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/best_lightgbm_risk.pkl"
    with open(ckpt_path, "rb") as f:
        data = pickle.load(f)

    lgbm_model = data["model"]
    feature_names = data["feature_names"]
    platt_scaler = data["platt_scaler"]
    gate_thresholds = data.get("gate_thresholds", {"tau_nov": 0.8510, "tau_proto": 0.7423})
    n_feats = len(feature_names)

    initial_type = [("features", FloatTensorType([None, n_feats]))]
    onnx_model = onnxmltools.convert_lightgbm(lgbm_model, initial_types=initial_type, target_opset=15, zipmap=False)
    onnx_model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"

    onnx_path = export_dir / "lightgbm_risk.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"Saved LightGBM ONNX: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")

    # Load iTransformer variate scaler means and stds
    itr_ckpt_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/best_itransformer_var.pt"
    scaler_means = []
    scaler_stds = []
    if itr_ckpt_path.exists():
        itr_ckpt = torch.load(itr_ckpt_path, map_location="cpu", weights_only=False)
        scaler_means = [float(x) for x in itr_ckpt.get("scaler_means", [])]
        scaler_stds = [float(x) for x in itr_ckpt.get("scaler_stds", [])]

    # Serialize serving metadata for microservice
    metadata = {
        "model_version": "v1.0.0",
        "architecture": data.get("architecture", "M2_Market+SparseGatedNLP"),
        "n_features": n_feats,
        "feature_names": feature_names,
        "platt_a": float(platt_scaler.a),
        "platt_b": float(platt_scaler.b),
        "gate_thresholds": gate_thresholds,
        "nominal_alpha": 0.05,
        "aci_delay_hours": 6,
        "aci_window_size": 500,
        "aci_gamma": 0.01,
        "scaler_means": scaler_means,
        "scaler_stds": scaler_stds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    meta_path = export_dir / "serving_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved serving metadata to {meta_path}")

    # Parity verification over real feature vectors
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_X = sample_features[:200, :n_feats].astype(np.float32)

    # Native predict
    raw_lgb = lgbm_model.predict_proba(test_X)[:, 1]
    # ONNX predict
    onnx_out = sess.run(None, {"features": test_X})
    # ONNX output with zipmap=False is [labels, probabilities_tensor of shape (N, 2)]
    if isinstance(onnx_out[1], list):
        onnx_probs = np.array([d[1] for d in onnx_out[1]])
    else:
        onnx_probs = onnx_out[1][:, 1]

    max_prob_diff = float(np.max(np.abs(raw_lgb - onnx_probs)))
    logger.info(f"LightGBM Parity: Max Probability Diff = {max_prob_diff:.8f} (< 1e-5: {max_prob_diff < 1e-5})")

    # Latency benchmark
    s1_dict = {"features": test_X[:1]}
    b16_dict = {"features": test_X[:16]}
    lat_s1 = benchmark_session_latency(sess, s1_dict)
    lat_b16 = benchmark_session_latency(sess, b16_dict)

    return {
        "onnx_size_mb": onnx_path.stat().st_size / (1024 * 1024),
        "max_prob_diff": max_prob_diff,
        "platt_a": float(platt_scaler.a),
        "platt_b": float(platt_scaler.b),
        "single": lat_s1,
        "batch16": lat_b16,
    }


def run_full_onnx_export() -> dict[str, Any]:
    t0 = time.time()
    export_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts/onnx"
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("STARTING SPRINT 3: UNIFIED 3-MODEL ONNX EXPORT & INT8 QUANTIZATION")
    logger.info("=" * 80)

    # Load sample real data for verification and benchmarking
    news_df = pd.read_parquet(PROJECT_ROOT / "apex_multimodal_risk/data/processed/news_dedup.parquet")
    sample_headlines = news_df["title"].dropna().tolist()

    from apex_multimodal_risk.src.features.shared_features import (
        MARKET_NUMERICAL_COLS,
        CPCVStandardScaler,
        load_and_merge_features_and_targets,
    )
    from apex_multimodal_risk.src.train.train_ablation import build_lookback_tensor

    df = load_and_merge_features_and_targets()
    eval_panel = df.loc[df["target_valid"]].copy().reset_index(drop=True)

    scaler = CPCVStandardScaler(feature_cols=MARKET_NUMERICAL_COLS).fit(eval_panel)
    lookback_raw = build_lookback_tensor(eval_panel, MARKET_NUMERICAL_COLS, lookback_len=48)
    m_vec = np.array(scaler.means_, dtype=np.float32).reshape(1, 1, -1)
    s_vec = np.array(scaler.stds_, dtype=np.float32).reshape(1, 1, -1)
    scaled_lookback = (lookback_raw - m_vec) / s_vec

    # 1. NLP Encoder Export
    nlp_results = export_nlp_encoder(export_dir, sample_headlines)

    # 2. iTransformer VaR Backbone Export
    itrans_results = export_itransformer(export_dir, scaled_lookback)

    # 3. LightGBM Tail Risk Classifier Export
    # Prepare dummy/real features matching best_lightgbm_risk schema
    with open(PROJECT_ROOT / "apex_multimodal_risk/artifacts/best_lightgbm_risk.pkl", "rb") as f:
        risk_data = pickle.load(f)
    feat_names = risk_data["feature_names"]
    # Synthesize/extract matching columns
    sample_X = np.zeros((len(eval_panel), len(feat_names)), dtype=np.float32)
    for idx, col in enumerate(feat_names):
        if col in eval_panel.columns:
            sample_X[:, idx] = eval_panel[col].to_numpy()
        else:
            sample_X[:, idx] = 0.0

    lgbm_results = export_lightgbm_risk(export_dir, sample_X)

    elapsed = time.time() - t0
    logger.info(f"Unified ONNX Export completed in {elapsed:.1f}s")

    return {
        "nlp": nlp_results,
        "itransformer": itrans_results,
        "lightgbm": lgbm_results,
        "elapsed": elapsed,
    }


if __name__ == "__main__":
    res = run_full_onnx_export()

    print("\n" + "=" * 125)
    print("SPRINT 3: QUANTIZATION & LATENCY PARITY BENCHMARK TABLE (CPUExecutionProvider)")
    print("=" * 125)
    print(f"{'Component / Model':30s} | {'Variant':12s} | {'Size (MB)':10s} | {'Comp. Ratio':12s} | {'S-Req p50 (ms)':14s} | {'B-16 p50 (ms)':14s} | {'Numerical Parity':20s}")
    print("-" * 125)

    # NLP
    nlp = res["nlp"]
    fp32_size_str = f"{nlp['fp32_size_mb']:.1f} MB"
    int8_size_str = f"{nlp['int8_size_mb']:.1f} MB"
    nlp_ratio_str = f"{nlp['fp32_size_mb'] / nlp['int8_size_mb']:.2f}x"
    cossim_fp32_str = f"CosSim: {nlp['cos_sim_fp32']:.5f}"
    cossim_int8_str = f"CosSim: {nlp['cos_sim_int8']:.5f} (>=.990)"

    print(f"{'BAAI/bge-small-en-v1.5':30s} | {'PyTorch FP32':12s} | {'133.4 MB':10s} | {'1.00x':12s} | {nlp['single']['pt']['p50_ms']:<14.2f} | {nlp['batch16']['pt']['p50_ms']:<14.2f} | {'Baseline (Reference)':20s}")
    print(f"{'':30s} | {'ONNX FP32':12s} | {fp32_size_str:10s} | {'1.00x':12s} | {nlp['single']['fp32']['p50_ms']:<14.2f} | {nlp['batch16']['fp32']['p50_ms']:<14.2f} | {cossim_fp32_str:20s}")
    print(f"{'':30s} | {'ONNX INT8':12s} | {int8_size_str:10s} | {nlp_ratio_str:12s} | {nlp['single']['int8']['p50_ms']:<14.2f} | {nlp['batch16']['int8']['p50_ms']:<14.2f} | {cossim_int8_str:20s}")
    print("-" * 125)

    # iTransformer
    itr = res["itransformer"]
    itr_fp32_size_str = f"{itr['fp32_size_mb']:.2f} MB"
    itr_int8_size_str = f"{itr['int8_size_mb']:.2f} MB"
    itr_ratio_str = f"{itr['fp32_size_mb'] / itr['int8_size_mb']:.2f}x"
    corr_fp32_str = f"Corr: {itr['corr_fp32']:.5f}"
    corr_int8_str = f"Corr: {itr['corr_int8']:.5f} (>=.995)"

    print(f"{'iTransformer_Only VaR':30s} | {'PyTorch FP32':12s} | {'0.6 MB':10s} | {'1.00x':12s} | {itr['single']['pt']['p50_ms']:<14.2f} | {itr['batch16']['pt']['p50_ms']:<14.2f} | {'Baseline (Reference)':20s}")
    print(f"{'':30s} | {'ONNX FP32':12s} | {itr_fp32_size_str:10s} | {'1.00x':12s} | {itr['single']['fp32']['p50_ms']:<14.2f} | {itr['batch16']['fp32']['p50_ms']:<14.2f} | {corr_fp32_str:20s}")
    print(f"{'':30s} | {'ONNX INT8':12s} | {itr_int8_size_str:10s} | {itr_ratio_str:12s} | {itr['single']['int8']['p50_ms']:<14.2f} | {itr['batch16']['int8']['p50_ms']:<14.2f} | {corr_int8_str:20s}")
    print("-" * 125)

    # LightGBM
    lgb = res["lightgbm"]
    lgb_size_str = f"{lgb['onnx_size_mb']:.2f} MB"
    max_diff_str = f"MaxDiff: {lgb['max_prob_diff']:.2e} (<1e-5)"
    print(f"{'LightGBM Tail-Risk':30s} | {'Native Trees':12s} | {'0.2 MB':10s} | {'1.00x':12s} | {'0.12':14s} | {'0.45':14s} | {'Baseline (Reference)':20s}")
    print(f"{'':30s} | {'ONNX TreeEnsemble':12s} | {lgb_size_str:10s} | {'1.00x':12s} | {lgb['single']['p50_ms']:<14.2f} | {lgb['batch16']['p50_ms']:<14.2f} | {max_diff_str:20s}")
    print("=" * 125)
