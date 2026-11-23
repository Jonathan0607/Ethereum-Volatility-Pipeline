"""
Async FastAPI Microservice with Dynamic Micro-Batching for APEX Multimodal Risk Engine.
Endpoints:
- GET  /health: Health check, ONNX execution providers, uptime, loaded models.
- POST /v1/score: Multimodal scoring (BGE INT8 NLP + iTransformer INT8 VaR + LightGBM ONNX + Platt + ACI).
- POST /v1/feedback: Delayed-feedback ACI quantile update (causal 6h realization).
"""

import asyncio
import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import UTC
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

try:
    from tokenizers import Tokenizer
    _HAS_TOKENIZERS = True
except ImportError:
    _HAS_TOKENIZERS = False

try:
    from transformers import AutoTokenizer
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

from apex_multimodal_risk.src.models.conformal import MultiAssetDelayedFeedbackACI
from apex_multimodal_risk.src.serving.drift_monitor import DriftMonitor
from apex_multimodal_risk.src.utils import PROJECT_ROOT, setup_logger

logger = setup_logger("serving_app")

# ---------------------------------------------------------------------------
# Prometheus Telemetry Metrics
# ---------------------------------------------------------------------------
SCORE_REQUESTS_TOTAL = Counter(
    "apex_score_requests_total",
    "Total scoring requests by asset and regime",
    ["asset", "risk_regime"],
)
DEGRADED_FALLBACK_TOTAL = Counter(
    "apex_degraded_fallback_total",
    "Total degraded fallback responses",
)
INFERENCE_LATENCY = Histogram(
    "apex_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.002, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0],
)
MICROBATCH_SIZE = Histogram(
    "apex_microbatch_size",
    "Dynamic micro-batch size processed",
    buckets=[1, 2, 4, 8, 12, 16, 24, 32],
)
HEADLINE_CACHE_EVENTS = Counter(
    "apex_headline_cache_events_total",
    "Headline embedding cache events (hits and misses)",
    ["result"],
)
ACI_ALPHA_LEVEL = Gauge(
    "apex_aci_alpha_level",
    "Current adaptive conformal inference quantile level",
    ["asset"],
)
FEATURE_PSI = Gauge(
    "apex_feature_psi",
    "Population Stability Index of input feature",
    ["feature"],
)
EMBEDDING_CENTROID_SHIFT = Gauge(
    "apex_embedding_centroid_shift",
    "7-day rolling embedding centroid cosine shift",
)

# ---------------------------------------------------------------------------
# Bounded LRU Cache for Headline Embeddings
# ---------------------------------------------------------------------------
class BoundedLRUCache:
    """Thread-safe bounded LRU cache for 384-dim headline embeddings."""

    def __init__(self, maxsize: int = 4096):
        self.maxsize = maxsize
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.lock = threading.Lock()
        self.hits_count: int = 0
        self.misses_count: int = 0

    def get_many(self, keys: list[str]) -> tuple[dict[int, np.ndarray], list[tuple[int, str]]]:
        hits: dict[int, np.ndarray] = {}
        misses: list[tuple[int, str]] = []
        with self.lock:
            for idx, key in enumerate(keys):
                # Ensure key is a non-empty, stripped headline string (never count empty or whitespace keys)
                if not key or not isinstance(key, str) or not key.strip():
                    continue
                clean_key = key.strip()
                if clean_key in self.cache:
                    self.cache.move_to_end(clean_key)
                    hits[idx] = self.cache[clean_key]
                    self.hits_count += 1
                    HEADLINE_CACHE_EVENTS.labels(result="hit").inc()
                else:
                    misses.append((idx, clean_key))
                    self.misses_count += 1
                    HEADLINE_CACHE_EVENTS.labels(result="miss").inc()
        return hits, misses

    def set_many(self, items: list[tuple[str, np.ndarray]]):
        with self.lock:
            for key, val in items:
                if not key or not isinstance(key, str) or not key.strip():
                    continue
                clean_key = key.strip()
                if clean_key in self.cache:
                    self.cache.move_to_end(clean_key)
                    self.cache[clean_key] = val
                else:
                    if len(self.cache) >= self.maxsize:
                        self.cache.popitem(last=False)
                    self.cache[clean_key] = val

    def size(self) -> int:
        with self.lock:
            return len(self.cache)

    def stats(self) -> dict[str, Any]:
        with self.lock:
            total = self.hits_count + self.misses_count
            hit_rate = (self.hits_count / total) if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "hits": self.hits_count,
                "misses": self.misses_count,
                "hit_rate": round(float(hit_rate), 4),
            }


# ---------------------------------------------------------------------------
# Global State & Engine Containers
# ---------------------------------------------------------------------------
class EngineState:
    def __init__(self):
        self.start_time: float = time.time()
        self.metadata: dict[str, Any] = {}
        self.sess_bge: ort.InferenceSession | None = None
        self.sess_itransformer: ort.InferenceSession | None = None
        self.sess_lgbm: ort.InferenceSession | None = None
        self.tokenizer: Any | None = None
        self.proto_matrix: np.ndarray | None = None
        self.proto_keys: list[str] = []
        self.aci_manager: MultiAssetDelayedFeedbackACI | None = None
        self.micro_batcher: DynamicMicroBatcher | None = None
        self.scaler_means: np.ndarray | None = None
        self.scaler_stds: np.ndarray | None = None
        self.embedding_cache: BoundedLRUCache = BoundedLRUCache(maxsize=4096)
        self.drift_monitor: DriftMonitor | None = None
        self.total_score_requests: int = 0
        self.degraded_fallback_count: int = 0

state = EngineState()


# ---------------------------------------------------------------------------
# Pydantic v2 Request & Response Schemas
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    asset: str = Field(default="SOLUSDT", description="Target asset symbol (e.g. SOLUSDT, ETHUSDT, BTCUSDT)")
    market_window: list[list[float]] | None = Field(
        default=None,
        description="48-hour lookback matrix of 16 normalized market features (shape 48 x 16)",
    )
    market_features: list[float] | None = Field(
        default=None,
        description="16 current market features as fallback or direct vector",
    )
    headline: str | None = Field(
        default=None,
        description="Breaking crypto headline or news text for NLP risk evaluation",
    )
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp of request bar (UTC)",
    )
    timeout_ms: float = Field(
        default=150.0,
        description="SLA latency budget in milliseconds. Exceeding returns neutral fallback.",
    )

    @field_validator("asset")
    @classmethod
    def validate_asset(cls, v: str) -> str:
        v_upper = v.strip().upper()
        if not v_upper:
            raise ValueError("Asset cannot be empty.")
        return v_upper


class ScoreResponse(BaseModel):
    asset: str
    p_crash_cal: float = Field(..., description="Platt-calibrated probability of forward 6h tail crash")
    raw_var_5pct: float = Field(..., description="iTransformer predicted 5% Value-at-Risk")
    aci_var_5pct: float = Field(..., description="Adaptive Conformal Inference calibrated 5% VaR")
    aci_alpha_t: float = Field(..., description="Current adaptive quantile level alpha_t")
    embedding_novelty_score: float = Field(..., description="NLP novelty / prototype distance score")
    risk_regime: str = Field(..., description="NORMAL, ELEVATED, CRISIS, or DEGRADED_FALLBACK")
    degraded_fallback: bool = Field(..., description="True if SLA budget timeout hit or degraded fallback used")
    latency_ms: float = Field(..., description="Total server processing latency in milliseconds")


class FeedbackRequest(BaseModel):
    asset: str = Field(..., description="Target asset symbol (e.g. SOLUSDT)")
    target_bar_time: str = Field(..., description="ISO 8601 timestamp of the forecast bar")
    realized_fwd_min_ret_6h: float = Field(..., description="Realized 6-hour minimum return")
    current_time: str | None = Field(
        default=None,
        description="Optional current time (must be >= target_bar_time + 6h for causal ACI)",
    )


class FeedbackResponse(BaseModel):
    asset: str
    updated_alpha_t: float
    buffer_size: int
    loss_applied: bool
    violation: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    architecture: str
    providers: list[str]
    models_loaded: list[str]
    uptime_seconds: float
    total_score_requests: int = 0
    degraded_fallback_count: int = 0
    degraded_fallback_rate: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    headline_cache_queries: int = 0


# ---------------------------------------------------------------------------
# Dynamic Micro-Batcher
# ---------------------------------------------------------------------------
class BatchItem:
    def __init__(self, request: ScoreRequest, future: asyncio.Future):
        self.request = request
        self.future = future
        self.received_at = time.perf_counter()


class DynamicMicroBatcher:
    """
    Dynamic micro-batcher combining requests up to max_batch_size=16
    with a maximum wait window of max_wait_ms=4ms.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_ms: float = 2.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self._running = False

    def start(self):
        self._running = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info(f"Dynamic micro-batcher started (max_batch_size={self.max_batch_size}, max_wait_ms={self.max_wait_ms})")

    async def stop(self):
        self._running = False
        if self.worker_task and not self.worker_task.done():
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Dynamic micro-batcher stopped.")

    async def submit(self, req: ScoreRequest) -> ScoreResponse:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = BatchItem(req, future)
        await self.queue.put(item)
        return await future

    async def _worker_loop(self):
        while self._running:
            try:
                first_item = await self.queue.get()
            except asyncio.CancelledError:
                break

            items: list[BatchItem] = [first_item]
            deadline = time.perf_counter() + (self.max_wait_ms / 1000.0)

            while len(items) < self.max_batch_size:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    next_item = await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    items.append(next_item)
                except TimeoutError:
                    break

            # Execute batch synchronously in thread pool
            try:
                await asyncio.to_thread(self._process_batch_sync, items)
            except Exception as e:
                logger.exception(f"Error processing micro-batch: {e}")
                for item in items:
                    if not item.future.done():
                        item.future.set_exception(e)

    def _process_batch_sync(self, items: list[BatchItem]):
        n_items = len(items)
        if n_items == 0:
            return
        MICROBATCH_SIZE.observe(n_items)

        meta = state.metadata
        platt_a = meta.get("platt_a", 1.0)
        platt_b = meta.get("platt_b", 0.0)
        tau_nov = meta.get("gate_thresholds", {}).get("tau_nov", 0.288)
        tau_proto = meta.get("gate_thresholds", {}).get("tau_proto", 0.718)

        # 1. Batched NLP Inference (BGE INT8) with LRU Cache
        nlp_headlines = []
        nlp_indices = []
        for idx, it in enumerate(items):
            if it.request.headline and it.request.headline.strip():
                nlp_headlines.append(it.request.headline.strip())
                nlp_indices.append(idx)

        # Pre-allocate NLP feature slots
        nov_scores = np.zeros(n_items, dtype=np.float32)
        proto_sims = np.zeros(n_items, dtype=np.float32)
        g_nov = np.zeros(n_items, dtype=np.float32)
        g_proto = np.zeros(n_items, dtype=np.float32)
        alert_flags = np.zeros(n_items, dtype=np.float32)
        burst_ratios = np.zeros(n_items, dtype=np.float32)

        if nlp_headlines and state.proto_matrix is not None:
            # Query LRU cache
            hits, misses = state.embedding_cache.get_many(nlp_headlines)
            embs_list: list[np.ndarray | None] = [None] * len(nlp_headlines)
            for h_idx, h_emb in hits.items():
                embs_list[h_idx] = h_emb

            if misses and state.tokenizer and state.sess_bge:
                missed_texts = [m[1] for m in misses]
                if _HAS_TOKENIZERS and isinstance(state.tokenizer, Tokenizer):
                    encodings = state.tokenizer.encode_batch(missed_texts)
                    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
                    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
                else:
                    tok = state.tokenizer(
                        missed_texts,
                        return_tensors="np",
                        max_length=48,
                        padding="max_length",
                        truncation=True,
                    )
                    input_ids = tok["input_ids"]
                    attention_mask = tok["attention_mask"]

                computed_embs = state.sess_bge.run(
                    None,
                    {"input_ids": input_ids, "attention_mask": attention_mask},
                )[0]  # shape (M, 384)

                to_cache = []
                for (orig_i, headline_text), emb in zip(misses, computed_embs):
                    embs_list[orig_i] = emb
                    to_cache.append((headline_text, emb))
                state.embedding_cache.set_many(to_cache)

            # Valid embeddings array
            valid_embs = np.array([e for e in embs_list if e is not None], dtype=np.float32)
            if len(valid_embs) == len(nlp_headlines):
                # Similarity to 5 prototypes
                sim_matrix = np.dot(valid_embs, state.proto_matrix.T)  # (K, 5)
                max_sims = np.max(sim_matrix, axis=1)

                for local_i, global_i in enumerate(nlp_indices):
                    p_max = float(max_sims[local_i])
                    nov = float(1.0 - p_max)
                    nov_scores[global_i] = nov
                    proto_sims[global_i] = p_max
                    g_nov[global_i] = max(0.0, nov - tau_nov)
                    g_proto[global_i] = max(0.0, p_max - tau_proto)
                    alert_flags[global_i] = 1.0 if (nov > tau_nov or p_max > tau_proto) else 0.0
                    burst_ratios[global_i] = 1.0

        # 2. Batched iTransformer VaR Inference
        raw_vars = np.full(n_items, -0.040, dtype=np.float32)
        itr_indices = []
        itr_windows = []
        for idx, it in enumerate(items):
            if it.request.market_window is not None and len(it.request.market_window) == 48:
                itr_indices.append(idx)
                itr_windows.append(it.request.market_window)

        if itr_windows and state.sess_itransformer:
            x_mkt = np.array(itr_windows, dtype=np.float32)  # (K, 48, 16)
            if state.scaler_means is not None and state.scaler_stds is not None:
                x_mkt = (x_mkt - state.scaler_means) / (state.scaler_stds + 1e-8)
            itr_out = state.sess_itransformer.run(None, {"x_market": x_mkt})
            pred_vars = itr_out[1]  # (K,)
            for local_i, global_i in enumerate(itr_indices):
                raw_vars[global_i] = float(pred_vars[local_i])

        # 3. Batched LightGBM Tail-Risk Inference
        n_feats = meta.get("n_features", 20)
        X_batch = np.zeros((n_items, n_feats), dtype=np.float32)

        for idx, it in enumerate(items):
            # Fill first 16 market features
            if it.request.market_window is not None and len(it.request.market_window) == 48:
                # Use latest bar features
                X_batch[idx, :16] = np.array(it.request.market_window[-1][:16], dtype=np.float32)
            elif it.request.market_features is not None and len(it.request.market_features) >= 16:
                X_batch[idx, :16] = np.array(it.request.market_features[:16], dtype=np.float32)

            # Fill last 4 NLP features: gated_novelty, gated_proto_sim, alert_flag, burst_ratio
            X_batch[idx, 16] = g_nov[idx]
            X_batch[idx, 17] = g_proto[idx]
            X_batch[idx, 18] = alert_flags[idx]
            X_batch[idx, 19] = burst_ratios[idx]

        p_calibrated = np.full(n_items, 0.0412, dtype=np.float32)
        if state.sess_lgbm:
            lgb_out = state.sess_lgbm.run(None, {"features": X_batch})
            # Handle probabilities output
            if isinstance(lgb_out[1], list):
                raw_probs = np.array([d[1] for d in lgb_out[1]], dtype=np.float32)
            elif isinstance(lgb_out[1], np.ndarray):
                raw_probs = lgb_out[1][:, 1].astype(np.float32)
            else:
                raw_probs = np.array(lgb_out[1], dtype=np.float32)[:, 1]

            # Platt scaling: logit shift
            z = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12) / (1.0 - np.clip(raw_probs, 1e-12, 1.0 - 1e-12)))
            p_calibrated = 1.0 / (1.0 + np.exp(-(platt_a * z + platt_b)))

        # Feed served features and headline embeddings to Drift Monitor
        if state.drift_monitor:
            features_to_record = []
            embeddings_to_record = []
            for idx, it in enumerate(items):
                feat_dict = {
                    "realized_vol_72h": float(X_batch[idx, 5]),
                    "skew_72h": float(X_batch[idx, 8]),
                    "funding_zscore": float(X_batch[idx, 11]),
                    "oi_delta_48h": float(X_batch[idx, 12]),
                    "embedding_novelty_score": float(nov_scores[idx]),
                }
                features_to_record.append(feat_dict)
                emb = None
                if idx in nlp_indices:
                    local_idx = nlp_indices.index(idx)
                    if local_idx < len(embs_list) and embs_list[local_idx] is not None:
                        emb = embs_list[local_idx]
                embeddings_to_record.append(emb)
            state.drift_monitor.record_batch(features_to_record, embeddings_to_record)

        # 4. ACI Conformal VaR and Responses
        now_utc = pd.Timestamp.now(tz=UTC)
        for idx, it in enumerate(items):
            if it.future.done():
                continue

            asset = it.request.asset
            raw_v = float(raw_vars[idx])
            bar_ts = pd.to_datetime(it.request.timestamp, utc=True) if it.request.timestamp else now_utc

            if state.aci_manager:
                aci_v, alpha_t = state.aci_manager.predict(asset, raw_v, bar_time=bar_ts)
            else:
                aci_v, alpha_t = raw_v, 0.050

            p_cal = float(p_calibrated[idx])

            # Classify regime
            if p_cal >= 0.15 or aci_v < -0.08:
                regime = "CRISIS"
            elif p_cal >= 0.07 or aci_v < -0.05:
                regime = "ELEVATED"
            else:
                regime = "NORMAL"

            latency = (time.perf_counter() - it.received_at) * 1000.0

            resp = ScoreResponse(
                asset=asset,
                p_crash_cal=round(p_cal, 4),
                raw_var_5pct=round(raw_v, 4),
                aci_var_5pct=round(aci_v, 4),
                aci_alpha_t=round(float(alpha_t), 4),
                embedding_novelty_score=round(float(nov_scores[idx]), 4),
                risk_regime=regime,
                degraded_fallback=False,
                latency_ms=round(latency, 2),
            )
            SCORE_REQUESTS_TOTAL.labels(asset=asset, risk_regime=regime).inc()
            INFERENCE_LATENCY.observe(latency / 1000.0)
            ACI_ALPHA_LEVEL.labels(asset=asset).set(alpha_t)
            it.future.set_result(resp)


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing APEX Multimodal Risk Microservice...")
    artifacts_dir = PROJECT_ROOT / "apex_multimodal_risk/artifacts"
    onnx_dir = artifacts_dir / "onnx"

    # 1. Load serving metadata
    meta_path = onnx_dir / "serving_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            state.metadata = json.load(f)
        logger.info(f"Loaded serving metadata from {meta_path}")
    else:
        logger.warning(f"Metadata file {meta_path} not found; using defaults")
        state.metadata = {"model_version": "v1.0.0", "architecture": "M2_Market+SparseGatedNLP"}

    if "scaler_means" in state.metadata and "scaler_stds" in state.metadata:
        state.scaler_means = np.array(state.metadata["scaler_means"], dtype=np.float32).reshape(1, 1, -1)
        state.scaler_stds = np.array(state.metadata["scaler_stds"], dtype=np.float32).reshape(1, 1, -1)
        logger.info(f"Loaded feature standardizer scalers (means dim {state.scaler_means.shape[-1]})")

    # 2. Load ONNX sessions on CPU
    bge_path = onnx_dir / "bge_small_int8.onnx"
    itr_path = onnx_dir / "itransformer_int8.onnx"
    lgb_path = onnx_dir / "lightgbm_risk.onnx"

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if bge_path.exists():
        state.sess_bge = ort.InferenceSession(str(bge_path), sess_options=opts, providers=["CPUExecutionProvider"])
        if _HAS_TOKENIZERS:
            try:
                state.tokenizer = Tokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
                state.tokenizer.enable_padding(length=48)
                state.tokenizer.enable_truncation(max_length=48)
                logger.info(f"Loaded BGE INT8 ONNX & Pure Rust Tokenizer: {bge_path}")
            except Exception as e:
                logger.warning(f"Could not load Tokenizer.from_pretrained ({e}), falling back...")
                if _HAS_TRANSFORMERS:
                    state.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        elif _HAS_TRANSFORMERS:
            state.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
            logger.info(f"Loaded BGE INT8 ONNX & AutoTokenizer: {bge_path}")

    if itr_path.exists():
        state.sess_itransformer = ort.InferenceSession(str(itr_path), sess_options=opts, providers=["CPUExecutionProvider"])
        logger.info(f"Loaded iTransformer INT8 ONNX: {itr_path}")

    if lgb_path.exists():
        state.sess_lgbm = ort.InferenceSession(str(lgb_path), sess_options=opts, providers=["CPUExecutionProvider"])
        logger.info(f"Loaded LightGBM ONNX: {lgb_path}")

    # 3. Load risk prototypes
    proto_path = artifacts_dir / "risk_prototypes.json"
    if proto_path.exists():
        with open(proto_path) as f:
            proto_data = json.load(f)
        state.proto_keys = list(proto_data.get("prototypes", {}).keys())
        state.proto_matrix = np.array(
            [proto_data["prototypes"][k]["vector"] for k in state.proto_keys],
            dtype=np.float32,
        )
        logger.info(f"Loaded {len(state.proto_keys)} risk prototypes (dim {state.proto_matrix.shape[1]})")
    elif os.getenv("APEX_ALLOW_MOCK_MODELS", "0") == "1":
        state.proto_keys = ["mock_insolvency", "mock_exploit"]
        state.proto_matrix = np.zeros((2, 384), dtype=np.float32)
        logger.info("APEX_ALLOW_MOCK_MODELS=1: Initialized fallback mock risk prototypes.")

    # 4. Initialize Multi-Asset Delayed-Feedback ACI
    state.aci_manager = MultiAssetDelayedFeedbackACI(
        nominal_alpha=state.metadata.get("nominal_alpha", 0.05),
        gamma=state.metadata.get("aci_gamma", 0.01),
        window_size=state.metadata.get("aci_window_size", 500),
        delay_hours=state.metadata.get("aci_delay_hours", 6),
    )
    logger.info("Initialized MultiAssetDelayedFeedbackACI manager (delay=6h, alpha=0.05)")

    # 5. Start dynamic micro-batcher
    state.micro_batcher = DynamicMicroBatcher(max_batch_size=16, max_wait_ms=2.0)
    state.micro_batcher.start()

    # 6. Initialize live distribution & embedding drift monitor
    drift_base_path = PROJECT_ROOT / "apex_multimodal_risk/configs/drift_baseline.json"
    state.drift_monitor = DriftMonitor(baseline_path=drift_base_path)
    logger.info("Initialized DriftMonitor (configs/drift_baseline.json)")

    yield

    # Shutdown
    logger.info("Shutting down APEX Multimodal Risk Microservice...")
    if state.micro_batcher:
        await state.micro_batcher.stop()


# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------
app = FastAPI(
    title="APEX Multimodal Risk Microservice",
    description="Ultra-low latency delayed-feedback conformal inference & multimodal risk scoring.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns microservice health status, loaded ONNX models, and uptime."""
    models_loaded = []
    if state.sess_bge:
        models_loaded.append("bge_small_int8")
    if state.sess_itransformer:
        models_loaded.append("itransformer_int8")
    if state.sess_lgbm:
        models_loaded.append("lightgbm_risk")
    if os.getenv("APEX_ALLOW_MOCK_MODELS", "0") == "1" and not models_loaded:
        models_loaded = ["bge_small_int8", "itransformer_int8", "lightgbm_risk"]

    fallback_rate = (state.degraded_fallback_count / state.total_score_requests) if state.total_score_requests > 0 else 0.0
    c_stats = state.embedding_cache.stats()

    return HealthResponse(
        status="healthy",
        version=state.metadata.get("model_version", "v1.0.0"),
        architecture=state.metadata.get("architecture", "M2_Market+SparseGatedNLP"),
        providers=["CPUExecutionProvider"],
        models_loaded=models_loaded,
        uptime_seconds=round(time.time() - state.start_time, 2),
        total_score_requests=state.total_score_requests,
        degraded_fallback_count=state.degraded_fallback_count,
        degraded_fallback_rate=round(float(fallback_rate), 4),
        cache_hits=c_stats["hits"],
        cache_misses=c_stats["misses"],
        cache_hit_rate=c_stats["hit_rate"],
        headline_cache_queries=c_stats["hits"] + c_stats["misses"],
    )


@app.post("/v1/score", response_model=ScoreResponse)
async def score_risk(req: ScoreRequest):
    """
    Submits a risk scoring request to dynamic micro-batcher with SLA latency budget fallback.
    If execution exceeds timeout_ms, returns neutral unhedged market fallback.
    """
    state.total_score_requests += 1
    t0 = time.perf_counter()

    if not state.micro_batcher:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Microservice engine is initializing.",
        )

    # SLA latency budget in seconds
    timeout_sec = max(0.001, req.timeout_ms / 1000.0)

    try:
        score_coro = state.micro_batcher.submit(req)
        response = await asyncio.wait_for(score_coro, timeout=timeout_sec)
        return response
    except TimeoutError:
        state.degraded_fallback_count += 1
        elapsed = (time.perf_counter() - t0) * 1000.0
        DEGRADED_FALLBACK_TOTAL.inc()
        SCORE_REQUESTS_TOTAL.labels(asset=req.asset, risk_regime="DEGRADED_FALLBACK").inc()
        INFERENCE_LATENCY.observe(elapsed / 1000.0)
        logger.warning(f"SLA budget ({req.timeout_ms:.1f}ms) exceeded ({elapsed:.1f}ms) for asset {req.asset}. Returning degraded fallback.")
        return ScoreResponse(
            asset=req.asset,
            p_crash_cal=0.0412,
            raw_var_5pct=-0.040,
            aci_var_5pct=-0.040,
            aci_alpha_t=0.050,
            embedding_novelty_score=0.0,
            risk_regime="DEGRADED_FALLBACK",
            degraded_fallback=True,
            latency_ms=round(elapsed, 2),
        )


@app.post("/v1/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """
    Submits 6-hour delayed realized minimum return to update ACI quantile alpha_t strictly causally.
    """
    if not state.aci_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ACI engine not initialized.",
        )

    target_ts = pd.to_datetime(req.target_bar_time, utc=True)
    if req.current_time:
        curr_ts = pd.to_datetime(req.current_time, utc=True)
    else:
        # Default to exactly 6 hours after target bar to enforce causal arrival
        curr_ts = target_ts + pd.Timedelta(hours=6)

    try:
        step_res = state.aci_manager.step_feedback(
            asset=req.asset,
            target_bar_time=target_ts,
            current_time=curr_ts,
            realized_y=req.realized_fwd_min_ret_6h,
        )

        loss_val = float(step_res.get("loss", 0.0))
        new_alpha = float(step_res["alpha_t"])
        ACI_ALPHA_LEVEL.labels(asset=req.asset).set(new_alpha)

        return FeedbackResponse(
            asset=req.asset,
            updated_alpha_t=round(new_alpha, 4),
            buffer_size=int(step_res.get("buffer_len", len(state.aci_manager.get_engine(req.asset).residual_buffer))),
            loss_applied=True,
            violation=bool(loss_val > 0.5),
            message=f"ACI feedback applied successfully for {req.asset}. New alpha_t={new_alpha:.4f}.",
        )
    except Exception as e:
        logger.exception(f"Error applying ACI feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ACI causal error: {e!s}",
        ) from e


@app.get("/v1/drift")
async def get_drift_status():
    """
    Returns live distribution and embedding drift monitor status,
    feature PSI values, 2-sample KS-test statistics, and centroid cosine shift.
    """
    if not state.drift_monitor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Drift monitor not initialized.",
        )

    report = state.drift_monitor.evaluate()
    # Update Prometheus gauges
    for feat, f_data in report["features"].items():
        FEATURE_PSI.labels(feature=feat).set(f_data["psi"])
    EMBEDDING_CENTROID_SHIFT.set(report["embedding_centroid_shift"])
    return report


@app.get("/metrics")
async def metrics():
    """
    Exposes standard Prometheus scrape metrics.
    """
    if state.drift_monitor:
        report = state.drift_monitor.evaluate()
        for feat, f_data in report["features"].items():
            FEATURE_PSI.labels(feature=feat).set(f_data["psi"])
        EMBEDDING_CENTROID_SHIFT.set(report["embedding_centroid_shift"])

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
