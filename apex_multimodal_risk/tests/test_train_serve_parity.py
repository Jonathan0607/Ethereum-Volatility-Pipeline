"""
Strict Online Train-Serve Feature Parity Test.
Verifies that incremental stateful feature calculator `OnlineFeatureState`:
1. Matches 16 causal market features against offline Pandas matrix within < 1e-6 max abs diff.
2. Matches 4 causal gated NLP features against offline matrix within < 1e-5 max abs diff.
3. Scoring both online and offline 20-dim feature vectors through LightGBM ONNX + Platt calibration
   matches p_crash_cal within < 1e-4 max abs diff across 500 consecutive historical bars.
"""

import json

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest

from apex_multimodal_risk.src.features.shared_features import (
    MARKET_NUMERICAL_COLS,
    OnlineFeatureState,
)
from apex_multimodal_risk.src.utils import PROJECT_ROOT, read_parquet_utc


@pytest.fixture(scope="module")
def setup_parity_data():
    from apex_multimodal_risk.src.train.exp002e_nlp_angles import engineer_nlp_features_and_targets
    eval_panel, _ = engineer_nlp_features_and_targets()
    news_df = read_parquet_utc("apex_multimodal_risk/data/processed/news_dedup.parquet")
    news_df = news_df.sort_values("ingested_at").reset_index(drop=True)

    meta_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/onnx/serving_metadata.json"
    with open(meta_path) as f:
        serving_meta = json.load(f)

    onnx_path = PROJECT_ROOT / "apex_multimodal_risk/artifacts/onnx/lightgbm_risk.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    return eval_panel, news_df, serving_meta, sess


@pytest.mark.requires_local_artifacts
def test_train_serve_feature_and_scoring_parity(setup_parity_data):
    """
    Feeds 500 consecutive historical bars step-by-step through OnlineFeatureState.
    Asserts strict numerical tolerances on market features, NLP features, and ONNX p_crash_cal.
    """
    eval_panel, news_df, serving_meta, sess = setup_parity_data

    tau_nov = serving_meta["gate_thresholds"]["tau_nov"]
    tau_proto = serving_meta["gate_thresholds"]["tau_proto"]
    platt_a = serving_meta["platt_a"]
    platt_b = serving_meta["platt_b"]

    ts_list = sorted(eval_panel["timestamp"].unique())
    assert len(ts_list) >= 1500, f"Expected at least 1500 unique timestamps, found {len(ts_list)}"

    # 750 bars for warmup, followed by 500 consecutive bars for parity audit
    warmup_ts = ts_list[250:1000]
    test_ts = ts_list[1000:1500]
    assert len(test_ts) == 500, f"Expected 500 test bars, got {len(test_ts)}"

    state = OnlineFeatureState()
    warm_df = eval_panel[eval_panel["timestamp"].isin(warmup_ts)]
    state.warmup(warm_df, news_df)

    news_secs = news_df["ingested_at"].to_numpy(dtype="datetime64[s]").astype("int64")
    warmup_end_sec = int(pd.to_datetime(warmup_ts[-1], utc=True).timestamp())
    prev_sec = warmup_end_sec

    max_mkt_diff = 0.0
    max_nlp_diff = 0.0
    max_p_diff = 0.0
    total_checked = 0

    for ts in test_ts:
        ts_sec = int(pd.to_datetime(ts, utc=True).timestamp())
        s_idx = np.searchsorted(news_secs, prev_sec, side="right")
        e_idx = np.searchsorted(news_secs, ts_sec, side="right")

        arriving_news = []
        if e_idx > s_idx:
            sub_n = news_df.iloc[s_idx:e_idx]
            for _, nr in sub_n.iterrows():
                sec_v = int(pd.to_datetime(nr["ingested_at"], utc=True).timestamp())
                arriving_news.append(
                    {
                        "timestamp_sec": sec_v,
                        "embedding": nr["embedding"],
                        "entity_sol": int(nr.get("entity_sol", 0)),
                        "entity_avax": int(nr.get("entity_avax", 0)),
                        "entity_link": int(nr.get("entity_link", 0)),
                        "entity_macro": int(nr.get("entity_macro", 0)),
                    }
                )
        prev_sec = ts_sec

        sub = eval_panel[eval_panel["timestamp"] == ts]
        bdict = {
            r["asset"]: {
                "close": float(r["close"]),
                "funding_rate": float(r.get("funding_rate", 0.0)),
                "open_interest": float(r.get("open_interest", r["close"] * 1000.0)),
                "garch_cond_vol": float(r["garch_cond_vol"]) if "garch_cond_vol" in r else None,
            }
            for _, r in sub.iterrows()
        }
        vecs = state.step(ts, bdict, news_items=arriving_news)

        for _, r in sub.iterrows():
            sym = r["asset"]
            online_vec = vecs[sym]  # (20,)

            # Build offline 20-dim feature vector
            off_mkt = r[MARKET_NUMERICAL_COLS].to_numpy(dtype=np.float32)
            nov = float(r["embedding_novelty_score"])
            p_max = float(r["max_proto_sim_24h"])
            g_nov = max(0.0, nov - tau_nov)
            g_proto = max(0.0, p_max - tau_proto)
            alert = 1.0 if (nov > tau_nov or p_max > tau_proto) else 0.0
            burst = float(r["news_burst_ratio"])
            off_vec = np.concatenate([off_mkt, [g_nov, g_proto, alert, burst]]).astype(np.float32)

            mkt_diff = float(np.max(np.abs(online_vec[:16] - off_vec[:16])))
            max_mkt_diff = max(max_mkt_diff, mkt_diff)

            nlp_diff = float(np.max(np.abs(online_vec[16:] - off_vec[16:])))
            max_nlp_diff = max(max_nlp_diff, nlp_diff)

            # Score both through LightGBM ONNX + Platt
            batch_x = np.stack([online_vec, off_vec], axis=0)
            lgb_out = sess.run(None, {"features": batch_x})
            raw_probs = lgb_out[1][:, 1].astype(np.float32)
            z = np.log(np.clip(raw_probs, 1e-12, 1.0 - 1e-12) / (1.0 - np.clip(raw_probs, 1e-12, 1.0 - 1e-12)))
            p_cal = 1.0 / (1.0 + np.exp(-(platt_a * z + platt_b)))
            p_diff = float(abs(p_cal[0] - p_cal[1]))
            max_p_diff = max(max_p_diff, p_diff)
            total_checked += 1

    print(f"\nChecked {total_checked} total asset-bars across 500 consecutive hours:")
    print(f"Max 16-Market Feature Abs Diff: {max_mkt_diff:.10e} (threshold < 1e-6)")
    print(f"Max 4-Gated NLP Feature Abs Diff: {max_nlp_diff:.10e} (threshold < 1e-5)")
    print(f"Max Calibrated p_crash Abs Diff:  {max_p_diff:.10e} (threshold < 1e-4)")

    assert max_mkt_diff < 1e-6, f"16 market features exceeded 1e-6 tolerance: {max_mkt_diff}"
    assert max_nlp_diff < 1e-5, f"NLP gated features exceeded 1e-5 tolerance: {max_nlp_diff}"
    assert max_p_diff < 1e-4, f"p_crash_cal exceeded 1e-4 tolerance: {max_p_diff}"


def test_online_feature_state_synthetic_math():
    """Verify OnlineFeatureState executes step() on synthetic OHLCV bars without local artifacts."""
    universe = ["SOLUSDT", "AVAXUSDT", "LINKUSDT"]
    state = OnlineFeatureState(universe=universe)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=50, freq="1h", tz="UTC")
    prices = {"SOLUSDT": 100.0, "AVAXUSDT": 30.0, "LINKUSDT": 15.0}

    for t in dates:
        bdict = {}
        for sym in universe:
            ret = rng.normal(0, 0.01)
            prices[sym] *= np.exp(ret)
            bdict[sym] = {
                "close": prices[sym],
                "open": prices[sym] * 0.999,
                "high": prices[sym] * 1.005,
                "low": prices[sym] * 0.995,
                "volume": float(rng.uniform(1e5, 5e5)),
                "funding_rate": float(rng.normal(0.0001, 0.0002)),
                "open_interest": float(rng.uniform(1e7, 2e7)),
                "garch_cond_vol": 2.0,
            }
        vecs = state.step(t, bdict)
        assert len(vecs) == len(universe)
        for sym in universe:
            assert vecs[sym].shape == (20,)
            assert not np.isnan(vecs[sym]).any()
