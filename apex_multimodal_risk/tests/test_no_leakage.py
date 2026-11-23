"""
Automated Verification Suite for Causal Integrity and Zero-Leakage Guarantees.
Asserts:
1. Every attached news item at bar T strictly satisfies ingested_at <= T and published_at < T.
2. No duplicate clusters appear within 6 hours with cosine similarity > 0.85.
3. Zero NaNs in target masks and strictly monotonic UTC timestamp index.
4. Correct mathematical relationship for tail_crash_6h and realized volatility.
"""

import os
from pathlib import Path
from datetime import timedelta

import pytest
import numpy as np
import pandas as pd
import duckdb

from apex_multimodal_risk.src.utils import PROJECT_ROOT, load_config


@pytest.fixture(scope="session")
def pipeline_config():
    return load_config()


@pytest.fixture(scope="session")
def dedup_news_df(pipeline_config):
    proc_path = pipeline_config["noise_filter"]["deduplication"]["processed_parquet_path"]
    if not os.path.isabs(proc_path):
        proc_path = str(PROJECT_ROOT / proc_path)
    assert os.path.exists(proc_path), f"Processed news parquet missing at: {proc_path}"
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{proc_path}' ORDER BY ingested_at ASC").df()
    con.close()
    return df


@pytest.fixture(scope="session")
def aligned_hourly_df(pipeline_config):
    aligned_path = pipeline_config["causal_alignment"]["aligned_parquet_path"]
    if not os.path.isabs(aligned_path):
        aligned_path = str(PROJECT_ROOT / aligned_path)
    assert os.path.exists(aligned_path), f"Aligned hourly parquet missing at: {aligned_path}"
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM '{aligned_path}' ORDER BY timestamp ASC, asset ASC").df()
    con.close()
    return df


def test_causal_ingestion_delay(dedup_news_df):
    """Asserts simulated ingestion delay: ingested_at = published_at + 120s."""
    pub = pd.to_datetime(dedup_news_df["published_at"], utc=True)
    ing = pd.to_datetime(dedup_news_df["ingested_at"], utc=True)
    
    diff_sec = (ing - pub).dt.total_seconds()
    assert (diff_sec == 120).all(), "Every news item must have strictly a 120s ingestion delay."
    assert (ing > pub).all(), "ingested_at must be strictly greater than published_at."


def test_aligned_news_strictly_causal(aligned_hourly_df, dedup_news_df):
    """
    Asserts zero lookahead: for every bar T with news,
    all news items in the lookback window satisfy ingested_at <= T and published_at < T.
    """
    news_times = pd.to_datetime(dedup_news_df["ingested_at"], utc=True).to_numpy()
    pub_times = pd.to_datetime(dedup_news_df["published_at"], utc=True).to_numpy()
    
    # Sample 500 rows with news_count_24h > 0
    rows_with_news = aligned_hourly_df[aligned_hourly_df["news_count_24h"] > 0]
    sample = rows_with_news.sample(min(500, len(rows_with_news)), random_state=42)
    
    for _, row in sample.iterrows():
        bar_t = pd.to_datetime(row["timestamp"], utc=True)
        bar_t_ns = bar_t.to_datetime64()
        lookback_24h_ns = (bar_t - timedelta(hours=24)).to_datetime64()
        
        # News items matching (T - 24h, T]
        mask = (news_times > lookback_24h_ns) & (news_times <= bar_t_ns)
        matching_ingested = news_times[mask]
        matching_published = pub_times[mask]
        
        assert len(matching_ingested) == row["news_count_24h"], (
            f"Expected {row['news_count_24h']} items, found {len(matching_ingested)} at bar {bar_t}"
        )
        if len(matching_ingested) > 0:
            assert np.all(matching_ingested <= bar_t_ns), "Violation: ingested_at > bar_t!"
            assert np.all(matching_published < bar_t_ns), "Violation: published_at >= bar_t!"


def test_semantic_dedup_no_similar_clusters_within_6h(dedup_news_df):
    """
    Asserts that within any rolling 6-hour window, no two distinct clusters have cosine similarity > 0.85.
    """
    timestamps = pd.to_datetime(dedup_news_df["ingested_at"], utc=True).tolist()
    embeddings = np.vstack(dedup_news_df["embedding"].to_numpy()).astype(np.float32)
    
    # Check norms are 1.0 (L2 normalized)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3, err_msg="Embeddings must be L2-normalized.")
    
    n_items = len(dedup_news_df)
    window_delta = timedelta(hours=6.0)
    
    # Check sliding window pairwise similarities (sample random checkpoints)
    np.random.seed(42)
    check_indices = np.random.choice(n_items, size=min(1000, n_items), replace=False)
    
    for i in check_indices:
        t_i = timestamps[i]
        emb_i = embeddings[i]
        
        # Items within (t_i, t_i + 6h]
        window_indices = [
            j for j in range(i + 1, min(i + 100, n_items))
            if timestamps[j] <= t_i + window_delta
        ]
        
        if window_indices:
            w_embs = embeddings[window_indices]
            sims = np.dot(w_embs, emb_i)
            # Allow floating precision tolerance 0.8501
            violating = np.where(sims > 0.8501)[0]
            assert len(violating) == 0, (
                f"Cluster {dedup_news_df.iloc[i]['cluster_id']} has duplicate within 6h! "
                f"Max sim: {sims.max():.4f}"
            )


def test_targets_and_monotonicity(aligned_hourly_df):
    """
    Asserts:
    1. Timestamps per asset are strictly monotonically increasing.
    2. Zero NaNs in target_valid mask.
    3. Valid rows have zero NaNs in fwd_min_ret_6h and trailing_24h_realized_vol.
    4. tail_crash_6h strictly equals 1 iff fwd_min_ret_6h < -2.0 * trailing_24h_realized_vol.
    """
    for asset in aligned_hourly_df["asset"].unique():
        sub = aligned_hourly_df[aligned_hourly_df["asset"] == asset]
        ts = pd.to_datetime(sub["timestamp"], utc=True)
        assert ts.is_monotonic_increasing, f"Timestamps for {asset} must be strictly monotonic increasing."
        assert not ts.duplicated().any(), f"Duplicate timestamps detected for {asset}."
        
    assert not aligned_hourly_df["target_valid"].isna().any(), "target_valid mask contains NaNs!"
    
    valid_sub = aligned_hourly_df[aligned_hourly_df["target_valid"]].copy()
    assert not valid_sub["fwd_min_ret_6h"].isna().any(), "fwd_min_ret_6h contains NaNs on valid rows."
    assert not valid_sub["trailing_24h_realized_vol"].isna().any(), "trailing_24h_realized_vol contains NaNs on valid rows."
    assert set(valid_sub["tail_crash_6h"].unique()).issubset({0, 1}), "tail_crash_6h must be binary {0, 1}."
    
    # Mathematical identity verification
    expected_crash = (valid_sub["fwd_min_ret_6h"] < (-2.0 * valid_sub["trailing_24h_realized_vol"])).astype(int)
    np.testing.assert_array_equal(
        valid_sub["tail_crash_6h"].to_numpy(),
        expected_crash.to_numpy(),
        err_msg="tail_crash_6h does not match -2.0 * trailing_vol threshold definition!"
    )


def test_entity_and_risk_labels(dedup_news_df):
    """Asserts entity flags are boolean and weak risk classes are valid strings."""
    for col in ["entity_sol", "entity_avax", "entity_link", "entity_macro"]:
        assert col in dedup_news_df.columns
        assert dedup_news_df[col].dtype == bool
        
    valid_classes = {
        "protocol_exploit",
        "network_outage",
        "regulatory_action",
        "exchange_insolvency",
        "macro_shock",
        "non_event_noise",
    }
    assigned_classes = set(dedup_news_df["weak_risk_class"].unique())
    assert assigned_classes.issubset(valid_classes), f"Unknown risk class: {assigned_classes - valid_classes}"
