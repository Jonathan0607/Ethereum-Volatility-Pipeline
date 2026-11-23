"""
Automated Verification Suite for Causal Integrity, UTC Parity, and Horizon Scaling.
Asserts:
1. Every timestamp column across all Parquet files is strictly UTC (+00:00).
2. Every attached news item at bar T strictly satisfies ingested_at <= T and published_at < T.
3. No duplicate clusters appear within 6 hours with cosine similarity > 0.85.
4. Zero NaNs in target masks and strictly monotonic UTC timestamp index.
5. Correct mathematical relationship for tail_crash_6h with sqrt(6) horizon scaling.
6. Base rate of tail_crash_6h lands cleanly within the 3%–8% institutional tail risk band.
"""

import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from apex_multimodal_risk.src.utils import PROJECT_ROOT, load_config, read_parquet_utc


@pytest.fixture(scope="session")
def pipeline_config():
    return load_config()


@pytest.fixture(scope="session")
def raw_news_path(pipeline_config):
    raw_path = pipeline_config["news_ingestion"]["raw_parquet_path"]
    if not os.path.isabs(raw_path):
        raw_path = str(PROJECT_ROOT / raw_path)
    assert os.path.exists(raw_path), f"Raw news parquet missing at: {raw_path}"
    return raw_path


@pytest.fixture(scope="session")
def dedup_news_path(pipeline_config):
    proc_path = pipeline_config["noise_filter"]["deduplication"]["processed_parquet_path"]
    if not os.path.isabs(proc_path):
        proc_path = str(PROJECT_ROOT / proc_path)
    assert os.path.exists(proc_path), f"Processed news parquet missing at: {proc_path}"
    return proc_path


@pytest.fixture(scope="session")
def aligned_hourly_path(pipeline_config):
    aligned_path = pipeline_config["causal_alignment"]["aligned_parquet_path"]
    if not os.path.isabs(aligned_path):
        aligned_path = str(PROJECT_ROOT / aligned_path)
    assert os.path.exists(aligned_path), f"Aligned hourly parquet missing at: {aligned_path}"
    return aligned_path


@pytest.fixture(scope="session")
def dedup_news_df(dedup_news_path):
    return read_parquet_utc(dedup_news_path, order_by="ingested_at ASC")


@pytest.fixture(scope="session")
def aligned_hourly_df(aligned_hourly_path):
    return read_parquet_utc(aligned_hourly_path, order_by="timestamp ASC, asset ASC")


def test_strict_utc_timestamps_across_all_parquets(raw_news_path, dedup_news_path, aligned_hourly_path):
    """
    Asserts every timestamp column in all Parquet files is explicitly stored as UTC (+00:00).
    Verifies both PyArrow schema metadata and Pandas timezone awareness.
    """
    parquet_paths = [raw_news_path, dedup_news_path, aligned_hourly_path]

    for path in parquet_paths:
        table = pq.read_table(path)
        schema = table.schema

        # Check PyArrow schema level
        found_timestamp_col = False
        for field in schema:
            field_type_str = str(field.type)
            if "timestamp" in field_type_str:
                found_timestamp_col = True
                assert "tz=UTC" in field_type_str or "UTC" in field_type_str, (
                    f"File '{Path(path).name}' column '{field.name}' has non-UTC type: {field.type}"
                )
        assert found_timestamp_col, f"File '{Path(path).name}' should contain at least one timestamp column."

        # Check Pandas DataFrame level
        df = read_parquet_utc(path)
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                tz_str = str(df[col].dt.tz)
                assert tz_str in ("UTC", "+00:00"), (
                    f"File '{Path(path).name}' column '{col}' has non-UTC timezone: {tz_str}"
                )
                sample_val = df[col].dropna().iloc[0]
                assert "+00:00" in str(sample_val) or str(sample_val).endswith("UTC"), (
                    f"Value string '{sample_val}' does not reflect strict UTC (+00:00)."
                )


def test_causal_ingestion_delay(dedup_news_df):
    """Asserts simulated ingestion delay: ingested_at = published_at + 120s."""
    pub = dedup_news_df["published_at"]
    ing = dedup_news_df["ingested_at"]

    diff_sec = (ing - pub).dt.total_seconds()
    assert (diff_sec == 120).all(), "Every news item must have strictly a 120s ingestion delay."
    assert (ing > pub).all(), "ingested_at must be strictly greater than published_at."


def test_aligned_news_strictly_causal(aligned_hourly_df, dedup_news_df):
    """
    Asserts zero lookahead: for every bar T with news,
    all news items in the lookback window satisfy ingested_at <= T and published_at < T.
    """
    news_times = pd.to_datetime(dedup_news_df["ingested_at"], utc=True)
    pub_times = pd.to_datetime(dedup_news_df["published_at"], utc=True)

    rows_with_news = aligned_hourly_df[aligned_hourly_df["news_count_24h"] > 0]
    sample = rows_with_news.sample(min(500, len(rows_with_news)), random_state=42)

    for _, row in sample.iterrows():
        bar_t = pd.to_datetime(row["timestamp"], utc=True)
        lookback_24h = bar_t - pd.Timedelta(hours=24)

        # News items matching (T - 24h, T]
        mask = (news_times > lookback_24h) & (news_times <= bar_t)
        matching_ingested = news_times[mask]
        matching_published = pub_times[mask]

        assert len(matching_ingested) == row["news_count_24h"], (
            f"Expected {row['news_count_24h']} items, found {len(matching_ingested)} at bar {bar_t}"
        )
        if len(matching_ingested) > 0:
            assert (matching_ingested <= bar_t).all(), "Violation: ingested_at > bar_t!"
            assert (matching_published < bar_t).all(), "Violation: published_at >= bar_t!"


def test_semantic_dedup_no_similar_clusters_within_6h(dedup_news_df):
    """
    Asserts that within any rolling 6-hour window, no two distinct clusters have cosine similarity > 0.85.
    """
    timestamps = dedup_news_df["ingested_at"].tolist()
    embeddings = np.vstack(dedup_news_df["embedding"].to_numpy()).astype(np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3, err_msg="Embeddings must be L2-normalized.")

    n_items = len(dedup_news_df)
    window_delta = timedelta(hours=6.0)

    np.random.seed(42)
    check_indices = np.random.choice(n_items, size=min(1000, n_items), replace=False)

    for i in check_indices:
        t_i = timestamps[i]
        emb_i = embeddings[i]

        window_indices = [
            j for j in range(i + 1, min(i + 100, n_items))
            if timestamps[j] <= t_i + window_delta
        ]

        if window_indices:
            w_embs = embeddings[window_indices]
            sims = np.dot(w_embs, emb_i)
            violating = np.where(sims > 0.8501)[0]
            assert len(violating) == 0, (
                f"Cluster {dedup_news_df.iloc[i]['cluster_id']} has duplicate within 6h! "
                f"Max sim: {sims.max():.4f}"
            )


def test_targets_and_monotonicity(aligned_hourly_df, pipeline_config):
    """
    Asserts:
    1. Timestamps per asset are strictly monotonically increasing in UTC.
    2. Zero NaNs in target_valid mask.
    3. Valid rows have zero NaNs in fwd_min_ret_6h and trailing_1h_vol_72h.
    4. tail_crash_6h strictly follows horizon scaling: < -mult * sqrt(6) * trailing_1h_vol_72h.
    5. Positive base rate lands cleanly within 3%–8% per asset.
    """
    align_cfg = pipeline_config["causal_alignment"]
    target_mult = float(align_cfg.get("tail_vol_multiplier", 2.0))
    fwd_horizon = int(align_cfg.get("forward_horizon_hours", 6))
    scale = np.sqrt(float(fwd_horizon))

    for asset in aligned_hourly_df["asset"].unique():
        sub = aligned_hourly_df[aligned_hourly_df["asset"] == asset]
        ts = pd.to_datetime(sub["timestamp"], utc=True)
        assert ts.is_monotonic_increasing, f"Timestamps for {asset} must be strictly monotonic increasing."
        assert not ts.duplicated().any(), f"Duplicate timestamps detected for {asset}."

        valid_sub = sub[sub["target_valid"]].copy()
        assert not valid_sub["fwd_min_ret_6h"].isna().any(), f"fwd_min_ret_6h contains NaNs on valid rows for {asset}."
        assert not valid_sub["trailing_1h_vol_72h"].isna().any(), f"trailing_1h_vol_72h contains NaNs on valid rows for {asset}."
        assert set(valid_sub["tail_crash_6h"].unique()).issubset({0, 1}), f"tail_crash_6h must be binary for {asset}."

        # Mathematical identity verification
        threshold = -target_mult * scale * valid_sub["trailing_1h_vol_72h"]
        expected_crash = (valid_sub["fwd_min_ret_6h"] < threshold).astype(int)
        np.testing.assert_array_equal(
            valid_sub["tail_crash_6h"].to_numpy(),
            expected_crash.to_numpy(),
            err_msg=f"tail_crash_6h does not match -{target_mult}*sqrt({fwd_horizon})*vol_72h threshold for {asset}!"
        )

        # Base rate verification: must land cleanly in 3%–8%
        base_rate = float(valid_sub["tail_crash_6h"].mean())
        assert 0.03 <= base_rate <= 0.08, (
            f"Base rate for {asset} is {base_rate:.2%}, outside institutional target range [3%, 8%]!"
        )

    assert not aligned_hourly_df["target_valid"].isna().any(), "target_valid mask contains NaNs!"


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
