"""
Unit tests for CPCV integrity and leak-free split generation.
Verifies:
1. Exactly 10 splits for N=5, k=2.
2. Zero timestamp overlap between train and test across all splits.
3. Minimum gap >= 504 hours between every train timestamp and every test timestamp.
4. Inner validation set is strictly disjoint and contained in training set.
"""

import numpy as np
import pandas as pd
import pytest

from apex_multimodal_risk.src.train.cpcv_runner import generate_cpcv_splits
from apex_multimodal_risk.src.utils import PROJECT_ROOT, read_parquet_utc


@pytest.fixture(scope="module")
def panel_timestamps():
    parquet_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/aligned_hourly.parquet"
    if parquet_path.exists():
        try:
            df = read_parquet_utc(parquet_path)
            return sorted(df["timestamp"].unique())
        except Exception:
            pass
    # Fallback for fresh-clone CI: generate synthetic hourly timestamps
    return list(pd.date_range("2021-01-01", periods=10000, freq="1h", tz="UTC"))


def test_cpcv_split_count_and_blocks(panel_timestamps):
    splits = generate_cpcv_splits(
        unique_timestamps=panel_timestamps,
        n_blocks=5,
        k_test=2,
        purge_embargo_hours=504,
    )
    assert len(splits) == 10, f"Expected 10 splits, got {len(splits)}"

    # Check that all test block combinations are unique
    test_combos = [s.test_block_ids for s in splits]
    assert len(test_combos) == len(set(test_combos)), "Duplicate test block combinations found"


def test_zero_timestamp_overlap(panel_timestamps):
    splits = generate_cpcv_splits(
        unique_timestamps=panel_timestamps,
        n_blocks=5,
        k_test=2,
        purge_embargo_hours=504,
    )
    for s in splits:
        train_set = set(s.train_timestamps)
        test_set = set(s.test_timestamps)
        overlap = train_set.intersection(test_set)
        assert len(overlap) == 0, f"Split {s.split_id} has {len(overlap)} overlapping timestamps!"


def test_minimum_504_hour_gap(panel_timestamps):
    splits = generate_cpcv_splits(
        unique_timestamps=panel_timestamps,
        n_blocks=5,
        k_test=2,
        purge_embargo_hours=504,
    )
    for s in splits:
        assert (
            s.min_gap_hours >= 504.0
        ), f"Split {s.split_id} minimum gap {s.min_gap_hours} < 504.0 hours!"

        # Exhaustive verification of pairwise distances
        test_secs = (
            pd.to_datetime(s.test_timestamps, utc=True)
            .tz_convert("UTC")
            .tz_localize(None)
            .astype("datetime64[s]")
            .astype("int64")
            .to_numpy()
        )
        train_secs = (
            pd.to_datetime(s.train_timestamps, utc=True)
            .tz_convert("UTC")
            .tz_localize(None)
            .astype("datetime64[s]")
            .astype("int64")
            .to_numpy()
        )

        pos = np.searchsorted(test_secs, train_secs)
        for i, p in enumerate(pos):
            if p < len(test_secs):
                gap_h = abs(train_secs[i] - test_secs[p]) / 3600.0
                assert gap_h >= 504.0, f"Split {s.split_id} right-gap {gap_h}h < 504h"
            if p > 0:
                gap_h = abs(train_secs[i] - test_secs[p - 1]) / 3600.0
                assert gap_h >= 504.0, f"Split {s.split_id} left-gap {gap_h}h < 504h"


def test_inner_val_disjoint_and_contained(panel_timestamps):
    splits = generate_cpcv_splits(
        unique_timestamps=panel_timestamps,
        n_blocks=5,
        k_test=2,
        purge_embargo_hours=504,
        inner_val_ratio=0.15,
    )
    for s in splits:
        it_set = set(s.inner_train_timestamps)
        iv_set = set(s.inner_val_timestamps)
        tr_set = set(s.train_timestamps)
        te_set = set(s.test_timestamps)

        # Disjoint inner train and inner val
        assert len(it_set.intersection(iv_set)) == 0, f"Split {s.split_id} inner train/val overlap!"
        # Both strictly contained in train
        assert it_set.issubset(tr_set), f"Split {s.split_id} inner_train not subset of train"
        assert iv_set.issubset(tr_set), f"Split {s.split_id} inner_val not subset of train"
        # Zero overlap with test
        assert len(iv_set.intersection(te_set)) == 0, f"Split {s.split_id} inner_val overlaps test!"
