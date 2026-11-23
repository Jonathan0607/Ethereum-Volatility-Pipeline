"""
Combinatorial Purged Cross-Validation (CPCV) Runner.
Implements Marcos Lopez de Prado's CPCV framework with:
- N = 5 contiguous time blocks
- k = 2 test blocks (C(5,2) = 10 combinatorial splits)
- Strict 504-hour (21-day) purge and 504-hour embargo around every test boundary
- Inner validation split for leakage-free temperature scaling and hyperparameter tuning
- Zero timestamp overlap and min_distance >= 504 hours mathematically guaranteed
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("cpcv_runner")


@dataclass
class CPCVSplit:
    split_id: int
    test_block_ids: Tuple[int, int]
    train_timestamps: np.ndarray
    inner_train_timestamps: np.ndarray
    inner_val_timestamps: np.ndarray
    test_timestamps: np.ndarray
    min_gap_hours: float
    test_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    train_intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]
    purge_embargo_td: pd.Timedelta = pd.Timedelta(hours=504)

    def get_train_mask(self, df_timestamps: pd.Series) -> np.ndarray:
        ts_set = set(self.train_timestamps)
        return df_timestamps.isin(ts_set).to_numpy()

    def get_inner_train_mask(self, df_timestamps: pd.Series) -> np.ndarray:
        ts_set = set(self.inner_train_timestamps)
        return df_timestamps.isin(ts_set).to_numpy()

    def get_inner_val_mask(self, df_timestamps: pd.Series) -> np.ndarray:
        ts_set = set(self.inner_val_timestamps)
        return df_timestamps.isin(ts_set).to_numpy()

    def get_test_mask(self, df_timestamps: pd.Series) -> np.ndarray:
        ts_set = set(self.test_timestamps)
        return df_timestamps.isin(ts_set).to_numpy()

    def get_news_test_mask(self, news_ts: pd.Series) -> np.ndarray:
        """Interval-based mask for continuous news events within test blocks."""
        mask = np.zeros(len(news_ts), dtype=bool)
        for start, end in self.test_intervals:
            mask = mask | ((news_ts >= start) & (news_ts <= end)).to_numpy()
        return mask

    def get_news_train_mask(self, news_ts: pd.Series) -> np.ndarray:
        """Interval-based mask for continuous news events in train blocks outside purge/embargo."""
        in_train = np.zeros(len(news_ts), dtype=bool)
        for start, end in self.train_intervals:
            in_train = in_train | ((news_ts >= start) & (news_ts <= end)).to_numpy()
        in_pe = np.zeros(len(news_ts), dtype=bool)
        for start, end in self.test_intervals:
            in_pe = in_pe | (
                (news_ts >= start - self.purge_embargo_td)
                & (news_ts <= end + self.purge_embargo_td)
            ).to_numpy()
        return in_train & (~in_pe)


def generate_cpcv_splits(
    unique_timestamps: Sequence[pd.Timestamp],
    n_blocks: int = 5,
    k_test: int = 2,
    purge_embargo_hours: int = 504,
    inner_val_ratio: float = 0.15,
) -> List[CPCVSplit]:
    """
    Generates N choose k CPCV splits with strict purge and embargo buffers.
    Guarantees min |t_train - t_test| >= purge_embargo_hours.
    """
    ts_array = np.array(sorted(unique_timestamps))
    n_total = len(ts_array)
    if n_total == 0:
        raise ValueError("unique_timestamps cannot be empty.")

    block_indices = np.array_split(np.arange(n_total), n_blocks)
    blocks = []
    for b_id, idx in enumerate(block_indices):
        blocks.append({
            "block_id": b_id,
            "indices": idx,
            "start": ts_array[idx[0]],
            "end": ts_array[idx[-1]],
            "timestamps": ts_array[idx],
        })

    purge_embargo_td = pd.Timedelta(hours=purge_embargo_hours)
    test_combinations = list(itertools.combinations(range(n_blocks), k_test))

    splits: List[CPCVSplit] = []

    for split_id, test_block_ids in enumerate(test_combinations):
        # 1. Test timestamps
        test_ts_parts = [blocks[b]["timestamps"] for b in test_block_ids]
        test_ts = np.concatenate(test_ts_parts)
        test_ts = np.array(sorted(test_ts))

        # 2. Candidate train timestamps from remaining blocks
        train_block_ids = [b for b in range(n_blocks) if b not in test_block_ids]
        train_cand_parts = [blocks[b]["timestamps"] for b in train_block_ids]
        train_cand_ts = np.concatenate(train_cand_parts)
        train_cand_ts = np.array(sorted(train_cand_ts))

        # 3. Purge & Embargo mask
        valid_train_mask = np.ones(len(train_cand_ts), dtype=bool)
        for b in test_block_ids:
            b_start = blocks[b]["start"]
            b_end = blocks[b]["end"]
            # Purge before test start: [b_start - purge_embargo_td, b_start)
            # Embargo after test end: (b_end, b_end + purge_embargo_td]
            in_purge_or_embargo = (train_cand_ts >= b_start - purge_embargo_td) & (
                train_cand_ts <= b_end + purge_embargo_td
            )
            valid_train_mask = valid_train_mask & (~in_purge_or_embargo)

        train_ts = train_cand_ts[valid_train_mask]

        # 4. Verify absolute minimum distance
        test_secs = pd.to_datetime(test_ts).astype("int64").to_numpy() // 10**9
        train_secs = pd.to_datetime(train_ts).astype("int64").to_numpy() // 10**9

        pos = np.searchsorted(test_secs, train_secs)
        diffs = []
        for i, p in enumerate(pos):
            cand = []
            if p < len(test_secs):
                cand.append(abs(train_secs[i] - test_secs[p]))
            if p > 0:
                cand.append(abs(train_secs[i] - test_secs[p - 1]))
            diffs.append(min(cand))
        min_dist_hours = min(diffs) / 3600.0

        if min_dist_hours < purge_embargo_hours:
            raise RuntimeError(
                f"Split {split_id} violated purge/embargo minimum distance: "
                f"{min_dist_hours:.2f}h < {purge_embargo_hours}h"
            )

        # 5. Partition train into inner_train and inner_val (last inner_val_ratio slice)
        val_size = int(len(train_ts) * inner_val_ratio)
        if val_size < 1:
            val_size = 1
        inner_train_ts = train_ts[:-val_size]
        inner_val_ts = train_ts[-val_size:]

        # Intervals
        test_intervals = [(blocks[b]["start"], blocks[b]["end"]) for b in test_block_ids]
        train_intervals = [(blocks[b]["start"], blocks[b]["end"]) for b in train_block_ids]

        splits.append(
            CPCVSplit(
                split_id=split_id,
                test_block_ids=test_block_ids,
                train_timestamps=train_ts,
                inner_train_timestamps=inner_train_ts,
                inner_val_timestamps=inner_val_ts,
                test_timestamps=test_ts,
                min_gap_hours=min_dist_hours,
                test_intervals=test_intervals,
                train_intervals=train_intervals,
                purge_embargo_td=purge_embargo_td,
            )
        )

    logger.info(
        f"Generated {len(splits)} CPCV splits (N={n_blocks}, k={k_test}, buffer={purge_embargo_hours}h). "
        f"Min gap across all splits: {min(s.min_gap_hours for s in splits):.1f}h"
    )
    return splits
