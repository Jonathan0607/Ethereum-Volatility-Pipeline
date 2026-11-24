#!/usr/bin/env python3
"""
Save a pinned snapshot of the multi-asset dataset so that all future
comparisons use the EXACT same data window.
"""
import os, sys, pickle, warnings
import pandas as pd
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research', 'archive'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'research'))
sys.path.insert(0, PROJECT_ROOT)

from multi_asset_ablation import build_multi_asset_dataset

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
WARMUP_BARS = 720
EVAL_BARS_4YR = 35040 + WARMUP_BARS

dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)

# Trim to exactly 35760 bars from the START to match what we had before
common_idx = dataset[UNIVERSE[0]].index
for s in UNIVERSE[1:]:
    common_idx = common_idx.intersection(dataset[s].index)
common_idx = common_idx.sort_values()

# Take the FIRST 35760 bars (anchoring to the earliest available data)
target_bars = EVAL_BARS_4YR
if len(common_idx) > target_bars:
    common_idx = common_idx[:target_bars]

for s in UNIVERSE:
    dataset[s] = dataset[s].loc[common_idx].copy()

out_path = os.path.join(PROJECT_ROOT, 'data', 'pinned_dataset_4yr.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(dataset, f)

print(f"\n[SAVED] Pinned dataset to {out_path}")
print(f"  Start: {common_idx[0]}")
print(f"  End:   {common_idx[-1]}")
print(f"  Bars per symbol: {len(common_idx)}")
