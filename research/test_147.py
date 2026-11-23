import os, sys, numpy as np, pandas as pd
from apex_v22_comprehensive_4yr import build_multi_asset_dataset, compute_indicators, cross_weights, run_directional, run_spread, run_breakout, fast_metrics, UNIVERSE, EVAL_BARS_4YR, WARMUP_BARS
dataset = build_multi_asset_dataset(UNIVERSE, total_bars=EVAL_BARS_4YR)
processed = compute_indicators(dataset, UNIVERSE)
first_df = processed[UNIVERSE[0]].iloc[WARMUP_BARS:]
eval_index = first_df.index
n = len(eval_index)
weights = cross_weights(processed, UNIVERSE, eval_index)
d_arr = run_directional(processed, weights, UNIVERSE, eval_index, n, 0.75)
s_arr = run_spread(processed, UNIVERSE, eval_index, n, 0.40)
b_arr = run_breakout(processed, weights, UNIVERSE, n, 0.25)
ens = 0.15 * d_arr + 0.60 * s_arr + 0.25 * b_arr
sh, ret, dd = fast_metrics(ens)
print(f"Test v22 config directly: Sharpe={sh:.2f}, Net={ret:.2f}%, MaxDD={dd:.2f}%")
