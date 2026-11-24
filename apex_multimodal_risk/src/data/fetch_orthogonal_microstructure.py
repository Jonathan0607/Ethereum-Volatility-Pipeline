#!/usr/bin/env python3
"""
fetch_orthogonal_microstructure.py
==================================
Fetches orthogonal Binance USD-M derivatives microstructure feeds across the 15-coin panel
for the full 4.17-year history (2022-06-01 to 2026-08-31 UTC, 37,272 hourly bars per coin).

Feeds:
  Feed A: Hourly Perp-vs-Spot Basis (monthly/premiumIndexKlines)
          -> premium_close, premium_high, premium_low
  Feed B: Open Interest & Whale-vs-Retail Positioning (daily/metrics)
          -> sum_open_interest, sum_open_interest_value, count_toptrader_long_short_ratio,
             sum_toptrader_long_short_ratio, count_long_short_ratio, sum_taker_long_short_vol_ratio

Outputs:
  - apex_multimodal_risk/data/raw/binance_basis/{SYMBOL}.parquet (resumable)
  - apex_multimodal_risk/data/raw/binance_metrics/{SYMBOL}.parquet (resumable)
  - apex_multimodal_risk/data/processed/universe_15_microstructure_4yr.parquet (gitignored)
"""

import asyncio
import io
import time
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "apex_multimodal_risk" / "data"
RAW_BASIS_DIR = DATA_DIR / "raw" / "binance_basis"
RAW_METRICS_DIR = DATA_DIR / "raw" / "binance_metrics"
PROCESSED_DIR = DATA_DIR / "processed"
BASE_PANEL_PATH = PROCESSED_DIR / "universe_15_hourly_4yr.parquet"
OUTPUT_PANEL_PATH = PROCESSED_DIR / "universe_15_microstructure_4yr.parquet"

SYMBOLS = [
    "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "DOTUSDT", "ETCUSDT", "ETHUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT",
]

START_DATE = datetime(2022, 6, 1, 0, 0, 0, tzinfo=UTC)
END_DATE = datetime(2026, 8, 31, 23, 0, 0, tzinfo=UTC)
EXPECTED_HOURLY_BARS = 37272  # 1553 days * 24h


def generate_year_months(start_dt: datetime, end_dt: datetime) -> list[str]:
    months = []
    cur = datetime(start_dt.year, start_dt.month, 1, tzinfo=UTC)
    end_month = datetime(end_dt.year, end_dt.month, 1, tzinfo=UTC)
    while cur <= end_month:
        months.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = datetime(cur.year + 1, 1, 1, tzinfo=UTC)
        else:
            cur = datetime(cur.year, cur.month + 1, 1, tzinfo=UTC)
    return months


def generate_dates(start_dt: datetime, end_dt: datetime) -> list[str]:
    dates = []
    cur = datetime(start_dt.year, start_dt.month, start_dt.day, tzinfo=UTC)
    end_day = datetime(end_dt.year, end_dt.month, end_dt.day, tzinfo=UTC)
    while cur <= end_day:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


async def fetch_basis_for_symbol(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    symbol: str,
    months: list[str],
) -> pd.DataFrame:
    """Fetches monthly premiumIndexKlines for a single symbol."""
    cache_path = RAW_BASIS_DIR / f"{symbol}.parquet"
    if cache_path.exists():
        print(f"[{symbol}] Basis feed cached at {cache_path}, loading...", flush=True)
        return pd.read_parquet(cache_path)

    print(f"[{symbol}] Downloading {len(months)} monthly basis files...", flush=True)
    frames = []

    async def fetch_month(ym: str):
        url = f"https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/{symbol}/1h/{symbol}-1h-{ym}.zip"
        async with sem:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            df_m = pd.read_csv(f, header=None)
                            # Handle potential header
                            if df_m.iloc[0, 0] == "open_time" or str(df_m.iloc[0, 0]).startswith("open"):
                                df_m = df_m.iloc[1:].copy()
                            # Columns: 0=open_time, 1=open, 2=high, 3=low, 4=close
                            df_sub = pd.DataFrame({
                                "timestamp": pd.to_datetime(df_m.iloc[:, 0].astype(int), unit="ms", utc=True),
                                "premium_high": df_m.iloc[:, 2].astype(float),
                                "premium_low": df_m.iloc[:, 3].astype(float),
                                "premium_close": df_m.iloc[:, 4].astype(float),
                            })
                            frames.append(df_sub)
            except Exception as e:
                print(f"[{symbol}] Error fetching basis {ym}: {e}", flush=True)

    tasks = [fetch_month(ym) for ym in months]
    await asyncio.gather(*tasks)

    if not frames:
        raise RuntimeError(f"[{symbol}] No basis data downloaded!")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= START_DATE) & (df["timestamp"] <= END_DATE)]
    df["asset"] = symbol

    # Forward fill missing within limit
    full_idx = pd.date_range(START_DATE, END_DATE, freq="1h", tz=UTC)
    df = df.set_index("timestamp").reindex(full_idx)
    df["asset"] = symbol
    df[["premium_high", "premium_low", "premium_close"]] = df[["premium_high", "premium_low", "premium_close"]].ffill(limit=24)
    # Neutral fill for any leading NaNs
    df[["premium_high", "premium_low", "premium_close"]] = df[["premium_high", "premium_low", "premium_close"]].fillna(0.0)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[{symbol}] Saved {len(df)} basis rows to {cache_path}", flush=True)
    return df


async def fetch_metrics_for_symbol(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    symbol: str,
    dates: list[str],
) -> pd.DataFrame:
    """Fetches daily metrics for a single symbol, aggregating to hourly bars."""
    cache_path = RAW_METRICS_DIR / f"{symbol}.parquet"
    if cache_path.exists():
        print(f"[{symbol}] Metrics feed cached at {cache_path}, loading...", flush=True)
        return pd.read_parquet(cache_path)

    print(f"[{symbol}] Downloading {len(dates)} daily metrics files...", flush=True)
    hourly_frames = []
    completed = 0
    t0 = time.time()

    async def fetch_day(d: str):
        nonlocal completed
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{d}.zip"
        async with sem:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with z.open(z.namelist()[0]) as f:
                            df_day = pd.read_csv(f)
                            df_day["create_time"] = pd.to_datetime(df_day["create_time"], utc=True)
                            df_day["timestamp"] = df_day["create_time"].dt.floor("1h")
                            # Hourly causal aggregation:
                            # Stocks (OI, ratios) -> last snapshot <= T
                            # Flow ratio (sum_taker_long_short_vol_ratio) -> mean across hour
                            agg_dict = {
                                "sum_open_interest": "last",
                                "sum_open_interest_value": "last",
                                "count_toptrader_long_short_ratio": "last",
                                "sum_toptrader_long_short_ratio": "last",
                                "count_long_short_ratio": "last",
                                "sum_taker_long_short_vol_ratio": "mean",
                            }
                            # Ensure columns exist
                            for col in agg_dict:
                                if col not in df_day.columns:
                                    df_day[col] = float("nan")
                                else:
                                    df_day[col] = pd.to_numeric(df_day[col], errors="coerce")

                            h_df = df_day.groupby("timestamp", as_index=False)[list(agg_dict.keys())].agg(agg_dict)
                            hourly_frames.append(h_df)
            except Exception:
                # Missing or error day will be handled by causal ffill
                pass
            finally:
                completed += 1
                if completed % 200 == 0 or completed == len(dates):
                    elapsed = time.time() - t0
                    print(f"[{symbol}] Metrics progress: {completed}/{len(dates)} days ({elapsed:.1f}s, {completed/elapsed:.1f} d/s)", flush=True)

    tasks = [fetch_day(d) for d in dates]
    await asyncio.gather(*tasks)

    if not hourly_frames:
        raise RuntimeError(f"[{symbol}] Zero metrics days loaded!")

    df = pd.concat(hourly_frames, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = df[(df["timestamp"] >= START_DATE) & (df["timestamp"] <= END_DATE)]

    # Align against full hourly index
    full_idx = pd.date_range(START_DATE, END_DATE, freq="1h", tz=UTC)
    raw_count = len(df)
    df = df.set_index("timestamp").reindex(full_idx)
    df["asset"] = symbol

    metric_cols = [
        "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    coverage = raw_count / EXPECTED_HOURLY_BARS * 100.0
    print(f"[{symbol}] Metrics Raw Coverage: {raw_count}/{EXPECTED_HOURLY_BARS} bars ({coverage:.2f}%)", flush=True)

    # Strictly causal forward fill: max 24h gap
    df[metric_cols] = df[metric_cols].ffill(limit=24)
    # Neutral defaults for any leading warmup before first snapshot
    defaults = {
        "sum_open_interest": 0.0,
        "sum_open_interest_value": 0.0,
        "count_toptrader_long_short_ratio": 1.0,
        "sum_toptrader_long_short_ratio": 1.0,
        "count_long_short_ratio": 1.0,
        "sum_taker_long_short_vol_ratio": 1.0,
    }
    df = df.fillna(value=defaults)
    df = df.reset_index().rename(columns={"index": "timestamp"})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"[{symbol}] Saved {len(df)} metrics rows to {cache_path}", flush=True)
    return df


async def main():
    RAW_BASIS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not BASE_PANEL_PATH.exists():
        raise FileNotFoundError(f"Base panel {BASE_PANEL_PATH} not found! Run extend_universe_4yr.py first.")

    print("=" * 80)
    print("EXP-014: FETCH ORTHOGONAL BINANCE DERIVATIVES MICROSTRUCTURE")
    print(f"Universe: {len(SYMBOLS)} assets across {START_DATE} to {END_DATE}")
    print("=" * 80, flush=True)

    months = generate_year_months(START_DATE, END_DATE)
    dates = generate_dates(START_DATE, END_DATE)

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=50)
    sem = asyncio.Semaphore(25)

    basis_dfs = []
    metrics_dfs = []

    async with httpx.AsyncClient(timeout=15.0, limits=limits) as client:
        # Phase 1: Basis (Monthly files, fast ~750 files total)
        print("\n--- PHASE 1: Perp-vs-Spot Basis (monthly/premiumIndexKlines) ---", flush=True)
        t_b0 = time.time()
        for sym in SYMBOLS:
            b_df = await fetch_basis_for_symbol(client, sem, sym, months)
            basis_dfs.append(b_df)
        print(f"Phase 1 complete in {time.time() - t_b0:.1f}s", flush=True)

        # Phase 2: Daily Metrics (~23,295 files total, cached per symbol)
        print("\n--- PHASE 2: Open Interest & Whale/Retail Positioning (daily/metrics) ---", flush=True)
        t_m0 = time.time()
        for sym in SYMBOLS:
            m_df = await fetch_metrics_for_symbol(client, sem, sym, dates)
            metrics_dfs.append(m_df)
        print(f"Phase 2 complete in {time.time() - t_m0:.1f}s", flush=True)

    print("\n--- PHASE 3: Merging into 4.17-Year Microstructure Panel ---", flush=True)
    all_basis = pd.concat(basis_dfs, ignore_index=True)
    all_metrics = pd.concat(metrics_dfs, ignore_index=True)

    base_panel = pd.read_parquet(BASE_PANEL_PATH)
    print(f"Base panel shape: {base_panel.shape}", flush=True)

    # Merge Basis
    merged = pd.merge(
        base_panel,
        all_basis[["timestamp", "asset", "premium_close", "premium_high", "premium_low"]],
        on=["timestamp", "asset"],
        how="left",
    )

    # Merge Metrics
    metric_cols = [
        "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    ]
    merged = pd.merge(
        merged,
        all_metrics[["timestamp", "asset"] + metric_cols],
        on=["timestamp", "asset"],
        how="left",
    )

    # Verify no row expansion
    assert len(merged) == len(base_panel), f"Row count mismatch! {len(merged)} vs {len(base_panel)}"
    assert not merged["timestamp"].isnull().any(), "Found null timestamps in merged panel!"

    # Forward fill any residual edge NaNs
    micro_cols = ["premium_close", "premium_high", "premium_low"] + metric_cols
    merged[micro_cols] = merged.groupby("asset")[micro_cols].ffill(limit=24)
    # Fill defaults
    merged[micro_cols] = merged[micro_cols].fillna(0.0)

    print("\nFinal Merged Microstructure Panel Verification:")
    print(f"Shape: {merged.shape}")
    print(f"Columns: {merged.columns.tolist()}")
    print(f"Date range: {merged['timestamp'].min()} to {merged['timestamp'].max()}")
    print("Null count per column:")
    print(merged.isnull().sum())

    merged.to_parquet(OUTPUT_PANEL_PATH, index=False)
    file_size_mb = OUTPUT_PANEL_PATH.stat().st_size / (1024 * 1024)
    print(f"\nSuccessfully wrote {OUTPUT_PANEL_PATH} ({file_size_mb:.2f} MB)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
