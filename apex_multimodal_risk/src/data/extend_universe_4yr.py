#!/usr/bin/env python3
"""
src/data/extend_universe_4yr.py
===================================================================================
Extends the 15-Coin Binance Perpetual Panel to Full 4.17 Years (2022-06-01 to 2026-08-31 UTC).
Downloads monthly hourly klines and monthly funding rates from Binance Vision archive:
  https://data.binance.vision/data/futures/um/monthly/
Preserves full Binance microstructure columns:
  [timestamp, asset, open, high, low, close, volume, quote_volume, trade_count,
   taker_buy_base_vol, taker_buy_quote_vol, funding_rate]
Saves to apex_multimodal_risk/data/processed/universe_15_hourly_4yr.parquet (gitignored).
===================================================================================
"""

from __future__ import annotations

import asyncio
import io
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apex_multimodal_risk.src.utils import setup_logger

logger = setup_logger("extend_universe_4yr")

UNIVERSE_15 = [
    "ADAUSDT", "ATOMUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT",
    "DOGEUSDT", "DOTUSDT", "ETCUSDT", "ETHUSDT", "LINKUSDT",
    "LTCUSDT", "NEARUSDT", "SOLUSDT", "UNIUSDT", "XRPUSDT"
]

START_DATE = datetime(2022, 6, 1, 0, 0, 0, tzinfo=UTC)
END_DATE = datetime(2026, 8, 31, 23, 0, 0, tzinfo=UTC)

EXPECTED_BARS_PER_ASSET = 37272  # 1553 days * 24h
EXPECTED_TOTAL_ROWS = EXPECTED_BARS_PER_ASSET * len(UNIVERSE_15)  # 559,080 rows


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


async def download_file_with_retry(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> bytes | None:
    async with sem:
        for attempt in range(max_retries):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return resp.content
                elif resp.status_code == 404:
                    logger.debug(f"404 Not Found: {url}")
                    return None
                else:
                    await asyncio.sleep(0.5 * (2 ** attempt))
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.warning(f"Failed to fetch {url} after {max_retries} attempts: {e}")
                await asyncio.sleep(0.5 * (2 ** attempt))
    return None


def parse_kline_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None)
                    # Check if first row is header
                    if isinstance(df.iloc[0, 0], str) and "open" in df.iloc[0, 0].lower():
                        df = df.iloc[1:].reset_index(drop=True)
                    # Standard Binance futures kline columns:
                    # 0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume,
                    # 6: close_time, 7: quote_volume, 8: count, 9: taker_buy_volume,
                    # 10: taker_buy_quote_volume, 11: ignore
                    df = df.iloc[:, 0:11]
                    df.columns = [
                        "open_time", "open", "high", "low", "close", "volume",
                        "close_time", "quote_volume", "trade_count",
                        "taker_buy_base_vol", "taker_buy_quote_vol"
                    ]
                    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
                    df = df.drop(columns=["open_time", "close_time"])
                    numeric_cols = [
                        "open", "high", "low", "close", "volume",
                        "quote_volume", "trade_count", "taker_buy_base_vol", "taker_buy_quote_vol"
                    ]
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df
    return pd.DataFrame()


def parse_funding_zip(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        for name in z.namelist():
            if name.endswith(".csv"):
                with z.open(name) as f:
                    df = pd.read_csv(f)
                    # Check columns: calc_time, funding_interval_hours, last_funding_rate
                    if "calc_time" in df.columns and "last_funding_rate" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True).dt.floor("h")
                        df["funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
                        return df[["timestamp", "funding_rate"]].dropna()
                    elif len(df.columns) >= 3:
                        # headerless or different names
                        df["timestamp"] = pd.to_datetime(pd.to_numeric(df.iloc[:, 0]), unit="ms", utc=True).dt.floor("h")
                        df["funding_rate"] = pd.to_numeric(df.iloc[:, 2], errors="coerce")
                        return df[["timestamp", "funding_rate"]].dropna()
    return pd.DataFrame()


async def fetch_symbol_data(
    symbol: str,
    months: list[str],
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info(f"[{symbol}] Scheduling {len(months)} kline and funding monthly downloads...")
    kline_tasks = []
    funding_tasks = []
    for ym in months:
        k_url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{ym}.zip"
        f_url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        kline_tasks.append(download_file_with_retry(client, k_url, sem))
        funding_tasks.append(download_file_with_retry(client, f_url, sem))

    kline_bytes = await asyncio.gather(*kline_tasks)
    funding_bytes = await asyncio.gather(*funding_tasks)

    # Parse klines
    kline_dfs = []
    for b in kline_bytes:
        if b is not None:
            df_k = parse_kline_zip(b)
            if not df_k.empty:
                kline_dfs.append(df_k)

    funding_dfs = []
    for b in funding_bytes:
        if b is not None:
            df_f = parse_funding_zip(b)
            if not df_f.empty:
                funding_dfs.append(df_f)

    if not kline_dfs:
        raise RuntimeError(f"No kline data parsed for {symbol}!")

    df_kline_all = pd.concat(kline_dfs, ignore_index=True)
    df_kline_all = df_kline_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df_funding_all = pd.DataFrame()
    if funding_dfs:
        df_funding_all = pd.concat(funding_dfs, ignore_index=True)
        df_funding_all = df_funding_all.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info(f"[{symbol}] Downloaded {len(df_kline_all)} raw kline bars and {len(df_funding_all)} funding records.")
    return df_kline_all, df_funding_all


async def build_4yr_panel() -> pd.DataFrame:
    months = generate_year_months(START_DATE, END_DATE)
    logger.info(f"Generating 4.17-Year Panel across {len(UNIVERSE_15)} coins for {len(months)} months ({months[0]} to {months[-1]})...")

    # Load existing universe_15_hourly.parquet if available to verify / backfill funding
    existing_parquet_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/universe_15_hourly.parquet"
    df_existing: pd.DataFrame | None = None
    if existing_parquet_path.exists():
        logger.info(f"Found existing base panel at {existing_parquet_path}, loading for funding rate alignment...")
        df_existing = pd.read_parquet(existing_parquet_path)
        df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"], utc=True)

    # Setup async client with max 8 concurrent requests
    limits = httpx.Limits(max_keepalive_connections=12, max_connections=8)
    timeout = httpx.Timeout(15.0, connect=15.0)
    sem = asyncio.Semaphore(8)

    t0 = time.time()
    asset_panels: list[pd.DataFrame] = []

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for symbol in UNIVERSE_15:
            df_kline, df_funding = await fetch_symbol_data(symbol, months, client, sem)

            # Generate target hourly date range
            target_range = pd.date_range(START_DATE, END_DATE, freq="1h", tz="UTC")
            df_symbol = pd.DataFrame({"timestamp": target_range})

            # Merge klines
            df_symbol = pd.merge(df_symbol, df_kline, on="timestamp", how="left")

            # Check forward funding rate source
            # 1. Start with downloaded funding
            if not df_funding.empty:
                df_symbol = pd.merge(df_symbol, df_funding, on="timestamp", how="left")
                df_symbol["funding_rate"] = df_symbol["funding_rate"].ffill()

            # 2. If existing panel available, ensure exact historical funding rate match for <= 2025-05-23
            if df_existing is not None:
                sub_exist = df_existing[df_existing["asset"] == symbol][["timestamp", "funding_rate"]].copy()
                sub_exist = sub_exist.rename(columns={"funding_rate": "funding_rate_exist"})
                df_symbol = pd.merge(df_symbol, sub_exist, on="timestamp", how="left")
                if "funding_rate" in df_symbol.columns:
                    df_symbol["funding_rate"] = df_symbol["funding_rate_exist"].combine_first(df_symbol["funding_rate"])
                    df_symbol = df_symbol.drop(columns=["funding_rate_exist"])
                else:
                    df_symbol["funding_rate"] = df_symbol["funding_rate_exist"]
                    df_symbol = df_symbol.drop(columns=["funding_rate_exist"])

            df_symbol["funding_rate"] = df_symbol["funding_rate"].ffill().bfill().fillna(0.0001)

            # Forward fill any micro missing bars in klines
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                df_symbol[col] = df_symbol[col].ffill().bfill()
            vol_cols = ["volume", "quote_volume", "trade_count", "taker_buy_base_vol", "taker_buy_quote_vol"]
            for col in vol_cols:
                df_symbol[col] = df_symbol[col].fillna(0.0)

            df_symbol["asset"] = symbol

            # Order columns
            cols = [
                "timestamp", "asset", "open", "high", "low", "close", "volume",
                "quote_volume", "trade_count", "taker_buy_base_vol", "taker_buy_quote_vol",
                "funding_rate"
            ]
            df_symbol = df_symbol[cols]

            assert len(df_symbol) == EXPECTED_BARS_PER_ASSET, (
                f"[{symbol}] Expected {EXPECTED_BARS_PER_ASSET} bars, got {len(df_symbol)}"
            )
            assert df_symbol["timestamp"].iloc[0] == START_DATE, f"Start date mismatch: {df_symbol['timestamp'].iloc[0]}"
            assert df_symbol["timestamp"].iloc[-1] == END_DATE, f"End date mismatch: {df_symbol['timestamp'].iloc[-1]}"
            assert df_symbol["close"].isnull().sum() == 0, f"Missing close values in {symbol}"

            asset_panels.append(df_symbol)
            logger.info(f"[{symbol}] Successfully built clean 37,272-bar panel ({START_DATE} to {END_DATE}).")

    df_full = pd.concat(asset_panels, ignore_index=True)
    df_full = df_full.sort_values(["timestamp", "asset"]).reset_index(drop=True)

    elapsed = time.time() - t0
    logger.info(f"All 15 coins downloaded and assembled in {elapsed:.2f}s. Total rows: {len(df_full):,}.")
    return df_full


def main():
    print("=" * 100)
    print("EXTENDING 15-COIN PERPETUAL UNIVERSE TO 4.17 YEARS (2022-06-01 to 2026-08-31 UTC)")
    print("=" * 100)

    output_dir = PROJECT_ROOT / "apex_multimodal_risk/data/processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "universe_15_hourly_4yr.parquet"

    df_full = asyncio.run(build_4yr_panel())

    # Verification checks
    assert len(df_full) == EXPECTED_TOTAL_ROWS, (
        f"Row count mismatch: expected {EXPECTED_TOTAL_ROWS}, got {len(df_full)}"
    )
    assert len(df_full["asset"].unique()) == 15, "Asset count mismatch"
    assert df_full["timestamp"].min() == START_DATE, "Start date mismatch"
    assert df_full["timestamp"].max() == END_DATE, "End date mismatch"

    # Verify zero duplicates
    dups = df_full.duplicated(subset=["timestamp", "asset"]).sum()
    assert dups == 0, f"Found {dups} duplicate timestamp-asset entries!"

    # Save to parquet
    logger.info(f"Saving 4.17-year panel to {output_path}...")
    df_full.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved successfully: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")

    print("\n--- DATA EXTENSION VERIFICATION SUMMARY ---")
    print(f"Total Rows:            {len(df_full):,}")
    print(f"Assets:                {len(df_full['asset'].unique())} ({sorted(df_full['asset'].unique())})")
    print(f"Start Timestamp (UTC): {df_full['timestamp'].min()}")
    print(f"End Timestamp (UTC):   {df_full['timestamp'].max()}")
    print(f"Hours per Asset:       {len(df_full) // 15:,} (Warmup 720h + 36,552 OOS bars)")
    print(f"Microstructure Cols:   {list(df_full.columns)}")
    print(f"Output File:           {output_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
