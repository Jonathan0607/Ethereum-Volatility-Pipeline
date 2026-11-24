"""
Universe 15 Market Data Ingestion Module.
Downloads hourly OHLCV klines and 8h funding rates directly from Binance Vision monthly archives
for the 15 liquid perpetuals:
BTCUSDT, ETHUSDT, SOLUSDT, AVAXUSDT, LINKUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LTCUSDT,
UNIUSDT, ATOMUSDT, NEARUSDT, DOTUSDT, ETCUSDT.
Date window: 2022-06-01 00:00:00 to 2025-05-23 23:00:00 UTC.
Saves panel dataset to apex_multimodal_risk/data/processed/universe_15_hourly.parquet.
"""

import io
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from apex_multimodal_risk.src.utils import PROJECT_ROOT, enforce_utc_columns, setup_logger

logger = setup_logger("fetch_universe_15")

TARGET_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "UNIUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "DOTUSDT",
    "ETCUSDT",
]

START_DATE = datetime(2022, 6, 1)
END_DATE = datetime(2025, 5, 23, 23, 59, 59)


def generate_month_tuples(start_date: datetime, end_date: datetime) -> list[str]:
    """Generates YYYY-MM strings for the target date range."""
    ym_list = []
    curr = start_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_limit = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while curr <= end_limit:
        ym_list.append(curr.strftime("%Y-%m"))
        if curr.month == 12:
            curr = curr.replace(year=curr.year + 1, month=1)
        else:
            curr = curr.replace(month=curr.month + 1)
    return ym_list


def download_zip_csv(url: str, session: requests.Session | None = None) -> pd.DataFrame | None:
    """Downloads a Binance Vision zip archive and reads its internal CSV."""
    s = session or requests.Session()
    try:
        resp = s.get(url, timeout=12)
        if resp.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for fname in z.namelist():
                if fname.endswith(".csv"):
                    with z.open(fname) as f:
                        return pd.read_csv(f)
    except Exception as e:
        logger.debug(f"Failed to fetch {url}: {e}")
        return None
    return None


def fetch_symbol_klines(symbol: str, ym_list: list[str], max_workers: int = 10) -> pd.DataFrame:
    """Fetches all monthly 1h klines for a given symbol."""
    urls = [
        f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/1h/{symbol}-1h-{ym}.zip"
        for ym in ym_list
    ]

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_zip_csv, url): url for url in urls}
        for future in as_completed(future_map):
            df = future.result()
            if df is not None and not df.empty:
                # Handle possible headerless CSVs or named columns
                if "open_time" not in df.columns:
                    df = df.iloc[:, 0:6]
                    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
                else:
                    df = df[["open_time", "open", "high", "low", "close", "volume"]]
                dfs.append(df)

    if not dfs:
        logger.warning(f"No klines retrieved for {symbol}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["open_time"].astype(np.int64), unit="ms", utc=True)
    combined = combined.drop(columns=["open_time"])
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = combined[col].astype(np.float64)

    # Sort and deduplicate by timestamp
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.set_index("timestamp")
    return combined


def fetch_symbol_funding(symbol: str, ym_list: list[str], max_workers: int = 10) -> pd.DataFrame:
    """Fetches all monthly funding rates for a given symbol."""
    urls = [
        f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        for ym in ym_list
    ]

    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_zip_csv, url): url for url in urls}
        for future in as_completed(future_map):
            df = future.result()
            if df is not None and not df.empty:
                if "calc_time" in df.columns:
                    df = df[["calc_time", "last_funding_rate"]]
                else:
                    df = df.iloc[:, [0, -1]]
                    df.columns = ["calc_time", "last_funding_rate"]
                dfs.append(df)

    if not dfs:
        logger.warning(f"No funding rates retrieved for {symbol}")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["calc_time"].astype(np.int64), unit="ms", utc=True)
    combined = combined.drop(columns=["calc_time"])
    combined["last_funding_rate"] = combined["last_funding_rate"].astype(np.float64)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.set_index("timestamp")
    return combined


def fetch_and_build_universe(
    symbols: list[str] = TARGET_SYMBOLS,
    start_date: datetime = START_DATE,
    end_date: datetime = END_DATE,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Downloads and aligns 15-coin universe market data into a clean Parquet panel.
    Filters to strictly [start_date, end_date] UTC.
    """
    if output_path is None:
        output_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/universe_15_hourly.parquet"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ym_list = generate_month_tuples(start_date, end_date)
    expected_full_index = pd.date_range(start_date, end_date, freq="1h", tz="UTC")
    total_expected_bars = len(expected_full_index)

    logger.info(f"Target Universe: {len(symbols)} coins from {start_date} to {end_date} ({total_expected_bars} bars)")

    symbol_panels = []
    coverage_report = {}

    for sym in symbols:
        logger.info(f"Fetching {sym} (Klines + Funding)...")
        df_kline = fetch_symbol_klines(sym, ym_list)
        df_funding = fetch_symbol_funding(sym, ym_list)

        if df_kline.empty:
            logger.warning(f"Skipping {sym}: zero kline data.")
            continue

        # Trim to exact target window
        df_kline = df_kline[(df_kline.index >= pd.Timestamp(start_date, tz="UTC")) &
                            (df_kline.index <= pd.Timestamp(end_date, tz="UTC"))]

        # Merge funding rate onto hourly index (backward/forward fill)
        if not df_funding.empty:
            df_merged = pd.merge_asof(
                df_kline.reset_index(),
                df_funding.reset_index(),
                on="timestamp",
                direction="backward",
            ).set_index("timestamp")
            df_merged["funding_rate"] = df_merged["last_funding_rate"].ffill().bfill().fillna(0.0)
            df_merged = df_merged.drop(columns=["last_funding_rate"])
        else:
            df_merged = df_kline
            df_merged["funding_rate"] = 0.0

        # Check coverage
        bar_count = len(df_merged)
        coverage_pct = (bar_count / total_expected_bars) * 100.0
        coverage_report[sym] = {"bars": bar_count, "coverage_pct": coverage_pct}

        if coverage_pct < 90.0:
            logger.warning(f"Skipping {sym}: coverage {coverage_pct:.1f}% is below 90% threshold.")
            continue

        df_merged["asset"] = sym
        symbol_panels.append(df_merged.reset_index())

    logger.info("=" * 60)
    logger.info("UNIVERSE COVERAGE REPORT:")
    for sym, stats in coverage_report.items():
        status = "INCLUDED" if stats["coverage_pct"] >= 90.0 else "EXCLUDED"
        logger.info(f"  {sym:10s}: {stats['bars']:5d} / {total_expected_bars} bars ({stats['coverage_pct']:5.1f}%) -> {status}")
    logger.info("=" * 60)

    if not symbol_panels:
        raise RuntimeError("No symbols met the coverage threshold.")

    panel_df = pd.concat(symbol_panels, ignore_index=True)
    panel_df = enforce_utc_columns(panel_df)
    panel_df = panel_df.sort_values(["timestamp", "asset"]).reset_index(drop=True)

    # Reorder columns
    cols = ["timestamp", "asset", "open", "high", "low", "close", "volume", "funding_rate"]
    panel_df = panel_df[cols]

    logger.info(f"Writing {len(panel_df):,} total panel rows across {len(symbol_panels)} assets to '{output_path}'...")
    panel_df.to_parquet(output_path, index=False)
    logger.info("Universe 15 hourly ingestion complete and verified.")
    return panel_df


if __name__ == "__main__":
    fetch_and_build_universe()
