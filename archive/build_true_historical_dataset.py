#!/usr/bin/env python3
"""
Build True Historical Dataset from Binance Vision CSV Archives.
Downloads Klines, Funding Rates, and Metrics (Open Interest) directly from 
Binance Vision to construct a 100% real data lake (no synthetic proxy).
"""

import os
import sys
import io
import time
import requests
import zipfile
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

UNIVERSE = ['SOLUSDT', 'AVAXUSDT', 'LINKUSDT']
START_DATE = datetime(2022, 6, 1) # ~4.2 years ago
END_DATE = datetime(2026, 8, 31)

def get_monthly_urls(base_path, symbol, prefix):
    urls = []
    current = START_DATE
    while current <= END_DATE:
        ym = current.strftime("%Y-%m")
        url = f"https://data.binance.vision/data/futures/um/monthly/{base_path}/{symbol}/{prefix}/{symbol}-{prefix}-{ym}.zip"
        # Funding rate has no prefix folder
        if base_path == "fundingRate":
            url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{ym}.zip"
        urls.append(url)
        # Add 1 month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return urls

def get_daily_urls(symbol):
    urls = []
    current = START_DATE
    while current <= END_DATE:
        ymd = current.strftime("%Y-%m-%d")
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{symbol}-metrics-{ymd}.zip"
        urls.append(url)
        current += pd.Timedelta(days=1)
    return urls

def download_and_parse_csv(url, is_klines=False, is_funding=False, is_metrics=False):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for filename in z.namelist():
                if filename.endswith(".csv"):
                    with z.open(filename) as f:
                        if is_klines:
                            df = pd.read_csv(f)
                            if 'open_time' in df.columns:
                                df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
                            else:
                                df = df.iloc[:, 0:6]
                                df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
                            
                            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
                            df = df.set_index('timestamp')
                            return df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                        elif is_funding:
                            df = pd.read_csv(f)
                            df['timestamp'] = pd.to_datetime(df['calc_time'], unit='ms', utc=True)
                            df = df.set_index('timestamp')
                            return df[['last_funding_rate']].astype(float)
                        elif is_metrics:
                            df = pd.read_csv(f)
                            # metrics has 'create_time' like 2022-01-01 00:00:00
                            df['timestamp'] = pd.to_datetime(df['create_time'], utc=True)
                            df = df.set_index('timestamp')
                            return df[['sum_open_interest']].astype(float)
    except Exception as e:
        return None
    return None

def fetch_concurrent(urls, parser_kwargs, max_workers=20):
    dfs = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_and_parse_csv, url, **parser_kwargs): url for url in urls}
        for future in as_completed(future_to_url):
            res = future.result()
            if res is not None and not res.empty:
                dfs.append(res)
    if not dfs:
        return pd.DataFrame()
    combined = pd.concat(dfs)
    return combined.sort_index()[~combined.index.duplicated(keep='last')]

def main():
    print("=" * 80)
    print("BUILDING TRUE HISTORICAL DATA LAKE FROM BINANCE VISION")
    print("=" * 80)
    
    final_dataset = {}
    
    for symbol in UNIVERSE:
        print(f"\n[{symbol}] Processing 4-Year History...")
        
        # 1. Klines (Monthly)
        kline_urls = get_monthly_urls("klines", symbol, "1h")
        print(f"  --> Downloading {len(kline_urls)} monthly Kline files...")
        df_kline = fetch_concurrent(kline_urls, {'is_klines': True})
        
        # 2. Funding Rates (Monthly)
        funding_urls = get_monthly_urls("fundingRate", symbol, "")
        print(f"  --> Downloading {len(funding_urls)} monthly Funding Rate files...")
        df_funding = fetch_concurrent(funding_urls, {'is_funding': True})
        
        # 3. Metrics/Open Interest (Daily)
        metrics_urls = get_daily_urls(symbol)
        print(f"  --> Downloading {len(metrics_urls)} daily Metrics (Open Interest) files...")
        df_metrics = fetch_concurrent(metrics_urls, {'is_metrics': True}, max_workers=40)
        
        if df_kline.empty or df_funding.empty or df_metrics.empty:
            print(f"  [ERROR] Failed to fetch complete data for {symbol}.")
            continue
            
        print(f"  --> Merging and resampling {symbol} data...")
        # Resample klines to exact 1h
        df_kline = df_kline[~df_kline.index.duplicated(keep='last')]
        df_kline = df_kline.resample('1h').last()
        
        # Resample funding rate to 1h (forward fill)
        df_funding = df_funding[~df_funding.index.duplicated(keep='last')]
        df_funding = df_funding.resample('1h').ffill()
        df_funding.columns = ['funding_rate']
        
        # Resample metrics to 1h (forward fill)
        df_metrics = df_metrics[~df_metrics.index.duplicated(keep='last')]
        df_metrics = df_metrics.resample('1h').ffill()
        df_metrics.columns = ['open_interest']
        
        # Merge all
        df_merged = df_kline.join(df_funding).join(df_metrics)
        
        # Forward fill any small gaps and drop na
        df_merged = df_merged.ffill().dropna()
        final_dataset[symbol] = df_merged
        
        print(f"  [OK] {symbol} complete. Bars: {len(df_merged)}")

    # Ensure common index
    print("\nSynchronizing indices across all assets...")
    common_idx = final_dataset[UNIVERSE[0]].index
    for s in UNIVERSE[1:]:
        common_idx = common_idx.intersection(final_dataset[s].index)
        
    for s in UNIVERSE:
        final_dataset[s] = final_dataset[s].loc[common_idx]
        
    print(f"[FINAL] Synchronized bars: {len(common_idx)}")
    print(f"[FINAL] Date Range: {common_idx[0]} to {common_idx[-1]}")
    
    out_path = os.path.join(DATA_DIR, 'true_historical_dataset_4yr.pkl')
    with open(out_path, 'wb') as f:
        import pickle
        pickle.dump(final_dataset, f)
        
    print(f"\n[SAVED] True dataset written to {out_path}")

if __name__ == '__main__':
    main()
