import requests, zipfile, io
import pandas as pd
from build_true_historical_dataset import get_monthly_urls, get_daily_urls, download_and_parse_csv

kline_urls = get_monthly_urls("klines", "SOLUSDT", "1h")
funding_urls = get_monthly_urls("fundingRate", "SOLUSDT", "")
metrics_urls = get_daily_urls("SOLUSDT")

print("Kline url 0:", kline_urls[0])
print("Funding url 0:", funding_urls[0])
print("Metrics url 0:", metrics_urls[0])

# test one of each
df1 = download_and_parse_csv(kline_urls[0], is_klines=True)
print("Kline 0 length:", len(df1) if df1 is not None else None)

df2 = download_and_parse_csv(funding_urls[0], is_funding=True)
print("Funding 0 length:", len(df2) if df2 is not None else None)

df3 = download_and_parse_csv(metrics_urls[0], is_metrics=True)
print("Metrics 0 length:", len(df3) if df3 is not None else None)

