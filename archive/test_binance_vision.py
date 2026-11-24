import requests
url = "https://data.binance.vision/data/futures/um/monthly/klines/SOLUSDT/1h/SOLUSDT-1h-2022-01.zip"
r = requests.head(url)
print("Kline 1h monthly status:", r.status_code)

url = "https://data.binance.vision/data/futures/um/monthly/fundingRate/SOLUSDT/SOLUSDT-fundingRate-2022-01.zip"
r = requests.head(url)
print("FundingRate monthly status:", r.status_code)

url = "https://data.binance.vision/data/futures/um/monthly/openInterestHist/SOLUSDT/SOLUSDT-openInterestHist-2022-01.zip"
r = requests.head(url)
print("OI Hist monthly status:", r.status_code)

url = "https://data.binance.vision/data/futures/um/monthly/metrics/SOLUSDT/SOLUSDT-metrics-2022-01.zip"
r = requests.head(url)
print("Metrics monthly status:", r.status_code)
