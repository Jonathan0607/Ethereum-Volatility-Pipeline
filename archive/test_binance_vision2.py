import requests
url = "https://data.binance.vision/data/futures/um/daily/openInterestHist/SOLUSDT/SOLUSDT-openInterestHist-2022-01-01.zip"
r = requests.head(url)
print("OI Hist daily status:", r.status_code)

url = "https://data.binance.vision/data/futures/um/daily/metrics/SOLUSDT/SOLUSDT-metrics-2022-01-01.zip"
r = requests.head(url)
print("Metrics daily status:", r.status_code)
