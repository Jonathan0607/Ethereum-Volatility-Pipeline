import requests, datetime
start_time = int(datetime.datetime(2022, 1, 1).timestamp() * 1000)
end_time = int(datetime.datetime(2022, 1, 20).timestamp() * 1000)
url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol=SOLUSDT&period=1h&limit=500&startTime={start_time}&endTime={end_time}"
r = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
print("Status:", r.status_code)
data = r.json()
if isinstance(data, list) and len(data) > 0:
    print("Got", len(data), "records. First record:", data[0])
else:
    print("Error/No data:", data)
