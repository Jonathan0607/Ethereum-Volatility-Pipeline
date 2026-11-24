import requests, zipfile, io
import pandas as pd

url = "https://data.binance.vision/data/futures/um/monthly/fundingRate/SOLUSDT/SOLUSDT-fundingRate-2022-01.zip"
r = requests.get(url)
print("status:", r.status_code)
if r.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for filename in z.namelist():
            print("File inside zip:", filename)
            with z.open(filename) as f:
                df = pd.read_csv(f, nrows=5)
                print(df.columns.tolist())
                print(df.head())
