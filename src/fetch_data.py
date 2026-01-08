import ccxt
import pandas as pd
import os

def fetch_eth_data(limit=1000):
    """
    Fetches historical OHLCV data for ETH/USDT from Binance.
    """
    print("Fetching data from Binance...")
    exchange = ccxt.binance()
    # Fetch 1-hour candles
    ohlcv = exchange.fetch_ohlcv('ETH/USDT', timeframe='1h', limit=limit)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Save to CSV for the next step
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/eth_hourly.csv', index=False)
    print(f"Data saved to data/eth_hourly.csv ({len(df)} rows)")
    return df

if __name__ == "__main__":
    fetch_eth_data()