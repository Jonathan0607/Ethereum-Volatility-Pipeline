import ccxt
import pandas as pd
import os

def fetch_eth_data(limit=1000):
    """
    Fetches historical OHLCV data for ETH/USD from Kraken 
    (Works in the US without API keys).
    """
    print("Fetching data from Kraken (US-Friendly)...")
    
    # CHANGE: Use Kraken instead of Binance
    exchange = ccxt.kraken()
    
    # Kraken uses 'ETH/USD' or 'ETH/USDT'. We will use ETH/USD for better data availability in US.
    symbol = 'ETH/USD'
    
    try:
        # Fetch 1-hour candles
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Save to CSV
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/eth_hourly.csv', index=False)
        print(f"Data saved to data/eth_hourly.csv ({len(df)} rows)")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_eth_data()