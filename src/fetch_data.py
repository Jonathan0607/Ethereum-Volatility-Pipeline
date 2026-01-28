import yfinance as yf
import pandas as pd
import os
from data_split import split_data 



def fetch_data():
    """
    Fetches 2 YEARS of hourly Ethereum data using Yahoo Finance.
    No API keys, no rate limits, no IP bans.
    """
    print("\nDownloading 2 Years of ETH Data via Yahoo Finance...")
    
    try:
        df = yf.download("ETH-USD", period="2y", interval="1h", progress=False, auto_adjust=True)
        
        df.reset_index(inplace=True)
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
            
        df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        }, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        if df.empty:
            print("ERROR: Yahoo Finance returned empty data. Check your internet.")
            return

        # Save to CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = os.path.join(data_dir, 'eth_hourly.csv')
        df.to_csv(output_path, index=False)
        
        print(f"SUCCESS: Saved {len(df)} rows to {output_path}\n")
        split_data(df, verbose=True)
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_data()