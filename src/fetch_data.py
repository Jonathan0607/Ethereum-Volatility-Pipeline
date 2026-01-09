import yfinance as yf
import pandas as pd
import os

def fetch_data():
    """
    Fetches 2 YEARS of hourly Ethereum data using Yahoo Finance.
    No API keys, no rate limits, no IP bans.
    """
    print("Downloading 2 Years of ETH Data via Yahoo Finance...")
    
    # Download data
    # ticker: ETH-USD
    # period: "2y" (Captures the 2024 Bull Run and the recent crash)
    # interval: "1h"
    try:
        df = yf.download("ETH-USD", period="2y", interval="1h", progress=False, auto_adjust=True)
        
        # Reset index so 'Datetime' becomes a column
        df.reset_index(inplace=True)
        
        # CLEANUP: Handle Yahoo's MultiIndex columns (if present)
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten columns if they look like ('Close', 'ETH-USD')
            df.columns = [col[0] for col in df.columns]
            
        # Rename standard columns to match our pipeline (lowercase)
        # Yahoo gives: Datetime, Open, High, Low, Close, Volume
        df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        }, inplace=True)
        
        # Standardize Timestamp (Remove timezone 'UTC-5' etc to avoid PyTorch warnings)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        
        # Ensure we have data
        if df.empty:
            print("ERROR: Yahoo Finance returned empty data. Check your internet.")
            return

        # Save to CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = os.path.join(data_dir, 'eth_hourly.csv')
        df.to_csv(output_path, index=False)
        
        print(f"\nSUCCESS: Saved {len(df)} rows to {output_path}")
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_data()