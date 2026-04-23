import yfinance as yf
import pandas as pd
import os
from data_split import split_data 

def fetch_data():
    """
    Fetches recent Ethereum data and appends it to the existing dataset.
    No API keys, no rate limits, no IP bans.
    """
    print("\nDownloading ETH Data via Yahoo Finance...")
    
    try:
        # 1. Fetch the maximum allowed hourly data window
        new_df = yf.download("ETH-USD", period="2y", interval="1h", progress=False, auto_adjust=True)
        
        new_df.reset_index(inplace=True)
        
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = [col[0] for col in new_df.columns]
            
        new_df.rename(columns={
            "Datetime": "timestamp",
            "Open": "open", 
            "High": "high", 
            "Low": "low", 
            "Close": "close", 
            "Volume": "volume"
        }, inplace=True)
        
        if new_df.empty:
            print("ERROR: Yahoo Finance returned empty data. Check your internet.")
            return

        # 2. File Path Management
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, 'eth_hourly.csv')
        
        # 3. Smart Merge Logic (Append & Deduplicate)
        if os.path.exists(output_path):
            print(f"Existing data found at {output_path}. Merging new data...")
            old_df = pd.read_csv(output_path)
            
            # Unify timezone formats to prevent concatenation crashes
            old_df['timestamp'] = pd.to_datetime(old_df['timestamp'], utc=True)
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], utc=True)
            
            # Combine, sort, and drop duplicates based on the timestamp
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            combined_df.sort_values(by='timestamp', inplace=True)
            final_df = combined_df
        else:
            print("No existing data found. Creating new dataset...")
            final_df = new_df

        # 4. Save to CSV
        final_df.to_csv(output_path, index=False)
        print(f"SUCCESS: Dataset now contains {len(final_df)} rows. Saved to {output_path}\n")
        
        split_data(final_df, verbose=True)
        
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_data()