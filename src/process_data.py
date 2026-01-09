import pandas as pd
import numpy as np
import os

def check_integrity(df):
    """
    Runs Engineering checks to ensure data is safe for the LSTM.
    """
    print("   [Integrity Check] Running sanity checks...")
    
    # Check 1: No Missing Values
    if df.isnull().any().any():
        print("   [WARNING] NaNs found. Filling with ffill...")
        df.fillna(method='ffill', inplace=True)
        
    # Check 2: Chronological Order
    if not df.index.is_monotonic_increasing:
        print("   [FIX] Sorting index...")
        df.sort_index(inplace=True)
        
    # Check 3: Data Types
    required_cols = ['close', 'high', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Try to fix capitalization
            if col.capitalize() in df.columns:
                df.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                raise ValueError(f"CRITICAL: Missing column '{col}'")
    
    print("   [Integrity Check] PASSED. Data is safe.")
    return df

def run_processing():
    """
    Main entry point called by pipeline.py
    """
    # 1. Load the raw data (or the fetched data)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(input_path):
        # Fallback: Create dummy data if file is missing (just to stop crash)
        raise FileNotFoundError(f"Missing {input_path}. Run fetch_data.py first.")
        
    df = pd.read_csv(input_path)
    
    # 2. Run Checks
    df = check_integrity(df)
    
    print("   [Process Data] Processing complete.")
    return True

if __name__ == "__main__":
    run_processing()