import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import warnings
import os

warnings.filterwarnings("ignore")

# =====================================================================
# FEATURE COLUMNS used by the ProgressiveModel (RNN→GRU→LSTM)
# These must stay in sync with strategy.py and api.py
# =====================================================================
FEATURE_COLS = ['volatility', 'log_return', 'rolling_vol_24h', 'ma_20_dist', 'ma_50_dist']

# --- fetch_data.py & process_data.py & data_split.py logic ---

def split_data(df: pd.DataFrame, verbose: bool = True):
    """
    Splits data chronologically:
    - First 75% = Train (History)
    - Last 25%  = Test (Future)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    split_idx = int(len(df) * 0.75)
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    if verbose:
        print(f"\n[Data Split] Configuration (Dynamic 80/20):")
        print(f"   - Total Rows:  {len(df)}")
        print(f"   - Train Size:  {len(train_df)} rows")
        print(f"   - Test Size:   {len(test_df)} rows")
    
    return train_df, test_df

def check_integrity(df):
    print("   [Integrity Check] Running sanity checks...")
    if df.isnull().any().any():
        print("   [WARNING] NaNs found. Filling with ffill...")
        df.fillna(method='ffill', inplace=True)
    if not df.index.is_monotonic_increasing:
        print("   [FIX] Sorting index...")
        df.sort_index(inplace=True)
    required_cols = ['close', 'high', 'low', 'volume']
    for col in required_cols:
        if col not in df.columns:
            if col.capitalize() in df.columns:
                df.rename(columns={col.capitalize(): col}, inplace=True)
            else:
                raise ValueError(f"CRITICAL: Missing column '{col}'")
    print("   [Integrity Check] PASSED. Data is safe.")
    return df

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

def run_processing():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing {input_path}. Run fetch_data() first.")
    df = pd.read_csv(input_path)
    df = check_integrity(df)
    print("   [Process Data] Processing complete.")
    return True

# =====================================================================
# FEATURE ENGINEERING — Stationary Features for Volatility-Scaled Sizing
# =====================================================================

def _compute_stationary_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 4 stationary features required by the ProgressiveModel.
    Called by both calculate_features() and calculate_features_test().

    Features:
        1. log_return      — np.log(close / close.shift(1))
        2. rolling_vol_24h — log_return.rolling(24).std()
        3. ma_20_dist      — (close - SMA_20) / SMA_20
        4. ma_50_dist      — (close - SMA_50) / SMA_50
    """
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['rolling_vol_24h'] = df['log_return'].rolling(window=24).std()

    sma_20 = df['close'].rolling(window=20).mean()
    sma_50 = df['close'].rolling(window=50).mean()
    df['ma_20_dist'] = (df['close'] - sma_20) / sma_20
    df['ma_50_dist'] = (df['close'] - sma_50) / sma_50

    return df


def calculate_features(df, train_df=None):
    """
    Full feature engineering for TRAINING data.
    Fits GARCH on training data and computes all features including
    the forward-looking target (fwd_vol_24h).
    """
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    # print("Fitting Dynamic GARCH(1,1) Model on TRAINING DATA...")

    garch_model = arch_model(df['log_ret'] * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    res = garch_model.fit(update_freq=0, disp='off')

    df['garch_vol'] = res.conditional_volatility / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()

    # --- Stationary features for ProgressiveModel ---
    df = _compute_stationary_features(df)

    # --- Forward-looking target: realized vol over the NEXT 24 hours ---
    df['fwd_vol_24h'] = df['log_return'].rolling(window=24).std().shift(-24)

    # Drop all NaN rows from rolling windows and forward target
    df.dropna(inplace=True)

    # print(f"Feature Engineering Complete. Columns: {list(df.columns)}")
    # print(f"   Stationary features: {FEATURE_COLS}")
    # print(f"   Forward target:      fwd_vol_24h")
    # print(f"   Rows after dropna:   {len(df)}")
    return df

def calculate_features_test(df, train_df=None):
    """
    Feature engineering for TEST / LIVE data.
    Uses pre-fitted GARCH params from train_df if provided to avoid look-ahead leak.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    if train_df is not None:
        train_copy = train_df.copy()
        train_copy['log_ret'] = np.log(train_copy['close'] / train_copy['close'].shift(1))
        train_copy.dropna(inplace=True)
        train_log_ret = train_copy['log_ret'] * 100
        train_garch = arch_model(train_log_ret, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        train_res = train_garch.fit(update_freq=0, disp='off')
        
        test_log_ret = df['log_ret'] * 100
        test_garch = arch_model(test_log_ret, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        res = test_garch.fix(train_res.params)
    else:
        garch_model = arch_model(df['log_ret'] * 100, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
        res = garch_model.fit(update_freq=0, disp='off')

    df['garch_vol'] = res.conditional_volatility / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()

    # --- Stationary features for ProgressiveModel ---
    df = _compute_stationary_features(df)

    # Drop all NaN rows from rolling windows
    df.dropna(inplace=True)
    return df



if __name__ == "__main__":
    fetch_data()
