import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import warnings
import os

warnings.filterwarnings("ignore")

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
    Fetches 2 YEARS of hourly Ethereum data using Yahoo Finance.
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
        
        if df.empty:
            print("ERROR: Yahoo Finance returned empty data. Check your internet.")
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        output_path = os.path.join(data_dir, 'eth_hourly.csv')
        df.to_csv(output_path, index=False)
        
        print(f"SUCCESS: Saved {len(df)} rows to {output_path}\n")
        split_data(df, verbose=True)
        
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

# --- features.py logic ---

def calculate_features(df, train_df=None):
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    print("Fitting GARCH(1,1) Model on TRAINING DATA only...")

    fit_target = train_df.copy() if train_df is not None else df.copy()
    fit_target['log_ret'] = np.log(fit_target['close'] / fit_target['close'].shift(1))
    fit_target.dropna(inplace=True)
    train_returns = fit_target['log_ret'] * 100

    garch_model = arch_model(train_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
    res = garch_model.fit(disp='off')

    current_dir = os.path.dirname(os.path.abspath(__file__))
    params = {
        'mu':    float(res.params['mu']),
        'omega': float(res.params['omega']),
        'alpha': float(res.params.get('alpha[1]', res.params.get('alpha', 0.05))),
        'beta':  float(res.params.get('beta[1]',  res.params.get('beta',  0.90))),
    }
    params_path = os.path.join(current_dir, '..', 'garch_params.npy')
    np.save(params_path, params)
    print(f"GARCH params saved → {params_path}")

    df['garch_vol'] = _apply_garch(df['log_ret'] * 100, params) / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()
    df.dropna(inplace=True)

    print("Feature Engineering Complete.")
    return df

def calculate_features_test(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'garch_params.npy')

    if not os.path.exists(params_path):
        raise FileNotFoundError("garch_params.npy not found. Fit GARCH on training data.")

    params = np.load(params_path, allow_pickle=True).item()
    print(f"Loaded GARCH params: {params}")

    df['garch_vol'] = _apply_garch(df['log_ret'] * 100, params) / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()
    df.dropna(inplace=True)

    print("Test Feature Engineering Complete (no GARCH refit).")
    return df

def _apply_garch(returns_pct: pd.Series, params: dict) -> pd.Series:
    mu, omega, alpha, beta = params['mu'], params['omega'], params['alpha'], params['beta']
    vals = returns_pct.values
    n = len(vals)
    uncond_var = omega / max(1 - alpha - beta, 1e-6)
    sigma2 = np.full(n, uncond_var)
    for t in range(1, n):
        eps2 = (vals[t - 1] - mu) ** 2
        sigma2[t] = omega + alpha * eps2 + beta * sigma2[t - 1]
    return pd.Series(np.sqrt(sigma2), index=returns_pct.index)

if __name__ == "__main__":
    fetch_data()
