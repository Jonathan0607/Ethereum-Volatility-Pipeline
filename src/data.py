import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import os
import sys

try:
    import ccxt
except ImportError:
    ccxt = None

warnings.filterwarnings("ignore")

FEATURE_COLS = ['volatility', 'log_return', 'rolling_vol_24h', 'ma_20_dist', 'ma_50_dist']


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
        print(f"\n[Data Split] Configuration (Dynamic 75/25):")
        print(f"   - Total Rows:  {len(df)}")
        print(f"   - Train Size:  {len(train_df)} rows")
        print(f"   - Test Size:   {len(test_df)} rows")
    
    return train_df, test_df


def check_integrity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures monotonic chronological ordering and handles any internal NaN gaps.
    """
    if df.isnull().any().any():
        df = df.ffill().bfill()
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def fetch_exchange_funding_rates(symbol='ETH/USDT:USDT', limit=500) -> pd.DataFrame:
    """
    Fetches real-time / recent perpetual funding rate history from CCXT exchanges (OKX, Binance, Bybit, Kraken).
    Returns an empty DataFrame if CCXT is unavailable or exchange queries fail.
    """
    if ccxt is None:
        return pd.DataFrame()

    for ex_name in ['okx', 'binance', 'bybit', 'kraken']:
        try:
            ex = getattr(ccxt, ex_name)({'enableRateLimit': True, 'timeout': 5000})
            if ex.has.get('fetchFundingRateHistory'):
                rates = ex.fetch_funding_rate_history(symbol, limit=limit)
                if rates and len(rates) > 0:
                    df_r = pd.DataFrame(rates)
                    df_r['timestamp'] = pd.to_datetime(df_r['timestamp'], unit='ms', utc=True)
                    df_r.set_index('timestamp', inplace=True)
                    df_r.sort_index(inplace=True)
                    return df_r[['fundingRate']]
        except Exception:
            continue
    return pd.DataFrame()


def compute_rolling_funding_zscore(funding_rates, window=72) -> np.ndarray:
    """
    Computes the 72-hour rolling Z-score of the perpetual funding rate:
        Z_funding = (FR - rolling_mean(FR, 72)) / max(rolling_std(FR, 72), 1e-5)
    Guarantees numerical stability with a 1e-5 minimum denominator floor when rates flatline.
    """
    fr_series = pd.Series(funding_rates, dtype=np.float64)
    rolling_mean = fr_series.rolling(window=window, min_periods=1).mean()
    rolling_std = fr_series.rolling(window=window, min_periods=1).std().fillna(0.0)
    # Maintain minimum denominator floor of 1e-5 to prevent zero division
    denom = np.maximum(rolling_std.values, 1e-5)
    z_funding = (fr_series.values - rolling_mean.values) / denom
    return np.nan_to_num(z_funding, nan=0.0, posinf=0.0, neginf=0.0)


def fetch_data() -> pd.DataFrame:
    """
    Downloads hourly ETH & BTC spot OHLCV via Yahoo Finance and attaches real CCXT perpetual funding rates.
    If exchange funding rates are unavailable, strictly fills with 0.0 (no proxy synthesis).
    """
    print("\nDownloading ETH & BTC Data via Yahoo Finance...")
    try:
        raw_df = yf.download(["ETH-USD", "BTC-USD"], period="2y", interval="1h", progress=False, auto_adjust=True)
        if raw_df.empty:
            print("ERROR: Yahoo Finance returned empty data.")
            return None

        eth_cols = {'Open': 'eth_open', 'High': 'eth_high', 'Low': 'eth_low', 'Close': 'eth_close', 'Volume': 'eth_volume'}
        btc_cols = {'Open': 'btc_open', 'High': 'btc_high', 'Low': 'btc_low', 'Close': 'btc_close', 'Volume': 'btc_volume'}

        eth_data = pd.DataFrame(index=raw_df.index)
        btc_data = pd.DataFrame(index=raw_df.index)

        for src_col, tgt_col in eth_cols.items():
            if (src_col, 'ETH-USD') in raw_df.columns:
                eth_data[tgt_col] = raw_df[(src_col, 'ETH-USD')].values
            elif src_col in raw_df.columns:
                eth_data[tgt_col] = raw_df[src_col].values

        for src_col, tgt_col in btc_cols.items():
            if (src_col, 'BTC-USD') in raw_df.columns:
                btc_data[tgt_col] = raw_df[(src_col, 'BTC-USD')].values
            elif src_col in raw_df.columns:
                btc_data[tgt_col] = raw_df[src_col].values

        merged_new = eth_data.join(btc_data, how='inner').dropna()
        merged_new['open'] = merged_new['eth_open']
        merged_new['high'] = merged_new['eth_high']
        merged_new['low'] = merged_new['eth_low']
        merged_new['close'] = merged_new['eth_close']
        merged_new['volume'] = merged_new['eth_volume']
        
        merged_new.reset_index(inplace=True)
        date_col = 'Datetime' if 'Datetime' in merged_new.columns else ('Date' if 'Date' in merged_new.columns else merged_new.columns[0])
        merged_new.rename(columns={date_col: 'timestamp'}, inplace=True)
        merged_new['timestamp'] = pd.to_datetime(merged_new['timestamp'], utc=True)
        merged_new.set_index('timestamp', inplace=True)

        # Pull real exchange perpetual funding rates via CCXT
        ex_df = fetch_exchange_funding_rates(limit=500)
        if not ex_df.empty:
            merged_new = merged_new.join(ex_df, how='left')
            merged_new['funding_rate'] = merged_new['fundingRate'].ffill().fillna(0.0)
            merged_new.drop(columns=['fundingRate'], inplace=True, errors='ignore')
        else:
            merged_new['funding_rate'] = 0.0

        # Calculate 72h rolling Z-score
        merged_new['z_funding'] = compute_rolling_funding_zscore(merged_new['funding_rate'].values, window=72)

        merged_new.reset_index(inplace=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        eth_btc_path = os.path.join(data_dir, 'eth_btc_hourly.csv')
        eth_path = os.path.join(data_dir, 'eth_hourly.csv')

        merged_new.to_csv(eth_btc_path, index=False)
        merged_new.to_csv(eth_path, index=False)

        print(f"SUCCESS: Dataset contains {len(merged_new)} synchronous rows with Real Perpetual Funding & Z-scores. Saved to {eth_btc_path}\n")
        return merged_new.set_index('timestamp')

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def load_pairs_data(data_path=None) -> pd.DataFrame:
    """
    Loads synchronous ETH/BTC dataset and enforces clean perpetual funding rate & Z-score features.
    Relies purely on real exchange data or 0.0 fallback (no synthetic proxy).
    """
    if data_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, '..', 'data', 'eth_btc_hourly.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')

    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')
    df.sort_index(inplace=True)
    
    if 'funding_rate' not in df.columns:
        ex_df = fetch_exchange_funding_rates(limit=500)
        if not ex_df.empty:
            df = df.join(ex_df, how='left')
            df['funding_rate'] = df['fundingRate'].ffill().fillna(0.0)
            df.drop(columns=['fundingRate'], inplace=True, errors='ignore')
        else:
            df['funding_rate'] = 0.0

    df['funding_rate'] = df['funding_rate'].fillna(0.0)
    df['z_funding'] = compute_rolling_funding_zscore(df['funding_rate'].values, window=72)

    return df


def load_microstructure_data(data_path=None) -> pd.DataFrame:
    """
    Alias for load_pairs_data to maintain compatibility across pipeline.
    """
    return load_pairs_data(data_path)


if __name__ == "__main__":
    fetch_data()
