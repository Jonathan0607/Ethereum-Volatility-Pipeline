"""
Path Signatures & Lead-Lag Embedding Module
Extracts the continuous geometry of cross-asset financial time series (ETH/BTC) using Rough Path Theory.
Constructs a 20-dimensional Lead-Lag tensor and extracts 420-dimensional depth-2 geometric signatures
capturing cross-asset Lévy areas and predicting cointegrated residual drift (mu_residual).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cointegration import compute_rolling_cointegration

try:
    import iisignature
except ImportError:
    iisignature = None


def normalize_ohlcv_window(df_or_array):
    """
    Normalizes a rolling OHLCV window for multi-asset (ETH & BTC) or single-asset time series:
    For each asset:
    - Relative price displacement relative to the first price P0: (P - P0) / (P0 + 1e-8)
    - Z-scored volume across the window: (V - mean(V)) / (std(V) + 1e-8)
    
    Returns a numpy array of shape:
        - (T, 10) for 2 assets [eth_open..eth_volume, btc_open..btc_volume]
        - (T, 5) for 1 asset [open..volume]
    """
    if isinstance(df_or_array, pd.DataFrame):
        df = df_or_array.copy()
        cols_lower = {c.lower(): c for c in df.columns}
        
        # Check for multi-asset columns (ETH and BTC)
        has_eth = all(f'eth_{k}' in cols_lower for k in ['open', 'high', 'low', 'close', 'volume'])
        has_btc = all(f'btc_{k}' in cols_lower for k in ['open', 'high', 'low', 'close', 'volume'])
        
        if has_eth and has_btc:
            eth_cols = [cols_lower[f'eth_{k}'] for k in ['open', 'high', 'low', 'close', 'volume']]
            btc_cols = [cols_lower[f'btc_{k}'] for k in ['open', 'high', 'low', 'close', 'volume']]
            
            eth_arr = df[eth_cols].values.astype(np.float64)
            btc_arr = df[btc_cols].values.astype(np.float64)
            
            # Normalize ETH
            if len(eth_arr) > 0:
                p0_eth = eth_arr[0, 0] if eth_arr[0, 0] > 0 else (eth_arr[0, 3] if eth_arr[0, 3] > 0 else 1.0)
                eth_arr[:, :4] = (eth_arr[:, :4] - p0_eth) / (p0_eth + 1e-8)
                v_eth = eth_arr[:, 4]
                v_std_eth = np.std(v_eth)
                eth_arr[:, 4] = (v_eth - np.mean(v_eth)) / (v_std_eth + 1e-8) if v_std_eth > 1e-8 else 0.0

            # Normalize BTC
            if len(btc_arr) > 0:
                p0_btc = btc_arr[0, 0] if btc_arr[0, 0] > 0 else (btc_arr[0, 3] if btc_arr[0, 3] > 0 else 1.0)
                btc_arr[:, :4] = (btc_arr[:, :4] - p0_btc) / (p0_btc + 1e-8)
                v_btc = btc_arr[:, 4]
                v_std_btc = np.std(v_btc)
                btc_arr[:, 4] = (v_btc - np.mean(v_btc)) / (v_std_btc + 1e-8) if v_std_btc > 1e-8 else 0.0

            return np.concatenate([eth_arr, btc_arr], axis=1)
        
        else:
            req = ['open', 'high', 'low', 'close', 'volume']
            missing = [r for r in req if r not in cols_lower]
            if missing:
                if 'eth_close' in cols_lower and 'btc_close' in cols_lower:
                    arr = df[[cols_lower['eth_open'], cols_lower['eth_high'], cols_lower['eth_low'], cols_lower['eth_close'], cols_lower['eth_volume'],
                              cols_lower['btc_open'], cols_lower['btc_high'], cols_lower['btc_low'], cols_lower['btc_close'], cols_lower['btc_volume']]].values.astype(np.float64)
                    return arr
                raise ValueError(f"Missing required columns in DataFrame: {missing}")
            
            arr = df[[cols_lower['open'], cols_lower['high'], cols_lower['low'], 
                      cols_lower['close'], cols_lower['volume']]].values.astype(np.float64)
            if len(arr) > 0:
                p0 = arr[0, 0] if arr[0, 0] > 0 else (arr[0, 3] if arr[0, 3] > 0 else 1.0)
                arr[:, :4] = (arr[:, :4] - p0) / (p0 + 1e-8)
                v = arr[:, 4]
                v_std = np.std(v)
                arr[:, 4] = (v - np.mean(v)) / (v_std + 1e-8) if v_std > 1e-8 else 0.0
            return arr

    else:
        arr = np.asarray(df_or_array, dtype=np.float64).copy()
        if arr.ndim != 2 or arr.shape[1] not in (5, 10):
            raise ValueError(f"Expected array of shape (T, 5) or (T, 10), got {arr.shape}")
        
        if arr.shape[1] == 10:
            if len(arr) > 0:
                p0_eth = arr[0, 0] if arr[0, 0] > 0 else 1.0
                arr[:, :4] = (arr[:, :4] - p0_eth) / (p0_eth + 1e-8)
                v_std1 = np.std(arr[:, 4])
                arr[:, 4] = (arr[:, 4] - np.mean(arr[:, 4])) / (v_std1 + 1e-8) if v_std1 > 1e-8 else 0.0

                p0_btc = arr[0, 5] if arr[0, 5] > 0 else 1.0
                arr[:, 5:9] = (arr[:, 5:9] - p0_btc) / (p0_btc + 1e-8)
                v_std2 = np.std(arr[:, 9])
                arr[:, 9] = (arr[:, 9] - np.mean(arr[:, 9])) / (v_std2 + 1e-8) if v_std2 > 1e-8 else 0.0
        else:
            if len(arr) > 0:
                p0 = arr[0, 0] if arr[0, 0] > 0 else 1.0
                arr[:, :4] = (arr[:, :4] - p0) / (p0 + 1e-8)
                v_std = np.std(arr[:, 4])
                arr[:, 4] = (arr[:, 4] - np.mean(arr[:, 4])) / (v_std + 1e-8) if v_std > 1e-8 else 0.0

        return arr


def lead_lag_transform(df_or_array):
    """
    Constructs a 20-dimensional (for 10D pairs input) or 10-dimensional (for 5D input)
    piecewise linear continuous path (lead and lag components).
    """
    norm_arr = normalize_ohlcv_window(df_or_array)
    t_len, d = norm_arr.shape
    if t_len == 0:
        return np.empty((0, 2 * d), dtype=np.float64)
    if t_len == 1:
        return np.concatenate([norm_arr, norm_arr], axis=1)

    out_len = 2 * t_len - 1
    ll_path = np.zeros((out_len, 2 * d), dtype=np.float64)

    for t in range(t_len - 1):
        ll_path[2 * t, :d] = norm_arr[t]
        ll_path[2 * t, d:] = norm_arr[t]
        ll_path[2 * t + 1, :d] = norm_arr[t + 1]
        ll_path[2 * t + 1, d:] = norm_arr[t]

    ll_path[out_len - 1, :d] = norm_arr[t_len - 1]
    ll_path[out_len - 1, d:] = norm_arr[t_len - 1]

    return ll_path


def compute_signatures(path_tensor, depth=2):
    """
    Computes truncated geometric path signatures using iisignature.
    For a 20D Lead-Lag tensor at depth=2:
    - 20 Level 1 terms
    - 400 Level 2 terms (iterated integrals / cross-asset Lévy areas)
    - Total = 420-dimensional signature vector.
    """
    global iisignature
    if iisignature is None:
        import iisignature

    if isinstance(path_tensor, list):
        sigs = [iisignature.sig(p, depth) for p in path_tensor]
        return np.array(sigs, dtype=np.float64)
    
    arr = np.asarray(path_tensor, dtype=np.float64)
    if arr.ndim == 2:
        return iisignature.sig(arr, depth)
    elif arr.ndim == 3:
        b, n, d = arr.shape
        sigs = [iisignature.sig(arr[i], depth) for i in range(b)]
        return np.array(sigs, dtype=np.float64)
    else:
        raise ValueError(f"Expected 2D or 3D array for path_tensor, got shape {arr.shape}")


class SignatureDriftPredictor:
    """
    Regularized Ridge Regression model on 20D rolling path signatures to predict
    either:
    - target_mode='residual': 4-hour forward change in cointegrated residual (y = eps_{t+4} - eps_t)
    - target_mode='eth_return': 4-hour forward cumulative return on ETH (y = ln(P_ETH,t+4 / P_ETH,t))
    - target_mode='spread_return': 4-hour forward spread return (y = ln(P_ETH,t+4/P_ETH,t) - ln(P_BTC,t+4/P_BTC,t))
    
    Uses a StandardScaler pipeline so Level 2 cross-asset Lévy Area terms are not dominated by L2 penalty.
    Explicitly sets fit_intercept=False to prevent memorizing baseline IS macro bull/bear trends.
    Applies EMA smoothing (span=3) to predicted drift.
    """
    def __init__(self, alpha=1.0, lookback_window=24, depth=2, ema_span=3, forward_horizon=4, target_mode='residual'):
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.depth = depth
        self.ema_span = ema_span
        self.forward_horizon = forward_horizon
        self.target_mode = target_mode
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=self.alpha, fit_intercept=False))
        ])
        self.is_fitted = False
        self.last_smoothed_mu_ = None

    def reset_smoother(self):
        """Resets the sequential EMA state."""
        self.last_smoothed_mu_ = None

    def extract_signatures_from_df(self, df):
        """
        Extracts 20D rolling window signatures and forward target from a DataFrame.
        """
        cols_lower = {c.lower(): c for c in df.columns}
        n = len(df)
        w = self.lookback_window
        h = self.forward_horizon

        if n <= w:
            raise ValueError(f"DataFrame length ({n}) must be greater than lookback window ({w})")

        # Compute target vector based on target_mode
        if self.target_mode == 'residual':
            if 'coint_residual' in df.columns:
                residuals = df['coint_residual'].values.astype(np.float64)
            else:
                coint_df = compute_rolling_cointegration(df, window=72)
                residuals = coint_df['coint_residual'].values.astype(np.float64)
            
            target_series = np.full(n, np.nan, dtype=np.float64)
            if n > h:
                target_series[:-h] = residuals[h:] - residuals[:-h]

        elif self.target_mode == 'eth_return':
            p_eth = df[cols_lower['eth_close']].values.astype(np.float64) if 'eth_close' in cols_lower else df[cols_lower['close']].values.astype(np.float64)
            target_series = np.full(n, np.nan, dtype=np.float64)
            if n > h:
                target_series[:-h] = np.log(p_eth[h:] / (p_eth[:-h] + 1e-8))

        elif self.target_mode == 'spread_return':
            p_eth = df[cols_lower['eth_close']].values.astype(np.float64) if 'eth_close' in cols_lower else df[cols_lower['close']].values.astype(np.float64)
            p_btc = df[cols_lower['btc_close']].values.astype(np.float64) if 'btc_close' in cols_lower else np.ones_like(p_eth)
            target_series = np.full(n, np.nan, dtype=np.float64)
            if n > h:
                fwd_eth = np.log(p_eth[h:] / (p_eth[:-h] + 1e-8))
                fwd_btc = np.log(p_btc[h:] / (p_btc[:-h] + 1e-8))
                target_series[:-h] = fwd_eth - fwd_btc
        else:
            raise ValueError(f"Unknown target_mode: {self.target_mode}")

        X_list = []
        y_list = []
        idx_list = []

        for t in range(w, n):
            window_df = df.iloc[t - w: t]
            ll_path = lead_lag_transform(window_df)
            sig = compute_signatures(ll_path, depth=self.depth)
            
            X_list.append(sig)
            last_bar_idx = t - 1
            if last_bar_idx < len(target_series):
                y_list.append(target_series[last_bar_idx])
            else:
                y_list.append(np.nan)
            idx_list.append(df.index[t])

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)
        return X, y, idx_list

    def fit(self, df):
        """
        Fits the Ridge regression model on in-sample rolling signatures.
        Aggressively drops NaN rows at the end of the training slice for zero look-ahead leak.
        """
        X, y, _ = self.extract_signatures_from_df(df)
        valid_mask = ~np.isnan(y)
        X_train = X[valid_mask]
        y_train = y[valid_mask]

        if len(X_train) == 0:
            raise ValueError("No valid training samples found for SignatureDriftPredictor")

        self.pipeline.set_params(ridge__alpha=self.alpha)
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        self.last_smoothed_mu_ = None
        return self

    def predict_drift_for_window(self, window_df, apply_smoothing=True):
        """
        Predicts expected forward change in cointegration residual (mu_residual) for a rolling window.
        Applies EMA smoothing (span=3) to dampen high-frequency oscillation.
        """
        if not self.is_fitted:
            raise RuntimeError("SignatureDriftPredictor must be fitted before calling predict_drift_for_window")
        
        ll_path = lead_lag_transform(window_df)
        sig = compute_signatures(ll_path, depth=self.depth)
        raw_mu = float(self.pipeline.predict(sig.reshape(1, -1))[0])

        if not apply_smoothing or self.ema_span <= 1:
            return raw_mu

        alpha_ema = 2.0 / (self.ema_span + 1.0)
        if self.last_smoothed_mu_ is None:
            self.last_smoothed_mu_ = raw_mu
        else:
            self.last_smoothed_mu_ = alpha_ema * raw_mu + (1.0 - alpha_ema) * self.last_smoothed_mu_

        return float(self.last_smoothed_mu_)

    def predict(self, X, apply_smoothing=True):
        """
        Predicts residual drift for an array of signatures, with optional EMA smoothing.
        """
        if not self.is_fitted:
            raise RuntimeError("SignatureDriftPredictor must be fitted before calling predict")
        raw_mu = self.pipeline.predict(X)
        if apply_smoothing and self.ema_span > 1:
            smoothed = pd.Series(raw_mu).ewm(span=self.ema_span, adjust=False).mean().values
            return smoothed
        return raw_mu
