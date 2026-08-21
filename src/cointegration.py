"""
Dynamic Rolling Cointegration Module
Implements rolling Ordinary Least Squares (OLS) regression over a 72-hour window:
    ln(P_ETH) = alpha + beta * ln(P_BTC) + epsilon
Extracts dynamic hedge ratios (beta_t), cointegration spread residuals (epsilon_t),
and rolling 72-hour Z-scores (Z_t) with strict backward-looking data integrity.
"""

import numpy as np
import pandas as pd


def compute_rolling_cointegration(df: pd.DataFrame, window: int = 72) -> pd.DataFrame:
    """
    Computes rolling OLS cointegration parameters, spread residuals, and Z-scores
    across a multi-asset DataFrame containing ETH and BTC price series.
    
    Formula:
        y_t = ln(P_ETH, t)
        x_t = ln(P_BTC, t)
        beta_t = Cov_72(y, x) / Var_72(x)
        alpha_t = Mean_72(y) - beta_t * Mean_72(x)
        epsilon_t = y_t - (alpha_t + beta_t * x_t)
        Z_t = (epsilon_t - Mean_72(epsilon)) / (Std_72(epsilon) + 1e-8)
        
    Parameters:
        df (pd.DataFrame): DataFrame with 'eth_close' and 'btc_close' (or 'close' and 'btc_close').
        window (int): Rolling lookback window in hours (default 72).
        
    Returns:
        pd.DataFrame: Contains ['coint_alpha', 'coint_beta', 'coint_residual', 'coint_zscore'].
    """
    cols_lower = {c.lower(): c for c in df.columns}
    
    if 'eth_close' in cols_lower:
        p_eth = df[cols_lower['eth_close']].astype(np.float64)
    elif 'close' in cols_lower:
        p_eth = df[cols_lower['close']].astype(np.float64)
    else:
        raise ValueError("DataFrame must contain 'eth_close' or 'close' column")

    if 'btc_close' in cols_lower:
        p_btc = df[cols_lower['btc_close']].astype(np.float64)
    else:
        # Fallback if single asset (degenerate beta=1.0)
        p_btc = pd.Series(np.ones(len(df)), index=df.index, dtype=np.float64)

    y = np.log(np.maximum(p_eth, 1e-8))
    x = np.log(np.maximum(p_btc, 1e-8))

    cov_xy = y.rolling(window=window, min_periods=max(10, window // 3)).cov(x)
    var_x = x.rolling(window=window, min_periods=max(10, window // 3)).var()
    
    # Dynamic hedge ratio (beta_t)
    beta = cov_xy / (var_x + 1e-12)
    beta = beta.clip(lower=-5.0, upper=10.0)

    # Rolling intercept (alpha_t)
    mean_y = y.rolling(window=window, min_periods=max(10, window // 3)).mean()
    mean_x = x.rolling(window=window, min_periods=max(10, window // 3)).mean()
    alpha = mean_y - beta * mean_x

    # Cointegration residual (epsilon_t)
    epsilon = y - (alpha + beta * x)

    # Rolling 72-hour Z-score of residual
    eps_mean = epsilon.rolling(window=window, min_periods=max(10, window // 3)).mean()
    eps_std = epsilon.rolling(window=window, min_periods=max(10, window // 3)).std().fillna(0.0)
    z_score = (epsilon - eps_mean) / (eps_std + 1e-8)
    z_score = z_score.clip(lower=-8.0, upper=8.0)

    res_df = pd.DataFrame(index=df.index)
    res_df['coint_alpha'] = alpha
    res_df['coint_beta'] = beta
    res_df['coint_residual'] = epsilon
    res_df['coint_zscore'] = z_score

    return res_df


def fit_ols_window(window_df: pd.DataFrame):
    """
    Fits OLS cointegration on a single rolling window slice and returns
    (alpha, beta, current_residual, current_zscore).
    """
    coint_df = compute_rolling_cointegration(window_df, window=len(window_df))
    last_row = coint_df.iloc[-1]
    return (
        float(last_row['coint_alpha']),
        float(last_row['coint_beta']),
        float(last_row['coint_residual']),
        float(last_row['coint_zscore'])
    )
