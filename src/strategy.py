import numpy as np
import pandas as pd


def compute_unified_signal(
    spot_df: pd.DataFrame, 
    funding_df: pd.DataFrame, 
    current_position: float
) -> float:
    """
    Stateless Signal Generator: Perpetual Funding Rate Squeeze Harvester + 7D Donchian.
    
    Parameters:
    -----------
    spot_df : pd.DataFrame
        Hourly spot OHLCV dataframe (must contain 'close' column and at least 168 rows).
    funding_df : pd.DataFrame
        8-hour perpetual funding rate dataframe (must contain 'fundingRate' column and at least 90 rows).
    current_position : float
        Current portfolio position (0.0 for FLAT/Cash, 1.0 for LONG).
        
    Returns:
    --------
    float: Target position (1.0 for LONG, 0.0 for FLAT/Cash).
    """
    # 1. Validation
    assert len(spot_df) >= 168, f"spot_df must contain at least 168 rows (received {len(spot_df)})"
    assert len(funding_df) >= 90, f"funding_df must contain at least 90 rows (received {len(funding_df)})"
    assert 'close' in spot_df.columns, "spot_df must contain 'close' column"

    # Handle funding rate column variations (e.g., fundingRate, funding_rate)
    fr_col = 'fundingRate' if 'fundingRate' in funding_df.columns else ('funding_rate' if 'funding_rate' in funding_df.columns else funding_df.columns[-1])

    # 2. Donchian 168-Hour (7-Day) Channel Bounds
    close_series = spot_df['close'].astype(float)
    # Channel bounds strictly prior to current bar (or over trailing 168 lookback)
    upper_168 = float(close_series.iloc[-168:].max())
    lower_168 = float(close_series.iloc[-168:].min())
    current_close = float(close_series.iloc[-1])

    # 3. Funding Rate 30-Day (90 8-hour epochs) Rolling Z-Score
    funding_series = funding_df[fr_col].astype(float)
    rolling_mean = float(funding_series.rolling(90, min_periods=30).mean().iloc[-1])
    rolling_std = float(funding_series.rolling(90, min_periods=30).std().iloc[-1])
    if rolling_std == 0 or np.isnan(rolling_std):
        rolling_std = 1e-6

    current_funding = float(funding_series.iloc[-1])
    z_f = float((current_funding - rolling_mean) / (rolling_std + 1e-8))

    # 4. State Machine Execution Logic
    curr_pos = float(current_position)

    if curr_pos == 0.0:
        # Long Entry: Donchian 7-Day Breakout OR Short-Squeeze Surge (Z_F < -2.0)
        if current_close > upper_168 or z_f < -2.0:
            return 1.0
        return 0.0

    elif curr_pos == 1.0:
        # Exit to Cash: Donchian 7-Day Breakdown OR Overheated Leverage Flush (Z_F > 2.5)
        if current_close < lower_168 or z_f > 2.5:
            return 0.0
        return 1.0

    else:
        return curr_pos
