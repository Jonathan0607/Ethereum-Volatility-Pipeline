import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Optional

def detect_regimes(df: pd.DataFrame, 
                   vol_col: str = 'volatility', 
                   n_components: int = 2) -> pd.DataFrame:
    """
    Detects market regimes using a Gaussian Mixture Model (GMM).
    
    Enforces deterministic labeling:
    - Regime 0: Low Volatility (Safe)
    - Regime 1: High Volatility (Crisis)
    
    Args:
        df (pd.DataFrame): Data containing the volatility feature.
        vol_col (str): Name of the volatility column.
        n_components (int): Number of regimes. Defaults to 2.

    Returns:
        pd.DataFrame: DataFrame with a new 'regime' column.
    """
    df = df.copy()
    
    if vol_col not in df.columns:
        raise ValueError(f"Column '{vol_col}' not found for regime detection.")
        
    # Reshape for Sklearn (samples, features)
    X = df[vol_col].values.reshape(-1, 1)
    
    # --- STRATEGY: Smart Initialization ---
    # We explicitly set the initial means of the clusters to ensure
    # Regime 0 is Low Vol and Regime 1 is High Vol.
    # We pick the 10th percentile (Low) and 90th percentile (High) as seeds.
    low_seed = np.percentile(X, 10)
    high_seed = np.percentile(X, 90)
    
    # Shape: (n_components, n_features) -> (2, 1)
    means_init = np.array([[low_seed], [high_seed]])
    
    # Initialize GMM with manual means
    gmm = GaussianMixture(
        n_components=n_components, 
        means_init=means_init, 
        random_state=42
    )
    
    gmm.fit(X)
    
    # Predict
    df['regime'] = gmm.predict(X)
    
    # Validation (Optional but recommended sanity check)
    vol_0 = df[df['regime'] == 0][vol_col].mean()
    vol_1 = df[df['regime'] == 1][vol_col].mean()
    
    if vol_0 > vol_1:
        # This should theoretically never happen with means_init, but strictly safe:
        print(f"[Warning] GMM Label Mismatch detected. Flipping labels...")
        df['regime'] = 1 - df['regime']
    
    print(f"[Quant] Regimes Detected. Low Vol Mean: {vol_0:.4f} | High Vol Mean: {vol_1:.4f}")
    
    return df