from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd

def detect_regimes(df, n_components=2):
    """
    Clusters market conditions into regimes using Gaussian Mixture Models.
    
    Args:
        df (pd.DataFrame): Dataframe containing a 'volatility' column.
        n_components (int): Number of regimes (2 = Stable/Volatile).
        
    Returns:
        pd.DataFrame: Dataframe with a new 'regime' column.
    """
    df = df.copy()
    
    # Check if volatility exists
    if 'volatility' not in df.columns:
        raise ValueError("Column 'volatility' missing. Run calculate_features() first.")
    
    # Reshape for Scikit-Learn (Requires 2D array)
    X = df['volatility'].values.reshape(-1, 1)
    
    # Initialize and Fit GMM
    # covariance_type='full' allows clusters to have different shapes/sizes
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X)
    
    # Predict soft labels (Regime 0 vs Regime 1)
    df['regime'] = gmm.predict(X)
    
    # --- ENGINEERING FIX: Enforce Consistency ---
    # GMM labels are random. We want Label 1 to ALWAYS be "High Volatility".
    # We compare the means of the two clusters.
    means = gmm.means_.flatten()
    if means[0] > means[1]:
        # If Cluster 0 is the high-vol one, flip labels so 1 becomes high-vol
        df['regime'] = 1 - df['regime']
    
    return df