import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

def detect_regimes(df, n_components=2):
    """
    Standard GMM:
    Regime 0: Low Volatility (Safe)
    Regime 1: High Volatility (Risk)
    """
    df = df.copy()
    X = df['volatility'].values.reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    df['regime'] = gmm.predict(X)
    
    # Force Regime 1 to be the High Volatility one
    vol_0 = df[df['regime'] == 0]['volatility'].mean()
    vol_1 = df[df['regime'] == 1]['volatility'].mean()
    
    if vol_0 > vol_1:
        df['regime'] = 1 - df['regime']
    
    print(f"   [Result] Regimes Detected (2-State).")
    return df