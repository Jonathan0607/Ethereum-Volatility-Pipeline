import pandas as pd
import numpy as np
import os
import ast
from sklearn.mixture import GaussianMixture
from data import split_data, FEATURE_COLS

def get_gmm_state(closes, z_window=20):
    """Calculates Z-Score and uses a fast GMM to determine mean-reversion state."""
    df = pd.DataFrame({'close': closes})
    df['roll_mean'] = df['close'].rolling(window=z_window).mean()
    df['roll_std'] = df['close'].rolling(window=z_window).std()
    df['z_score'] = (df['close'] - df['roll_mean']) / (df['roll_std'] + 1e-8)
    df.dropna(inplace=True)
    
    if len(df) < 50:
        return 0.0, 1
        
    features = df[['z_score']].values
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(features)
    
    # 0: Oversold (Buy), 1: Neutral, 2: Overbought (Sell)
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    
    current_z = features[-1][0]
    current_cluster = gmm.predict([features[-1]])[0]
    
    # Map cluster to logical state
    if current_cluster == sorted_indices[0]: return current_z, 0 # Oversold
    if current_cluster == sorted_indices[2]: return current_z, 2 # Overbought
    return current_z, 1 # Neutral

def load_best_params():
    import json
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    if not os.path.exists(params_path):
        return {'breakout_window': 48, 'hmm_chop_max': 0.40, 'hmm_trend_min': 0.60, 'rebalance_threshold': 0.15}
    with open(params_path, 'r') as f:
        content = f.read()
        try:
            params = json.loads(content)
        except Exception:
            params = ast.literal_eval(content)
    return params
