import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

def get_hmm_features(closes):
    """Calculates log returns and rolling volatility for the HMM."""
    df = pd.DataFrame({'close': closes})
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['roll_vol'] = df['log_ret'].rolling(24).std()
    df.dropna(inplace=True)
    return df

def get_high_vol_probability(closes):
    """Fits a 2-State HMM and returns the probability of the current tick being in the High-Vol State."""
    df = get_hmm_features(closes)
    if len(df) < 100:
        return 0.0
    
    features = df[['log_ret', 'roll_vol']].values
    
    try:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        # Fit the dynamic HMM
        hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, tol=0.01, random_state=42)
        hmm.fit(features)
        
        # Identify which state is High Volatility (the one with the higher mean rolling volatility)
        high_vol_state = np.argmax(hmm.means_[:, 1])
        
        # Return the probability of being in the High Vol state for the latest tick
        hidden_states_probs = hmm.predict_proba(features)
        return float(hidden_states_probs[-1][high_vol_state])
    except Exception:
        # Return neutral probability on convergence / numerical errors
        return 0.5
