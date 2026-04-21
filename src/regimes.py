import pandas as pd
import numpy as np
import pickle
import os
from sklearn.mixture import GaussianMixture


def fit_and_save_regimes(train_df: pd.DataFrame,
                         vol_col: str = 'volatility',
                         n_components: int = 2) -> pd.DataFrame:
    """
    Fits a GMM on TRAINING DATA ONLY and saves the fitted model to disk.

    Fix: Previously detect_regimes() was called on the test set, which means
    the GMM learned regime boundaries from future (test) data — a form of
    lookahead bias. Now we separate fitting (train) from predicting (test).

    Call this from train.py after calculate_features().

    Returns:
        train_df with a 'regime' column added.
    """
    train_df = train_df.copy()

    if vol_col not in train_df.columns:
        raise ValueError(f"Column '{vol_col}' not found for regime detection.")

    X = train_df[vol_col].values.reshape(-1, 1)

    low_seed  = np.percentile(X, 10)
    high_seed = np.percentile(X, 90)
    means_init = np.array([[low_seed], [high_seed]])

    gmm = GaussianMixture(
        n_components=n_components,
        means_init=means_init,
        random_state=42
    )
    gmm.fit(X)

    # Enforce label convention: 0 = Low Vol, 1 = High Vol
    gmm = _enforce_label_order(gmm)

    # Save fitted GMM for use in backtest
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gmm_path = os.path.join(current_dir, '..', 'gmm_model.pkl')
    with open(gmm_path, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"[Regimes] Fitted GMM saved → {gmm_path}")

    train_df['regime'] = gmm.predict(X)

    vol_0 = train_df[train_df['regime'] == 0][vol_col].mean()
    vol_1 = train_df[train_df['regime'] == 1][vol_col].mean()
    print(f"[Regimes] Train — Low Vol Mean: {vol_0:.4f} | High Vol Mean: {vol_1:.4f}")

    return train_df


def predict_regimes(df: pd.DataFrame,
                    vol_col: str = 'volatility') -> pd.DataFrame:
    """
    Loads the saved GMM and predicts regimes on df (test set or any new data).
    Does NOT refit — strictly uses the boundaries learned from training data.

    Call this from backtest.py instead of detect_regimes().
    """
    df = df.copy()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    gmm_path = os.path.join(current_dir, '..', 'gmm_model.pkl')

    if not os.path.exists(gmm_path):
        raise FileNotFoundError(
            "gmm_model.pkl not found. Run train.py first to fit the GMM on training data."
        )

    with open(gmm_path, 'rb') as f:
        gmm = pickle.load(f)

    if vol_col not in df.columns:
        raise ValueError(f"Column '{vol_col}' not found for regime prediction.")

    X = df[vol_col].values.reshape(-1, 1)
    df['regime'] = gmm.predict(X)

    vol_0 = df[df['regime'] == 0][vol_col].mean()
    vol_1 = df[df['regime'] == 1][vol_col].mean()
    print(f"[Regimes] Test  — Low Vol Mean: {vol_0:.4f} | High Vol Mean: {vol_1:.4f}")

    return df


# ---------------------------------------------------------------------------
# Legacy wrapper — kept so any code still calling detect_regimes() on training
# data doesn't break, but it now saves the GMM as a side-effect.
# ---------------------------------------------------------------------------
def detect_regimes(df: pd.DataFrame,
                   vol_col: str = 'volatility',
                   n_components: int = 2) -> pd.DataFrame:
    """
    Backward-compatible wrapper. Safe to call on TRAINING data only.
    Fits the GMM and saves it — equivalent to fit_and_save_regimes().

    DO NOT call this on test data. Use predict_regimes() instead.
    """
    print("[Regimes] detect_regimes() called — fitting on provided data and saving GMM.")
    return fit_and_save_regimes(df, vol_col=vol_col, n_components=n_components)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _enforce_label_order(gmm: GaussianMixture) -> GaussianMixture:
    """
    Ensures component 0 has the lower mean (Low Vol) and component 1 has the
    higher mean (High Vol) by swapping internals if necessary.
    """
    means = gmm.means_.flatten()
    if means[0] <= means[1]:
        return gmm  # Already correct

    print("[Regimes] Swapping GMM components to enforce Low=0 / High=1 convention.")

    # Swap means, covariances, weights, and precisions
    gmm.means_             = gmm.means_[[1, 0]]
    gmm.covariances_       = gmm.covariances_[[1, 0]]
    gmm.weights_           = gmm.weights_[[1, 0]]
    gmm.precisions_        = gmm.precisions_[[1, 0]]
    gmm.precisions_chol_   = gmm.precisions_chol_[[1, 0]]

    return gmm