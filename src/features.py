import pandas as pd
import numpy as np
from arch import arch_model
import warnings
import os

warnings.filterwarnings("ignore")

def calculate_features(df, train_df=None):
    """
    Feature Engineering:
    - Log Returns
    - GARCH(1,1) Conditional Volatility fitted ONLY on train data
    - Momentum features for the LSTM

    Args:
        df:       The full dataframe (or test-only slice) to engineer features on.
        train_df: The training slice used to FIT the GARCH model. If None, GARCH
                  is fitted on df itself (only safe when df IS the training set).

    Fix: Previously GARCH was fit on the full dataset before the train/test split,
    leaking future volatility information into the test set. Now the model is fit
    strictly on train_df, and conditional volatility for df is recovered via
    a filtered recursion using the saved GARCH parameters.
    """
    df = df.copy()

    # 1. Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    # 2. GARCH(1,1) — fit on training data only
    print("Fitting GARCH(1,1) Model on TRAINING DATA only...")

    fit_target = train_df.copy() if train_df is not None else df.copy()
    fit_target['log_ret'] = np.log(fit_target['close'] / fit_target['close'].shift(1))
    fit_target.dropna(inplace=True)

    train_returns = fit_target['log_ret'] * 100

    garch_model = arch_model(
        train_returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal'
    )
    res = garch_model.fit(disp='off')

    # Save GARCH params so backtest.py can load them without refitting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params = {
        'mu':    float(res.params['mu']),
        'omega': float(res.params['omega']),
        'alpha': float(res.params.get('alpha[1]', res.params.get('alpha', 0.05))),
        'beta':  float(res.params.get('beta[1]',  res.params.get('beta',  0.90))),
    }
    params_path = os.path.join(current_dir, '..', 'garch_params.npy')
    np.save(params_path, params)
    print(f"GARCH params saved → {params_path}")

    # 3. Roll GARCH variance forward over df using saved params
    # This is a strict causal filter — no future data is used.
    df['garch_vol'] = _apply_garch(df['log_ret'] * 100, params) / 100

    # 4. Overwrite 'volatility' column with GARCH estimate
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']

    # 5. Rolling vol (fallback / comparison only)
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()

    df.dropna(inplace=True)

    print("Feature Engineering Complete.")
    return df


def calculate_features_test(df):
    """
    Applies saved GARCH params to the test set WITHOUT refitting.
    Call this in backtest.py instead of calculate_features() for the test slice.
    """
    df = df.copy()

    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'garch_params.npy')

    if not os.path.exists(params_path):
        raise FileNotFoundError(
            "garch_params.npy not found. Run train.py first to fit GARCH on training data."
        )

    params = np.load(params_path, allow_pickle=True).item()
    print(f"Loaded GARCH params: {params}")

    df['garch_vol'] = _apply_garch(df['log_ret'] * 100, params) / 100
    df['volatility'] = df['garch_vol']
    df['returns']    = df['log_ret']
    df['rolling_vol'] = df['log_ret'].rolling(window=24).std()

    df.dropna(inplace=True)

    print("Test Feature Engineering Complete (no GARCH refit).")
    return df


def _apply_garch(returns_pct: pd.Series, params: dict) -> pd.Series:
    """
    Rolls GARCH(1,1) variance forward using fixed params.
    Initialises variance at the unconditional variance of the series.

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}
    """
    mu    = params['mu']
    omega = params['omega']
    alpha = params['alpha']
    beta  = params['beta']

    vals = returns_pct.values
    n    = len(vals)

    # Unconditional variance as starting point
    uncond_var = omega / max(1 - alpha - beta, 1e-6)
    sigma2 = np.full(n, uncond_var)

    for t in range(1, n):
        eps2       = (vals[t - 1] - mu) ** 2
        sigma2[t]  = omega + alpha * eps2 + beta * sigma2[t - 1]

    return pd.Series(np.sqrt(sigma2), index=returns_pct.index)