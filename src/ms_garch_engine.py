"""
Markov-Switching GARCH(1,1)-X Engine with Time-Varying Transition Probabilities (TVTP)
Implements Maximum Likelihood Estimation (MLE) with Haas et al. variance recursions.
Uses Garman-Klass volatility as exogenous variance driver and volume-shock TVTP.
Enforces a fail-closed risk-off prior [0.0, 1.0] upon any MLE divergence or solver failure.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
import warnings

warnings.filterwarnings("ignore")


def compute_garman_klass_volatility(df: pd.DataFrame) -> np.ndarray:
    """
    Computes the Garman-Klass volatility estimator using OHLC data:
    sigma_GK^2 = 0.5 * (ln(H/L))^2 - (2*ln(2) - 1) * (ln(C/O))^2
    
    For multi-asset pairs (ETH and BTC), computes combined spread variance:
    sigma_GK_spread^2 = sigma_GK_ETH^2 + sigma_GK_BTC^2
    
    Returns:
        np.ndarray: 1D array of non-negative Garman-Klass variance estimates.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    const = 2.0 * np.log(2.0) - 1.0  # ~0.38629436
    eps = 1e-8

    has_eth = all(f'eth_{k}' in cols_lower for k in ['open', 'high', 'low', 'close'])
    has_btc = all(f'btc_{k}' in cols_lower for k in ['open', 'high', 'low', 'close'])

    if has_eth and has_btc:
        o1 = np.maximum(df[cols_lower['eth_open']].values.astype(np.float64), eps)
        h1 = np.maximum(df[cols_lower['eth_high']].values.astype(np.float64), eps)
        l1 = np.maximum(df[cols_lower['eth_low']].values.astype(np.float64), eps)
        c1 = np.maximum(df[cols_lower['eth_close']].values.astype(np.float64), eps)

        h1 = np.maximum(h1, np.maximum(o1, np.maximum(c1, l1)))
        l1 = np.minimum(l1, np.minimum(o1, np.minimum(c1, h1)))
        gk1 = 0.5 * (np.log(h1 / l1) ** 2) - const * (np.log(c1 / o1) ** 2)

        o2 = np.maximum(df[cols_lower['btc_open']].values.astype(np.float64), eps)
        h2 = np.maximum(df[cols_lower['btc_high']].values.astype(np.float64), eps)
        l2 = np.maximum(df[cols_lower['btc_low']].values.astype(np.float64), eps)
        c2 = np.maximum(df[cols_lower['btc_close']].values.astype(np.float64), eps)

        h2 = np.maximum(h2, np.maximum(o2, np.maximum(c2, l2)))
        l2 = np.minimum(l2, np.minimum(o2, np.minimum(c2, h2)))
        gk2 = 0.5 * (np.log(h2 / l2) ** 2) - const * (np.log(c2 / o2) ** 2)

        return np.maximum(gk1 + gk2, 1e-12)

    req = ['open', 'high', 'low', 'close']
    missing = [r for r in req if r not in cols_lower]
    if missing:
        if 'close' in cols_lower:
            c = df[cols_lower['close']].values.astype(np.float64)
            ret = np.diff(np.log(np.maximum(c, eps)))
            v = np.var(ret) if len(ret) > 0 else 1e-4
            return np.full(len(df), max(v, 1e-6))
        raise ValueError(f"Missing required OHLC columns: {missing}")

    o = np.maximum(df[cols_lower['open']].values.astype(np.float64), eps)
    h = np.maximum(df[cols_lower['high']].values.astype(np.float64), eps)
    l = np.maximum(df[cols_lower['low']].values.astype(np.float64), eps)
    c = np.maximum(df[cols_lower['close']].values.astype(np.float64), eps)

    h = np.maximum(h, np.maximum(o, np.maximum(c, l)))
    l = np.minimum(l, np.minimum(o, np.minimum(c, h)))

    gk_var = 0.5 * (np.log(h / l) ** 2) - const * (np.log(c / o) ** 2)
    return np.maximum(gk_var, 1e-12)


def compute_volume_shock(df: pd.DataFrame, window=24, ema_span=12) -> np.ndarray:
    """
    Computes normalized rolling volume shock with 12-hour EMA smoothing:
    z_t = EMA_12((V_t - MA_24(V)) / (Std_24(V) + 1e-8))
    Adds persistence to the Markov chain and mitigates hour-over-hour regime chattering.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    
    if 'eth_volume' in cols_lower and 'btc_volume' in cols_lower:
        v1 = pd.Series(df[cols_lower['eth_volume']].values.astype(np.float64))
        v2 = pd.Series(df[cols_lower['btc_volume']].values.astype(np.float64))
        z1 = (v1 - v1.rolling(window, min_periods=1).mean()) / (v1.rolling(window, min_periods=1).std().fillna(0.0) + 1e-8)
        z2 = (v2 - v2.rolling(window, min_periods=1).mean()) / (v2.rolling(window, min_periods=1).std().fillna(0.0) + 1e-8)
        z_raw = 0.5 * (z1.clip(-5.0, 5.0) + z2.clip(-5.0, 5.0))
        return z_raw.ewm(span=ema_span, adjust=False).mean().values

    if 'volume' not in cols_lower:
        return np.zeros(len(df), dtype=np.float64)
    
    v = df[cols_lower['volume']].values.astype(np.float64)
    v_series = pd.Series(v)
    roll_mean = v_series.rolling(window=window, min_periods=1).mean()
    roll_std = v_series.rolling(window=window, min_periods=1).std().fillna(0.0)
    z_raw = (v_series - roll_mean) / (roll_std + 1e-8)
    z_raw = z_raw.clip(lower=-5.0, upper=5.0)
    
    return z_raw.ewm(span=ema_span, adjust=False).mean().values


def extract_model_returns(df: pd.DataFrame) -> np.ndarray:
    """
    Extracts returns for MS-GARCH-X.
    If multi-asset (ETH & BTC), returns spread return r_ETH - r_BTC.
    Otherwise returns single asset log return.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    eps = 1e-8
    
    if 'eth_close' in cols_lower and 'btc_close' in cols_lower:
        c_eth = np.maximum(df[cols_lower['eth_close']].values.astype(np.float64), eps)
        c_btc = np.maximum(df[cols_lower['btc_close']].values.astype(np.float64), eps)
        r_eth = np.diff(np.log(c_eth))
        r_btc = np.diff(np.log(c_btc))
        return r_eth - r_btc
    elif 'close' in cols_lower:
        c = np.maximum(df[cols_lower['close']].values.astype(np.float64), eps)
        return np.diff(np.log(c))
    else:
        raise ValueError("DataFrame must contain 'eth_close' and 'btc_close' or 'close'")


class MSGARCHX:
    """
    Two-State Markov-Switching GARCH(1,1)-X with Time-Varying Transition Probabilities (TVTP).
    
    State 0: Low-Volatility Regime
    State 1: High-Volatility Regime (enforced by post-fit state ordering)
    
    Variance equations:
        sigma_{k, t}^2 = omega_k + alpha_k * eps_{k, t-1}^2 + beta_k * sigma_{k, t-1}^2 + gamma_k * sigma_GK_{t-1}^2
    
    TVTP Transitions:
        p_{00, t} = sigmoid(a0 + b0 * z_t)
        p_{11, t} = sigmoid(a1 + b1 * z_t)
        
    Fail-Closed Policy:
        Upon any MLE divergence or optimization failure, falls back to unconditional variance
        and forces regime probabilities to [0.0, 1.0] (100% High-Vol / Risk-Off).
    """
    
    PARAM_NAMES = [
        'mu0', 'omega0', 'alpha0', 'beta0', 'gamma0',
        'mu1', 'omega1', 'alpha1', 'beta1', 'gamma1',
        'a0', 'b0', 'a1', 'b1'
    ]

    def __init__(self):
        self.fitted_params_ = None
        self.is_fitted = False
        self.mle_failed = False
        self.unconditional_var_ = 1e-4
        self.last_state_prob_ = np.array([0.0, 1.0])  # Default to risk-off prior
        self.last_variances_ = None
        self.last_gk_var_ = 1e-6
        self.last_return_ = 0.0

    def _unpack_params(self, params):
        (mu0, omega0, alpha0, beta0, gamma0,
         mu1, omega1, alpha1, beta1, gamma1,
         a0, b0, a1, b1) = params
        return {
            0: {'mu': mu0, 'omega': omega0, 'alpha': alpha0, 'beta': beta0, 'gamma': gamma0},
            1: {'mu': mu1, 'omega': omega1, 'alpha': alpha1, 'beta': beta1, 'gamma': gamma1},
            'tvtp': {'a0': a0, 'b0': b0, 'a1': a1, 'b1': b1}
        }

    def _forward_filter_fast(self, returns, gk_vars, z_shocks, params, return_full=True):
        """
        High-performance scalar Hamilton filter with Haas et al. variance recursions.
        """
        mu0, omega0, alpha0, beta0, gamma0 = params[0:5]
        mu1, omega1, alpha1, beta1, gamma1 = params[5:10]
        a0, b0, a1, b1 = params[10:14]

        # Parameter stationarity checks
        if (omega0 <= 1e-10 or alpha0 < 0 or beta0 < 0 or gamma0 < 0 or
            omega1 <= 1e-10 or alpha1 < 0 or beta1 < 0 or gamma1 < 0):
            return -1e10, None, None, None
        if (alpha0 + beta0 >= 0.9999 or alpha1 + beta1 >= 0.9999):
            return -1e10, None, None, None

        T = len(returns)
        mean_gk = np.mean(gk_vars) if len(gk_vars) > 0 else 1e-5
        sample_var = np.var(returns) if len(returns) > 0 else 1e-4

        v0 = max(1e-8, (omega0 + gamma0 * mean_gk) / max(1e-4, 1.0 - alpha0 - beta0))
        v1 = max(1e-8, (omega1 + gamma1 * mean_gk) / max(1e-4, 1.0 - alpha1 - beta1))
        v0 = np.clip(v0, 1e-8, 10.0 * sample_var + 1e-4)
        v1 = np.clip(v1, 1e-8, 50.0 * sample_var + 1e-4)

        # Precompute TVTP transition probabilities
        p00_arr = np.clip(expit(a0 + b0 * z_shocks), 1e-6, 1.0 - 1e-6)
        p11_arr = np.clip(expit(a1 + b1 * z_shocks), 1e-6, 1.0 - 1e-6)

        c_norm = 1.0 / np.sqrt(2.0 * np.pi)
        pi0, pi1 = 0.5, 0.5
        log_lik = 0.0

        if return_full:
            filtered_probs = np.zeros((T, 2), dtype=np.float64)
            variances = np.zeros((T, 2), dtype=np.float64)
            unified_vars = np.zeros(T, dtype=np.float64)
        else:
            filtered_probs = None
            variances = None
            unified_vars = None

        for t in range(T):
            r_t = returns[t]
            if t > 0:
                r_prev = returns[t - 1]
                gk_prev = gk_vars[t - 1]
                v0 = max(1e-8, omega0 + alpha0 * ((r_prev - mu0) ** 2) + beta0 * v0 + gamma0 * gk_prev)
                v1 = max(1e-8, omega1 + alpha1 * ((r_prev - mu1) ** 2) + beta1 * v1 + gamma1 * gk_prev)

            p00 = p00_arr[t]
            p11 = p11_arr[t]

            # Prior state probabilities
            pi_prior0 = p00 * pi0 + (1.0 - p11) * pi1
            pi_prior1 = (1.0 - p00) * pi0 + p11 * pi1

            # Densities
            f0 = max(1e-30, (c_norm / np.sqrt(v0)) * np.exp(-0.5 * ((r_t - mu0) ** 2) / v0))
            f1 = max(1e-30, (c_norm / np.sqrt(v1)) * np.exp(-0.5 * ((r_t - mu1) ** 2) / v1))

            marginal = max(1e-30, pi_prior0 * f0 + pi_prior1 * f1)
            log_lik += np.log(marginal)

            # Posterior state probabilities
            pi0 = (pi_prior0 * f0) / marginal
            pi1 = (pi_prior1 * f1) / marginal

            if return_full:
                filtered_probs[t, 0] = pi0
                filtered_probs[t, 1] = pi1
                variances[t, 0] = v0
                variances[t, 1] = v1
                mean_mu = pi0 * mu0 + pi1 * mu1
                unified_vars[t] = max(1e-8, pi0 * (v0 + (mu0 - mean_mu) ** 2) + pi1 * (v1 + (mu1 - mean_mu) ** 2))

        return log_lik, filtered_probs, variances, unified_vars

    def _get_initial_params(self, returns, gk_vars, warm_start_params=None):
        if warm_start_params is not None and len(warm_start_params) == len(self.PARAM_NAMES):
            return np.array(warm_start_params, dtype=np.float64)

        sample_var = np.var(returns) if len(returns) > 0 else 1e-4

        # Regime 0 (Low Vol)
        mu0 = 0.0
        omega0 = max(1e-6, 0.05 * sample_var)
        alpha0 = 0.05
        beta0 = 0.85
        gamma0 = 0.05

        # Regime 1 (High Vol)
        mu1 = 0.0
        omega1 = max(1e-5, 0.20 * sample_var)
        alpha1 = 0.15
        beta1 = 0.70
        gamma1 = 0.10

        # TVTP parameters
        a0 = 2.0
        b0 = -0.5
        a1 = 1.5
        b1 = 0.5

        return np.array([
            mu0, omega0, alpha0, beta0, gamma0,
            mu1, omega1, alpha1, beta1, gamma1,
            a0, b0, a1, b1
        ], dtype=np.float64)

    def _get_bounds(self, sample_var):
        sv = max(sample_var, 1e-6)
        return [
            (-0.05, 0.05), (1e-8, 5.0 * sv), (0.0, 0.5), (0.0, 0.98), (0.0, 0.5),  # Regime 0
            (-0.05, 0.05), (1e-8, 20.0 * sv), (0.0, 0.7), (0.0, 0.95), (0.0, 0.5), # Regime 1
            (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)                    # TVTP
        ]

    def _enforce_state_ordering(self, params, returns, gk_vars, z_shocks):
        """
        Enforces post-fit constraint: Regime 0 is Low-Vol, Regime 1 is High-Vol.
        """
        p = self._unpack_params(params)
        mean_gk = np.mean(gk_vars) if len(gk_vars) > 0 else 1e-5
        
        unc_var0 = (p[0]['omega'] + p[0]['gamma'] * mean_gk) / max(1e-4, 1.0 - p[0]['alpha'] - p[0]['beta'])
        unc_var1 = (p[1]['omega'] + p[1]['gamma'] * mean_gk) / max(1e-4, 1.0 - p[1]['alpha'] - p[1]['beta'])

        if unc_var0 > unc_var1:
            return np.array([
                p[1]['mu'], p[1]['omega'], p[1]['alpha'], p[1]['beta'], p[1]['gamma'],
                p[0]['mu'], p[0]['omega'], p[0]['alpha'], p[0]['beta'], p[0]['gamma'],
                p['tvtp']['a1'], p['tvtp']['b1'], p['tvtp']['a0'], p['tvtp']['b0']
            ], dtype=np.float64)
        return params

    def fit(self, df_or_returns, initial_params=None):
        """
        Fits 2-State MS-GARCH(1,1)-X via bounded Maximum Likelihood Estimation.
        Fails closed to [0.0, 1.0] (100% High-Vol) and unconditional variance if solver diverges.
        """
        if isinstance(df_or_returns, pd.DataFrame):
            df = df_or_returns.copy()
            returns = extract_model_returns(df)
            gk_vars = compute_garman_klass_volatility(df)[1:]
            z_shocks = compute_volume_shock(df)[1:]
        else:
            returns = np.asarray(df_or_returns, dtype=np.float64)
            gk_vars = np.full_like(returns, np.var(returns) if len(returns) > 0 else 1e-4)
            z_shocks = np.zeros_like(returns)

        sample_var = float(np.var(returns)) if len(returns) > 0 else 1e-4
        self.unconditional_var_ = max(sample_var, 1e-6)

        if len(returns) < 50:
            # Insufficient data -> fail closed
            self.is_fitted = True
            self.mle_failed = True
            self.last_state_prob_ = np.array([0.0, 1.0])
            self.last_variances_ = np.array([self.unconditional_var_, self.unconditional_var_])
            self.last_gk_var_ = np.mean(gk_vars) if len(gk_vars) > 0 else 1e-6
            self.last_return_ = returns[-1] if len(returns) > 0 else 0.0
            return self

        if len(returns) > 4000:
            fit_returns = returns[-4000:]
            fit_gk = gk_vars[-4000:]
            fit_z = z_shocks[-4000:]
        else:
            fit_returns = returns
            fit_gk = gk_vars
            fit_z = z_shocks

        fit_sample_var = np.var(fit_returns)
        warm_p = initial_params if initial_params is not None else self.fitted_params_
        init_p = self._get_initial_params(fit_returns, fit_gk, warm_p)
        bounds = self._get_bounds(fit_sample_var)

        def objective(p):
            penalty = 0.0
            for k in (0, 5):
                persist = p[k + 2] + p[k + 3]
                if persist >= 0.999:
                    penalty += 1e5 * (persist - 0.998) ** 2
            
            log_lik, _, _, _ = self._forward_filter_fast(fit_returns, fit_gk, fit_z, p, return_full=False)
            if not np.isfinite(log_lik):
                return 1e10
            return -log_lik + penalty

        try:
            res = minimize(
                objective,
                init_p,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-4, 'disp': False}
            )

            if not res.success and res.fun >= 1e9:
                # MLE Diverged -> Fail closed
                self.mle_failed = True
                self.is_fitted = True
                self.fitted_params_ = None
                self.last_state_prob_ = np.array([0.0, 1.0])
                self.last_variances_ = np.array([self.unconditional_var_, self.unconditional_var_])
                self.last_gk_var_ = gk_vars[-1] if len(gk_vars) > 0 else 1e-6
                self.last_return_ = returns[-1] if len(returns) > 0 else 0.0
                return self

            best_p = res.x
            best_p = self._enforce_state_ordering(best_p, fit_returns, fit_gk, fit_z)
            self.fitted_params_ = best_p
            self.mle_failed = False
            self.is_fitted = True

            _, filtered_probs, variances, _ = self._forward_filter_fast(returns, gk_vars, z_shocks, self.fitted_params_, return_full=True)
            if filtered_probs is not None:
                self.last_state_prob_ = filtered_probs[-1]
                self.last_variances_ = variances[-1]
                self.last_gk_var_ = gk_vars[-1] if len(gk_vars) > 0 else 1e-6
                self.last_return_ = returns[-1] if len(returns) > 0 else 0.0
            else:
                self.mle_failed = True
                self.last_state_prob_ = np.array([0.0, 1.0])

        except Exception:
            # Catch unexpected solver crash and fail closed
            self.mle_failed = True
            self.is_fitted = True
            self.fitted_params_ = None
            self.last_state_prob_ = np.array([0.0, 1.0])
            self.last_variances_ = np.array([self.unconditional_var_, self.unconditional_var_])
            self.last_gk_var_ = gk_vars[-1] if len(gk_vars) > 0 else 1e-6
            self.last_return_ = returns[-1] if len(returns) > 0 else 0.0

        return self

    def predict_volatility(self, df_or_returns=None):
        """
        Predicts 1-step ahead regime probabilities and forecasted variance.
        Returns:
            tuple: (regime_probabilities [prob_low_vol, prob_high_vol], forecasted_variance)
        """
        if not self.is_fitted:
            raise RuntimeError("MSGARCHX model must be fitted before calling predict_volatility")

        if self.mle_failed or self.fitted_params_ is None:
            return np.array([0.0, 1.0]), float(self.unconditional_var_)

        if df_or_returns is None:
            p = self._unpack_params(self.fitted_params_)
            var_k = np.zeros(2)
            for k in (0, 1):
                eps_prev = self.last_return_ - p[k]['mu']
                var_k[k] = (p[k]['omega'] + 
                            p[k]['alpha'] * (eps_prev ** 2) + 
                            p[k]['beta'] * self.last_variances_[k] + 
                            p[k]['gamma'] * self.last_gk_var_)
                var_k[k] = max(1e-8, var_k[k])
            
            p00 = expit(p['tvtp']['a0'])
            p11 = expit(p['tvtp']['a1'])
            P_mat = np.array([[p00, 1.0 - p00],
                              [1.0 - p11, p11]])
            pi_fwd = P_mat.T @ self.last_state_prob_
            pi_fwd = np.clip(pi_fwd, 1e-6, 1.0 - 1e-6)
            pi_fwd /= np.sum(pi_fwd)

            mean_mu = pi_fwd[0] * p[0]['mu'] + pi_fwd[1] * p[1]['mu']
            unified_var = (pi_fwd[0] * (var_k[0] + (p[0]['mu'] - mean_mu) ** 2) +
                           pi_fwd[1] * (var_k[1] + (p[1]['mu'] - mean_mu) ** 2))
            return pi_fwd, float(unified_var)

        if isinstance(df_or_returns, pd.DataFrame):
            df = df_or_returns.copy()
            returns = extract_model_returns(df)
            gk_vars = compute_garman_klass_volatility(df)[1:]
            z_shocks = compute_volume_shock(df)[1:]
        else:
            returns = np.asarray(df_or_returns, dtype=np.float64)
            gk_vars = np.full_like(returns, np.var(returns) if len(returns) > 0 else 1e-4)
            z_shocks = np.zeros_like(returns)

        _, filtered_probs, variances, unified_vars = self._forward_filter_fast(
            returns, gk_vars, z_shocks, self.fitted_params_, return_full=True
        )
        
        if filtered_probs is None or len(filtered_probs) == 0:
            return np.array([0.0, 1.0]), float(self.unconditional_var_)

        return filtered_probs[-1], float(unified_vars[-1])

    def filter_full_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies forward filter across the entire time-series.
        Returns:
            pd.DataFrame: ['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']
        """
        if not self.is_fitted:
            raise RuntimeError("MSGARCHX model must be fitted first")

        if self.mle_failed or self.fitted_params_ is None:
            res_df = pd.DataFrame(index=df.index[1:])
            res_df['prob_low_vol'] = 0.0
            res_df['prob_high_vol'] = 1.0
            res_df['ms_garch_variance'] = self.unconditional_var_
            res_df['ms_garch_volatility'] = np.sqrt(self.unconditional_var_)
            return res_df

        returns = extract_model_returns(df)
        gk_vars = compute_garman_klass_volatility(df)[1:]
        z_shocks = compute_volume_shock(df)[1:]

        _, filtered_probs, _, unified_vars = self._forward_filter_fast(
            returns, gk_vars, z_shocks, self.fitted_params_, return_full=True
        )

        res_df = pd.DataFrame(index=df.index[1:])
        res_df['prob_low_vol'] = filtered_probs[:, 0]
        res_df['prob_high_vol'] = filtered_probs[:, 1]
        res_df['ms_garch_variance'] = unified_vars
        res_df['ms_garch_volatility'] = np.sqrt(unified_vars)
        return res_df
