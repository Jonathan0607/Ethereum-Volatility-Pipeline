"""
The Strategy Zoo Module
Implements three mathematically distinct and isolated quantitative execution models:
1. Model A (Cointegrated StatArb): ADF-gated 72h rolling OLS pairs trading with discrete state machine.
2. Model B (Volatility Risk Premium): Directional long model buying underpriced volatility (GARCH < 24h Hist Vol).
3. Model C (Pure Rough Path Momentum): 20D Lead-Lag continuous path signature momentum breakouts with Fractional Kelly sizing.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from scipy.stats import norm, skew, kurtosis
import optuna

from cointegration import compute_rolling_cointegration
from rough_paths import lead_lag_transform, compute_signatures, SignatureDriftPredictor
from ms_garch_engine import MSGARCHX
from kelly_execution import execute_discrete_state_machine, compute_volatility_multiplier, calculate_target_position, execute_kelly_rebalance

FEE_PCT = 0.0020  # 20 bps transaction fee per leg


def evaluate_model_performance(net_returns, active_positions, step_num=0, n_trials=300):
    """
    Computes annualized Sharpe, Sortino, Max Drawdown, Total Return, and Deflated Sharpe Ratio (DSR).
    """
    if net_returns is None or len(net_returns) < 50 or np.std(net_returns) < 1e-10:
        return {'sharpe': -999.0, 'sortino': -999.0, 'max_drawdown': -999.0, 'strategy_return': -999.0, 'dsr': 0.0}

    total_trades = np.sum(np.abs(np.diff(active_positions)) > 1e-4)
    if total_trades == 0:
        return {'sharpe': -999.0, 'sortino': -999.0, 'max_drawdown': -999.0, 'strategy_return': -999.0, 'dsr': 0.0}

    mean_ret = np.mean(net_returns) * 24 * 365
    std_ret = np.std(net_returns) * np.sqrt(24 * 365)
    sharpe = mean_ret / (std_ret + 1e-9)

    negative_returns = np.minimum(net_returns, 0.0)
    downside_dev = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(24 * 365)
    sortino = mean_ret / (downside_dev + 1e-9)

    cum_ret = np.cumprod(1.0 + net_returns)
    strat_return = cum_ret[-1] - 1.0

    peak = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - peak) / peak
    max_dd = np.min(drawdown)

    sr_observed = np.mean(net_returns) / (np.std(net_returns) + 1e-9)
    n = len(net_returns)
    skewness = skew(net_returns)
    excess_kurt = kurtosis(net_returns, fisher=True)

    std_sr_unann = np.sqrt(max(1e-9, (1.0 + 0.5 * sr_observed**2 - skewness * sr_observed + (excess_kurt / 4.0) * sr_observed**2) / (n - 1.0)))
    std_sr_ann = std_sr_unann * np.sqrt(24 * 365)

    n_trials_total = max(1, step_num) * n_trials
    var_sharpes = 0.5
    gamma_const = 0.57721566490153286
    expected_max_ann = 0.0 + np.sqrt(var_sharpes) * (
        (1.0 - gamma_const) * norm.ppf(1.0 - 1.0 / n_trials_total) +
        gamma_const * norm.ppf(1.0 - 1.0 / (n_trials_total * np.e))
    )
    dsr = norm.cdf((sharpe - expected_max_ann) / (std_sr_ann + 1e-9))

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_drawdown': float(max_dd),
        'strategy_return': float(strat_return),
        'dsr': float(dsr)
    }


# =====================================================================
# MODEL A: Cointegrated StatArb
# =====================================================================
class ModelA_CointegratedStatArb:
    name = "Model A (Cointegrated StatArb)"

    @staticmethod
    def simulate_in_sample(X_scaled, y_4h_delta_eps, z_scores, betas, sigmas_spread, r_eth_1h, r_btc_1h, alpha, entry_threshold, stop_loss_threshold, target_vol):
        if len(X_scaled) < 100:
            return None

        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_scaled, y_4h_delta_eps)
        raw_mu = ridge.predict(X_scaled)
        mu_residuals = pd.Series(raw_mu).ewm(span=3, adjust=False).mean().values

        pos_eth, pos_btc, directions, _ = execute_discrete_state_machine(
            z_scores=z_scores,
            mu_residuals=mu_residuals,
            betas=betas,
            sigmas_spread=sigmas_spread,
            entry_threshold=entry_threshold,
            stop_loss_threshold=stop_loss_threshold,
            target_vol=target_vol
        )

        pos_eth_exec = np.roll(pos_eth, 1)
        pos_btc_exec = np.roll(pos_btc, 1)
        pos_eth_exec[0] = 0.0
        pos_btc_exec[0] = 0.0

        gross_returns = pos_eth_exec * r_eth_1h + pos_btc_exec * r_btc_1h
        turnover = np.abs(np.diff(pos_eth_exec, prepend=0.0)) + np.abs(np.diff(pos_btc_exec, prepend=0.0))
        trans_costs = turnover * FEE_PCT
        net_returns = gross_returns - trans_costs
        return net_returns, directions, pos_eth_exec, pos_btc_exec

    @staticmethod
    def run_oos(slice_df, test_start, test_end, params, trained_predictor, trained_msgarch, initial_state=None):
        lookback_window = int(params.get('lookback_window', 24))
        entry_threshold = float(params.get('entry_threshold', 1.75))
        stop_loss_threshold = float(params.get('stop_loss_threshold', 3.50))
        target_vol = float(params.get('target_vol', 0.20))
        coint_window = int(params.get('coint_window', 72))

        trained_predictor.reset_smoother()

        coint_res = compute_rolling_cointegration(slice_df, window=coint_window)
        slice_df = slice_df.join(coint_res, how='left')

        filtered_garch = trained_msgarch.filter_full_series(slice_df)
        slice_df = slice_df.join(filtered_garch[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
        slice_df['prob_high_vol'] = slice_df['prob_high_vol'].ffill().fillna(0.0)
        slice_df['ms_garch_volatility'] = slice_df['ms_garch_volatility'].ffill().fillna(0.01)

        test_df = slice_df[slice_df.index >= test_start].copy()
        if test_df.empty:
            return test_df, initial_state or {}

        curr_dir = initial_state.get('direction', 0.0) if initial_state else 0.0
        locked_beta = initial_state.get('locked_beta', 1.0) if initial_state else 1.0

        timestamps = slice_df.index
        test_indices = np.where(timestamps >= test_start)[0]

        pos_eth_list = []
        pos_btc_list = []
        dir_list = []

        for idx in test_indices:
            window_slice = slice_df.iloc[idx - lookback_window: idx]
            mu_res = trained_predictor.predict_drift_for_window(window_slice, apply_smoothing=True)
            z_t = float(slice_df['coint_zscore'].iloc[idx])
            beta_t = float(slice_df['coint_beta'].iloc[idx]) if not np.isnan(slice_df['coint_beta'].iloc[idx]) else locked_beta
            vol_spread = float(slice_df['ms_garch_volatility'].iloc[idx])

            if curr_dir == 0.0:
                if z_t < -entry_threshold and mu_res > 0:
                    curr_dir = 1.0
                    locked_beta = beta_t
                elif z_t > entry_threshold and mu_res < 0:
                    curr_dir = -1.0
                    locked_beta = beta_t
            elif curr_dir == 1.0:
                if z_t >= 0.0 or abs(z_t) > stop_loss_threshold:
                    curr_dir = 0.0
            elif curr_dir == -1.0:
                if z_t <= 0.0 or abs(z_t) > stop_loss_threshold:
                    curr_dir = 0.0

            g_mult = compute_volatility_multiplier(vol_spread, target_vol=target_vol)
            p_eth = g_mult * curr_dir if curr_dir != 0.0 else 0.0
            p_btc = -g_mult * curr_dir * locked_beta if curr_dir != 0.0 else 0.0

            pos_eth_list.append(p_eth)
            pos_btc_list.append(p_btc)
            dir_list.append(curr_dir)

        test_df['direction'] = dir_list
        test_df['target_eth'] = pos_eth_list
        test_df['target_btc'] = pos_btc_list
        test_df['eth_position'] = test_df['target_eth'].shift(1).fillna(initial_state.get('pos_eth', 0.0) if initial_state else 0.0)
        test_df['btc_position'] = test_df['target_btc'].shift(1).fillna(initial_state.get('pos_btc', 0.0) if initial_state else 0.0)

        eth_c = test_df['eth_close'] if 'eth_close' in test_df.columns else test_df['close']
        btc_c = test_df['btc_close']
        test_df['eth_returns'] = eth_c.pct_change().fillna(0.0)
        test_df['btc_returns'] = btc_c.pct_change().fillna(0.0)

        test_df['market_returns'] = 0.5 * test_df['eth_returns'] + 0.5 * test_df['btc_returns']
        test_df['gross_strategy_returns'] = test_df['eth_position'] * test_df['eth_returns'] + test_df['btc_position'] * test_df['btc_returns']
        
        test_df['turnover_eth'] = test_df['eth_position'].diff().abs().fillna(0.0)
        test_df['turnover_btc'] = test_df['btc_position'].diff().abs().fillna(0.0)
        test_df['transaction_costs'] = (test_df['turnover_eth'] + test_df['turnover_btc']) * FEE_PCT
        test_df['strategy_returns'] = test_df['gross_strategy_returns'] - test_df['transaction_costs']
        test_df['model_selected'] = ModelA_CointegratedStatArb.name

        next_state = {
            'direction': curr_dir,
            'locked_beta': locked_beta,
            'pos_eth': pos_eth_list[-1] if len(pos_eth_list) > 0 else 0.0,
            'pos_btc': pos_btc_list[-1] if len(pos_btc_list) > 0 else 0.0,
            'msgarch_params': trained_msgarch.fitted_params_
        }
        return test_df, next_state


# =====================================================================
# MODEL B: Volatility Risk Premium (VRP)
# =====================================================================
class ModelB_VolatilityRiskPremium:
    name = "Model B (Volatility Risk Premium)"

    @staticmethod
    def simulate_in_sample(garch_vol_series, prob_high_vol_series, hist_vol_24h_series, r_eth_1h, vol_discount, target_leverage, exit_prob_high_vol):
        n = len(r_eth_1h)
        if n < 50:
            return None

        pos_eth = np.zeros(n, dtype=np.float64)
        curr_state = 0.0

        for t in range(n):
            g_vol = garch_vol_series[t]
            h_vol = hist_vol_24h_series[t]
            p_high = prob_high_vol_series[t]

            # Entry: GARCH vol is discounted relative to 24h realized vol (buying underpriced vol)
            if curr_state == 0.0:
                if g_vol < h_vol * (1.0 - vol_discount) and p_high < exit_prob_high_vol:
                    curr_state = 1.0
            elif curr_state == 1.0:
                if g_vol >= h_vol or p_high >= exit_prob_high_vol:
                    curr_state = 0.0

            ann_g_vol = g_vol * np.sqrt(24 * 365)
            f_star = np.clip(target_leverage / (ann_g_vol + 1e-8), 0.1, 1.0)
            pos_eth[t] = curr_state * f_star

        pos_eth_exec = np.roll(pos_eth, 1)
        pos_eth_exec[0] = 0.0

        gross_returns = pos_eth_exec * r_eth_1h
        turnover = np.abs(np.diff(pos_eth_exec, prepend=0.0))
        trans_costs = turnover * FEE_PCT
        net_returns = gross_returns - trans_costs

        return net_returns, pos_eth_exec, pos_eth_exec, np.zeros(n)

    @staticmethod
    def run_oos(slice_df, test_start, test_end, params, trained_msgarch, initial_state=None):
        vol_discount = float(params.get('vol_discount', 0.15))
        target_leverage = float(params.get('target_leverage', 0.25))
        exit_prob_high_vol = float(params.get('exit_prob_high_vol', 0.60))

        # Filter MS-GARCH-X on ETH
        filtered_garch = trained_msgarch.filter_full_series(slice_df)
        slice_df = slice_df.join(filtered_garch[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
        slice_df['prob_high_vol'] = slice_df['prob_high_vol'].ffill().fillna(0.0)
        slice_df['ms_garch_volatility'] = slice_df['ms_garch_volatility'].ffill().fillna(0.01)

        # 24h historical realized volatility
        eth_c = slice_df['eth_close'] if 'eth_close' in slice_df.columns else slice_df['close']
        log_ret = np.diff(np.log(eth_c.values), prepend=0.0)
        slice_df['hist_vol_24h'] = pd.Series(log_ret, index=slice_df.index).rolling(24, min_periods=1).std().fillna(0.01)

        test_df = slice_df[slice_df.index >= test_start].copy()
        if test_df.empty:
            return test_df, initial_state or {}

        curr_state = initial_state.get('vrp_state', 0.0) if initial_state else 0.0
        timestamps = slice_df.index
        test_indices = np.where(timestamps >= test_start)[0]

        pos_eth_list = []
        for idx in test_indices:
            g_vol = float(slice_df['ms_garch_volatility'].iloc[idx])
            h_vol = float(slice_df['hist_vol_24h'].iloc[idx])
            p_high = float(slice_df['prob_high_vol'].iloc[idx])

            if curr_state == 0.0:
                if g_vol < h_vol * (1.0 - vol_discount) and p_high < exit_prob_high_vol:
                    curr_state = 1.0
            elif curr_state == 1.0:
                if g_vol >= h_vol or p_high >= exit_prob_high_vol:
                    curr_state = 0.0

            ann_g_vol = g_vol * np.sqrt(24 * 365)
            f_star = float(np.clip(target_leverage / (ann_g_vol + 1e-8), 0.1, 1.0))
            pos_eth_list.append(curr_state * f_star)

        test_df['target_eth'] = pos_eth_list
        test_df['target_btc'] = 0.0
        test_df['eth_position'] = test_df['target_eth'].shift(1).fillna(initial_state.get('pos_eth', 0.0) if initial_state else 0.0)
        test_df['btc_position'] = 0.0

        test_df['eth_returns'] = eth_c.loc[test_df.index].pct_change().fillna(0.0)
        test_df['btc_returns'] = (test_df['btc_close'].pct_change().fillna(0.0) if 'btc_close' in test_df.columns else pd.Series(0.0, index=test_df.index))
        test_df['market_returns'] = 0.5 * test_df['eth_returns'] + 0.5 * test_df['btc_returns']

        test_df['gross_strategy_returns'] = test_df['eth_position'] * test_df['eth_returns']
        test_df['turnover_eth'] = test_df['eth_position'].diff().abs().fillna(0.0)
        test_df['transaction_costs'] = test_df['turnover_eth'] * FEE_PCT
        test_df['strategy_returns'] = test_df['gross_strategy_returns'] - test_df['transaction_costs']
        test_df['model_selected'] = ModelB_VolatilityRiskPremium.name

        next_state = {
            'vrp_state': curr_state,
            'pos_eth': pos_eth_list[-1] if len(pos_eth_list) > 0 else 0.0,
            'pos_btc': 0.0,
            'msgarch_params': trained_msgarch.fitted_params_
        }
        return test_df, next_state


# =====================================================================
# MODEL C: Pure Rough Path Momentum
# =====================================================================
class ModelC_PureRoughPathMomentum:
    name = "Model C (Pure Rough Path Momentum)"

    @staticmethod
    def simulate_in_sample(X_scaled, y_4h_eth_ret, sigmas_eth, r_eth_1h, alpha, target_leverage, rebalance_deadband):
        if len(X_scaled) < 100:
            return None

        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(X_scaled, y_4h_eth_ret)
        raw_mu = ridge.predict(X_scaled)
        mu_pred = pd.Series(raw_mu).ewm(span=3, adjust=False).mean().values

        raw_targets = calculate_target_position(
            mu=mu_pred,
            sigma=sigmas_eth,
            target_leverage=target_leverage,
            noise_threshold=1e-4
        )

        n = len(raw_targets)
        active_pos = np.zeros(n)
        curr_pos = 0.0
        for i in range(n):
            tgt = raw_targets[i]
            curr_pos, _ = execute_kelly_rebalance(curr_pos, tgt, rebalance_deadband=rebalance_deadband)
            active_pos[i] = curr_pos

        pos_eth_exec = np.roll(active_pos, 1)
        pos_eth_exec[0] = 0.0

        gross_returns = pos_eth_exec * r_eth_1h
        turnover = np.abs(np.diff(pos_eth_exec, prepend=0.0))
        trans_costs = turnover * FEE_PCT
        net_returns = gross_returns - trans_costs

        return net_returns, active_pos, pos_eth_exec, np.zeros(n)

    @staticmethod
    def run_oos(slice_df, test_start, test_end, params, trained_predictor, trained_msgarch, initial_state=None):
        lookback_window = int(params.get('lookback_window', 24))
        target_leverage = float(params.get('target_leverage', 0.25))
        rebalance_deadband = float(params.get('rebalance_deadband', 0.15))

        trained_predictor.reset_smoother()

        filtered_garch = trained_msgarch.filter_full_series(slice_df)
        slice_df = slice_df.join(filtered_garch[['prob_low_vol', 'prob_high_vol', 'ms_garch_variance', 'ms_garch_volatility']], how='left')
        slice_df['prob_high_vol'] = slice_df['prob_high_vol'].ffill().fillna(0.0)
        slice_df['ms_garch_volatility'] = slice_df['ms_garch_volatility'].ffill().fillna(0.01)

        test_df = slice_df[slice_df.index >= test_start].copy()
        if test_df.empty:
            return test_df, initial_state or {}

        curr_pos = initial_state.get('pos_eth', 0.0) if initial_state else 0.0
        timestamps = slice_df.index
        test_indices = np.where(timestamps >= test_start)[0]

        pos_eth_list = []
        for idx in test_indices:
            window_slice = slice_df.iloc[idx - lookback_window: idx]
            mu_eth = trained_predictor.predict_drift_for_window(window_slice, apply_smoothing=True)
            vol_eth = float(slice_df['ms_garch_volatility'].iloc[idx])

            target_f = calculate_target_position(
                mu=mu_eth,
                sigma=vol_eth,
                target_leverage=target_leverage,
                noise_threshold=1e-4
            )
            new_pos, _ = execute_kelly_rebalance(curr_pos, target_f, rebalance_deadband=rebalance_deadband)
            curr_pos = new_pos
            pos_eth_list.append(curr_pos)

        test_df['target_eth'] = pos_eth_list
        test_df['target_btc'] = 0.0
        test_df['eth_position'] = test_df['target_eth'].shift(1).fillna(initial_state.get('pos_eth', 0.0) if initial_state else 0.0)
        test_df['btc_position'] = 0.0

        eth_c = test_df['eth_close'] if 'eth_close' in test_df.columns else test_df['close']
        btc_c = test_df['btc_close'] if 'btc_close' in test_df.columns else eth_c
        test_df['eth_returns'] = eth_c.pct_change().fillna(0.0)
        test_df['btc_returns'] = btc_c.pct_change().fillna(0.0)

        test_df['market_returns'] = 0.5 * test_df['eth_returns'] + 0.5 * test_df['btc_returns']
        test_df['gross_strategy_returns'] = test_df['eth_position'] * test_df['eth_returns']
        test_df['turnover_eth'] = test_df['eth_position'].diff().abs().fillna(0.0)
        test_df['transaction_costs'] = test_df['turnover_eth'] * FEE_PCT
        test_df['strategy_returns'] = test_df['gross_strategy_returns'] - test_df['transaction_costs']
        test_df['model_selected'] = ModelC_PureRoughPathMomentum.name

        next_state = {
            'pos_eth': pos_eth_list[-1] if len(pos_eth_list) > 0 else 0.0,
            'pos_btc': 0.0,
            'msgarch_params': trained_msgarch.fitted_params_
        }
        return test_df, next_state
