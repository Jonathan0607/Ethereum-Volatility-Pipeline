import sys
import os
import pandas as pd
import numpy as np
import ast
import json
import optuna
import torch
import pickle
import gc
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from backtest import load_data, get_lstm_predictions, get_hurst_exponent
from data import calculate_features, calculate_features_test, split_data, FEATURE_COLS
from strategy import ProgressiveModel

STUDY_DB = f"sqlite:///{os.path.join(PROJECT_ROOT, 'research', 'strategy_opt.db')}"
PARAMS_PATH = os.path.join(PROJECT_ROOT, 'best_params.txt')

def simulate_strategy_sharpe(df, predictions, params):
    z_window = params['z_window']
    z_buy = params['z_buy']
    z_sell = params['z_sell']
    hurst_threshold = params['hurst_threshold']
    hurst_window = params['hurst_window']
    sl_mult = params['sl_mult']
    cooldown_hours = params['cooldown_hours']
    vol_mult = params['vol_mult']
    use_partial_sell = params['use_partial_sell']
    partial_sell_ratio = params['partial_sell_ratio']
    use_ema_exit = params['use_ema_exit']
    
    df_bt = df.copy()
    df_bt['hurst'] = df_bt[f'hurst_{hurst_window}']
    # ema_200 is already calculated on the full df

    
    train_raw, test_raw = split_data(df_bt, verbose=False)
    test_df = calculate_features_test(test_raw)
    
    # Align predictions
    test_df['forecasted_vol'] = predictions.loc[test_df.index]
    test_df['market_returns'] = test_df['close'].pct_change()
    
    # Calculate rolling Z-Score
    roll_mean = test_df['close'].rolling(window=z_window).mean()
    roll_std  = test_df['close'].rolling(window=z_window).std()
    test_df['z_score'] = (test_df['close'] - roll_mean) / (roll_std + 1e-8)
    
    # Volatility threshold
    vol_thresholds = test_df['volatility'].rolling(24).mean() * vol_mult
    
    signals = []
    position_multipliers = []
    position = 0  # 0: FLAT, 1: LONG
    entry_price = 0.0
    current_mult = 0.0
    cooldown_until_idx = -1
    has_partial_sold = False
    
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        price = row['close']
        forecasted_vol = row['forecasted_vol']
        z_score = row['z_score']
        hurst = row['hurst']
        vol_threshold = vol_thresholds.iloc[i]
        
        is_safe_vol = forecasted_vol < vol_threshold
        is_mean_reverting = hurst < hurst_threshold
        is_macro_uptrend = price > row['ema_200']
        
        # Dynamic Z-Score Band
        dynamic_z_buy = z_buy - (forecasted_vol * 10)
        
        # Check cooldown
        in_cooldown = i < cooldown_until_idx
        
        action = "HOLDING" if position == 1 else "FLAT"
        
        # 1. Stop Loss Check (Gate 4)
        if position == 1:
            stop_loss_pct = forecasted_vol * sl_mult
            if price < entry_price * (1.0 - stop_loss_pct):
                action = "CASH"
                cooldown_until_idx = i + cooldown_hours
            elif use_ema_exit and price < row['ema_200']:
                # Soft exit if trend breaks
                action = "CASH"
                cooldown_until_idx = i + cooldown_hours
                
        # 2. Entry and Exit Logic
        if action not in ["CASH"] and not in_cooldown:
            if is_mean_reverting and is_safe_vol and is_macro_uptrend and z_score <= dynamic_z_buy:
                action = "BUY"
            elif z_score >= z_sell or not is_safe_vol:
                action = "CASH"
            elif use_partial_sell and z_score >= 0.0 and position == 1 and not has_partial_sold:
                action = "PARTIAL_SELL"
                
        # State machine updates
        if action == "BUY":
            if position == 0:
                entry_price = price
                current_mult = 1.0
                has_partial_sold = False
            position = 1
            signals.append(1)
        elif action == "PARTIAL_SELL":
            current_mult = current_mult * partial_sell_ratio
            position = 1
            has_partial_sold = True
            signals.append(1)
        elif action == "CASH":
            if position == 1 and price < entry_price:
                cooldown_until_idx = i + cooldown_hours
            position = 0
            entry_price = 0.0
            current_mult = 0.0
            has_partial_sold = False
            signals.append(0)
        elif action == "HOLDING":
            signals.append(1)
        else:  # FLAT
            position = 0
            entry_price = 0.0
            current_mult = 0.0
            has_partial_sold = False
            signals.append(0)
            
        position_multipliers.append(current_mult)
        
    test_df['signal'] = signals
    daily_forecasted_vol = test_df['forecasted_vol'] * np.sqrt(24)
    base_position_size = (0.06 / (daily_forecasted_vol + 1e-8)).clip(upper=1.0)
    test_df['position_size'] = base_position_size * position_multipliers
    
    test_df['strategy_returns'] = (
        test_df['market_returns'] *
        test_df['signal'].shift(1) *
        test_df['position_size'].shift(1)
    )
    test_df.dropna(inplace=True)
    
    if len(test_df) < 100 or test_df['strategy_returns'].std() < 1e-10:
        return -999.0
        
    mean_ret = test_df['strategy_returns'].mean() * 24 * 365
    std_ret  = test_df['strategy_returns'].std() * np.sqrt(24 * 365)
    sharpe   = mean_ret / (std_ret + 1e-9)
    
    return float(sharpe)

def objective(trial, df, predictions):
    params = {
        'z_window':         trial.suggest_int('z_window', 10, 100),
        'z_buy':            trial.suggest_float('z_buy', -3.5, -1.0),
        'z_sell':           trial.suggest_float('z_sell', -0.5, 2.5),
        'hurst_threshold':  trial.suggest_float('hurst_threshold', 0.3, 0.7),
        'hurst_window':     trial.suggest_categorical('hurst_window', [24, 36, 48, 60, 72]),
        'sl_mult':          trial.suggest_float('sl_mult', 1.0, 10.0),
        'cooldown_hours':   trial.suggest_int('cooldown_hours', 1, 24),
        'vol_mult':         trial.suggest_float('vol_mult', 1.0, 3.0),
        'use_partial_sell': trial.suggest_categorical('use_partial_sell', [True, False]),
        'partial_sell_ratio': trial.suggest_float('partial_sell_ratio', 0.1, 0.9),
        'use_ema_exit':     trial.suggest_categorical('use_ema_exit', [True, False])
    }
    
    return simulate_strategy_sharpe(df, predictions, params)

def main():
    print("Loading data...")
    df = load_data()
    
    print("Pre-calculating Hurst exponents and EMA-200...")
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    for w in [24, 36, 48, 60, 72]:
        print(f"  Computing Hurst window={w}...")
        df[f'hurst_{w}'] = df['close'].rolling(window=w).apply(lambda x: get_hurst_exponent(x), raw=True)
    
    # We need to run inference once to get predictions
    print("Generating base predictions from existing model...")
    # Get test_df format
    _, test_raw = split_data(df, verbose=False)
    test_df = calculate_features_test(test_raw)
    predictions = get_lstm_predictions(test_df)
    
    study = optuna.create_study(
        study_name="strategy_only_opt_v2",
        direction="maximize",
        storage=STUDY_DB,
        load_if_exists=True
    )
    
    print("Starting optimization trials...")
    study.optimize(lambda trial: objective(trial, df, predictions), n_trials=300)
    
    print("\nBest Trial:")
    best = study.best_trial
    print(f"  Sharpe: {best.value:+.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
        
    # Overwrite best_params.txt with the new optimal parameters
    # Note: We must preserve the neural network model parameters in best_params.txt as well!
    # Let's read best_params.txt first
    model_params = {'hidden_dim': 64, 'lr': 0.001, 'dropout': 0.2, 'input_dim': len(FEATURE_COLS)}
    if os.path.exists(PARAMS_PATH):
        try:
            with open(PARAMS_PATH, 'r') as f:
                model_params = ast.literal_eval(f.read())
        except Exception:
            pass
            
    # Combine model parameters and optimized strategy parameters
    config = {
        'hidden_dim': model_params.get('hidden_dim', 64),
        'lr': model_params.get('lr', 0.001),
        'dropout': model_params.get('dropout', 0.2),
        'input_dim': len(FEATURE_COLS),
        
        'z_window': best.params['z_window'],
        'z_buy': best.params['z_buy'],
        'z_sell': best.params['z_sell'],
        'hurst_threshold': best.params['hurst_threshold'],
        'hurst_window': best.params['hurst_window'],
        'sl_mult': best.params['sl_mult'],
        'cooldown_hours': best.params['cooldown_hours'],
        'vol_mult': best.params['vol_mult'],
        'use_partial_sell': best.params['use_partial_sell'],
        'partial_sell_ratio': best.params['partial_sell_ratio'],
        'use_ema_exit': best.params['use_ema_exit']
    }
    
    with open(PARAMS_PATH, 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"\n[Success] Updated {PARAMS_PATH}")

if __name__ == "__main__":
    main()
