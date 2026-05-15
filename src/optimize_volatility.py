import sys
import os
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings("ignore")

# Add src to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from backtest import (
    load_data, 
    split_data, 
    calculate_features_test, 
    predict_regimes, 
    get_lstm_predictions, 
    calculate_metrics
)

def evaluate_target_volatility(test_df, target_volatility):
    """
    Applies the specified target volatility to the pre-computed test set
    to calculate position sizing and backtest metrics without running 
    the LSTM forward pass multiple times.
    """
    df = test_df.copy()
    
    base_signal = (
        (df['regime_gmm'] == 0) &
        (df['close'] > df['fast_trend']) &
        (df['volume'] > df['vol_ma'])
    )
    vol_threshold = df['volatility'].rolling(24).mean() * 1.8
    lstm_risk_on = df['forecasted_vol'] > vol_threshold
    parabolic_move = df['close'] > (df['slow_trend'] * 1.02)

    long_condition = base_signal & (~lstm_risk_on | parabolic_move)
    df['signal'] = np.where(long_condition, 1, 0)

    # Convert forecasted hourly volatility to daily volatility
    daily_forecasted_vol = df['forecasted_vol'] * np.sqrt(24)

    # Volatility-Scaled Position Sizing
    uncapped_size = target_volatility / (daily_forecasted_vol + 1e-8)
    df['position_size'] = uncapped_size.clip(upper=1.0)
    df.loc[df['signal'] == 0, 'position_size'] = 0.0

    # Weighted strategy returns
    df['strategy_returns'] = (
        df['market_returns'] *
        df['signal'].shift(1) *
        df['position_size'].shift(1)
    )
    df['cumulative_market'] = (1 + df['market_returns']).cumprod()
    df['cumulative_strategy'] = (1 + df['strategy_returns']).cumprod()
    
    # Calculate metrics with verbosity disabled
    return calculate_metrics(df, verbose=False)

def main():
    print("==========================================")
    print("   VOLATILITY-SCALED SIZING OPTIMIZATION  ")
    print("==========================================\n")
    print("Initializing Data and generating AI forecasts...")
    
    # Load and prepare data ONCE to save massive compute time (No retraining)
    df = load_data()
    _, test_raw = split_data(df, verbose=False)
    
    test_df = calculate_features_test(test_raw)
    test_df = predict_regimes(test_df)
    test_df['forecasted_vol'] = get_lstm_predictions(test_df)
    test_df['market_returns'] = test_df['close'].pct_change()
    test_df['fast_trend'] = test_df['close'].ewm(span=20, adjust=False).mean()
    test_df['slow_trend'] = test_df['close'].ewm(span=50, adjust=False).mean()
    test_df['vol_ma'] = test_df['volume'].rolling(window=12).mean()

    results = []
    
    # 1. The Grid: Iterate through target volatility from 0.01 to 0.10 in increments of 0.005
    target_vols = np.arange(0.01, 0.1001, 0.005)
    
    print("\nStarting 1D Grid Search Optimization...")
    for tv in target_vols:
        # 2. Integration: calculate metrics for each target_vol
        metrics = evaluate_target_volatility(test_df, tv)
        
        # 3. Data Logging: Store results in a list of dictionaries
        results.append({
            'Target Volatility': tv,
            'Total Return (%)': metrics['strategy_return'],
            'Sharpe Ratio': metrics['sharpe'],
            'Max Drawdown (%)': metrics['max_drawdown']
        })

    # Convert to DataFrame for matrix presentation
    results_df = pd.DataFrame(results)
    
    # 4. Console Output: Print a clean matrix
    # Formatting columns for display
    display_df = results_df.copy()
    display_df['Target Volatility'] = display_df['Target Volatility'].apply(lambda x: f"{x:.3f} ({x*100:.1f}%)")
    display_df['Total Return (%)'] = display_df['Total Return (%)'].apply(lambda x: f"{x:.2f}%")
    display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.3f}")
    display_df['Max Drawdown (%)'] = display_df['Max Drawdown (%)'].apply(lambda x: f"{x:.2f}%")
    
    print("\n=== OPTIMIZATION MATRIX ===")
    print(display_df.to_string(index=False))

    # 5. The Objective Function: Programmatically identify the optimal run based on highest Sharpe
    optimal_idx = results_df['Sharpe Ratio'].idxmax()
    optimal_run = results_df.loc[optimal_idx]

    print("\n" + "="*40)
    print("   OPTIMAL PARAMETERS IDENTIFIED   ")
    print("="*40)
    print(f"Target Volatility: {optimal_run['Target Volatility']:.3f} ({optimal_run['Target Volatility']*100:.1f}%)")
    print(f"Total Return:      {optimal_run['Total Return (%)']:.2f}%")
    print(f"Sharpe Ratio:      {optimal_run['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown:      {optimal_run['Max Drawdown (%)']:.2f}%")
    print("========================================\n")

if __name__ == "__main__":
    main()
