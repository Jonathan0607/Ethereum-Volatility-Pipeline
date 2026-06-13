import os
import sys
import pandas as pd

# Add local path to import sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules
import data
import strategy
import backtest

def run_pipeline():
    print("==========================================")
    print("   ETHEREUM QUANT STRATEGY PIPELINE       ")
    print("==========================================\n")
    
    # 1. FETCH DATA (The Miner)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    print("[1/4] Fetching fresh data for the training cycle...")
    data.fetch_data()

    # 2. VERIFY HYPERPARAMETERS EXIST
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    
    if not os.path.exists(params_path):
        print("\n[2/3] ERROR: best_params.txt not found!")
        print("       Run the Optuna sandbox first:  python research/optuna_tuner.py")
        return
    else:
        print("\n[2/3] Optimized params found. Proceeding to training.")

    # 3. BACKTEST & VISUALIZE (The Auditor)
    print("\n[3/4] Running Out-of-Sample Backtest & Dashboard...")
    try:
        # Load and Backtest
        df = backtest.load_data()
        results = backtest.run_backtest(df)
        metrics = backtest.calculate_metrics(results)
        backtest.export_json(results, metrics)
        backtest.plot_results(results) 
        
        # Generate Detailed Dashboard
        backtest.plot_dashboard(results)
        
    except Exception as e:
        print(f"CRITICAL ERROR during backtesting: {e}")
        return

    print("\n==========================================")
    print("   PIPELINE SUCCESSFUL.                   ")
    print("   Artifacts Generated:                   ")
    print("   - backtest_results.png                 ")
    print("   - model_dashboard.png                  ")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()