import os
import sys
import pandas as pd

# Add local path to import sibling scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modules
import fetch_data
import tune
import train
import backtest
import visualize

def run_pipeline():
    print("==========================================")
    print("   ETHEREUM QUANT STRATEGY PIPELINE       ")
    print("   Phase: Production (End-to-End)         ")
    print("==========================================\n")
    
    # 1. FETCH DATA (The Miner)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        print("[1/4] Data missing. Fetching fresh data...")
        fetch_data.fetch_data()
    else:
        print("[1/4] Data found. Skipping download.")

    # 2. HYPERPARAMETER TUNING (The Architect)
    # We only tune if we haven't already found the best params.
    # To force re-tuning, just delete 'best_params.txt'.
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    
    if not os.path.exists(params_path):
        print("\n[2/4] No params found. Running Bayesian Optimization...")
        try:
            tune.run_optimization()
        except Exception as e:
            print(f"CRITICAL ERROR during tuning: {e}")
            return
    else:
        print("\n[2/4] optimized params found. Skipping tuning.")

    # 3. TRAIN MODEL (The Teacher)
    print("\n[3/4] Training Model on History...")
    try:
        train.train_model()
    except Exception as e:
        print(f"CRITICAL ERROR during training: {e}")
        return

    # 4. BACKTEST & VISUALIZE (The Auditor)
    print("\n[4/4] Running Out-of-Sample Backtest & Dashboard...")
    try:
        # Load and Backtest
        df = backtest.load_data()
        results = backtest.run_backtest(df)
        backtest.calculate_metrics(results)
        backtest.plot_results(results) # Saves backtest_results.png
        
        # Generate Detailed Dashboard
        # We re-run the visualizer logic on the full dataset for the final chart
        visualize.run_visualizer()
        
    except Exception as e:
        print(f"CRITICAL ERROR during backtesting: {e}")
        return

    print("\n==========================================")
    print("   PIPELINE SUCCESSFUL.                   ")
    print("   Artifacts Generated:                   ")
    print("   - lstm_model.pth                       ")
    print("   - backtest_results.png                 ")
    print("   - model_dashboard.png                  ")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()