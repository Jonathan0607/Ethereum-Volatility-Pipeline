import sys
import os

# Add 'src' to python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tune
import train
import fetch_data
import process_data

def run_pipeline():
    print("==========================================")
    print("   ETHEREUM VOLATILITY PIPELINE ENGINE    ")
    print("==========================================")

    # --- PHASE 1: Data Engineering ---
    print("\n[1/3] Checking Data Integrity...")
    if not os.path.exists("data/eth_hourly.csv"):
        print("Data missing. Fetching from Kraken...")
        fetch_data.fetch_latest() # Assuming you wrap fetch logic in a function
    else:
        print("Data found. Proceeding.")

    # --- PHASE 2: Optimization (The Brain) ---
    print("\n[2/3] Starting Hyperparameter Optimization...")
    # This calls the function we modified in Step 1
    best_params = tune.run_optimization()
    
    print(f"\n>>> OPTIMIZATION WINNER IDENTIFIED: {best_params}")
    print("Pass these parameters to Production Training...")

    # --- PHASE 3: Production Training (The Muscle) ---
    print("\n[3/3] Starting Production Training...")
    # This passes the dictionary into the function we modified in Step 2
    train.train_model(hyperparams=best_params)

    print("\n==========================================")
    print("   PIPELINE COMPLETE. MODEL SAVED.        ")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()