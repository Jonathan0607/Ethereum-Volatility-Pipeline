import os
import sys
import ast

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import calculate_features
import tune
import train  # <--- WE IMPORT THE TRAINER MODULE

# --- CONFIG ---
# We still need these for data loading checks, but the heavy lifting moves to train.py
import pandas as pd

def load_best_params():
    """
    Loads hyperparameters from best_params.txt.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(current_dir, '..', 'best_params.txt')
    
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            params = ast.literal_eval(f.read())
        print(f"\n[INFO] Loaded Optimized Hyperparameters: {params}")
        return params
    else:
        print("\n[INFO] No params found. Running Optimization first...")
        return None

def run_pipeline():
    print("==========================================")
    print("   ETHEREUM VOLATILITY PIPELINE ENGINE    ")
    print("==========================================\n")
    
    # 1. Data Check
    print("[1/3] Verifying Data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'eth_hourly.csv')
    
    if not os.path.exists(data_path):
        print("Error: Data not found. Run 'python src/fetch_data.py'")
        return

    # We briefly load df just to check if we need to feature engineer, 
    # but we can rely on tune/train to handle the heavy loading.
    
    # 2. Hyperparameter Tuning
    best_params = load_best_params()
    
    if best_params is None:
        print("[2/3] Starting Hyperparameter Optimization...")
        best_params = tune.run_optimization()
    else:
        print("[2/3] Skipping Tuning (Using saved params).")

    # 3. Production Training
    print("\n[3/3] Handing off to Trainer...")
    
    # --- THE REFACTOR ---
    # Instead of rewriting the training loop here, we just call the function.
    # We pass the best_params we found/loaded directly to the trainer.
    train.train_model(hyperparams=best_params)

    print("\n==========================================")
    print("   PIPELINE COMPLETE. MODEL SAVED.        ")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()