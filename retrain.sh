#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "--- Starting Chronos Weekly Retraining Cycle: $(date) ---"

# 1. Navigate to your absolute local project directory
# 2. Run the full data fetch and training pipeline
# (If you use a virtual environment like conda or venv, activate it here first)
. venv/bin/activate
/usr/bin/env python3 src/pipeline.py

echo ">>> Training complete. Pushing fresh brain to production node..."

# 3. Securely copy the updated weights to the DigitalOcean server
scp *.npy *.pkl *.pth *.json best_params.txt root@142.93.189.92:~/Ethereum-Volatility-Pipeline/src/

echo ">>> Weights transferred. Hot-swapping the production API..."

# 4. SSH into the server and restart ONLY the API container to load the new memory
# (The stream_engine stays online so you don't drop any live WebSocket ticks)
ssh root@142.93.189.92 "cd ~/Ethereum-Volatility-Pipeline && docker compose restart api"

echo "--- Cycle Complete: $(date) ---"