#!/bin/bash
# =====================================================================
# Cron wrapper script for Daily live_recon.py
# Runs daily to cross-reference executions, calculate slippage and fees,
# and check strategy decay limits.
# =====================================================================

# Resolve project directory absolute path
PROJECT_DIR="/Users/kingebenezer/Desktop/Coding/Projects/Ethereum Tracker"
cd "$PROJECT_DIR"

# Execute with the virtual environment Python interpreter
venv/bin/python3 live_recon.py >> data/cron_recon.log 2>&1
