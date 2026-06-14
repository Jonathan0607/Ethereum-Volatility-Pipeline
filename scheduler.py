import os
import time
import subprocess
import schedule

print("[Scheduler] Daemon started.")

def run_daily_recon():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Daily Reconciliation Run...")
    res = subprocess.run(["python", "live_recon.py"], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[Error] Reconciliation Stderr:", res.stderr)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Daily Reconciliation Completed.")

def run_weekly_retraining():
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting Weekly Walk-Forward Retraining Run...")
    res = subprocess.run(["python", "research/optuna_tuner.py"], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("[Error] Retraining Stderr:", res.stderr)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Weekly Walk-Forward Retraining Completed.")

# Run Recon every day after the daily close
schedule.every().day.at("17:01").do(run_daily_recon)

# Run Walk-Forward Retraining every Sunday at 02:00
schedule.every().sunday.at("02:00").do(run_weekly_retraining)

print("[Scheduler] Tasks registered successfully:")
print("  - Daily Reconciliation: Every day at 17:01")
print("  - Weekly Retraining: Every Sunday at 02:00")

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(1)
