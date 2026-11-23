"""
Automated Runner for Headless Locust Load Benchmark.
Spawns Uvicorn server, waits for health check, runs 30s benchmark with 25 concurrent users,
terminates server, and parses latency & throughput results.
"""

import sys
import time
import subprocess
import signal
from pathlib import Path
import pandas as pd
import httpx

from apex_multimodal_risk.src.utils import setup_logger, PROJECT_ROOT

logger = setup_logger("run_benchmark")


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            resp = httpx.get(f"{url}/health", timeout=1.0)
            if resp.status_code == 200:
                logger.info(f"Microservice is healthy at {url}")
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    host = "127.0.0.1"
    port = 8000
    base_url = f"http://{host}:{port}"
    results_dir = PROJECT_ROOT / "apex_multimodal_risk/benchmarks/results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_prefix = results_dir / "locust"

    # 1. Start Uvicorn microservice in background
    cmd_server = [
        sys.executable,
        "-m",
        "uvicorn",
        "apex_multimodal_risk.src.serving.app:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    logger.info(f"Starting microservice: {' '.join(cmd_server)}")
    proc = subprocess.Popen(cmd_server, cwd=str(PROJECT_ROOT))

    try:
        # 2. Wait for /health
        healthy = wait_for_server(base_url, timeout=20.0)
        if not healthy:
            logger.error("Microservice failed to become healthy within timeout.")
            return 1

        # 3. Run Locust headless (25 users, ramp rate 5/s, 30s runtime)
        locustfile = PROJECT_ROOT / "apex_multimodal_risk/benchmarks/locustfile.py"
        cmd_locust = [
            sys.executable,
            "-m",
            "locust",
            "-f",
            str(locustfile),
            "--headless",
            "-u",
            "25",
            "-r",
            "5",
            "--run-time",
            "30s",
            "--host",
            base_url,
            "--csv",
            str(csv_prefix),
        ]
        logger.info(f"Executing Locust benchmark: {' '.join(cmd_locust)}")
        locust_proc = subprocess.run(cmd_locust, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        logger.info("Locust benchmark finished.")
        if locust_proc.returncode != 0:
            logger.warning(f"Locust returned code {locust_proc.returncode}: {locust_proc.stderr}")

    finally:
        # Cleanly terminate Uvicorn
        logger.info("Stopping Uvicorn microservice...")
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()

    # 4. Parse results and print benchmark summary
    stats_file = results_dir / "locust_stats.csv"
    if not stats_file.exists():
        logger.error(f"Stats file {stats_file} not found.")
        return 1

    df_stats = pd.read_csv(stats_file)
    logger.info(f"Loaded Locust stats from {stats_file}")

    print("\n" + "=" * 115)
    print("SPRINT 3: HEADLESS LOCUST LOAD BENCHMARK (25 Concurrent Users, 30s Duration, CPUExecutionProvider)")
    print("=" * 115)
    print(f"{'Endpoint':25s} | {'Req Count':10s} | {'RPS':8s} | {'Fail %':8s} | {'p50 (ms)':10s} | {'p95 (ms)':10s} | {'p99 (ms)':10s} | {'Max (ms)':10s}")
    print("-" * 115)

    for _, row in df_stats.iterrows():
        name = str(row["Name"]) if pd.notna(row["Name"]) else "Aggregated"
        req_count = int(row["Request Count"])
        fail_count = int(row["Failure Count"])
        fail_pct = (fail_count / req_count * 100.0) if req_count > 0 else 0.0
        rps = float(row["Requests/s"])
        p50 = float(row.get("50%", row.get("Median Response Time", 0.0)))
        p95 = float(row.get("95%", 0.0))
        p99 = float(row.get("99%", 0.0))
        max_time = float(row["Max Response Time"])

        print(f"{name:25s} | {req_count:<10d} | {rps:<8.1f} | {fail_pct:<8.2f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f} | {max_time:<10.2f}")

    print("=" * 115)
    return 0


if __name__ == "__main__":
    sys.exit(main())
