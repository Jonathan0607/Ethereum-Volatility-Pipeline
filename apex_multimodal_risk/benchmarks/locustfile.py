"""
Headless Locust Load Benchmark for APEX Multimodal Risk Microservice.
Supports TWO profiles:
- Profile 1 (Cold / Worst-Case Unique Headlines): 100% unique synthetic headlines, testing raw throughput without cache hits.
- Profile 2 (Realistic Production Shared-Macro Load): 70% market-only, 25% breaking shared headlines (high cache hits), 5% feedback.
"""

import os
import uuid
import random
import numpy as np
from locust import HttpUser, task, between

ASSETS = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "AVAXUSDT", "LINKUSDT"]

SHARED_CRISIS_HEADLINES = [
    "Major exchange halts crypto withdrawals citing severe liquidity deficit",
    "Smart contract exploit drains 75M from leading DeFi lending protocol",
    "SEC announces immediate enforcement action and subpoena for crypto market makers",
    "Validators report sudden consensus stalling causing mainnet transaction halt",
    "Federal Reserve raises rates unexpectedly causing global risk-off selloff",
    "Whale wallet dumps 50,000 ETH onto spot markets triggering cascading liquidations",
    "Stablecoin depegs to 0.82 following mass collateral liquidation spiral",
]

SHARED_NORMAL_HEADLINES = [
    "Ethereum layer 2 transactions reach new daily active address high",
    "Solana ecosystem hackathon concludes with record developer participation",
    "Bitcoin institutional inflows surge across ETF products this week",
    "DeFi governance forum votes in favor of protocol parameter upgrade",
]

PROFILE = os.environ.get("LOCUST_PROFILE", "profile2").lower()


if PROFILE == "profile1":
    class Profile1ColdUser(HttpUser):
        """100% unique synthetic headlines to test raw throughput with 0% cache hits."""
        wait_time = between(0.005, 0.025)

        @task
        def score_unique_headline(self):
            asset = random.choice(ASSETS)
            dummy_window = np.random.randn(48, 16).tolist()
            unique_headline = f"Breaking crypto event {uuid.uuid4().hex[:12]}: unusual volatility spike and spread widening detected"
            payload = {
                "asset": asset,
                "market_window": dummy_window,
                "headline": unique_headline,
                "timeout_ms": 150.0,
            }
            with self.client.post("/v1/score", json=payload, catch_response=True) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("degraded_fallback", False):
                        resp.failure("SLA exceeded: degraded fallback returned")
                    else:
                        resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}: {resp.text}")

else:
    class Profile2ProductionUser(HttpUser):
        """Realistic production workload: 70% market-only, 25% shared headlines, 5% causal feedback."""
        wait_time = between(0.005, 0.025)

        @task(14)
        def score_market_only(self):
            asset = random.choice(ASSETS)
            dummy_window = np.random.randn(48, 16).tolist()
            payload = {
                "asset": asset,
                "market_window": dummy_window,
                "timeout_ms": 100.0,
            }
            with self.client.post("/v1/score", json=payload, catch_response=True) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("degraded_fallback", False):
                        resp.failure("SLA exceeded: degraded fallback returned")
                    else:
                        resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}: {resp.text}")

        @task(5)
        def score_shared_headline(self):
            asset = random.choice(ASSETS)
            dummy_window = np.random.randn(48, 16).tolist()
            headline = random.choice(SHARED_CRISIS_HEADLINES if random.random() < 0.7 else SHARED_NORMAL_HEADLINES)
            payload = {
                "asset": asset,
                "market_window": dummy_window,
                "headline": headline,
                "timeout_ms": 100.0,
            }
            with self.client.post("/v1/score", json=payload, catch_response=True) as resp:
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("degraded_fallback", False):
                        resp.failure("SLA exceeded: degraded fallback returned")
                    else:
                        resp.success()
                else:
                    resp.failure(f"Status {resp.status_code}: {resp.text}")

        @task(1)
        def send_causal_feedback(self):
            asset = random.choice(ASSETS)
            payload = {
                "asset": asset,
                "target_bar_time": "2024-08-05T00:00:00Z",
                "current_time": "2024-08-05T06:00:00Z",
                "realized_fwd_min_ret_6h": float(np.random.uniform(-0.15, 0.02)),
            }
            with self.client.post("/v1/feedback", json=payload, catch_response=True) as resp:
                if resp.status_code == 200:
                    resp.success()
                else:
                    resp.failure(f"Feedback failure {resp.status_code}: {resp.text}")
