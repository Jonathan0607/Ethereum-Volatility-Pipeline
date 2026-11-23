"""
Headless Locust Load Benchmark for APEX Multimodal Risk Microservice (EXP-003e).
Supports TWO profiles:
- Profile 1 (Cold / Worst-Case Unique Headlines): 100% unique synthetic headlines, testing raw throughput without cache hits.
- Profile 2 (Realistic Production Shared-Macro Load): 70% market-only, 25% headlines sampled from a 250-headline realistic pool
  (mixing shared broadcast macro breaking news + asset-specific micro news), 5% causal delayed-feedback.
"""

import os
import random
import uuid

import numpy as np
from locust import HttpUser, between, task

ASSETS = ["SOLUSDT", "ETHUSDT", "BTCUSDT", "AVAXUSDT", "LINKUSDT"]

# ---------------------------------------------------------------------------
# Realistic 250-Headline Pool (50 Crisis Macro + 50 Normal Macro + 150 Asset Micro)
# ---------------------------------------------------------------------------
CRISIS_TEMPLATES = [
    "Major crypto exchange halts all token withdrawals citing severe liquidity deficit in collateral pool",
    "Smart contract reentrancy exploit drains 75M from leading cross-chain lending protocol",
    "SEC issues immediate enforcement subpoenas to top institutional crypto market makers",
    "Validators report sudden consensus stalling causing mainnet block production halt",
    "Federal Reserve announces emergency rate hike triggering cross-asset margin liquidations",
    "Whale wallet dumps 50,000 ETH on spot books causing cascading perpetual liquidations",
    "Algorithmic stablecoin depegs to 0.84 following liquidity drain in primary curve pool",
    "CFTC files civil enforcement action alleging unregistered derivatives trading and fraud",
    "Leading digital asset market maker reports 200M unexpected loss from counterparty default",
    "Critical oracle latency failure leads to bad debt accumulation across decentralized exchanges",
]
SHARED_CRISIS_HEADLINES: list[str] = [
    f"{CRISIS_TEMPLATES[i % len(CRISIS_TEMPLATES)]} [macro-shock-{i+1:02d}]" for i in range(50)
]

NORMAL_MACRO_TEMPLATES = [
    "Institutional ETF products record net inflows surpassing 850M this trading week",
    "Layer 2 daily active addresses reach new all-time high amid global scaling adoption",
    "Global crypto liquidity metrics turn positive as central banks ease bank reserve ratios",
    "Institutional prime broker onboards 40 new systematic hedge fund clients this quarter",
    "Decentralized spot exchange aggregate volume overtakes centralized trading venues",
    "Cross-chain settlement volume sets record with zero protocol security vulnerabilities",
    "Bipartisan digital asset regulatory framework bill advances through key Senate committee",
    "Venture capital deployment in Web3 infrastructure rebounds 45% quarter over quarter",
    "Global asset management firm submits revised application for multi-asset staking ETF",
    "Leading payment processing network settles 10B in merchant stablecoin transactions",
]
SHARED_NORMAL_HEADLINES: list[str] = [
    f"{NORMAL_MACRO_TEMPLATES[i % len(NORMAL_MACRO_TEMPLATES)]} [macro-flow-{i+1:02d}]" for i in range(50)
]

SHARED_BROADCAST_MACRO = SHARED_CRISIS_HEADLINES + SHARED_NORMAL_HEADLINES  # 100 headlines

ASSET_TOPICS = [
    "core development team releases client performance optimization upgrade",
    "governance council approves annual treasury diversification grant allocation",
    "staking participation rate reaches new record percentage of circulating supply",
    "ecosystem foundation announces developer incentive grant program",
    "network throughput hits sustained TPS milestone following protocol upgrade",
    "institutional staking node operators onboard multi-party computation security",
]
ASSET_SPECIFIC_HEADLINES: list[str] = []
for sym in ["SOL", "ETH", "BTC", "AVAX", "LINK"]:
    for top in ASSET_TOPICS:
        for v in range(5):
            ASSET_SPECIFIC_HEADLINES.append(f"{sym} ecosystem: {top} [release-{v+1}]")  # 150 headlines

TOTAL_HEADLINES_POOL = SHARED_BROADCAST_MACRO + ASSET_SPECIFIC_HEADLINES  # Exactly 250 distinct headlines

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
                "timeout_ms": 500.0,
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
        """Realistic production workload: 70% market-only, 25% headlines (250-pool), 5% causal feedback."""
        wait_time = between(0.005, 0.025)

        @task(14)
        def score_market_only(self):
            asset = random.choice(ASSETS)
            dummy_window = np.random.randn(48, 16).tolist()
            payload = {
                "asset": asset,
                "market_window": dummy_window,
                "timeout_ms": 250.0,
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
        def score_headline(self):
            asset = random.choice(ASSETS)
            dummy_window = np.random.randn(48, 16).tolist()
            # 70% shared macro broadcast (high cross-asset sharing), 30% asset-specific long-tail
            if random.random() < 0.70:
                headline = random.choice(SHARED_BROADCAST_MACRO)
            else:
                headline = random.choice(ASSET_SPECIFIC_HEADLINES)

            payload = {
                "asset": asset,
                "market_window": dummy_window,
                "headline": headline,
                "timeout_ms": 250.0,
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
