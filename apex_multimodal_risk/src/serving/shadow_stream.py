"""
Redis Streams Shadow Scoring Worker.
Operates in strictly non-executing shadow mode (zero real trading orders):
1. Ingests normalized UTC market bars and breaking news events into Redis Streams:
   - Stream 'market:bars'
   - Stream 'news:raw'
2. Consumes stream events, runs 4-stage noise filtering and OnlineFeatureState.
3. Submits POST /v1/score for real-time tail risk assessment.
4. Tracks 6-hour delayed realization and submits causal POST /v1/feedback.
5. Emits structured JSON shadow telemetry.

Supports TWO modes:
- `--mode live`: Polls public Binance hourly klines + crypto RSS feeds.
- `--mode replay`: Replays 25+ historical/synthetic bars through Redis Streams
  (automatically falling back to an in-memory queue if local Redis is unavailable).
"""

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import UTC
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Ensure apex_multimodal_risk is importable
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import feedparser
import httpx
import numpy as np
import pandas as pd
import redis.asyncio as aioredis

from apex_multimodal_risk.src.features.shared_features import OnlineFeatureState
from apex_multimodal_risk.src.utils import PROJECT_ROOT, read_parquet_utc, setup_logger

logger = setup_logger("shadow_stream")

STREAM_MARKET_BARS = "market:bars"
STREAM_NEWS_RAW = "news:raw"

ALLOWLIST_DOMAINS = {
    "reuters.com", "bloomberg.com", "sec.gov", "coindesk.com", "theblock.co",
    "theblockcrypto.com", "blockworks.co", "dlnews.com", "wsj.com", "ft.com",
    "cointelegraph.com", "decrypt.co", "cryptoslate.com", "cryptobriefing.com",
    "beincrypto.com", "bitcoinist.com", "newsbtc.com", "cryptopotato.com",
    "u.today", "dailyhodl.com", "zycrypto.com", "ambcrypto.com", "finbold.com",
    "dailycoin.com",
}

CLICKBAIT_PATTERNS = [
    re.compile(r"\bprice\s+prediction\b", re.IGNORECASE),
    re.compile(r"\bcould\s+hit(?:\s*\$)?\b", re.IGNORECASE),
    re.compile(r"\b100x\b", re.IGNORECASE),
    re.compile(r"\bto\s+the\s+moon\b", re.IGNORECASE),
    re.compile(r"\bweekly\s+recap\b", re.IGNORECASE),
    re.compile(r"\btop\s+5\b", re.IGNORECASE),
    re.compile(r"\bsponsored\b", re.IGNORECASE),
    re.compile(r"\bpress\s+release\b", re.IGNORECASE),
]

ENTITY_REGEX = {
    "sol": re.compile(r"\b(sol|solana)\b", re.IGNORECASE),
    "avax": re.compile(r"\b(avax|avalanche)\b", re.IGNORECASE),
    "link": re.compile(r"\b(link|chainlink)\b", re.IGNORECASE),
    "macro": re.compile(
        r"\b(btc|bitcoin|eth|ethereum|binance|cz|ftx|sbf|sec|gensler|fed|federal\s+reserve|powell|cpi|fomc|inflation|rate\s+hike|rate\s+cut|treasury|liquidity|usdt|tether)\b",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Stream Client Abstraction (Redis vs. In-Memory Fallback)
# ---------------------------------------------------------------------------
class AbstractStreamClient:
    async def publish(self, stream_name: str, payload: dict[str, Any]) -> str:
        raise NotImplementedError

    async def read_stream(
        self, stream_name: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        raise NotImplementedError

    async def close(self):
        pass


class RedisStreamClient(AbstractStreamClient):
    """Production Redis Streams client using aioredis."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: aioredis.Redis | None = None

    async def connect(self):
        self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
        await self.redis.ping()
        logger.info(f"Connected to Redis at {self.redis_url}")

    async def publish(self, stream_name: str, payload: dict[str, Any]) -> str:
        if self.redis is None:
            await self.connect()
        # Serialize all values to string for Redis
        serialized = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) for k, v in payload.items()}
        msg_id = await self.redis.xadd(stream_name, serialized)
        return str(msg_id)

    async def read_stream(
        self, stream_name: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        if self.redis is None:
            await self.connect()
        entries = await self.redis.xread({stream_name: last_id}, count=count, block=50)
        results = []
        if entries:
            for s_name, msg_list in entries:
                for m_id, m_dict in msg_list:
                    # Parse deserialized values
                    parsed = {}
                    for k, v in m_dict.items():
                        try:
                            parsed[k] = json.loads(v)
                        except (json.JSONDecodeError, TypeError):
                            parsed[k] = v
                    results.append((m_id, parsed))
        return results

    async def close(self):
        if self.redis:
            await self.redis.aclose()


class InMemoryStreamClient(AbstractStreamClient):
    """In-memory streaming fallback using ordered lists and monotonic IDs."""

    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict[str, Any]]]] = {
            STREAM_MARKET_BARS: [],
            STREAM_NEWS_RAW: [],
        }
        self.seq = 0
        self.lock = asyncio.Lock()

    async def publish(self, stream_name: str, payload: dict[str, Any]) -> str:
        async with self.lock:
            self.seq += 1
            msg_id = f"{int(time.time() * 1000)}-{self.seq}"
            if stream_name not in self.streams:
                self.streams[stream_name] = []
            self.streams[stream_name].append((msg_id, payload.copy()))
            return msg_id

    async def read_stream(
        self, stream_name: str, last_id: str = "0-0", count: int = 100
    ) -> list[tuple[str, dict[str, Any]]]:
        async with self.lock:
            items = self.streams.get(stream_name, [])
            if last_id == "0-0":
                results = items[:count]
            else:
                idx = -1
                for i, (m_id, _) in enumerate(items):
                    if m_id == last_id:
                        idx = i
                        break
                results = items[idx + 1: idx + 1 + count] if idx != -1 else items[:count]
            return results

    async def close(self):
        self.streams.clear()


async def get_stream_client(
    redis_url: str = "redis://localhost:6379/0", force_in_memory: bool = False
) -> AbstractStreamClient:
    """Factory creating RedisStreamClient or falling back to InMemoryStreamClient."""
    if force_in_memory:
        logger.info("Using in-memory stream client (forced).")
        return InMemoryStreamClient()

    try:
        client = RedisStreamClient(redis_url)
        await client.connect()
        return client
    except Exception as e:
        logger.warning(f"Could not connect to Redis at {redis_url} ({e}). Falling back to InMemoryStreamClient.")
        return InMemoryStreamClient()


# ---------------------------------------------------------------------------
# Noise Filtering Helper
# ---------------------------------------------------------------------------
def filter_news_headline(
    title: str,
    source_url: str,
    published_at: str,
) -> dict[str, Any] | None:
    """Applies domain allowlist, clickbait regex, and entity linking to incoming headline."""
    if not title or not isinstance(title, str) or not title.strip():
        return None

    # 1. Domain verification
    try:
        domain = urlparse(source_url).netloc.lower().replace("www.", "")
    except Exception:
        domain = ""

    # Allow if domain or known source matched, or default to general crypto source
    if domain and not any(allowed in domain for allowed in ALLOWLIST_DOMAINS):
        return None

    clean_title = title.strip()

    # 2. Clickbait check
    for pat in CLICKBAIT_PATTERNS:
        if pat.search(clean_title):
            return None

    # 3. Entity linking
    entity_sol = 1 if ENTITY_REGEX["sol"].search(clean_title) else 0
    entity_avax = 1 if ENTITY_REGEX["avax"].search(clean_title) else 0
    entity_link = 1 if ENTITY_REGEX["link"].search(clean_title) else 0
    entity_macro = 1 if ENTITY_REGEX["macro"].search(clean_title) else 0

    if entity_sol == 0 and entity_avax == 0 and entity_link == 0 and entity_macro == 0:
        # Default to macro relevance if no specific altcoin mentioned
        entity_macro = 1

    return {
        "title": clean_title,
        "source": domain or "rss_feed",
        "published_at": published_at,
        "entity_sol": entity_sol,
        "entity_avax": entity_avax,
        "entity_link": entity_link,
        "entity_macro": entity_macro,
    }


# ---------------------------------------------------------------------------
# Shadow Streaming Worker
# ---------------------------------------------------------------------------
class ShadowStreamWorker:
    """
    Shadow worker consuming market and news streams, executing OnlineFeatureState,
    submitting scoring requests to FastAPI service, and applying causal delayed feedback.
    """

    def __init__(
        self,
        service_url: str = "http://localhost:8000",
        redis_url: str = "redis://localhost:6379/0",
        universe: list[str] | None = None,
        force_in_memory: bool = False,
        http_client: httpx.AsyncClient | None = None,
    ):
        self.service_url = service_url.rstrip("/")
        self.redis_url = redis_url
        self.universe = universe or ["SOLUSDT", "AVAXUSDT", "LINKUSDT"]
        self.force_in_memory = force_in_memory
        self.custom_http_client = http_client

        self.stream_client: AbstractStreamClient | None = None
        self.feature_state: OnlineFeatureState = OnlineFeatureState(universe=self.universe)
        # Buffer of (target_bar_ts, asset, close_T) awaiting delayed feedback at T+6h
        self.feedback_delay_queue: list[dict[str, Any]] = []
        self.history_close_lookup: dict[str, dict[pd.Timestamp, float]] = {s: {} for s in self.universe}

        self.last_market_id = "0-0"
        self.last_news_id = "0-0"
        self.running = False
        self.telemetry_log: list[dict[str, Any]] = []

    async def initialize(self):
        if self.stream_client is None:
            self.stream_client = await get_stream_client(self.redis_url, self.force_in_memory)

    async def get_http_client(self) -> httpx.AsyncClient:
        if self.custom_http_client is not None:
            return self.custom_http_client
        return httpx.AsyncClient(base_url=self.service_url, timeout=5.0)

    async def process_bar(
        self,
        timestamp: pd.Timestamp,
        ohlcv_bar_dict: dict[str, dict[str, float]],
        news_items: list[dict[str, Any]] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> list[dict[str, Any]]:
        """
        Calculates incremental features via OnlineFeatureState, sends /v1/score,
        checks delayed feedback at T >= T_target + 6h, and logs shadow telemetry.
        """
        close_client = False
        if client is None:
            client = await self.get_http_client()
            if self.custom_http_client is None:
                close_client = True

        bar_results = []
        try:
            # Step OnlineFeatureState
            vecs = self.feature_state.step(timestamp, ohlcv_bar_dict, news_items=news_items)

            # Record current close for realized return calculations
            for sym in self.universe:
                if sym in ohlcv_bar_dict:
                    c_val = float(ohlcv_bar_dict[sym]["close"])
                    self.history_close_lookup[sym][timestamp] = c_val
                    # Queue for 6h delayed feedback
                    self.feedback_delay_queue.append(
                        {
                            "target_bar_ts": timestamp,
                            "asset": sym,
                            "close_T": c_val,
                        }
                    )

            # Call POST /v1/score for each asset
            for sym in self.universe:
                vec = vecs[sym]
                _ = self.feature_state.get_named_features(sym)
                active_headline = None
                if news_items:
                    prefix = sym.replace("USDT", "").lower()
                    matched = [
                        n["title"] for n in news_items
                        if (n.get(f"entity_{prefix}", 0) == 1 or n.get("entity_macro", 0) == 1)
                    ]
                    if matched:
                        active_headline = matched[-1]

                score_payload = {
                    "asset": sym,
                    "market_features": vec[:16].tolist(),
                    "headline": active_headline,
                    "timestamp": timestamp.isoformat(),
                    "timeout_ms": 1000.0,
                }

                resp = await client.post("/v1/score", json=score_payload)
                if resp.status_code == 200:
                    data = resp.json()
                    telemetry = {
                        "event": "shadow_score",
                        "timestamp": timestamp.isoformat(),
                        "asset": sym,
                        "p_crash_cal": data["p_crash_cal"],
                        "raw_var_5pct": data["raw_var_5pct"],
                        "aci_var_5pct": data["aci_var_5pct"],
                        "aci_alpha_t": data["aci_alpha_t"],
                        "risk_regime": data["risk_regime"],
                        "latency_ms": data["latency_ms"],
                    }
                    logger.info(json.dumps(telemetry))
                    self.telemetry_log.append(telemetry)
                    bar_results.append(data)
                else:
                    logger.error(f"Score request failed for {sym}: {resp.status_code} {resp.text}")

            # Check and process causal 6-hour delayed feedback
            # Feedback for bar T is eligible when current timestamp >= T + 6h
            remaining_queue = []
            for item in self.feedback_delay_queue:
                t_bar = item["target_bar_ts"]
                asset = item["asset"]
                c_0 = item["close_T"]

                if timestamp >= t_bar + pd.Timedelta(hours=6):
                    # Compute realized minimum return over [T+1, T+6]
                    future_closes = [
                        self.history_close_lookup[asset].get(t_bar + pd.Timedelta(hours=h))
                        for h in range(1, 7)
                    ]
                    valid_closes = [c for c in future_closes if c is not None and c > 0]
                    if valid_closes:
                        min_future_close = min(valid_closes)
                        realized_min_ret = float((min_future_close - c_0) / c_0)
                    else:
                        realized_min_ret = 0.0

                    fb_payload = {
                        "asset": asset,
                        "target_bar_time": t_bar.isoformat(),
                        "current_time": timestamp.isoformat(),
                        "realized_fwd_min_ret_6h": realized_min_ret,
                    }

                    fb_resp = await client.post("/v1/feedback", json=fb_payload)
                    if fb_resp.status_code == 200:
                        fb_data = fb_resp.json()
                        fb_telemetry = {
                            "event": "shadow_feedback",
                            "target_bar": t_bar.isoformat(),
                            "current_time": timestamp.isoformat(),
                            "asset": asset,
                            "realized_min_ret": round(realized_min_ret, 4),
                            "updated_alpha_t": fb_data["updated_alpha_t"],
                        }
                        logger.info(json.dumps(fb_telemetry))
                        self.telemetry_log.append(fb_telemetry)
                    else:
                        logger.warning(f"Feedback rejected for {asset} at {t_bar}: {fb_resp.text}")
                else:
                    remaining_queue.append(item)

            self.feedback_delay_queue = remaining_queue

        finally:
            if close_client:
                await client.aclose()

        return bar_results

    async def run_replay(
        self,
        n_bars: int = 25,
        delay_sec: float = 0.01,
        synthetic: bool = False,
    ) -> dict[str, Any]:
        """
        Replays n_bars through the streaming worker against FastAPI service.
        Runs with real historical data if present, or synthetic data for fresh-clone CI.
        """
        await self.initialize()
        assert self.stream_client is not None

        aligned_path = PROJECT_ROOT / "apex_multimodal_risk/data/processed/aligned_hourly.parquet"
        use_real = not synthetic and aligned_path.exists()

        if use_real:
            logger.info(f"Replay mode: Loading {n_bars} historical bars from {aligned_path}...")
            df = read_parquet_utc(str(aligned_path))
            unique_ts = sorted(df["timestamp"].unique())
            replay_ts = unique_ts[1000: 1000 + n_bars] if len(unique_ts) >= 1000 + n_bars else unique_ts[:n_bars]
            sample_df = df[df["timestamp"].isin(replay_ts)].copy()
        else:
            logger.info(f"Replay mode: Generating {n_bars} synthetic bars for fresh-clone simulation...")
            dates = pd.date_range("2024-01-01", periods=n_bars, freq="1h", tz=UTC)
            rows = []
            rng = np.random.default_rng(42)
            prices = {"SOLUSDT": 100.0, "AVAXUSDT": 35.0, "LINKUSDT": 15.0}

            for t in dates:
                for sym in self.universe:
                    ret = rng.normal(0, 0.01)
                    prices[sym] *= np.exp(ret)
                    rows.append(
                        {
                            "timestamp": t,
                            "asset": sym,
                            "close": prices[sym],
                            "open": prices[sym] * 0.999,
                            "high": prices[sym] * 1.005,
                            "low": prices[sym] * 0.995,
                            "volume": float(rng.uniform(1e5, 5e5)),
                            "funding_rate": float(rng.normal(0.0001, 0.0002)),
                            "open_interest": float(rng.uniform(1e7, 2e7)),
                            "garch_cond_vol": float(rng.uniform(1.0, 3.0)),
                            "embedding_novelty_score": 0.0,
                            "news_count_24h": 0,
                        }
                    )
            sample_df = pd.DataFrame(rows)
            replay_ts = dates.tolist()

        # 1. Publish bars to Redis Stream 'market:bars'
        for ts in replay_ts:
            sub = sample_df[sample_df["timestamp"] == ts]
            bdict = {}
            for _, r in sub.iterrows():
                bdict[r["asset"]] = {
                    "close": float(r["close"]),
                    "funding_rate": float(r.get("funding_rate", 0.0)),
                    "open_interest": float(r.get("open_interest", r["close"] * 1000.0)),
                    "garch_cond_vol": float(r["garch_cond_vol"]) if "garch_cond_vol" in r else None,
                }
            await self.stream_client.publish(
                STREAM_MARKET_BARS,
                {"timestamp": pd.to_datetime(ts, utc=True).isoformat(), "bars": bdict},
            )

        # 2. Publish sample news items to 'news:raw'
        sample_headlines = [
            ("SEC announces digital asset regulatory update", "sec.gov"),
            ("Decentralized lending protocol audits complete", "coindesk.com"),
            ("Solana core developers release validator performance patch", "theblock.co"),
        ]
        for i, (head, src) in enumerate(sample_headlines):
            if i < len(replay_ts):
                await self.stream_client.publish(
                    STREAM_NEWS_RAW,
                    {
                        "title": head,
                        "source": src,
                        "published_at": pd.to_datetime(replay_ts[i], utc=True).isoformat(),
                    },
                )

        # 3. Consume from streams and execute scoring loop
        client = await self.get_http_client()
        scores_count = 0
        feedbacks_count = 0

        try:
            market_entries = await self.stream_client.read_stream(STREAM_MARKET_BARS, "0-0", count=len(replay_ts))
            news_entries = await self.stream_client.read_stream(STREAM_NEWS_RAW, "0-0", count=100)

            raw_news_list = []
            for _, n_payload in news_entries:
                filtered = filter_news_headline(
                    n_payload.get("title", ""),
                    n_payload.get("source", ""),
                    n_payload.get("published_at", ""),
                )
                if filtered:
                    # Synthetic 384-dim embedding
                    filtered["embedding"] = np.random.randn(384).astype(np.float32)
                    filtered["timestamp"] = pd.to_datetime(filtered["published_at"], utc=True)
                    raw_news_list.append(filtered)

            for _, m_payload in market_entries:
                bar_ts = pd.to_datetime(m_payload["timestamp"], utc=True)
                bdict = m_payload["bars"]

                # Attaching arriving news for this hour
                news_for_bar = [n for n in raw_news_list if n["timestamp"] <= bar_ts]

                results = await self.process_bar(bar_ts, bdict, news_items=news_for_bar, client=client)
                scores_count += len(results)
                if delay_sec > 0:
                    await asyncio.sleep(delay_sec)

            feedbacks_count = sum(1 for e in self.telemetry_log if e.get("event") == "shadow_feedback")

        finally:
            if self.custom_http_client is None:
                await client.aclose()
            await self.stream_client.close()

        logger.info(f"Replay complete: {len(replay_ts)} bars, {scores_count} scores, {feedbacks_count} feedbacks.")
        return {
            "bars_processed": len(replay_ts),
            "scores_submitted": scores_count,
            "feedbacks_submitted": feedbacks_count,
            "telemetry_count": len(self.telemetry_log),
        }

    async def run_live(self, poll_interval_sec: float = 60.0):
        """Live shadow loop polling Binance and RSS, publishing to streams, and scoring."""
        await self.initialize()
        assert self.stream_client is not None
        self.running = True
        logger.info(f"Starting live shadow stream worker (poll interval: {poll_interval_sec}s)...")

        async with httpx.AsyncClient(timeout=10.0) as http_c:
            while self.running:
                try:
                    now_utc = pd.Timestamp.now(tz=UTC)
                    bdict = {}

                    # 1. Fetch public Binance klines (1h)
                    for sym in self.universe:
                        url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1h&limit=2"
                        try:
                            r = await http_c.get(url)
                            if r.status_code == 200:
                                klines = r.json()
                                if klines:
                                    last_k = klines[-1]
                                    bdict[sym] = {
                                        "close": float(last_k[4]),
                                        "open": float(last_k[1]),
                                        "high": float(last_k[2]),
                                        "low": float(last_k[3]),
                                        "volume": float(last_k[5]),
                                        "funding_rate": 0.0001,
                                        "open_interest": float(last_k[4]) * 1000.0,
                                    }
                        except Exception as e:
                            logger.warning(f"Error fetching Binance klines for {sym}: {e}")

                    # 2. Fetch public Tier-1 RSS feeds
                    rss_feeds = [
                        "https://www.coindesk.com/arc/outboundfeeds/rss/",
                        "https://cointelegraph.com/rss",
                    ]
                    arriving_news = []
                    for feed_url in rss_feeds:
                        try:
                            parsed_rss = feedparser.parse(feed_url)
                            for entry in parsed_rss.entries[:5]:
                                filtered = filter_news_headline(
                                    entry.get("title", ""),
                                    feed_url,
                                    now_utc.isoformat(),
                                )
                                if filtered:
                                    filtered["embedding"] = np.random.randn(384).astype(np.float32)
                                    arriving_news.append(filtered)
                        except Exception as e:
                            logger.warning(f"Error reading RSS from {feed_url}: {e}")

                    if bdict:
                        # Publish and process bar
                        await self.stream_client.publish(
                            STREAM_MARKET_BARS,
                            {"timestamp": now_utc.isoformat(), "bars": bdict},
                        )
                        await self.process_bar(now_utc, bdict, news_items=arriving_news)

                except Exception as e:
                    logger.exception(f"Unexpected error in live shadow loop: {e}")

                await asyncio.sleep(poll_interval_sec)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
async def async_main():
    parser = argparse.ArgumentParser(description="APEX Multimodal Risk Redis Streams Shadow Scoring Worker")
    parser.add_argument("--mode", choices=["live", "replay"], default="replay", help="Worker mode: live or replay")
    parser.add_argument("--bars", type=int, default=25, help="Number of bars to replay in replay mode")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--service-url", type=str, default="http://localhost:8000", help="Inference microservice URL")
    parser.add_argument("--in-memory", action="store_true", help="Force in-memory streaming client")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic replay data")
    args = parser.parse_args()

    worker = ShadowStreamWorker(
        service_url=args.service_url,
        redis_url=args.redis_url,
        force_in_memory=args.in_memory,
    )

    if args.mode == "replay":
        stats = await worker.run_replay(n_bars=args.bars, synthetic=args.synthetic)
        print(f"\n[REPLAY COMPLETE] Stats: {stats}")
    else:
        await worker.run_live()


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
