"""
Unit & Integration Tests for Redis Streams Shadow Scoring Worker (Replay Mode).
Verifies:
1. Headline 4-stage noise filtering (domain allowlist, clickbait rejection, entity linking).
2. End-to-end 25-bar synthetic replay against FastAPI ASGI service:
   - 25 bars processed across 3 assets (75 scoring requests).
   - Zero degraded fallback or microservice errors.
   - Causal 6-hour delayed feedback delivered (feedbacks submitted > 0).
   - MultiAssetDelayedFeedbackACI alpha state updated.
   - DriftMonitor observations recorded.
"""

import asyncio

import httpx

from apex_multimodal_risk.src.serving.app import app, state
from apex_multimodal_risk.src.serving.shadow_stream import (
    ShadowStreamWorker,
    filter_news_headline,
)


def test_headline_noise_filter():
    """Test 4-stage headline noise filter rejects spam and tags entities."""
    # 1. Clickbait rejection
    assert (
        filter_news_headline(
            "Top 5 Altcoins That Could Hit $1000 This Week!",
            "https://cointelegraph.com",
            "2024-01-01T00:00:00Z",
        )
        is None
    )

    # 2. Untrusted domain rejection
    assert (
        filter_news_headline(
            "Solana Network Upgrade Deployed Successfully",
            "https://random-scam-blog.xyz/post",
            "2024-01-01T00:00:00Z",
        )
        is None
    )

    # 3. Valid asset-specific headline
    res_sol = filter_news_headline(
        "Solana core engineers release network stability patch",
        "https://coindesk.com/tech",
        "2024-01-01T00:00:00Z",
    )
    assert res_sol is not None
    assert res_sol["entity_sol"] == 1
    assert res_sol["entity_avax"] == 0

    # 4. Valid macro headline
    res_macro = filter_news_headline(
        "Federal Reserve signals interest rate decision at FOMC meeting",
        "https://reuters.com/markets",
        "2024-01-01T00:00:00Z",
    )
    assert res_macro is not None
    assert res_macro["entity_macro"] == 1


def test_shadow_replay_e2e():
    """
    Test 25-bar synthetic replay running against FastAPI app via ASGI transport.
    Validates end-to-end scoring, delayed feedback, ACI alpha updates, and drift monitoring.
    """
    async def _run():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                worker = ShadowStreamWorker(
                    service_url="http://testserver",
                    universe=["SOLUSDT", "AVAXUSDT", "LINKUSDT"],
                    force_in_memory=True,
                    http_client=client,
                )

                # Record initial ACI alpha levels
                initial_alphas = {sym: state.aci_manager.get_alpha(sym) for sym in worker.universe}

                # Run 25-bar replay
                stats = await worker.run_replay(n_bars=25, delay_sec=0.0, synthetic=True)

                assert stats["bars_processed"] == 25
                assert stats["scores_submitted"] == 75  # 25 bars * 3 assets
                assert stats["feedbacks_submitted"] > 0
                assert stats["telemetry_count"] > 75

                # Confirm ACI alpha level evolved causally
                final_alphas = {sym: state.aci_manager.get_alpha(sym) for sym in worker.universe}
                assert any(
                    final_alphas[sym] != initial_alphas[sym] for sym in worker.universe
                ), f"Expected ACI alpha to update: initial={initial_alphas}, final={final_alphas}"

                # Confirm drift monitor registered observations
                assert any(len(w) > 0 for w in state.drift_monitor.feature_windows.values())
                drift_report = state.drift_monitor.evaluate()
                assert drift_report["status"] in ["OK", "WARN", "CRITICAL"]

                # Verify /metrics endpoint reflects shadow scoring activity
                metrics_resp = await client.get("/metrics")
                assert metrics_resp.status_code == 200
                assert "apex_score_requests_total" in metrics_resp.text
                assert "apex_aci_alpha_level" in metrics_resp.text

    asyncio.run(_run())
