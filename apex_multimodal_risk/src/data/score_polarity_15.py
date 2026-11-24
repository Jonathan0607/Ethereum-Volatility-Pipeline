"""
EXP-005 Polarity Scoring and Causal Sentiment Feature Engineering.

Sequential model execution:
1. Load ElKulako/cryptobert, score 18,939 headlines on MPS/CPU -> s_crypto
2. Unload cryptobert, free memory.
3. Load ProsusAI/finbert, score 18,939 headlines on MPS/CPU -> s_fin
4. Unload finbert, free memory.
5. Entity re-linking across 15 Binance perpetual assets + MACRO.
6. Save scored clusters to apex_multimodal_risk/data/processed/news_polarity_15.parquet
7. Causal rolling alignment onto universe_15_hourly.parquet:
   - Causal filtering (ingested_at <= bar_timestamp)
   - Decayed sentiment (sent_24h, sent_72h)
   - Sentiment surprise (sent_24h - sent_7d_trailing, shift 1)
   - Cross-sectional z-scores (rel_sent, rel_surprise)
8. Save feature panel to apex_multimodal_risk/data/processed/universe_15_features.parquet
"""

import gc
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Strict timeout guardrails
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "apex_multimodal_risk" / "data" / "processed"

ASSETS_15 = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT",
    "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT",
    "UNIUSDT", "ATOMUSDT", "NEARUSDT", "DOTUSDT", "ETCUSDT"
]

ENTITY_REGEXES = {
    "BTCUSDT": re.compile(r"\b(btc|bitcoin)\b", re.IGNORECASE),
    "ETHUSDT": re.compile(r"\b(eth|ethereum|ether)\b", re.IGNORECASE),
    "SOLUSDT": re.compile(r"\b(sol|solana)\b", re.IGNORECASE),
    "AVAXUSDT": re.compile(r"\b(avax|avalanche)\b", re.IGNORECASE),
    "LINKUSDT": re.compile(r"\b(chainlink|\$link|link\s+token)\b", re.IGNORECASE),
    "BNBUSDT": re.compile(r"\b(bnb|binance\s+coin)\b", re.IGNORECASE),
    "XRPUSDT": re.compile(r"\b(xrp|ripple)\b", re.IGNORECASE),
    "ADAUSDT": re.compile(r"\b(ada|cardano)\b", re.IGNORECASE),
    "DOGEUSDT": re.compile(r"\b(doge|dogecoin)\b", re.IGNORECASE),
    "LTCUSDT": re.compile(r"\b(ltc|litecoin)\b", re.IGNORECASE),
    "UNIUSDT": re.compile(r"\b(uni|uniswap)\b", re.IGNORECASE),
    "ATOMUSDT": re.compile(r"\b(cosmos|\$atom|atom\s+token)\b", re.IGNORECASE),
    "NEARUSDT": re.compile(r"\b(near\s+protocol|near\s+foundation|\$near)\b", re.IGNORECASE),
    "DOTUSDT": re.compile(r"\b(polkadot|\$dot|dot\s+token)\b", re.IGNORECASE),
    "ETCUSDT": re.compile(r"\b(etc|ethereum\s+classic)\b", re.IGNORECASE),
    "MACRO": re.compile(r"\b(fed|fomc|powell|sec|cpi|inflation|rate\s+hike|treasury|yields?|macro)\b", re.IGNORECASE),
}

# Also check uppercase tickers for coins with short names
UPPER_REGEXES = {
    "LINKUSDT": re.compile(r"\bLINK\b"),
    "ATOMUSDT": re.compile(r"\bATOM\b"),
    "NEARUSDT": re.compile(r"\bNEAR\b"),
    "DOTUSDT": re.compile(r"\bDOT\b"),
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_cryptobert_sequential(texts: list[str], batch_size: int = 32, max_length: int = 64) -> np.ndarray:
    """Step A: Load ONLY cryptobert, score texts, then free memory."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device()
    logger.info(f"Step A: Loading ElKulako/cryptobert on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ElKulako/cryptobert", use_safetensors=False
    ).to(device)
    model.eval()

    n = len(texts)
    scores = np.zeros(n, dtype=np.float32)
    logger.info(f"Scoring {n} headlines with CryptoBERT (batch_size={batch_size})...")

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = texts[start_idx:end_idx]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # cryptobert id2label: {0: 'Bearish', 1: 'Neutral', 2: 'Bullish'}
            s_crypto = probs[:, 2] - probs[:, 0]
            scores[start_idx:end_idx] = s_crypto

            if (start_idx // batch_size) % 65 == 0 or end_idx == n:
                print(f"[CryptoBERT] Progress: {end_idx}/{n} headlines scored...", flush=True)

    del model, tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    logger.info("Step A complete. CryptoBERT unloaded from memory.")
    return scores


def score_finbert_sequential(texts: list[str], batch_size: int = 32, max_length: int = 64) -> np.ndarray:
    """Step B: Load ONLY finbert, score texts, then free memory."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device()
    logger.info(f"Step B: Loading ProsusAI/finbert on {device}...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert", use_safetensors=False
    ).to(device)
    model.eval()

    n = len(texts)
    scores = np.zeros(n, dtype=np.float32)
    logger.info(f"Scoring {n} headlines with FinBERT (batch_size={batch_size})...")

    with torch.no_grad():
        for start_idx in range(0, n, batch_size):
            end_idx = min(start_idx + batch_size, n)
            batch = texts[start_idx:end_idx]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # finbert id2label: {0: 'positive', 1: 'negative', 2: 'neutral'}
            s_fin = probs[:, 0] - probs[:, 1]
            scores[start_idx:end_idx] = s_fin

            if (start_idx // batch_size) % 65 == 0 or end_idx == n:
                print(f"[FinBERT] Progress: {end_idx}/{n} headlines scored...", flush=True)

    del model, tokenizer
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    logger.info("Step B complete. FinBERT unloaded from memory.")
    return scores


def link_entities(df: pd.DataFrame) -> pd.DataFrame:
    """Link 15 assets + macro using regex patterns."""
    titles = df["title"].fillna("").tolist()
    descs = df["description"].fillna("").tolist()
    full_texts = [f"{t} {d}" for t, d in zip(titles, descs)]

    for asset in ASSETS_15:
        pat = ENTITY_REGEXES[asset]
        upper_pat = UPPER_REGEXES.get(asset, None)
        matches = []
        for raw in full_texts:
            m = bool(pat.search(raw))
            if not m and upper_pat:
                m = bool(upper_pat.search(raw))
            matches.append(m)
        df[f"entity_{asset}"] = matches
        logger.info(f"Entity count {asset}: {sum(matches)}")

    macro_pat = ENTITY_REGEXES["MACRO"]
    macro_matches = [bool(macro_pat.search(raw)) for raw in full_texts]
    df["entity_MACRO"] = macro_matches
    logger.info(f"Entity count MACRO: {sum(macro_matches)}")

    return df


def build_causal_features(df_news: pd.DataFrame, df_market: pd.DataFrame) -> pd.DataFrame:
    """
    Causally align news polarity onto 15-coin hourly panel:
    - Half-lives: 24h and 72h exponential decay
    - Bayesian neutral prior shrinkage: sum(w*s) / (sum(w) + 1.0)
    - Sentiment surprise: sent_24h - trailing_7d_mean(sent_24h) (strictly shift 1 bar)
    - Cross-sectional z-scores: rel_sent, rel_surprise across 15 coins at each hour
    """
    logger.info("Building causal rolling features across 15-coin panel...")
    df_market = df_market.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    df_news = df_news.sort_values("ingested_at").reset_index(drop=True)

    decay_24 = np.log(2.0) / 24.0
    decay_72 = np.log(2.0) / 72.0

    # Pre-allocate feature columns
    n_market = len(df_market)
    sent_24h = np.zeros(n_market, dtype=np.float32)
    sent_72h = np.zeros(n_market, dtype=np.float32)

    for asset in ASSETS_15:
        logger.info(f"Aligning news for {asset}...")
        asset_mask = df_news[f"entity_{asset}"].values
        macro_mask = df_news["entity_MACRO"].values
        relevant_mask = asset_mask | macro_mask

        sub_news = df_news[relevant_mask].copy().reset_index(drop=True)
        if len(sub_news) == 0:
            continue

        # Effective score: asset-specific uses s_crypto, macro uses 0.5*s_crypto + 0.5*s_fin
        is_asset = sub_news[f"entity_{asset}"].values
        s_c = sub_news["s_crypto"].values
        s_f = sub_news["s_fin"].values
        eff_scores = np.where(is_asset, s_c, 0.5 * s_c + 0.5 * s_f).astype(np.float32)
        sub_ts = sub_news["ingested_at"].values

        # Market panel rows for this asset
        mkt_asset_idx = np.where(df_market["asset"].values == asset)[0]
        bar_ts = df_market.loc[mkt_asset_idx, "timestamp"].values

        # Causal indices using searchsorted
        idx_now = np.searchsorted(sub_ts, bar_ts, side="right")
        idx_72h = np.searchsorted(sub_ts, bar_ts - np.timedelta64(72, "h"), side="right")

        # Fast calculation for each bar
        for i in range(len(bar_ts)):
            e = idx_now[i]
            s = idx_72h[i]
            if e > s:
                # News items in window
                t_news = sub_ts[s:e]
                scores = eff_scores[s:e]
                # delta in hours
                dt = (bar_ts[i] - t_news).astype("timedelta64[s]").astype(np.float64) / 3600.0
                dt = np.maximum(dt, 0.0)

                w24 = np.exp(-decay_24 * dt)
                w72 = np.exp(-decay_72 * dt)

                sum_w24 = np.sum(w24)
                sum_w72 = np.sum(w72)

                # Prior shrinkage of +1.0 neutral weight
                sent_24h[mkt_asset_idx[i]] = np.sum(w24 * scores) / (sum_w24 + 1.0)
                sent_72h[mkt_asset_idx[i]] = np.sum(w72 * scores) / (sum_w72 + 1.0)

    df_market["sent_24h"] = sent_24h
    df_market["sent_72h"] = sent_72h

    logger.info("Computing sentiment surprise (strictly trailing 7d mean, shifted by 1)...")
    df_market = df_market.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    # Trailing 7-day (168-hour) mean strictly shifted by 1
    df_market["sent_7d_trailing"] = (
        df_market.groupby("asset")["sent_24h"]
        .transform(lambda s: s.shift(1).rolling(168, min_periods=24).mean())
        .fillna(0.0)
    )
    df_market["sent_surprise"] = df_market["sent_24h"] - df_market["sent_7d_trailing"]

    logger.info("Computing cross-sectional relative sentiment (rel_sent, rel_surprise)...")
    # Group by timestamp across the 15 assets
    time_grouped = df_market.groupby("timestamp")

    mean_sent = time_grouped["sent_24h"].transform("mean")
    std_sent = time_grouped["sent_24h"].transform("std").fillna(1.0)
    std_sent = np.where(std_sent < 1e-4, 1.0, std_sent)
    df_market["rel_sent"] = (df_market["sent_24h"] - mean_sent) / std_sent

    mean_surp = time_grouped["sent_surprise"].transform("mean")
    std_surp = time_grouped["sent_surprise"].transform("std").fillna(1.0)
    std_surp = np.where(std_surp < 1e-4, 1.0, std_surp)
    df_market["rel_surprise"] = (df_market["sent_surprise"] - mean_surp) / std_surp

    # Compute forward returns for IC and backtests
    logger.info("Computing forward asset and cross-sectional returns (1h, 24h, 48h, 72h)...")
    df_market = df_market.sort_values(["asset", "timestamp"]).reset_index(drop=True)
    for h in [1, 12, 24, 48, 72]:
        df_market[f"fwd_ret_{h}h"] = (
            df_market.groupby("asset")["close"]
            .transform(lambda s, h=h: s.shift(-h) / s - 1.0)
        )

    # Cross-sectional excess forward returns
    time_grouped_rets = df_market.groupby("timestamp")
    for h in [24, 48, 72]:
        mean_fwd = time_grouped_rets[f"fwd_ret_{h}h"].transform("mean")
        df_market[f"xs_ret_{h}h"] = df_market[f"fwd_ret_{h}h"] - mean_fwd

    # Sort cleanly by timestamp then asset
    df_market = df_market.sort_values(["timestamp", "asset"]).reset_index(drop=True)
    return df_market


def main():
    news_parquet = DATA_DIR / "news_dedup.parquet"
    if not news_parquet.exists():
        raise FileNotFoundError(f"Missing {news_parquet}")

    df_news = pd.read_parquet(news_parquet)
    logger.info(f"Loaded {len(df_news)} news clusters from {news_parquet}")

    # Use title for polarity scoring
    headlines = df_news["title"].fillna("").astype(str).tolist()

    # Step A: CryptoBERT
    s_crypto = score_cryptobert_sequential(headlines, batch_size=32, max_length=64)
    df_news["s_crypto"] = s_crypto

    # Step B: FinBERT
    s_fin = score_finbert_sequential(headlines, batch_size=32, max_length=64)
    df_news["s_fin"] = s_fin

    # Step C: Entity Linking
    df_news = link_entities(df_news)

    # Save news polarity parquet
    news_out = DATA_DIR / "news_polarity_15.parquet"
    df_news.to_parquet(news_out, index=False)
    logger.info(f"Saved scored news to {news_out} (shape: {df_news.shape})")

    # Step D: Align onto universe_15_hourly
    mkt_parquet = DATA_DIR / "universe_15_hourly.parquet"
    if not mkt_parquet.exists():
        raise FileNotFoundError(f"Missing {mkt_parquet}")

    df_mkt = pd.read_parquet(mkt_parquet)
    logger.info(f"Loaded {len(df_mkt)} market bars from {mkt_parquet}")

    df_features = build_causal_features(df_news, df_mkt)

    feat_out = DATA_DIR / "universe_15_features.parquet"
    df_features.to_parquet(feat_out, index=False)
    logger.info(f"Saved feature panel to {feat_out} (shape: {df_features.shape})")
    print("ALL DONE! Polarity scoring and feature alignment complete.", flush=True)


if __name__ == "__main__":
    main()
