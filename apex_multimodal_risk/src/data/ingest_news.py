"""
Historical News Ingestion Module.
Downloads ExponentialScience/DLT-Sentiment-News from Hugging Face,
normalizes schema, enforces UTC timestamps, simulates causal 120s ingestion delay,
and writes to Parquet via DuckDB.
"""

import os
import json
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
from datasets import load_dataset
import duckdb

from apex_multimodal_risk.src.utils import setup_logger, load_config, PROJECT_ROOT

logger = setup_logger("ingest_news")


def extract_root_domain(url: Optional[str]) -> str:
    """Extracts lowercase root domain from source_url, stripping www."""
    if not url or not isinstance(url, str):
        return ""
    try:
        netloc = urlparse(url).netloc.lower().strip()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


def ingest_historical_news(config_path: Optional[str] = None) -> pd.DataFrame:
    """
    Ingests and normalizes the DLT-Sentiment-News dataset.
    Returns normalized DataFrame and writes raw Parquet via DuckDB.
    """
    config = load_config(config_path)
    ingest_cfg = config["news_ingestion"]
    dataset_name = ingest_cfg["dataset_name"]
    causal_delay_sec = ingest_cfg.get("causal_delay_seconds", 120)
    raw_parquet_path = ingest_cfg["raw_parquet_path"]
    
    if not os.path.isabs(raw_parquet_path):
        raw_parquet_path = str(PROJECT_ROOT / raw_parquet_path)
        
    os.makedirs(os.path.dirname(raw_parquet_path), exist_ok=True)
    
    logger.info(f"Downloading dataset '{dataset_name}' from Hugging Face Hub...")
    hf_ds = load_dataset(dataset_name)["train"]
    logger.info(f"Loaded {len(hf_ds):,} raw records from Hugging Face.")
    
    # Extract exact requested columns
    required_cols = [
        "timestamp",
        "title",
        "description",
        "source_url",
        "market_direction",
        "content_characteristics",
        "vote_counts",
        "total_votes",
    ]
    
    records = []
    for item in hf_ds:
        row = {col: item.get(col) for col in required_cols}
        records.append(row)
        
    df = pd.DataFrame(records)
    
    # Parse timestamp strictly as UTC
    logger.info("Enforcing UTC timestamps and applying causal 120s ingestion delay...")
    df["published_at"] = pd.to_datetime(df["timestamp"], utc=True)
    df["ingested_at"] = df["published_at"] + pd.Timedelta(seconds=causal_delay_sec)
    
    # Extract domain
    df["domain"] = df["source_url"].apply(extract_root_domain)
    
    # Normalize vote_counts as structured fields & string for DuckDB
    votes_df = pd.DataFrame(list(df["vote_counts"]))
    df["votes_bearish"] = votes_df["bearish"].fillna(0).astype(int)
    df["votes_bullish"] = votes_df["bullish"].fillna(0).astype(int)
    df["votes_important"] = votes_df["important"].fillna(0).astype(int)
    df["votes_liked"] = votes_df["liked"].fillna(0).astype(int)
    df["votes_disliked"] = votes_df["disliked"].fillna(0).astype(int)
    df["vote_counts_json"] = df["vote_counts"].apply(lambda v: json.dumps(v) if isinstance(v, dict) else str(v))
    
    # Ensure titles and descriptions are strings
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    
    # Chronological sort strictly by published_at
    df = df.sort_values(by="published_at", ascending=True).reset_index(drop=True)
    
    # Persist via DuckDB
    logger.info(f"Persisting {len(df):,} normalized records to '{raw_parquet_path}' using DuckDB...")
    con = duckdb.connect()
    con.register("raw_news_df", df)
    con.execute(f"COPY raw_news_df TO '{raw_parquet_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.close()
    
    logger.info(
        f"Raw news ingestion complete: {len(df):,} rows spanning "
        f"{df['published_at'].min()} to {df['published_at'].max()} saved."
    )
    return df


if __name__ == "__main__":
    ingest_historical_news()
